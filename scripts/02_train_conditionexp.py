#!/usr/bin/env python3
"""
ConditionEXP: Tissue-specific gene expression prediction via foundation model
conditioning on epigenomic state.

Trains on cached Botanic-1-S embeddings + epigenomic features with FiLM
conditioning on tissue type. Directly comparable to DeepWheat (Ma et al., 2025).

Ablation variants:
  A: Botanic-1-S embeddings + epi features, simple concatenation (DeepWheat-style)
  B: + cross-attention between sequence and epi features
  C: + FiLM conditioning on tissue token
  D: + contrastive auxiliary loss between tissues

Usage:
    python scripts/02_train_conditionexp.py \
        --emb data/embeddings/spike_botanic1s.npz \
        --epi data/processed/spike.epi.tsv \
        --tpm data/processed/spike.logp_TPM \
        --test-genes data/processed/4700_gene.txt \
        --tissue spike \
        --variant C \
        --out results/

Multi-tissue training (variants C/D):
    python scripts/02_train_conditionexp.py \
        --emb data/embeddings/spike_botanic1s.npz data/embeddings/leaf_botanic1s.npz \
        --epi data/processed/spike.epi.tsv data/processed/leaf.epi.tsv \
        --tpm data/processed/spike.logp_TPM data/processed/leaf.logp_TPM \
        --tissue spike leaf \
        --variant D \
        --out results/
"""
import argparse, json, logging, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="ConditionEXP training")
    p.add_argument("--emb", nargs="+", required=True, help="Botanic-1-S embedding .npz files")
    p.add_argument("--epi", nargs="+", required=True, help="Epigenomic TSV files")
    p.add_argument("--tpm", nargs="+", required=True, help="log1p(TPM) files")
    p.add_argument("--test-genes", default="", help="Held-out test gene list")
    p.add_argument("--tissue", nargs="+", required=True, help="Tissue names (one per file)")
    p.add_argument("--variant", choices=["A", "B", "C", "D", "E"], default="C",
                   help="Model variant: A=concat, B=+cross-attn, C=+FiLM, D=+contrastive, E=epi-only")
    p.add_argument("--out", default="results/", help="Output directory")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--warmup-epochs", type=int, default=5)
    # Train/val split: use fraction of data (0.0 = use n-train/n-val absolute counts)
    p.add_argument("--train-frac", type=float, default=0.8,
                   help="Fraction of non-test2 data for training (0.8 = 80%%)")
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="Fraction for validation (rest goes to test)")
    # Legacy absolute sizes (used only if train-frac <= 0)
    p.add_argument("--n-train", type=int, default=70000)
    p.add_argument("--n-val", type=int, default=22000)
    return p.parse_args()


# ---------------------------------------------------------------------------
# 2. Dataset
# ---------------------------------------------------------------------------
class ExpressionDataset(Dataset):
    def __init__(self, seq_emb, epi_feat, labels, tissue_ids):
        self.seq_emb = torch.from_numpy(seq_emb).float()
        self.epi_feat = torch.from_numpy(epi_feat.astype(np.float16)).half()  # keep as float16
        self.labels = torch.from_numpy(labels).float()
        self.tissue_ids = torch.from_numpy(tissue_ids).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.seq_emb[idx], self.epi_feat[idx].float(), self.labels[idx], self.tissue_ids[idx]


# ---------------------------------------------------------------------------
# 3. Model components
# ---------------------------------------------------------------------------
class EpigenomicEncoder(nn.Module):
    """Conv1D encoder for epigenomic signals (5 marks × ~4200 positions)."""

    def __init__(self, n_marks=5, n_positions=4200, hidden=128, out_dim=512):
        super().__init__()
        # Reshape input: (B, n_marks * n_positions) → (B, n_marks, n_positions)
        self.n_marks = n_marks
        self.n_positions = n_positions
        self.conv = nn.Sequential(
            nn.Conv1d(n_marks, hidden, kernel_size=15, padding=7),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Conv1d(hidden, hidden, kernel_size=11, padding=5),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Conv1d(hidden, hidden, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),  # → (B, hidden, 1)
        )
        self.fc = nn.Linear(hidden, out_dim)

    def forward(self, x):
        # x: (B, n_marks * n_positions)
        B = x.shape[0]
        x = x.view(B, self.n_marks, self.n_positions)
        x = self.conv(x).squeeze(-1)  # (B, hidden)
        return self.fc(x)  # (B, out_dim)


class EpigenomicTokenEncoder(nn.Module):
    """Encode epi signals into token-level features for cross-attention."""

    def __init__(self, n_marks=5, n_positions=4200, hidden=128, n_tokens=64):
        super().__init__()
        self.n_marks = n_marks
        self.n_positions = n_positions
        self.n_tokens = n_tokens
        self.conv = nn.Sequential(
            nn.Conv1d(n_marks, hidden, kernel_size=15, padding=7),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(n_tokens),  # → (B, hidden, n_tokens)
        )

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, self.n_marks, self.n_positions)
        x = self.conv(x)  # (B, hidden, n_tokens)
        return x.transpose(1, 2)  # (B, n_tokens, hidden)


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: γ * x + β, conditioned on tissue."""

    def __init__(self, n_tissues, feature_dim, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Embedding(n_tissues, hidden),
        )
        self.gamma_fc = nn.Linear(hidden, feature_dim)
        self.beta_fc = nn.Linear(hidden, feature_dim)
        # Initialize near identity: γ≈1, β≈0
        nn.init.ones_(self.gamma_fc.bias)
        nn.init.zeros_(self.gamma_fc.weight)
        nn.init.zeros_(self.beta_fc.bias)
        nn.init.zeros_(self.beta_fc.weight)

    def forward(self, x, tissue_id):
        emb = self.mlp(tissue_id)  # (B, hidden)
        gamma = self.gamma_fc(emb)  # (B, D)
        beta = self.beta_fc(emb)    # (B, D)
        return gamma * x + beta


class CrossAttentionBlock(nn.Module):
    """Cross-attention: sequence queries attend to epigenomic keys/values."""

    def __init__(self, seq_dim, epi_dim, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=seq_dim, num_heads=n_heads,
            kdim=epi_dim, vdim=epi_dim, batch_first=True,
        )
        self.norm = nn.LayerNorm(seq_dim)

    def forward(self, seq_emb, epi_tokens):
        # seq_emb: (B, 1, seq_dim) — treat pooled embedding as single query
        # epi_tokens: (B, N, epi_dim)
        attended, _ = self.attn(seq_emb, epi_tokens, epi_tokens)
        return self.norm(seq_emb + attended)


# ---------------------------------------------------------------------------
# 4. Full model variants
# ---------------------------------------------------------------------------
class ConditionEXP(nn.Module):
    """
    ConditionEXP: foundation model embeddings + epigenomic integration + FiLM.

    Variants:
      A: concat(seq_emb, epi_emb) → MLP → expression
      B: cross-attn(seq, epi_tokens) → concat → MLP → expression
      C: B + FiLM conditioning on tissue
      D: C + contrastive loss (handled in training loop)
      E: epi-only baseline (no sequence embeddings)
    """

    def __init__(self, seq_dim, epi_input_dim, n_marks, n_positions,
                 n_tissues, variant="C", epi_hidden=128, epi_out=512):
        super().__init__()
        self.variant = variant
        self.seq_dim = seq_dim
        self.n_tissues = n_tissues

        # Epigenomic encoder (pooled, for concat)
        self.epi_encoder = EpigenomicEncoder(
            n_marks=n_marks, n_positions=n_positions,
            hidden=epi_hidden, out_dim=epi_out,
        )

        # Cross-attention components (variants B, C, D)
        if variant in ("B", "C", "D"):
            self.epi_token_encoder = EpigenomicTokenEncoder(
                n_marks=n_marks, n_positions=n_positions,
                hidden=epi_hidden, n_tokens=64,
            )
            self.cross_attn = CrossAttentionBlock(
                seq_dim=seq_dim, epi_dim=epi_hidden, n_heads=4,
            )

        # Combined dimension: epi-only (E) vs seq+epi
        combined_dim = epi_out if variant == "E" else seq_dim + epi_out

        # FiLM conditioning (variants C, D, E)
        if variant in ("C", "D", "E"):
            self.film = FiLMLayer(n_tissues, combined_dim)

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, seq_emb, epi_feat, tissue_id):
        """
        seq_emb:   (B, seq_dim) — pre-computed Botanic-1-S embedding
        epi_feat:  (B, n_marks * n_positions) — raw epigenomic signal
        tissue_id: (B,) — tissue index
        """
        # Epigenomic pooled embedding
        epi_emb = self.epi_encoder(epi_feat)  # (B, epi_out)

        if self.variant == "E":
            # Epi-only baseline — no sequence information
            combined = epi_emb
        else:
            # Cross-attention (variants B, C, D)
            if self.variant in ("B", "C", "D"):
                epi_tokens = self.epi_token_encoder(epi_feat)  # (B, 64, epi_hidden)
                seq_query = seq_emb.unsqueeze(1)  # (B, 1, seq_dim)
                attended = self.cross_attn(seq_query, epi_tokens)  # (B, 1, seq_dim)
                seq_emb = attended.squeeze(1)  # (B, seq_dim) — enriched with epi info
            # Concatenate seq + epi
            combined = torch.cat([seq_emb, epi_emb], dim=1)  # (B, seq_dim + epi_out)

        # FiLM conditioning (variants C, D, E)
        if self.variant in ("C", "D", "E"):
            combined = self.film(combined, tissue_id)

        # Predict expression
        return self.head(combined).squeeze(-1)  # (B,)

    def get_embedding(self, seq_emb, epi_feat, tissue_id):
        """Return internal representation (for contrastive loss, variant D)."""
        epi_emb = self.epi_encoder(epi_feat)
        if self.variant in ("B", "C", "D"):
            epi_tokens = self.epi_token_encoder(epi_feat)
            seq_query = seq_emb.unsqueeze(1)
            attended = self.cross_attn(seq_query, epi_tokens)
            seq_emb = attended.squeeze(1)
        combined = torch.cat([seq_emb, epi_emb], dim=1)
        if self.variant in ("C", "D"):
            combined = self.film(combined, tissue_id)
        return combined


# ---------------------------------------------------------------------------
# 5. Data loading helpers
# ---------------------------------------------------------------------------
def _get_epi_column_indices():
    """Return column indices for DeepWheat epi positions.
    TSS: 1000-4500 + TTS: 7500-8200 = 4200 positions per mark, 5 marks = 21000."""
    ATAC = np.concatenate((np.arange(1000, 4500), np.arange(7500, 8200)))
    K27a = np.concatenate((np.arange(11000, 14500), np.arange(17500, 18200)))
    K27  = np.concatenate((np.arange(21000, 24500), np.arange(27500, 28200)))
    K36  = np.concatenate((np.arange(31000, 34500), np.arange(37500, 38200)))
    K4   = np.concatenate((np.arange(41000, 44500), np.arange(47500, 48200)))
    return np.concatenate((ATAC, K27a, K27, K36, K4))


def load_data(args):
    """Load and merge multi-tissue data into unified arrays.

    Supports two data layouts:
      1. Zenodo H5 format: --epi points to .epi.h5 files (shape N×50000)
      2. Processed TSV format: --epi points to .epi.tsv files (comma-separated)
    """
    import h5py

    all_seq_emb, all_epi, all_labels, all_tissue_ids, all_gene_ids = [], [], [], [], []

    # Load shared gene list (for Zenodo layout)
    zenodo_dir = None
    gene_list = None
    if args.epi[0].endswith(".h5"):
        zenodo_dir = os.path.dirname(args.epi[0])
        gene_list_path = os.path.join(zenodo_dir, "gene.list.txt")
        if os.path.exists(gene_list_path):
            gene_list = pd.read_csv(gene_list_path, header=None)[0].tolist()

    # Pre-compute epi column indices for selective H5 reading
    epi_col_idx = _get_epi_column_indices()

    # Load embeddings once (same file for all tissues)
    emb_path_0 = args.emb[0]
    log.info(f"Loading shared embeddings from {emb_path_0}")
    emb_data = np.load(emb_path_0, allow_pickle=True)
    emb_gene_ids = emb_data["gene_ids"]
    seq_emb_all = np.concatenate([emb_data["tss_emb"], emb_data["tts_emb"]], axis=1)
    del emb_data

    for i, (emb_path, epi_path, tpm_path, tissue) in enumerate(
        zip(args.emb, args.epi, args.tpm, args.tissue)
    ):
        log.info(f"Loading tissue '{tissue}'")
        seq_emb = seq_emb_all  # shared reference

        # Epigenomic features — H5 or TSV
        if epi_path.endswith(".h5"):
            with h5py.File(epi_path, "r") as f:
                epi_raw = f["dataset_1"][:, epi_col_idx].astype(np.float16)  # read only needed cols
            n_h5_genes = epi_raw.shape[0]
            epi_gene_ids = gene_list[:n_h5_genes] if gene_list is not None else list(range(n_h5_genes))
        else:
            epi_df = pd.read_csv(epi_path, sep="\t", dtype=str)
            epi_cols = [c for c in epi_df.columns if c != "gene_id"]
            epi_gene_ids = epi_df["gene_id"].tolist()

        # Expression labels
        tpm_df = pd.read_csv(tpm_path, sep="\t", header=0, dtype=str)
        tpm_df.columns = ["gene_id", "TPM"]
        tpm_df["TPM"] = tpm_df["TPM"].astype(float)

        # Merge on gene_id
        emb_genes = set(emb_gene_ids)
        epi_genes_set = set(epi_gene_ids) if not isinstance(epi_gene_ids[0], int) else set(range(len(epi_gene_ids)))
        tpm_genes = set(tpm_df["gene_id"])
        common = sorted(emb_genes & epi_genes_set & tpm_genes)
        log.info(f"  {tissue}: {len(common)} genes with all three data types")

        # Build aligned arrays
        emb_gene_to_idx = {g: j for j, g in enumerate(emb_gene_ids)}
        tpm_df_indexed = tpm_df.set_index("gene_id")

        emb_idxs = [emb_gene_to_idx[g] for g in common]
        se = seq_emb[emb_idxs]

        if epi_path.endswith(".h5"):
            epi_gene_to_idx = {g: j for j, g in enumerate(epi_gene_ids)}
            epi_idxs = [epi_gene_to_idx[g] for g in common]
            ep = epi_raw[epi_idxs]  # already float16 from H5 read
        else:
            epi_df_indexed = epi_df.set_index("gene_id")
            epi_rows = []
            for g in common:
                row = epi_df_indexed.loc[g, epi_cols]
                vec = []
                for cell in row:
                    vec.extend(map(float, str(cell).split(",")))
                epi_rows.append(vec)
            ep = np.array(epi_rows, dtype=np.float32)

        lab = np.array([np.log1p(tpm_df_indexed.loc[g, "TPM"]) for g in common], dtype=np.float32)

        all_seq_emb.append(se)
        all_epi.append(ep)
        all_labels.append(lab)
        all_tissue_ids.append(np.full(len(common), i, dtype=np.int64))
        all_gene_ids.append(np.array(common))

    # Concatenate all tissues
    seq_emb = np.concatenate(all_seq_emb); del all_seq_emb
    epi = np.concatenate(all_epi); del all_epi
    labels = np.concatenate(all_labels); del all_labels
    tissue_ids = np.concatenate(all_tissue_ids)
    gene_ids = np.concatenate(all_gene_ids)

    # Normalize epi features (MinMax per feature, like DeepWheat)
    # Work in float16 to save memory, convert to float32 after
    epi_min = epi.min(axis=0, keepdims=True)
    epi_max = epi.max(axis=0, keepdims=True)
    epi_range = np.clip((epi_max - epi_min).astype(np.float32), 1e-8, None)
    epi = ((epi.astype(np.float32) - epi_min.astype(np.float32)) / epi_range).astype(np.float16)

    log.info(f"Total: {len(labels)} samples, seq_emb {seq_emb.shape}, epi {epi.shape}")

    n_marks = 5
    n_positions = epi.shape[1] // n_marks

    return {
        "seq_emb": seq_emb, "epi": epi, "labels": labels,
        "tissue_ids": tissue_ids, "gene_ids": gene_ids,
        "embed_dim": seq_emb.shape[1], "epi_dim": epi.shape[1],
        "n_marks": n_marks, "n_positions": n_positions,
        "n_tissues": len(args.tissue),
    }


def split_data(data, args):
    """Split into train/val/test + test2, matching DeepWheat's protocol."""
    N = len(data["labels"])

    # Test2 genes (held-out, if provided)
    test2_mask = np.zeros(N, dtype=bool)
    if args.test_genes and os.path.exists(args.test_genes):
        t2_genes = set(pd.read_csv(args.test_genes, header=None)[0].tolist())
        test2_mask = np.array([g in t2_genes for g in data["gene_ids"]])
        log.info(f"Test2 genes: {test2_mask.sum()}")

    # Remaining genes: shuffle and split
    rest_idx = np.where(~test2_mask)[0]
    rng = np.random.RandomState(args.seed)
    rng.shuffle(rest_idx)

    if args.train_frac > 0:
        n_tr = int(len(rest_idx) * args.train_frac)
        n_va = int(len(rest_idx) * args.val_frac)
    else:
        n_tr = min(args.n_train, len(rest_idx) - args.n_val - 1000)
        n_va = min(args.n_val, len(rest_idx) - n_tr - 1000)
    train_idx = rest_idx[:n_tr]
    val_idx = rest_idx[n_tr:n_tr + n_va]
    test_idx = rest_idx[n_tr + n_va:]
    test2_idx = np.where(test2_mask)[0]

    splits = {}
    for name, idx in [("train", train_idx), ("val", val_idx),
                       ("test", test_idx), ("test2", test2_idx)]:
        if len(idx) == 0:
            continue
        splits[name] = ExpressionDataset(
            data["seq_emb"][idx], data["epi"][idx],
            data["labels"][idx], data["tissue_ids"][idx],
        )
        log.info(f"  {name}: {len(idx)} samples")

    return splits


# ---------------------------------------------------------------------------
# 6. Training loop
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for seq, epi, lab, tid in loader:
        seq, epi, lab, tid = seq.to(device), epi.to(device), lab.to(device), tid.to(device)
        pred = model(seq, epi, tid)
        loss = F.mse_loss(pred, lab)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(lab)
        n += len(lab)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    for seq, epi, lab, tid in loader:
        seq, epi, tid = seq.to(device), epi.to(device), tid.to(device)
        pred = model(seq, epi, tid)
        preds.append(pred.cpu().numpy())
        trues.append(lab.numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    pcc, _ = pearsonr(trues, preds)
    r2 = r2_score(trues, preds)
    mse = float(np.mean((trues - preds) ** 2))
    # Also evaluate in TPM space
    true_tpm = np.expm1(trues)
    pred_tpm = np.expm1(np.clip(preds, -10, 20))  # safety clip
    pcc_tpm, _ = pearsonr(true_tpm, pred_tpm)
    r2_tpm = r2_score(true_tpm, pred_tpm)
    return {
        "mse": mse, "pcc_log": pcc, "r2_log": r2,
        "pcc_tpm": pcc_tpm, "r2_tpm": r2_tpm,
    }


def train(model, splits, args, device):
    train_loader = DataLoader(splits["train"], batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(splits["val"], batch_size=args.batch_size * 2,
                            num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs])

    best_pcc = -1
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        history.append({"epoch": epoch, "train_mse": train_loss, **val_metrics})
        log.info(f"Epoch {epoch:3d} | train_mse {train_loss:.4f} | "
                 f"val_pcc {val_metrics['pcc_log']:.4f} | val_r2 {val_metrics['r2_log']:.4f} | "
                 f"pcc_tpm {val_metrics['pcc_tpm']:.4f} | {elapsed:.1f}s")

        if val_metrics["pcc_log"] > best_pcc:
            best_pcc = val_metrics["pcc_log"]
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.out, "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log.info(f"Early stopping at epoch {epoch}")
                break

    return history


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cpu")  # default fallback
    if args.device != "auto":
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info(f"Device: {device}")

    # Load data
    data = load_data(args)
    splits = split_data(data, args)

    # Build model
    model = ConditionEXP(
        seq_dim=data["embed_dim"],
        epi_input_dim=data["epi_dim"],
        n_marks=data["n_marks"],
        n_positions=data["n_positions"],
        n_tissues=data["n_tissues"],
        variant=args.variant,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model variant {args.variant}: {n_params:,} trainable params")

    # Train
    history = train(model, splits, args, device)

    # Load best and evaluate on test sets
    model.load_state_dict(torch.load(os.path.join(args.out, "best_model.pt"),
                                      map_location=device, weights_only=True))

    results = {}
    for split_name in ["test", "test2"]:
        if split_name not in splits:
            continue
        loader = DataLoader(splits[split_name], batch_size=args.batch_size * 2,
                            num_workers=2, pin_memory=True)
        metrics = evaluate(model, loader, device)
        results[split_name] = metrics
        log.info(f"\n=== {split_name} results ===")
        log.info(f"  PCC (log1p): {metrics['pcc_log']:.4f}")
        log.info(f"  R²  (log1p): {metrics['r2_log']:.4f}")
        log.info(f"  PCC (TPM):   {metrics['pcc_tpm']:.4f}")
        log.info(f"  R²  (TPM):   {metrics['r2_tpm']:.4f}")

    # Save results
    summary = {
        "variant": args.variant,
        "tissues": args.tissue,
        "n_params": n_params,
        "best_val_pcc": float(max(h["pcc_log"] for h in history)),
        "results": results,
        "history": history,
    }
    out_json = os.path.join(args.out, f"conditionexp_{args.variant}_results.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    log.info(f"\nResults saved to {out_json}")

    # Comparison table
    log.info("\n" + "=" * 60)
    log.info("COMPARISON WITH DEEPWHEAT (Ma et al., Genome Biology 2025)")
    log.info("=" * 60)
    log.info(f"DeepWheat (Basenji2 + CNN concat):  PCC 0.82 - 0.88")
    log.info(f"DeepWheat (seq only, Basenji2):     PCC < 0.66")
    if "test" in results:
        log.info(f"ConditionEXP variant {args.variant}:       "
                 f"PCC {results['test']['pcc_log']:.3f} (log1p), "
                 f"{results['test']['pcc_tpm']:.3f} (TPM)")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
