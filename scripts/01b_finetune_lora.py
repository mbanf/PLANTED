#!/usr/bin/env python3
"""
Fine-tune Botanic-1-S with LoRA on wheat gene expression prediction.

Stage 1 of the ConditionEXP v2 pipeline:
  1. Load Botanic-1-S + LoRA adapters
  2. Train: sequence → mean-pool → linear head → log1p(TPM)
  3. Save LoRA adapter weights
  4. Re-extract embeddings with fine-tuned model

Usage:
    python scripts/01b_finetune_lora.py \
        --seq data/zenodo/DeepExp/data/seq.tsv \
        --tpm data/zenodo/DeepExp/data/spike.TPM data/zenodo/DeepExp/data/leaf.TPM \
        --tissue spike leaf \
        --out data/lora_adapted \
        --device cuda
"""
import argparse, os, sys, time, logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

log = logging.getLogger("lora_finetune")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seq", required=True, help="seq.tsv (gene_id, sequence)")
    p.add_argument("--tpm", nargs="+", required=True, help="TPM files per tissue")
    p.add_argument("--tissue", nargs="+", required=True, help="Tissue names")
    p.add_argument("--model", default="living-models/Botanic1-S")
    p.add_argument("--out", default="data/lora_adapted")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16,
                   help="Gradient accumulation steps (effective batch = batch-size * grad-accum)")
    p.add_argument("--max-seq-len", type=int, default=2048,
                   help="Max sequence length for tokenization (reduces Mamba2 memory)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-train", type=int, default=50000,
                   help="Max training samples (subset for speed)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--emb-batch-size", type=int, default=64,
                   help="Batch size for embedding extraction (no gradients)")
    return p.parse_args()


def get_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_hidden_states(model, input_ids, attention_mask=None):
    """Extract hidden states from model, handling PEFT-wrapped Botanic-1-S.

    PEFT wraps the full model and returns (loss, logits) as a tuple,
    losing last_hidden_state. We use a forward hook on post_encoder_norm
    to capture the hidden states before the LM head.
    """
    hidden = {}

    def hook_fn(module, inp, out):
        hidden["h"] = out

    # Find post_encoder_norm in the model hierarchy
    target = None
    for name, mod in model.named_modules():
        if "post_encoder_norm" in name:
            target = mod
            break

    if target is not None:
        handle = target.register_forward_hook(hook_fn)
        try:
            model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            handle.remove()
        return hidden["h"]

    # Fallback for non-Botanic models
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state
    return outputs[0]


class SeqExprDataset(Dataset):
    """Dataset of (sequence_string, log1p_tpm, tissue_id) tuples."""
    def __init__(self, sequences, labels, tissue_ids):
        self.sequences = sequences
        self.labels = labels
        self.tissue_ids = tissue_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.tissue_ids[idx]


def collate_fn(batch, tokenizer, max_length=4096):
    """Tokenize sequences and stack labels."""
    seqs, labels, tids = zip(*batch)
    encoded = tokenizer(
        list(seqs), return_tensors="pt", padding=True,
        truncation=True, max_length=max_length,
    )
    labels = torch.tensor(labels, dtype=torch.float32)
    tids = torch.tensor(tids, dtype=torch.long)
    return encoded, labels, tids


def apply_lora(model, rank=16, alpha=32):
    """Apply LoRA adapters to linear layers in the model."""
    from peft import LoraConfig, get_peft_model, TaskType

    # Find target modules — look for linear layers in the backbone
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(
            k in name for k in ["in_proj", "out_proj", "fc1", "fc2",
                                "linear", "dense", "query", "key", "value",
                                "q_proj", "k_proj", "v_proj", "o_proj"]
        ):
            target_modules.append(name)

    # If no specific matches, target all Linear layers with 'proj' or 'linear'
    if not target_modules:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.weight.shape[0] >= 256:
                target_modules.append(name)

    log.info(f"LoRA targets: {len(target_modules)} modules")
    if target_modules:
        log.info(f"  Examples: {target_modules[:5]}")

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules[:50],  # cap to avoid memory issues
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


class ExpressionHead(nn.Module):
    """Simple head: mean-pooled embedding → expression."""
    def __init__(self, embed_dim, n_tissues):
        super().__init__()
        self.tissue_emb = nn.Embedding(n_tissues, 64)
        self.head = nn.Sequential(
            nn.Linear(embed_dim + 64, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, pooled, tissue_id):
        t = self.tissue_emb(tissue_id)
        x = torch.cat([pooled, t], dim=1)
        return self.head(x).squeeze(-1)


def extract_windows(sequence):
    """Extract TSS (2000bp) window from 10kb sequence for LoRA training."""
    seq = sequence.upper()
    if len(seq) < 8200:
        seq = seq + "N" * (10000 - len(seq))
    # TSS ±1000bp = positions 2000-4000 (2000bp) — shorter for memory
    return seq[2000:4000]


def main():
    args = parse_args()
    device = get_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    # === Load sequences ===
    log.info(f"Loading sequences from {args.seq}")
    seq_df = pd.read_csv(args.seq, sep="\t", dtype=str)
    gene_seqs = {row.gene_id: extract_windows(row.sequence)
                 for _, row in seq_df.iterrows()}
    log.info(f"  {len(gene_seqs)} genes loaded")

    # === Load TPM labels per tissue ===
    all_seqs, all_labels, all_tissue_ids = [], [], []
    for i, (tpm_path, tissue) in enumerate(zip(args.tpm, args.tissue)):
        tpm_df = pd.read_csv(tpm_path, sep="\t", header=0, dtype=str)
        tpm_df.columns = ["gene_id", "TPM"]
        tpm_df["TPM"] = tpm_df["TPM"].astype(float)

        for _, row in tpm_df.iterrows():
            if row.gene_id in gene_seqs:
                all_seqs.append(gene_seqs[row.gene_id])
                all_labels.append(np.log1p(row.TPM))
                all_tissue_ids.append(i)

    log.info(f"Total samples: {len(all_seqs)} across {len(args.tissue)} tissues")

    # === Subsample for training speed ===
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(all_seqs))
    n_train = min(args.max_train, int(len(idx) * 0.8))
    n_val = min(10000, len(idx) - n_train)

    train_seqs = [all_seqs[j] for j in idx[:n_train]]
    train_labels = [all_labels[j] for j in idx[:n_train]]
    train_tids = [all_tissue_ids[j] for j in idx[:n_train]]
    val_seqs = [all_seqs[j] for j in idx[n_train:n_train + n_val]]
    val_labels = [all_labels[j] for j in idx[n_train:n_train + n_val]]
    val_tids = [all_tissue_ids[j] for j in idx[n_train:n_train + n_val]]

    del all_seqs, all_labels, all_tissue_ids  # free memory

    log.info(f"Training on {n_train} samples, validating on {n_val}")

    # === Load model + LoRA ===
    from transformers import AutoModel, AutoTokenizer

    log.info(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # Load in bfloat16 to halve memory footprint (318M params: 1.2GB → 0.6GB)
    load_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    backbone = AutoModel.from_pretrained(args.model, trust_remote_code=True,
                                         torch_dtype=load_dtype)
    backbone = backbone.to(device)

    n_total = sum(p.numel() for p in backbone.parameters())
    log.info(f"Backbone: {n_total/1e6:.0f}M params")

    # Apply LoRA
    backbone = apply_lora(backbone, rank=args.lora_rank, alpha=args.lora_alpha)

    # Note: Botanic1 does not support gradient_checkpointing_enable()

    # Get embed dim from a test forward pass
    test_enc = tokenizer(["ACGT" * 10], return_tensors="pt", padding=True)
    with torch.no_grad():
        test_hidden = get_hidden_states(backbone, test_enc["input_ids"].to(device))
        embed_dim = test_hidden.shape[-1]
    log.info(f"Embed dim: {embed_dim}")

    # Expression head
    head = ExpressionHead(embed_dim, len(args.tissue)).to(device)

    # === Training ===
    from functools import partial
    collate = partial(collate_fn, tokenizer=tokenizer, max_length=args.max_seq_len)

    train_ds = SeqExprDataset(train_seqs, train_labels, train_tids)
    val_ds = SeqExprDataset(val_seqs, val_labels, val_tids)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2,
                            collate_fn=collate, num_workers=0)

    # Only LoRA params + head are trainable
    trainable_params = [p for p in backbone.parameters() if p.requires_grad]
    trainable_params += list(head.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Mixed precision for memory efficiency
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    log.info(f"Mixed precision: {use_amp} (dtype={amp_dtype})")
    log.info(f"Effective batch size: {args.batch_size} × {args.grad_accum} = {args.batch_size * args.grad_accum}")

    log.info(f"\n=== LoRA Fine-tuning ({args.epochs} epochs) ===")
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # Train
        backbone.train()
        head.train()
        total_loss, n = 0.0, 0
        t0 = time.time()
        optimizer.zero_grad()

        for batch_idx, (encoded, labels, tissue_ids) in enumerate(train_loader):
            input_ids = encoded["input_ids"].to(device)
            attn_mask = encoded.get("attention_mask")
            if attn_mask is not None:
                attn_mask = attn_mask.to(device)
            labels = labels.to(device)
            tissue_ids = tissue_ids.to(device)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                hidden = get_hidden_states(backbone, input_ids, attn_mask)

                if attn_mask is not None:
                    mask = attn_mask.unsqueeze(-1).float()
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                else:
                    pooled = hidden.mean(dim=1)

                pred = head(pooled, tissue_ids)
                loss = F.mse_loss(pred, labels) / args.grad_accum

            scaler.scale(loss).backward()

            if (batch_idx + 1) % args.grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * args.grad_accum * len(labels)
            n += len(labels)

            if (batch_idx + 1) % 100 == 0:
                log.info(f"  Epoch {epoch} batch {batch_idx+1}/{len(train_loader)} "
                         f"loss={total_loss/n:.4f}")

        scheduler.step()
        train_loss = total_loss / n
        elapsed = time.time() - t0

        # Validate
        backbone.eval()
        head.eval()
        val_loss, vn = 0.0, 0
        with torch.no_grad():
            for encoded, labels, tissue_ids in val_loader:
                input_ids = encoded["input_ids"].to(device)
                attn_mask = encoded.get("attention_mask")
                if attn_mask is not None:
                    attn_mask = attn_mask.to(device)
                labels = labels.to(device)
                tissue_ids = tissue_ids.to(device)

                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    hidden = get_hidden_states(backbone, input_ids, attn_mask)
                    if attn_mask is not None:
                        mask = attn_mask.unsqueeze(-1).float()
                        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                    else:
                        pooled = hidden.mean(dim=1)

                    pred = head(pooled, tissue_ids)
                    val_loss += F.mse_loss(pred, labels).item() * len(labels)
                vn += len(labels)

        val_loss /= vn
        log.info(f"Epoch {epoch} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | {elapsed:.0f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            backbone.save_pretrained(os.path.join(args.out, "lora_adapter"))
            torch.save(head.state_dict(), os.path.join(args.out, "expression_head.pt"))
            log.info(f"  Saved best adapter (val_loss={val_loss:.4f})")

    log.info(f"\n=== LoRA fine-tuning complete. Best val_loss: {best_val_loss:.4f} ===")

    # === Stage 2: Re-extract embeddings with fine-tuned model ===
    log.info("\n=== Re-extracting embeddings with LoRA-adapted Botanic-1-S ===")

    # Reload best adapter
    from peft import PeftModel
    base_model = AutoModel.from_pretrained(args.model, trust_remote_code=True).to(device)
    model = PeftModel.from_pretrained(base_model, os.path.join(args.out, "lora_adapter"))
    model.eval()

    # Extract TSS + TTS embeddings for all genes
    gene_ids = seq_df["gene_id"].values
    tss_seqs = [extract_windows(s) for s in seq_df["sequence"]]
    tts_seqs = []
    for s in seq_df["sequence"]:
        s = s.upper()
        if len(s) < 8200:
            s = s + "N" * (10000 - len(s))
        tts_seqs.append(s[7500:8200])

    emb_bs = args.emb_batch_size

    def embed_all(sequences, desc):
        all_embs = []
        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), emb_bs), desc=desc):
                batch = sequences[i:i + emb_bs]
                encoded = tokenizer(
                    batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=8192,
                )
                input_ids = encoded["input_ids"].to(device)
                attn_mask = encoded.get("attention_mask")
                if attn_mask is not None:
                    attn_mask = attn_mask.to(device)

                hidden = get_hidden_states(model, input_ids, attn_mask)

                if attn_mask is not None:
                    mask = attn_mask.unsqueeze(-1).float()
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                else:
                    pooled = hidden.mean(dim=1)

                all_embs.append(pooled.cpu().float().numpy())
        return np.concatenate(all_embs)

    log.info(f"Embedding TSS windows ({len(tss_seqs)} × 3500bp)...")
    t0 = time.time()
    tss_embs = embed_all(tss_seqs, "TSS")
    log.info(f"TSS done: {tss_embs.shape}, {time.time()-t0:.0f}s")

    log.info(f"Embedding TTS windows ({len(tts_seqs)} × 700bp)...")
    t0 = time.time()
    tts_embs = embed_all(tts_seqs, "TTS")
    log.info(f"TTS done: {tts_embs.shape}, {time.time()-t0:.0f}s")

    # Save
    emb_path = os.path.join(args.out, "all_botanic1s_lora.npz")
    np.savez_compressed(
        emb_path,
        gene_ids=gene_ids,
        tss_emb=tss_embs,
        tts_emb=tts_embs,
        embed_dim=tss_embs.shape[1],
    )
    size_mb = os.path.getsize(emb_path) / 1e6
    log.info(f"Saved LoRA embeddings: {emb_path} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
