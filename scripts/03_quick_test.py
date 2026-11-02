#!/usr/bin/env python3
"""
Quick end-to-end test: load Zenodo data, create fake embeddings, train 3 epochs.
Validates the full pipeline before running on GPU with real Botanic-1-S embeddings.

Usage:
    python scripts/03_quick_test.py
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import h5py
import torch
from scipy.stats import pearsonr

# Add scripts to path
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "scripts"))

from importlib.machinery import SourceFileLoader
train_mod = SourceFileLoader("train", os.path.join(PROJ, "scripts", "02_train_conditionexp.py")).load_module()

ZENODO = os.path.join(PROJ, "data", "zenodo", "DeepExp", "data")


def load_zenodo_tissue(tissue, max_genes=500):
    """Load one tissue from the Zenodo H5/TSV/TPM files."""
    print(f"\nLoading {tissue} (max {max_genes} genes)...")

    # Gene list
    genes = pd.read_csv(f"{ZENODO}/gene.list.txt", header=None)[0].tolist()
    n = min(max_genes, len(genes))
    genes = genes[:n]

    # Sequences (for embedding — here we just use random embeddings)
    # In real pipeline, 01_precompute_embeddings.py creates these
    embed_dim = 2048  # TSS (1024) + TTS (1024)
    fake_seq_emb = np.random.randn(n, embed_dim).astype(np.float32) * 0.1

    # Epigenomic features from H5
    with h5py.File(f"{ZENODO}/{tissue}.epi.h5", "r") as f:
        epi = f["dataset_1"][:n]  # (N, 50000)
    print(f"  Epi: {epi.shape}")

    # Subset to same positions as DeepWheat training code
    # TSS: 1000-4500 + TTS: 7500-8200 = 4200 positions per mark
    ATAC = np.concatenate((np.arange(1000, 4500), np.arange(7500, 8200)))
    K27a = np.concatenate((np.arange(11000, 14500), np.arange(17500, 18200)))
    K27  = np.concatenate((np.arange(21000, 24500), np.arange(27500, 28200)))
    K36  = np.concatenate((np.arange(31000, 34500), np.arange(37500, 38200)))
    K4   = np.concatenate((np.arange(41000, 44500), np.arange(47500, 48200)))
    epi_idx = np.concatenate((ATAC, K27a, K27, K36, K4))
    epi = epi[:, epi_idx]  # (N, 21000) = 5 marks × 4200 positions
    print(f"  Epi (subsetted): {epi.shape}")

    # Normalize
    epi_min = epi.min(axis=0, keepdims=True)
    epi_max = epi.max(axis=0, keepdims=True)
    epi_range = np.clip(epi_max - epi_min, 1e-8, None)
    epi = ((epi - epi_min) / epi_range).astype(np.float32)

    # TPM
    tpm_df = pd.read_csv(f"{ZENODO}/{tissue}.TPM", sep="\t")
    tpm_df.columns = ["gene_id", "TPM"]
    tpm_df = tpm_df.set_index("gene_id")
    labels = np.array([
        np.log1p(float(tpm_df.loc[g, "TPM"])) if g in tpm_df.index else 0.0
        for g in genes
    ], dtype=np.float32)

    return genes, fake_seq_emb, epi, labels


def main():
    print("=" * 60)
    print("  ConditionEXP Quick Test (fake embeddings, 3 epochs)")
    print("=" * 60)

    # Load two tissues
    tissues = ["spike", "leaf"]
    all_emb, all_epi, all_lab, all_tid, all_genes = [], [], [], [], []

    for i, tissue in enumerate(tissues):
        genes, emb, epi, lab = load_zenodo_tissue(tissue, max_genes=500)
        all_emb.append(emb)
        all_epi.append(epi)
        all_lab.append(lab)
        all_tid.append(np.full(len(genes), i, dtype=np.int64))
        all_genes.append(genes)

    seq_emb = np.concatenate(all_emb)
    epi_feat = np.concatenate(all_epi)
    labels = np.concatenate(all_lab)
    tissue_ids = np.concatenate(all_tid)
    print(f"\nCombined: {len(labels)} samples, epi {epi_feat.shape}")

    # Split: 70% train, 15% val, 15% test
    n = len(labels)
    idx = np.random.permutation(n)
    n_tr = int(0.7 * n)
    n_va = int(0.15 * n)
    train_idx, val_idx, test_idx = idx[:n_tr], idx[n_tr:n_tr+n_va], idx[n_tr+n_va:]

    device = torch.device("cpu")

    # Test each variant
    for variant in ["A", "B", "C"]:
        print(f"\n--- Variant {variant} ---")
        n_marks = 5
        n_positions = epi_feat.shape[1] // n_marks

        model = train_mod.ConditionEXP(
            seq_dim=seq_emb.shape[1],
            epi_input_dim=epi_feat.shape[1],
            n_marks=n_marks,
            n_positions=n_positions,
            n_tissues=len(tissues),
            variant=variant,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Params: {n_params:,}")

        # Quick train
        ds_train = train_mod.ExpressionDataset(
            seq_emb[train_idx], epi_feat[train_idx],
            labels[train_idx], tissue_ids[train_idx])
        ds_val = train_mod.ExpressionDataset(
            seq_emb[val_idx], epi_feat[val_idx],
            labels[val_idx], tissue_ids[val_idx])
        ds_test = train_mod.ExpressionDataset(
            seq_emb[test_idx], epi_feat[test_idx],
            labels[test_idx], tissue_ids[test_idx])

        loader_tr = torch.utils.data.DataLoader(ds_train, batch_size=32, shuffle=True)
        loader_va = torch.utils.data.DataLoader(ds_val, batch_size=64)
        loader_te = torch.utils.data.DataLoader(ds_test, batch_size=64)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        for epoch in range(1, 4):
            loss = train_mod.train_epoch(model, loader_tr, optimizer, device)
            val = train_mod.evaluate(model, loader_va, device)
            print(f"  Epoch {epoch}: loss={loss:.4f}, val_pcc={val['pcc_log']:.3f}")

        test = train_mod.evaluate(model, loader_te, device)
        print(f"  Test PCC: {test['pcc_log']:.3f} (random embeddings — expect ~0)")

    print("\n" + "=" * 60)
    print("  Quick test PASSED — pipeline is functional")
    print("  With real Botanic-1-S embeddings, expect PCC >> 0")
    print("=" * 60)


if __name__ == "__main__":
    main()
