#!/usr/bin/env python3
"""
Pre-compute Botanic-1-S embeddings for all wheat genes.

Reads the DeepWheat-format sequence TSV (gene_id, sequence), tokenizes each
gene's TSS and TTS windows through frozen Botanic-1-S, and saves mean-pooled
embeddings to disk for fast downstream training.

Usage:
    python scripts/01_precompute_embeddings.py \
        --seq data/processed/spike.seq.tsv \
        --out data/embeddings/spike_botanic1s.npz \
        --model living-models/Botanic1-S \
        --batch-size 8
"""
import argparse, os, sys, time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Pre-compute Botanic-1-S embeddings")
    p.add_argument("--seq", required=True, help="DeepWheat seq TSV (gene_id, sequence)")
    p.add_argument("--out", required=True, help="Output .npz file")
    p.add_argument("--model", default="living-models/Botanic1-S", help="HF model ID")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-genes", type=int, default=0, help="Limit genes (0=all)")
    p.add_argument("--device", default="auto", help="cuda / cpu / mps / auto")
    return p.parse_args()


def get_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_id, device):
    """Load Botanic-1-S with its custom tokenizer."""
    from transformers import AutoModel, AutoTokenizer

    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model = model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {model_id}: {n_params/1e6:.0f}M params on {device}")
    return model, tokenizer


def extract_windows(sequence, tss_start=1000, tss_end=4500, tts_start=7500, tts_end=8200):
    """
    Extract the same windows DeepWheat uses from the 10kb concatenated sequence.
    TSS region: positions 1000-4500 (3500 bp) = TSS-2000 to TSS+1500
    TTS region: positions 7500-8200 (700 bp)  = TTS-500 to TTS+200
    """
    seq = sequence.upper()
    tss_seq = seq[tss_start:tss_end]  # 3500 bp
    tts_seq = seq[tts_start:tts_end]  # 700 bp
    return tss_seq, tts_seq


@torch.no_grad()
def embed_sequences(sequences, model, tokenizer, device, batch_size=8):
    """
    Tokenize and embed a list of DNA sequences through Botanic-1-S.
    Returns mean-pooled embeddings [N, embed_dim].
    """
    all_embs = []
    for i in tqdm(range(0, len(sequences), batch_size), desc="Embedding"):
        batch_seqs = sequences[i:i + batch_size]
        # Tokenize
        encoded = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Get hidden states — handle different output formats
        if hasattr(outputs, "last_hidden_state"):
            hidden = outputs.last_hidden_state  # [B, L, D]
        elif isinstance(outputs, tuple):
            hidden = outputs[0]
        else:
            hidden = outputs

        # Mean pool (respecting attention mask if present)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden.mean(dim=1)

        all_embs.append(pooled.cpu().float().numpy())

    return np.concatenate(all_embs, axis=0)


def main():
    args = parse_args()
    device = get_device(args.device)

    # Load sequences
    print(f"Loading sequences from {args.seq}...")
    df = pd.read_csv(args.seq, sep="\t", dtype=str)
    assert "gene_id" in df.columns and "sequence" in df.columns, \
        f"Expected columns 'gene_id' and 'sequence', got {df.columns.tolist()}"

    if args.max_genes > 0:
        df = df.head(args.max_genes)
    print(f"Processing {len(df)} genes")

    # Extract TSS and TTS windows
    print("Extracting TSS (3500bp) and TTS (700bp) windows...")
    tss_seqs, tts_seqs = [], []
    for seq in df["sequence"]:
        if len(seq) < 8200:
            # Short sequence — pad with N
            seq = seq + "N" * (10000 - len(seq))
        tss, tts = extract_windows(seq)
        tss_seqs.append(tss)
        tts_seqs.append(tts)

    # Load model
    model, tokenizer = load_model(args.model, device)

    # Embed TSS windows
    print(f"\n=== Embedding TSS windows ({len(tss_seqs)} × 3500bp) ===")
    t0 = time.time()
    tss_embs = embed_sequences(tss_seqs, model, tokenizer, device, args.batch_size)
    t_tss = time.time() - t0
    print(f"TSS embeddings: {tss_embs.shape}, took {t_tss:.0f}s")

    # Embed TTS windows
    print(f"\n=== Embedding TTS windows ({len(tts_seqs)} × 700bp) ===")
    t0 = time.time()
    tts_embs = embed_sequences(tts_seqs, model, tokenizer, device, args.batch_size)
    t_tts = time.time() - t0
    print(f"TTS embeddings: {tts_embs.shape}, took {t_tts:.0f}s")

    # Save
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        gene_ids=df["gene_id"].values,
        tss_emb=tss_embs,
        tts_emb=tts_embs,
        embed_dim=tss_embs.shape[1],
    )
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"\nSaved embeddings to {args.out} ({size_mb:.1f} MB)")
    print(f"  TSS: {tss_embs.shape}, TTS: {tts_embs.shape}")
    print(f"  Total time: {t_tss + t_tts:.0f}s")


if __name__ == "__main__":
    main()
