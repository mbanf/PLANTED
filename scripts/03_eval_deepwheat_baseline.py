#!/usr/bin/env python3
"""
Evaluate DeepWheat pretrained Keras models on our ConditionEXP test split.

Loads the original DeepWheat models from Zenodo and evaluates them on the
exact same test genes used by ConditionEXP, enabling fair head-to-head
comparison.

Usage:
    python scripts/03_eval_deepwheat_baseline.py \
        --data-dir data/zenodo/DeepExp/data \
        --model-dir data/zenodo/DeepExp/model \
        --out results/deepwheat_baseline \
        --train-frac 0.8 --val-frac 0.1 --seed 42
"""
import argparse, os, json, time
import numpy as np
import pandas as pd
import h5py
import scipy.stats
from sklearn.metrics import r2_score


def build_deepwheat_model(k1, k2, ksz, L2, dr, nrb, rksz, act, epi_dim, seq_shape):
    """Reconstruct DeepWheat model architecture from best_params.

    Matches model_train.py build_model() exactly.
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, backend as K

    def res_block(x, filters, kernel_size, l2, dropout_rate, activation):
        shortcut = x
        y = layers.Conv2D(filters, kernel_size, padding="same",
                          kernel_regularizer=keras.regularizers.l2(l2))(x)
        y = layers.BatchNormalization()(y)
        y = layers.Activation(activation)(y)
        y = layers.Dropout(dropout_rate)(y)
        y = layers.Conv2D(filters, kernel_size, padding="same",
                          kernel_regularizer=keras.regularizers.l2(l2))(y)
        y = layers.BatchNormalization()(y)
        if K.int_shape(shortcut)[-1] != filters:
            shortcut = layers.Conv2D(filters, (1, 1), padding="same",
                                     kernel_regularizer=keras.regularizers.l2(l2))(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        y = layers.Add()([y, shortcut])
        y = layers.Activation(activation)(y)
        y = layers.Dropout(dropout_rate)(y)
        return y

    # Epi branch
    ei = keras.Input((epi_dim,), name="epi_input")
    x = layers.Reshape((5, epi_dim // 5, 1))(ei)
    x = layers.Permute((2, 1, 3))(x)
    x = layers.ZeroPadding2D(((10, 9), (0, 0)))(x)
    x = layers.Conv2D(k1, (ksz, 5), use_bias=False, padding="valid",
                      kernel_regularizer=keras.regularizers.l2(L2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(act)(x)
    x = layers.Dropout(dr)(x)
    x = layers.ZeroPadding2D(((5, 4), (0, 0)))(x)
    x = layers.Conv2D(k2, (ksz, 1), padding="valid",
                      kernel_regularizer=keras.regularizers.l2(L2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(act)(x)
    x = layers.Dropout(dr)(x)
    x = layers.AveragePooling2D((5, 1), strides=5)(x)
    epi_out = x

    # Seq branch
    si = keras.Input((seq_shape[0], seq_shape[1], 1), name="seq_input")
    y = layers.ZeroPadding2D(((10, 9), (0, 0)))(si)
    y = layers.Conv2D(k1, (ksz, seq_shape[1]), use_bias=False, padding="valid",
                      kernel_regularizer=keras.regularizers.l2(L2))(y)
    y = layers.BatchNormalization()(y)
    y = layers.Activation(act)(y)
    y = layers.Dropout(dr)(y)
    y = layers.ZeroPadding2D(((5, 4), (0, 0)))(y)
    y = layers.Conv2D(k2, (ksz, 1), padding="valid",
                      kernel_regularizer=keras.regularizers.l2(L2))(y)
    y = layers.BatchNormalization()(y)
    y = layers.Activation(act)(y)
    y = layers.Dropout(dr)(y)
    y = layers.AveragePooling2D((5, 1), strides=5)(y)
    seq_out = y

    # Fusion + residual blocks
    z = layers.Concatenate(axis=2)([epi_out, seq_out])
    for _ in range(nrb):
        z = res_block(z, k1, (rksz, rksz), L2, dr, act)
    z = layers.Flatten()(z)
    for d in (256, 64, 16):
        z = layers.Dense(d, kernel_regularizer=keras.regularizers.l2(L2))(z)
        z = layers.BatchNormalization()(z)
        z = layers.Activation(act)(z)
        z = layers.Dropout(dr)(z)
    o = layers.Dense(1, activation="linear",
                     kernel_regularizer=keras.regularizers.l2(L2))(z)
    o = layers.Lambda(lambda t: tf.squeeze(t, -1))(o)
    model = keras.Model([ei, si], o)
    return model


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate DeepWheat baseline on our test split")
    p.add_argument("--data-dir", required=True, help="DeepWheat data dir with .epi.h5, .TPM, seq.tsv")
    p.add_argument("--model-dir", required=True, help="DeepWheat model dir with *.model.h5")
    p.add_argument("--out", default="results/deepwheat_baseline")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_epi_subset(h5_path, gene_list):
    """Load epigenomic features and subset to DeepWheat's column indices."""
    with h5py.File(h5_path, "r") as f:
        epi_raw = f["dataset_1"][:]

    # DeepWheat uses 5 marks × (TSS 3500 + TTS 700) = 21000 features
    # Column indices from their model_train.py lines 133-138
    ATAC = np.concatenate((np.arange(1000, 4500), np.arange(7500, 8200)))
    K27a = np.concatenate((np.arange(11000, 14500), np.arange(17500, 18200)))
    K27  = np.concatenate((np.arange(21000, 24500), np.arange(27500, 28200)))
    K36  = np.concatenate((np.arange(31000, 34500), np.arange(37500, 38200)))
    K4   = np.concatenate((np.arange(41000, 44500), np.arange(47500, 48200)))
    epi_idx = np.concatenate((ATAC, K27a, K27, K36, K4))

    # Handle gene count mismatch (some tissues have fewer genes)
    n_genes = min(len(gene_list), epi_raw.shape[0])
    epi = epi_raw[:n_genes, :]

    # Subset columns
    if epi.shape[1] > max(epi_idx):
        epi = epi[:, epi_idx]

    return epi.astype(np.float32), n_genes


def load_seq_onehot(data_dir, n_genes):
    """Load one-hot encoded sequences from seq.h5 or seq.tsv.

    Uses the pre-computed H5 file if available (fast), otherwise falls back
    to TSV parsing (slow).
    """
    seq_h5 = os.path.join(data_dir, "seq.h5")
    seq_idx = np.concatenate((np.arange(1000, 4500), np.arange(7500, 8200)))

    if os.path.exists(seq_h5):
        print(f"  Loading from seq.h5 (fast path)...")
        with h5py.File(seq_h5, "r") as f:
            # seq.h5 has dataset_2: [N, 10000, 4] one-hot
            seq_all = f["dataset_2"][:n_genes]
        return seq_all[:, seq_idx, :].astype(np.float32)

    # Fallback: vectorized TSV parsing
    print(f"  Loading from seq.tsv (vectorized)...")
    seq_tsv = os.path.join(data_dir, "seq.tsv")
    seq_df = pd.read_csv(seq_tsv, sep="\t", dtype=str)
    seqs = seq_df["sequence"].values[:n_genes]

    L = len(seq_idx)
    dna = np.zeros((n_genes, L, 4), dtype=np.float32)
    base_to_idx = {b: i for i, b in enumerate("ACGT")}

    for i, s in enumerate(seqs):
        s = s.upper()
        arr = np.array([base_to_idx.get(s[p], -1) for p in seq_idx if p < len(s)])
        valid = arr >= 0
        rows = np.where(valid)[0]
        dna[i, rows, arr[valid]] = 1.0

    return dna


def make_split(n_genes, train_frac, val_frac, seed):
    """Reproduce the same train/val/test split as ConditionEXP."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n_genes)
    n_train = int(n_genes * train_frac)
    n_val = int(n_genes * val_frac)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Lazy import TF (slow startup)
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")

    tissues = ["spike", "leaf", "vl", "e5", "e7", "e9"]
    gene_list_path = os.path.join(args.data_dir, "gene.list.txt")
    gene_list = pd.read_csv(gene_list_path, header=None)[0].tolist()
    seq_tsv = os.path.join(args.data_dir, "seq.tsv")

    all_results = {}

    for tissue in tissues:
        model_path = os.path.join(args.model_dir, f"{tissue}.model.h5")
        epi_path = os.path.join(args.data_dir, f"{tissue}.epi.h5")
        tpm_path = os.path.join(args.data_dir, f"{tissue}.TPM")

        if not os.path.exists(model_path):
            print(f"  Skipping {tissue}: model not found")
            continue
        if not os.path.exists(epi_path) or not os.path.exists(tpm_path):
            print(f"  Skipping {tissue}: data not found")
            continue

        print(f"\n=== {tissue} ===")
        t0 = time.time()

        # Load epi features
        print(f"  Loading epi from {epi_path}...")
        epi, n_genes = load_epi_subset(epi_path, gene_list)
        print(f"  Epi shape: {epi.shape}")

        # Load sequence one-hot
        print(f"  Loading sequences...")
        seq = load_seq_onehot(args.data_dir, n_genes)
        print(f"  Seq shape: {seq.shape}")

        # Load TPM labels
        tpm_df = pd.read_csv(tpm_path, sep="\t", header=0, dtype=str)
        tpm_df.columns = ["gene_id", "TPM"]
        tpm_df["TPM"] = tpm_df["TPM"].astype(float)
        # Match to gene list order
        tpm_map = dict(zip(tpm_df["gene_id"], tpm_df["TPM"]))
        labels = np.array([np.log1p(tpm_map.get(g, 0.0)) for g in gene_list[:n_genes]],
                          dtype=np.float32)

        # Make split (same as ConditionEXP)
        train_idx, val_idx, test_idx = make_split(n_genes, args.train_frac,
                                                   args.val_frac, args.seed)
        print(f"  Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

        # MinMaxScaler fitted on ALL data (matching DeepWheat's approach)
        # Note: DeepWheat scales before splitting (line 116 of model_train.py)
        # This is mild data leakage but we must match their preprocessing
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        epi_scaled = scaler.fit_transform(epi)

        # Prepare test data
        test_epi = epi_scaled[test_idx]
        test_seq = seq[test_idx]
        test_labels = labels[test_idx]
        test_genes = [gene_list[i] for i in test_idx]

        # Load DeepWheat model by reconstructing architecture + loading weights
        print(f"  Loading model from {model_path}...")
        params_path = os.path.join(args.data_dir,
                                    f"{tissue}_best_params.json" if tissue != "spike"
                                    else "spike_best_params.json")
        # Handle naming inconsistency
        if not os.path.exists(params_path):
            params_path = os.path.join(args.data_dir, f"{tissue}.best_params.json")
        if not os.path.exists(params_path):
            print(f"  Skipping {tissue}: best_params.json not found")
            continue

        with open(params_path) as pf:
            bp = json.load(pf)
        try:
            # First try: exact architecture match
            model = build_deepwheat_model(
                bp["k1"], bp["k2"], bp["ksz"], bp["L2"], bp["dr"],
                bp["nrb"], bp["rksz"], bp["act"],
                epi_dim=test_epi.shape[1], seq_shape=(test_seq.shape[1], test_seq.shape[2]))
            try:
                model.load_weights(model_path)
            except ValueError:
                # Layer count mismatch — try different nrb values
                loaded = False
                for try_nrb in range(2, 11):
                    if try_nrb == bp["nrb"]:
                        continue
                    try:
                        model = build_deepwheat_model(
                            bp["k1"], bp["k2"], bp["ksz"], bp["L2"], bp["dr"],
                            try_nrb, bp["rksz"], bp["act"],
                            epi_dim=test_epi.shape[1],
                            seq_shape=(test_seq.shape[1], test_seq.shape[2]))
                        model.load_weights(model_path)
                        print(f"  Loaded with nrb={try_nrb} (params said {bp['nrb']})")
                        loaded = True
                        break
                    except (ValueError, Exception):
                        continue
                if not loaded:
                    print(f"  ERROR: could not match architecture for {tissue}")
                    continue
        except Exception as e:
            print(f"  ERROR loading model: {e}")
            continue

        print(f"  Model loaded: {model.count_params():,} params")

        # Reshape inputs to match model expectations
        # DeepWheat expects: epi as [N, epi_dim], seq as [N, L, 4, 1]
        test_seq_4d = test_seq[..., np.newaxis]  # [N, L, 4] -> [N, L, 4, 1]

        # Run inference
        print(f"  Running inference on {len(test_idx)} test genes...")
        pred_log = model.predict([test_epi, test_seq_4d], batch_size=64, verbose=0).flatten()

        # Compute metrics
        pcc_log, _ = scipy.stats.pearsonr(test_labels, pred_log)
        r2_log = r2_score(test_labels, pred_log)
        true_tpm = np.expm1(test_labels)
        pred_tpm = np.expm1(pred_log)
        pcc_tpm, _ = scipy.stats.pearsonr(true_tpm, pred_tpm)
        r2_tpm = r2_score(true_tpm, pred_tpm)

        elapsed = time.time() - t0
        print(f"  PCC (log1p): {pcc_log:.4f}")
        print(f"  R2  (log1p): {r2_log:.4f}")
        print(f"  PCC (TPM):   {pcc_tpm:.4f}")
        print(f"  R2  (TPM):   {r2_tpm:.4f}")
        print(f"  Time: {elapsed:.1f}s")

        all_results[tissue] = {
            "pcc_log": float(pcc_log),
            "r2_log": float(r2_log),
            "pcc_tpm": float(pcc_tpm),
            "r2_tpm": float(r2_tpm),
            "n_test": len(test_idx),
            "n_params": int(model.count_params()),
        }

        # Save per-gene predictions
        pred_df = pd.DataFrame({
            "gene_id": test_genes,
            "true_log1p": test_labels,
            "pred_log1p": pred_log,
            "true_tpm": true_tpm,
            "pred_tpm": pred_tpm,
        })
        pred_df.to_csv(os.path.join(args.out, f"deepwheat_{tissue}_predictions.csv"),
                       index=False)

        # Clear model
        del model
        tf.keras.backend.clear_session()

    # Summary
    print("\n" + "=" * 60)
    print("  DeepWheat Baseline Results (on ConditionEXP test split)")
    print("=" * 60)
    for tissue, r in all_results.items():
        print(f"  {tissue:6s}: PCC={r['pcc_log']:.4f}  R2={r['r2_log']:.4f}  "
              f"params={r['n_params']:,}")

    mean_pcc = np.mean([r["pcc_log"] for r in all_results.values()])
    mean_r2 = np.mean([r["r2_log"] for r in all_results.values()])
    print(f"  {'MEAN':6s}: PCC={mean_pcc:.4f}  R2={mean_r2:.4f}")
    print("=" * 60)

    # Save summary
    summary = {"results": all_results, "mean_pcc_log": float(mean_pcc),
               "mean_r2_log": float(mean_r2),
               "split": {"train_frac": args.train_frac, "val_frac": args.val_frac,
                          "seed": args.seed}}
    with open(os.path.join(args.out, "deepwheat_baseline_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {args.out}/")


if __name__ == "__main__":
    main()
