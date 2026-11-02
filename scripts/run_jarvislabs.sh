#!/usr/bin/env bash
# ConditionEXP — full pipeline on Jarvislabs A100
#
# Uses pre-processed Zenodo data (H5 epi + TPM) + Botanic-1-S embeddings.
# Auto-pauses the instance when training completes.
#
# Launch from local machine:
#   jl create --gpu A100 --name conditionexp
#   # Upload project files, then SSH in:
#   jl ssh <id>
#   cd /workspace/conditionexp
#   HF_TOKEN=hf_xxx bash scripts/run_jarvislabs.sh
#
# Or quick local launch:
#   cd conditionexp && bash scripts/launch_remote.sh

set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"

# Ensure ~/.local/bin is on PATH (pip installs scripts there on VMs)
export PATH="$HOME/.local/bin:$PATH"

echo "============================================"
echo "  ConditionEXP — GPU Training Pipeline"
echo "============================================"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo ""

# 0. Install dependencies
echo "=== Installing dependencies ==="
pip3 install -q -r requirements.txt || pip install -q -r requirements.txt

# HuggingFace login for Botanic-1-S (gated model)
if [ -n "${HF_TOKEN:-}" ]; then
    huggingface-cli login --token "$HF_TOKEN" 2>/dev/null \
        || python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN')" \
        || echo "HF login failed, will use HF_TOKEN env var directly"
fi

# 1. Download Zenodo data if not present
echo ""
echo "=== Step 1: Data preparation ==="
ZENODO_DIR="data/zenodo/DeepExp/data"
if [ ! -d "$ZENODO_DIR" ]; then
    echo "Downloading DeepWheat Zenodo archive..."
    mkdir -p data/zenodo
    curl -L -o data/zenodo/deepwheat_zenodo.zip \
        "https://zenodo.org/records/15761553/files/DeepWheat.zip?download=1"
    cd data/zenodo && unzip -q deepwheat_zenodo.zip && cd "$PROJ_DIR"
fi

# Verify key files
for tissue in spike leaf vl e5 e7 e9; do
    if [ -f "$ZENODO_DIR/${tissue}.epi.h5" ]; then
        echo "  Found: ${tissue}.epi.h5"
    fi
done

# 2. Build seq.tsv from Zenodo if needed (gene_id + sequence)
SEQ_TSV="$ZENODO_DIR/seq.tsv"
if [ ! -f "$SEQ_TSV" ]; then
    echo "ERROR: $SEQ_TSV not found. Check Zenodo download."
    exit 1
fi
echo "  Seq TSV: $(wc -l < "$SEQ_TSV") genes"

# 3. Pre-compute frozen Botanic-1-S embeddings (baseline)
echo ""
echo "=== Step 2a: Frozen Botanic-1-S embeddings ==="
EMB_DIR="data/embeddings"
mkdir -p "$EMB_DIR"

EMB_FILE="$EMB_DIR/all_botanic1s.npz"
if [ ! -f "$EMB_FILE" ]; then
    echo "Computing frozen embeddings for all genes..."
    python3 scripts/01_precompute_embeddings.py \
        --seq "$SEQ_TSV" \
        --out "$EMB_FILE" \
        --model living-models/Botanic1-S \
        --batch-size 64 \
        --device cuda
else
    echo "Frozen embeddings already computed: $EMB_FILE"
fi

# 3b. LoRA fine-tune Botanic-1-S + re-extract embeddings
echo ""
echo "=== Step 2b: LoRA fine-tune Botanic-1-S ==="
LORA_DIR="data/lora_adapted"
LORA_EMB="$LORA_DIR/all_botanic1s_lora.npz"
if [ ! -f "$LORA_EMB" ]; then
    # Build TPM args for all tissues
    TPM_ARGS=()
    TISSUE_ARGS=()
    for tissue in spike leaf vl e5 e7 e9; do
        TPM="$ZENODO_DIR/${tissue}.TPM"
        if [ -f "$TPM" ]; then
            TPM_ARGS+=("$TPM")
            TISSUE_ARGS+=("$tissue")
        fi
    done

    echo "Fine-tuning on ${#TISSUE_ARGS[@]} tissues..."
    python3 scripts/01b_finetune_lora.py \
        --seq "$SEQ_TSV" \
        --tpm "${TPM_ARGS[@]}" \
        --tissue "${TISSUE_ARGS[@]}" \
        --out "$LORA_DIR" \
        --lora-rank 16 \
        --epochs 3 \
        --batch-size 1 \
        --grad-accum 16 \
        --lr 2e-4 \
        --max-train 50000 \
        --emb-batch-size 16 \
        --device cuda
else
    echo "LoRA embeddings already computed: $LORA_EMB"
fi

# 4. Train all ablation variants
echo ""
echo "=== Step 3: Training ablation variants ==="

# Build tissue arrays
TISSUES=()
EPIS=()
TPMS=()
for tissue in spike leaf vl e5 e7 e9; do
    EPI="$ZENODO_DIR/${tissue}.epi.h5"
    TPM="$ZENODO_DIR/${tissue}.TPM"
    if [ -f "$EPI" ] && [ -f "$TPM" ]; then
        TISSUES+=("$tissue")
        EPIS+=("$EPI")
        TPMS+=("$TPM")
        echo "  Tissue: $tissue"
    fi
done

if [ ${#TISSUES[@]} -eq 0 ]; then
    echo "ERROR: No tissue data found. Check Zenodo download."
    exit 1
fi
echo "  Total tissues: ${#TISSUES[@]}"

TEST_GENES=""
if [ -f "$ZENODO_DIR/4700_gene.txt" ]; then
    TEST_GENES="--test-genes $ZENODO_DIR/4700_gene.txt"
elif [ -f "data/processed/4700_gene.txt" ]; then
    TEST_GENES="--test-genes data/processed/4700_gene.txt"
fi

COMMON_ARGS="--epi ${EPIS[*]} --tpm ${TPMS[*]} --tissue ${TISSUES[*]} \
    --epochs 100 --batch-size 64 --lr 3e-4 --warmup-epochs 5 \
    --train-frac 0.8 --val-frac 0.1 --device cuda $TEST_GENES"

# Build per-tissue embedding arrays for LoRA
LORA_EMBS=()
for tissue in "${TISSUES[@]}"; do
    LORA_EMBS+=("$LORA_EMB")
done

# --- v3: LoRA-adapted embeddings ---
echo ""
echo "========== v3: LoRA-adapted embeddings =========="
for variant in E A C D; do
    echo ""
    echo "--- v3 LoRA: variant $variant ---"
    OUTDIR="results/v3_lora_${variant}"
    mkdir -p "$OUTDIR"
    python3 scripts/02_train_conditionexp.py \
        --emb "${LORA_EMBS[@]}" \
        --epi "${EPIS[@]}" \
        --tpm "${TPMS[@]}" \
        --tissue "${TISSUES[@]}" \
        --variant "$variant" \
        --out "$OUTDIR" \
        --epochs 100 --batch-size 64 --lr 3e-4 --warmup-epochs 5 \
        --train-frac 0.8 --val-frac 0.1 --device cuda \
        $TEST_GENES
done

# 5. Summary
echo ""
echo "============================================"
echo "  Training complete! Results:"
echo "============================================"
for prefix in v3_lora; do
for variant in E A C D; do
    JSON="results/${prefix}_${variant}/conditionexp_${variant}_results.json"
    if [ -f "$JSON" ]; then
        echo "${prefix} variant $variant:"
        python3 -c "
import json
d = json.load(open('$JSON'))
r = d.get('results', {}).get('test', {})
print(f'  PCC (log1p): {r.get(\"pcc_log\", 0):.4f}')
print(f'  R2  (log1p): {r.get(\"r2_log\", 0):.4f}')
print(f'  PCC (TPM):   {r.get(\"pcc_tpm\", 0):.4f}')
print(f'  Params:      {d.get(\"n_params\", 0):,}')
"
    fi
done
done
echo ""
echo "DeepWheat reference: PCC 0.82-0.88 (seq+epi), <0.66 (seq only)"
echo "============================================"

# 6. Tar results for easy download
echo ""
echo "=== Packaging results ==="
tar czf results_conditionexp.tar.gz results/
echo "Results packaged: results_conditionexp.tar.gz"
echo "Download with: jl download <instance-id> results_conditionexp.tar.gz"

# 7. Auto-pause instance to stop billing
echo ""
echo "=== Auto-pausing instance to save costs ==="
echo "Training finished at $(date). Pausing in 60 seconds..."
echo "Cancel with: kill $$"
sleep 60
# Jarvislabs pause command (works from within instance)
if command -v jl &>/dev/null; then
    jl pause
elif [ -f /etc/jarvislabs/pause.sh ]; then
    bash /etc/jarvislabs/pause.sh
else
    echo "Could not auto-pause. Please manually pause/stop the instance!"
    # Fallback: shutdown the instance
    sudo shutdown -h +1 "ConditionEXP training complete. Shutting down to save costs."
fi
