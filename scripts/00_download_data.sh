#!/usr/bin/env bash
# Download wheat AK58 genome, annotation, epigenomic data, and expression data
# for the ConditionEXP project (improved DeepWheat).
#
# Data sources (from Ma et al., Genome Biology 2025):
#   - AK58 genome: NCBI GCA_025895885.1
#   - ChIP-seq / ATAC-seq: GEO GSE289179
#   - RNA-seq: GEO GSE287695 + GSE289179
#   - Additional: GSE232430 (vernalization), GSE14739 (grain development)
#
# Usage: bash scripts/00_download_data.sh [DATA_DIR]

set -euo pipefail

DATA_DIR="${1:-data/raw}"
mkdir -p "$DATA_DIR"/{genome,geo}

echo "=== Step 1: Download AK58 wheat reference genome ==="
GENOME_DIR="$DATA_DIR/genome"
if [ ! -f "$GENOME_DIR/AK58.fa" ]; then
    echo "Downloading AK58 genome assembly from NCBI..."
    # NCBI Assembly: GCA_025895885.1 (Aikang 58)
    wget -q -O "$GENOME_DIR/AK58_genomic.fna.gz" \
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/025/895/885/GCA_025895885.1_ASM2589588v1/GCA_025895885.1_ASM2589588v1_genomic.fna.gz"
    gunzip -k "$GENOME_DIR/AK58_genomic.fna.gz"
    mv "$GENOME_DIR/AK58_genomic.fna" "$GENOME_DIR/AK58.fa"
    echo "Done. Genome: $GENOME_DIR/AK58.fa"

    # Download GFF annotation
    wget -q -O "$GENOME_DIR/AK58.gff3.gz" \
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/025/895/885/GCA_025895885.1_ASM2589588v1/GCA_025895885.1_ASM2589588v1_genomic.gff.gz"
    gunzip -k "$GENOME_DIR/AK58.gff3.gz"
    echo "Done. Annotation: $GENOME_DIR/AK58.gff3"
else
    echo "AK58 genome already present, skipping."
fi

echo ""
echo "=== Step 2: Download epigenomic and expression data from GEO ==="
GEO_DIR="$DATA_DIR/geo"

# GSE289179: Main dataset (ChIP-seq + ATAC-seq + RNA-seq for AK58 spike/leaf)
# This accession contains processed bedGraph and TPM files
if [ ! -d "$GEO_DIR/GSE289179" ]; then
    mkdir -p "$GEO_DIR/GSE289179"
    echo "Downloading GSE289179 supplementary files..."
    # Download the supplementary tar bundle
    wget -q -P "$GEO_DIR/GSE289179/" \
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE289nnn/GSE289179/suppl/" \
        --recursive --no-parent --no-host-directories --cut-dirs=6 \
        -A "*.bedGraph.gz,*.bed.gz,*.txt.gz,*.tsv.gz,*.csv.gz" \
        2>/dev/null || echo "wget recursive failed, trying GEOquery fallback..."

    # Fallback: use GEO's FTP for individual files
    # The exact filenames depend on what the authors uploaded
    echo "Note: If wget failed, download manually from:"
    echo "  https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE289179"
    echo "  Place bedGraph files in $GEO_DIR/GSE289179/"
else
    echo "GSE289179 data already present, skipping."
fi

# GSE287695: Additional epigenomic data
if [ ! -d "$GEO_DIR/GSE287695" ]; then
    mkdir -p "$GEO_DIR/GSE287695"
    echo "Downloading GSE287695 supplementary files..."
    wget -q -P "$GEO_DIR/GSE287695/" \
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE287nnn/GSE287695/suppl/" \
        --recursive --no-parent --no-host-directories --cut-dirs=6 \
        -A "*.bedGraph.gz,*.bed.gz,*.txt.gz,*.tsv.gz" \
        2>/dev/null || echo "wget recursive failed for GSE287695"
else
    echo "GSE287695 data already present, skipping."
fi

echo ""
echo "=== Step 3: Check DeepWheat pre-processed data (Zenodo) ==="
ZENODO_DIR="$DATA_DIR/zenodo"
if [ ! -d "$ZENODO_DIR" ]; then
    mkdir -p "$ZENODO_DIR"
    echo "Downloading DeepWheat trained models and processed data from Zenodo..."
    # Zenodo DOI: 10.5281/zenodo.15761553
    curl -L -o "$ZENODO_DIR/deepwheat_zenodo.zip" \
        "https://zenodo.org/records/15761553/files/DeepWheat.zip?download=1" \
        || wget -O "$ZENODO_DIR/deepwheat_zenodo.zip" \
            "https://zenodo.org/records/15761553/files/DeepWheat.zip?download=1" \
        || echo "Zenodo download may require manual access"
    if [ -f "$ZENODO_DIR/deepwheat_zenodo.zip" ]; then
        cd "$ZENODO_DIR" && unzip -q deepwheat_zenodo.zip && cd -
    fi
else
    echo "Zenodo data already present, skipping."
fi

echo ""
echo "=== Step 4: Alternative — use DeepWheat repo's data processing ==="
echo "If raw GEO downloads are incomplete, use the DeepWheat pipeline:"
echo "  1. cd ../DeepWheat/DeepEXP/Data_Preparation/"
echo "  2. python get_seq.py -g $GENOME_DIR/AK58.fa -a $GENOME_DIR/AK58.gff3 -o spike"
echo "  3. python get_epi.py -g $GENOME_DIR/AK58.gff3 -p <bedGraph files...> -o spike"
echo ""
echo "Required bedGraph files per tissue (normalized):"
echo "  - <tissue>-atac.norm.bedGraph   (ATAC-seq / TIS)"
echo "  - <tissue>-K27ac.norm.bedGraph  (H3K27ac)"
echo "  - <tissue>-K27.norm.bedGraph    (H3K27me3)"
echo "  - <tissue>-K36.norm.bedGraph    (H3K36me3)"
echo "  - <tissue>-K4.norm.bedGraph     (H3K4me3)"
echo ""
echo "=== Download complete ==="
echo "Next: python scripts/01_precompute_embeddings.py"
