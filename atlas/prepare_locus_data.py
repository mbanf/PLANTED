#!/usr/bin/env python3
"""Prepare locus browser data for PLANTED atlas.

For each of the 372 strawberry editing targets, extract:
1. Gene model (exons, UTRs, CDS) from GFF3
2. ChromHMM states in a 4kb window (2kb upstream TSS to 2kb into gene body)
3. Promoter sequence (2kb upstream) for CRISPR guide design
4. Pre-designed guide RNAs (top 3 per target)

Output: public/data/locus/ directory with one JSON per gene + an index.
"""

import json, gzip, csv, re, os, sys
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent.parent  # planted/
ATLAS = Path(__file__).resolve().parent         # planted/atlas/
DATA = BASE / "data" / "strawberry"
RESULTS = BASE / "results"

GFF = DATA / "genome" / "Fragaria_vesca_v6_genome.gff"
CHROMHMM_RED = DATA / "chipseq" / "Redv6_ChromatinStates.bed"
CHROMHMM_PRET = DATA / "chipseq" / "preTv6_ChromatinStates.bed"
FASTA = DATA / "genome" / "Fragaria_vesca_v6_genome.fasta"
CATALOG = RESULTS / "strawberry_editing_catalog" / "editing_target_catalog.tsv"
JASPAR = BASE / "data" / "motifs" / "jaspar_plants_meme.txt"
OUTDIR = ATLAS / "public" / "data" / "locus"

# ChromHMM state names and mark composition
STATE_NAMES = {
    1: "Active TSS",       2: "Flanking TSS",    3: "Transcribed (5')",
    4: "Strong Txn",       5: "Weak Txn",        6: "Txn Transition",
    7: "Enhancer",         8: "Weak Enhancer",   9: "Bivalent TSS",
    10: "Bivalent Enhancer", 11: "Polycomb Weak",  12: "Polycomb",
    13: "Polycomb Strong", 14: "H3K9me2 Repressed", 15: "Het. Low",
    16: "Het. H3K9me2",    17: "Quiescent",
}

# Color palette matching ChromHMM convention
STATE_COLORS = {
    1: "#FF0000", 2: "#FF6600", 3: "#008000", 4: "#006400",
    5: "#90EE90", 6: "#C0FFC0", 7: "#FFFF00", 8: "#FFFFCC",
    9: "#CC66CC", 10: "#CC99CC", 11: "#808080", 12: "#999999",
    13: "#666666", 14: "#404040", 15: "#65D1FF", 16: "#65D1FF",
    17: "#F5F5F5",
}

# Mark emission per state (which marks are present in each state)
# Simplified: 7 marks → binary presence
STATE_MARKS = {
    1:  {"H3K4me3": 1, "H3K27ac": 1, "H3K9K14ac": 1},
    2:  {"H3K4me3": 1, "H3K27ac": 0.5},
    3:  {"H3K36me3": 1, "H3K4me3": 0.5},
    4:  {"H3K36me3": 1},
    5:  {"H3K36me3": 0.5},
    6:  {},
    7:  {"H3K4me1": 1, "H3K27ac": 1},
    8:  {"H3K4me1": 0.5},
    9:  {"H3K4me3": 1, "H3K27me3": 1},
    10: {"H3K4me1": 1, "H3K27me3": 1},
    11: {"H3K27me3": 0.5},
    12: {"H3K27me3": 1},
    13: {"H3K27me3": 1, "H3K9me2": 0.5},
    14: {"H3K9me2": 1},
    15: {"H3K9me2": 0.3},
    16: {"H3K9me2": 1},
    17: {},
}

MARKS = ["H3K4me3", "H3K27ac", "H3K9K14ac", "H3K36me3", "H3K4me1", "H3K27me3", "H3K9me2"]
MARK_COLORS = {
    "H3K4me3": "#FF0000", "H3K27ac": "#FF8C00", "H3K9K14ac": "#FFD700",
    "H3K36me3": "#008000", "H3K4me1": "#4169E1", "H3K27me3": "#808080",
    "H3K9me2": "#2F4F4F",
}

WINDOW_UP = 2000   # bp upstream of TSS
WINDOW_DOWN = 2000  # bp downstream of TSS (into gene)


def parse_gff(gff_path, gene_ids):
    """Parse GFF3 to extract gene models for target genes."""
    genes = {}
    current_gene = None
    features = defaultdict(list)

    with open(gff_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue
            chrom, source, ftype, start, end, score, strand, phase, attrs = parts
            start, end = int(start), int(end)

            # Parse attributes
            attr_dict = {}
            for a in attrs.split(";"):
                if "=" in a:
                    k, v = a.split("=", 1)
                    attr_dict[k] = v

            if ftype == "gene":
                gid = attr_dict.get("ID", "")
                if gid in gene_ids:
                    current_gene = gid
                    genes[gid] = {
                        "chrom": chrom, "start": start, "end": end,
                        "strand": strand, "features": []
                    }
                else:
                    current_gene = None
            elif current_gene and ftype in ("mRNA", "exon", "CDS", "five_prime_UTR", "three_prime_UTR"):
                genes[current_gene]["features"].append({
                    "type": ftype, "start": start, "end": end
                })

    return genes


def parse_chromhmm(bed_path):
    """Parse ChromHMM BED into per-chromosome interval trees (simple list)."""
    states = defaultdict(list)
    with open(bed_path) as f:
        for line in f:
            if line.startswith("track") or line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            chrom = parts[0]
            start, end = int(parts[1]), int(parts[2])
            state = int(parts[3])
            states[chrom].append((start, end, state))
    # Sort by start
    for chrom in states:
        states[chrom].sort()
    return states


def query_chromhmm(states, chrom, qstart, qend):
    """Get ChromHMM states overlapping a region, return list of (rel_start, rel_end, state)."""
    result = []
    for start, end, state in states.get(chrom, []):
        if end <= qstart:
            continue
        if start >= qend:
            break
        s = max(start, qstart) - qstart
        e = min(end, qend) - qstart
        result.append({"start": s, "end": e, "state": state})
    return result


def load_fasta_index(fasta_path):
    """Build a simple index: chrom -> (offset, line_len, line_bytes)."""
    fai_path = str(fasta_path) + ".fai"
    if os.path.exists(fai_path):
        idx = {}
        with open(fai_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                idx[parts[0]] = {
                    "length": int(parts[1]),
                    "offset": int(parts[2]),
                    "line_bases": int(parts[3]),
                    "line_bytes": int(parts[4]),
                }
        return idx
    return None


def fetch_sequence(fasta_path, idx, chrom, start, end):
    """Fetch sequence from indexed FASTA. 0-based half-open."""
    if idx is None or chrom not in idx:
        return ""
    info = idx[chrom]
    seq_chars = []
    with open(fasta_path, "rb") as f:
        for pos in range(start, end):
            line_num = pos // info["line_bases"]
            line_off = pos % info["line_bases"]
            file_pos = info["offset"] + line_num * info["line_bytes"] + line_off
            f.seek(file_pos)
            seq_chars.append(f.read(1).decode().upper())
    return "".join(seq_chars)


def design_guides(seq, strand, dominant_mark, action):
    """Design top CRISPR guide RNAs for the promoter region.

    For epigenome editing, we target the promoter/enhancer region:
    - Find all NGG PAM sites (SpCas9)
    - Score by: GC content (40-60% optimal), position relative to TSS, avoiding poly-T (Pol III terminator)
    - Return top 3 guides with metadata
    """
    guides = []
    seq = seq.upper()
    n = len(seq)

    # SpCas9 NGG PAM — scan both strands
    for i in range(n - 22):
        # Forward: 20nt guide + NGG at positions i..i+22
        if seq[i+21:i+23] == "GG":
            guide_seq = seq[i:i+20]
            pam = seq[i+20:i+23]
            g_strand = "+"
        # Reverse: CCN + 20nt guide
        elif seq[i:i+2] == "CC":
            guide_seq = complement(seq[i+3:i+23])[::-1]
            pam = complement(seq[i:i+3])[::-1]
            g_strand = "-"
        else:
            continue

        if "N" in guide_seq or len(guide_seq) != 20:
            continue

        # Score
        gc = (guide_seq.count("G") + guide_seq.count("C")) / 20
        gc_score = 1.0 - abs(gc - 0.5) * 2  # peaks at 50%

        # Penalize poly-T (>= 4 T's in a row terminates Pol III)
        polyt_penalty = -0.5 if "TTTT" in guide_seq else 0

        # Position score: prefer -200 to -50 from TSS (end of the window)
        pos_from_tss = i - (n - 200)  # approximate TSS at end of window
        if -400 < pos_from_tss < -30:
            pos_score = 0.3
        elif -800 < pos_from_tss < -20:
            pos_score = 0.15
        else:
            pos_score = 0

        score = gc_score + pos_score + polyt_penalty

        guides.append({
            "sequence": guide_seq,
            "pam": pam,
            "strand": g_strand,
            "position": i,
            "gc_content": round(gc, 3),
            "score": round(score, 3),
        })

    # Sort by score descending, take top 3
    guides.sort(key=lambda g: -g["score"])
    top = guides[:3]

    # Add the editing tool recommendation
    tool_map = {
        "H3K4me1": ("dCas9-LSD1", "Remove H3K4me1 enhancer marks"),
        "H3K4me3": ("dCas9-KDM5B", "Remove H3K4me3 active promoter marks"),
        "H3K27ac": ("dCas9-HDAC3", "Remove H3K27ac activation marks"),
        "H3K27me3": ("dCas9-JMJD3", "Remove H3K27me3 repression marks"),
        "H3K36me3": ("dCas9-KDM4A", "Remove H3K36me3 transcription marks"),
        "H3K9me2": ("dCas9-KDM3A", "Remove H3K9me2 heterochromatin marks"),
        "H3K9K14ac": ("dCas9-HDAC", "Remove H3K9/14ac activation marks"),
    }

    if action == "activate":
        # To activate, we add activating marks or remove repressive ones
        add_tools = {
            "H3K27me3": ("dCas9-JMJD3", "Remove repressive H3K27me3 to activate"),
            "H3K9me2": ("dCas9-KDM3A", "Remove heterochromatin H3K9me2 to activate"),
        }
        tool_info = add_tools.get(dominant_mark, tool_map.get(dominant_mark, ("dCas9-VP64", "Transcriptional activation")))
    else:
        tool_info = tool_map.get(dominant_mark, ("dCas9-KRAB", "Transcriptional silencing"))

    for g in top:
        g["tool"] = tool_info[0]
        g["mechanism"] = tool_info[1]

    return top


def complement(seq):
    comp = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
    return "".join(comp.get(c, "N") for c in seq)


def marks_from_states(chromhmm_intervals, window_size, bin_count=40):
    """Convert ChromHMM state intervals to per-mark signal tracks.

    Returns dict of mark_name -> list of bin_count floats (0-1).
    """
    bin_size = window_size / bin_count
    tracks = {m: [0.0] * bin_count for m in MARKS}

    for interval in chromhmm_intervals:
        s, e, state = interval["start"], interval["end"], interval["state"]
        mark_emissions = STATE_MARKS.get(state, {})
        for bi in range(bin_count):
            bin_start = bi * bin_size
            bin_end = (bi + 1) * bin_size
            overlap = max(0, min(e, bin_end) - max(s, bin_start))
            frac = overlap / bin_size
            if frac > 0:
                for mark, strength in mark_emissions.items():
                    tracks[mark][bi] = max(tracks[mark][bi], strength * frac)

    # Round for JSON size
    for m in tracks:
        tracks[m] = [round(v, 3) for v in tracks[m]]

    return tracks


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Load catalog
    print("Loading editing catalog...")
    targets = []
    with open(CATALOG) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            targets.append(row)
    gene_ids = {t["gene_id"] for t in targets}
    print(f"  {len(targets)} targets")

    # Parse GFF
    print("Parsing GFF3...")
    genes = parse_gff(GFF, gene_ids)
    print(f"  {len(genes)} genes found in GFF")

    # Load ChromHMM
    print("Loading ChromHMM states...")
    red_states = parse_chromhmm(CHROMHMM_RED)
    pret_states = parse_chromhmm(CHROMHMM_PRET)
    print(f"  Red: {sum(len(v) for v in red_states.values())} intervals")
    print(f"  PreT: {sum(len(v) for v in pret_states.values())} intervals")

    # Index FASTA
    print("Indexing FASTA...")
    fasta_idx = load_fasta_index(FASTA)
    if fasta_idx is None:
        print("  No .fai index found, building with samtools...")
        os.system(f'samtools faidx "{FASTA}"')
        fasta_idx = load_fasta_index(FASTA)
    if fasta_idx:
        print(f"  {len(fasta_idx)} chromosomes indexed")
    else:
        print("  WARNING: Cannot index FASTA, primers will be unavailable")

    # Process each target
    print("Processing targets...")
    index = []
    missing = 0

    for i, target in enumerate(targets):
        gid = target["gene_id"]
        if gid not in genes:
            missing += 1
            continue

        gene = genes[gid]
        chrom = gene["chrom"]
        strand = gene["strand"]

        # Define window: 2kb upstream of TSS to 2kb into gene body
        if strand == "+":
            tss = gene["start"]
            win_start = max(0, tss - WINDOW_UP)
            win_end = tss + WINDOW_DOWN
        else:
            tss = gene["end"]
            win_start = max(0, tss - WINDOW_DOWN)
            win_end = tss + WINDOW_UP

        window_size = win_end - win_start

        # Gene model features (relative to window)
        model_features = []
        for feat in gene["features"]:
            if feat["type"] in ("exon", "CDS", "five_prime_UTR", "three_prime_UTR"):
                fs = max(feat["start"], win_start) - win_start
                fe = min(feat["end"], win_end) - win_start
                if fe > fs:
                    model_features.append({
                        "type": feat["type"],
                        "start": fs,
                        "end": fe,
                    })

        # ChromHMM states in window
        red_intervals = query_chromhmm(red_states, chrom, win_start, win_end)
        pret_intervals = query_chromhmm(pret_states, chrom, win_start, win_end)

        # Convert to mark tracks (40 bins across the 4kb window = 100bp resolution)
        red_tracks = marks_from_states(red_intervals, window_size)
        pret_tracks = marks_from_states(pret_intervals, window_size)

        # Fetch promoter sequence for primer design
        if strand == "+":
            prom_start = max(0, tss - WINDOW_UP)
            prom_end = tss + 200  # include a bit downstream
        else:
            prom_start = max(0, tss - 200)
            prom_end = tss + WINDOW_UP

        prom_seq = ""
        guides = []
        if fasta_idx:
            # FASTA is 0-based, GFF is 1-based
            prom_seq = fetch_sequence(FASTA, fasta_idx, chrom, prom_start - 1, prom_end - 1)
            if prom_seq:
                guides = design_guides(
                    prom_seq, strand,
                    target.get("dominant_mark", ""),
                    target.get("edit_action", "silence")
                )

        # Motif position from catalog
        motif_pos_raw = target.get("motif_pos", "")
        motif_info = None
        if motif_pos_raw and motif_pos_raw != "nan":
            try:
                mpos = float(motif_pos_raw)
                # motif_pos is relative to TSS, convert to window coords
                if strand == "+":
                    motif_win = WINDOW_UP + mpos
                else:
                    motif_win = WINDOW_UP - mpos
                motif_info = {
                    "name": target.get("tf_motif", target.get("best_motif", "")),
                    "position": round(motif_win),
                    "relative_to_tss": round(mpos),
                }
            except (ValueError, TypeError):
                pass

        # Build locus JSON
        locus = {
            "gene_id": gid,
            "chrom": chrom,
            "strand": strand,
            "gene_start": gene["start"],
            "gene_end": gene["end"],
            "tss": tss,
            "window": {"start": win_start, "end": win_end, "size": window_size},
            "gene_model": model_features,
            "chromhmm": {
                "red": red_intervals,
                "pret": pret_intervals,
            },
            "mark_tracks": {
                "red": red_tracks,
                "pret": pret_tracks,
            },
            "motif": motif_info,
            "guides": guides,
            "promoter_seq": prom_seq[:200] + "..." if len(prom_seq) > 200 else prom_seq,
            # Catalog metadata
            "dominant_mark": target.get("dominant_mark", ""),
            "edit_action": target.get("edit_action", ""),
            "tissue": target.get("most_affected_tissue", ""),
            "effect": float(target.get("max_effect", 0)),
        }

        # Write per-gene file
        outfile = OUTDIR / f"{gid}.json"
        with open(outfile, "w") as f:
            json.dump(locus, f, separators=(",", ":"))

        index.append({
            "gene_id": gid,
            "chrom": chrom,
            "tss": tss,
            "has_guides": len(guides) > 0,
        })

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(targets)} processed")

    # Write index
    with open(OUTDIR / "index.json", "w") as f:
        json.dump(index, f, separators=(",", ":"))

    print(f"\nDone! {len(index)} locus files written to {OUTDIR}")
    if missing:
        print(f"  ({missing} targets not found in GFF)")


if __name__ == "__main__":
    main()
