#!/usr/bin/env python3
"""Build web-ready catalog JSONs for wheat, rice, arabidopsis, tomato.

Reads each species' gene_editability_catalog.tsv, enriches with GO annotations,
generates phenotype predictions, and outputs JSON matching the strawberry catalog format.

For large catalogs (wheat, tomato), we take top ~500 targets by effect × specificity.
"""

import json
import os
import re
from collections import defaultdict

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "atlas", "public", "data")

# ── Tissue context per species ───────────────────────────────────────
TISSUE_CONTEXT = {
    "rice": {
        "young_leaf": "Vegetative growth, early photosynthesis",
        "mature_leaf": "Photosynthesis, source tissue for assimilate",
        "panicle": "Grain yield, spikelet development, flowering",
        "root": "Root architecture, nutrient/water uptake",
    },
    "arabidopsis": {
        "leaf": "Photosynthesis, vegetative growth",
        "root": "Root development, nutrient uptake",
        "flower": "Floral organ identity, reproductive success",
        "seed": "Seed development, dormancy, storage reserves",
        "fruit": "Silique development, seed dispersal",
        "seedling": "Early establishment, light response",
    },
    "tomato": {
        "flower_bud": "Floral meristem, flower initiation",
        "flower_open": "Pollination, petal/sepal identity",
        "fruit_1cm": "Fruit set, early cell division",
        "fruit_2cm": "Fruit expansion, cell division",
        "fruit_3cm": "Fruit growth, metabolite accumulation",
        "fruit_mature_green": "Pre-ripening, chloroplast-to-chromoplast",
        "fruit_breaker": "Ripening onset, ethylene burst, color change",
        "fruit_turning": "Active ripening, carotenoid/lycopene synthesis",
        "leaf": "Photosynthesis, vegetative growth",
        "root": "Root architecture, nutrient uptake",
    },
    "wheat": {
        "leaf": "Flag leaf photosynthesis, source capacity",
        "spike": "Grain development, spikelet fertility, yield",
        "vl": "Vegetative leaf, early growth, canopy",
        "e5": "Endosperm 5 DPA, early grain filling",
        "e7": "Endosperm 7 DPA, starch deposition onset",
        "e9": "Endosperm 9 DPA, storage protein accumulation",
    },
}

# ── Mark strategy ────────────────────────────────────────────────────
MARK_STRATEGY = {
    "H3K4me1": ("enhancer priming", "dCas9-LSD1 to remove; dCas9-MLL to add"),
    "H3K4me3": ("promoter activation", "dCas9-LSD1 to silence; dCas9-SET1 to activate"),
    "H3K27me3": ("Polycomb repression", "dCas9-EZH2 to silence; dCas9-UTX to de-repress"),
    "H3K27ac": ("active enhancer", "dCas9-HDAC to silence; dCas9-p300 to activate"),
    "H3K36me3": ("transcription elongation", "dCas9-SETD2 to mark"),
    "H3K9me2": ("heterochromatin", "dCas9-SUV39H to silence; dCas9-KDM4 to open"),
    "ATAC": ("open chromatin", "dCas9-KRAB to close; CRISPRa to enhance"),
}

# ── GO-based phenotype keywords ──────────────────────────────────────
PHENO_KEYWORDS = {
    "Fruit quality": ["fruit", "ripen", "flavor", "sugar", "color", "softening", "anthocyanin", "aroma", "sweetness", "acidity", "taste", "carotenoid", "lycopene"],
    "Stress tolerance": ["stress", "drought", "salt", "cold", "heat", "defense", "pathogen", "disease", "resistance", "immune", "oxidative", "wounding"],
    "Nutrition": ["nutrient", "vitamin", "lipid", "protein", "amino", "nitrogen", "mineral", "antioxidant", "phenolic", "starch", "storage"],
    "Shelf life": ["shelf", "storage", "senescence", "decay", "postharvest", "firmness", "perishab", "cell wall", "pectin"],
    "Yield & growth": ["yield", "growth", "biomass", "size", "flower", "pollen", "seed", "fertility", "root", "vegetative", "branching", "tillering", "grain", "spikelet"],
    "Hormone signaling": ["hormone", "ethylene", "auxin", "gibberellin", "abscisic", "jasmonate", "cytokinin", "brassinosteroid"],
    "Cell wall & structure": ["cell wall", "lignin", "cellulose", "pectin", "xylan", "structural", "cutin", "suberin"],
    "Signal transduction": ["signal", "kinase", "receptor", "calcium", "phosphorylation", "MAPK"],
    "Transport": ["transport", "allocation", "loading", "channel", "transporter", "efflux", "ABC"],
}


def load_go_annotations(species):
    """Load GO term names per gene for a species."""
    go_dir = os.path.join(BASE, "data", "go_annotations")
    gene_go = defaultdict(set)

    if species == "rice":
        path = os.path.join(go_dir, "rice_go.tsv")
        if os.path.exists(path):
            df = pd.read_csv(path, sep="\t")
            for _, row in df.iterrows():
                gid = str(row.iloc[0])
                name = str(row.get("GO term name", ""))
                if name and name != "nan":
                    gene_go[gid].add(name)

    elif species == "arabidopsis":
        path = os.path.join(go_dir, "tair.gaf")
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    if line.startswith("!"):
                        continue
                    parts = line.strip().split("\t")
                    if len(parts) >= 10:
                        gid = parts[1]
                        go_name = parts[4] if len(parts) > 4 else ""
                        if gid.startswith("AT") and go_name:
                            gene_go[gid].add(go_name)

    elif species == "tomato":
        path = os.path.join(go_dir, "tomato_go_fixed.tsv")
        if os.path.exists(path):
            df = pd.read_csv(path, sep="\t")
            for _, row in df.iterrows():
                gid = str(row.iloc[0]).split(".")[0]  # strip version
                name = str(row.get("GO term name", ""))
                if name and name != "nan":
                    gene_go[gid].add(name)

    elif species == "wheat":
        path = os.path.join(go_dir, "wheat_go_transferred.tsv")
        if os.path.exists(path):
            df = pd.read_csv(path, sep="\t")
            for _, row in df.iterrows():
                gid = str(row.iloc[0])
                name = str(row.get("GO term name", ""))
                if name and name != "nan":
                    gene_go[gid].add(name)

    return gene_go


def categorize_go(go_terms):
    """Assign a simple GO category from term names."""
    text = " ".join(go_terms).lower()
    if any(w in text for w in ["transcription", "dna-binding", "regulation of transcription"]):
        return "transcription"
    if any(w in text for w in ["kinase", "signal", "receptor"]):
        return "signaling"
    if any(w in text for w in ["metabol", "biosynthes", "catabol"]):
        return "metabolism"
    if any(w in text for w in ["transport", "channel", "carrier"]):
        return "transport"
    if any(w in text for w in ["development", "morphogenesis", "differentiation"]):
        return "development"
    if any(w in text for w in ["cell cycle", "mitosis", "division"]):
        return "cell_cycle"
    if any(w in text for w in ["binding", "protein binding"]):
        return "binding"
    return "unknown"


def predict_phenotype(tissue, mark, action, effect, go_terms, go_cat, species):
    """Generate a phenotype prediction string."""
    ctx = TISSUE_CONTEXT.get(species, {})
    tissue_desc = ctx.get(tissue, tissue.replace("_", " "))

    mark_info = MARK_STRATEGY.get(mark, ("chromatin modification", "epigenome editing"))
    mark_desc, _ = mark_info

    parts = [f"Tissue context: {tissue_desc}"]

    go_text = "; ".join(sorted(go_terms)[:5]) if go_terms else ""
    if go_text:
        parts.append(f"Function: {go_text}")

    if go_cat == "transcription":
        parts.append("Regulatory: may alter downstream gene networks")
    elif go_cat == "signaling":
        parts.append("Signal transduction: may affect hormone/stress response")
    elif go_cat == "metabolism":
        parts.append("Metabolic: may alter composition or nutritional content")
    elif go_cat == "transport":
        parts.append("Transport: may affect nutrient allocation or ion homeostasis")
    elif go_cat == "development":
        parts.append("Developmental: direct role in organ formation")

    parts.append(
        f"Editing goal: {'increase' if action == 'activate' else 'decrease'} expression "
        f"via {mark_desc} (effect={effect:.2f})"
    )
    return "; ".join(parts)


def assign_phenotype_categories(phenotype_text):
    """Assign phenotype categories from keyword matching."""
    cats = []
    text = phenotype_text.lower()
    for cat, kws in PHENO_KEYWORDS.items():
        if any(kw in text for kw in kws):
            cats.append(cat)
    return cats if cats else ["Other"]


def compute_tier(effect, specificity, n_tissues, has_tf):
    """Assign tier 1-4 based on evidence strength."""
    score = abs(effect) * 2
    if specificity == "highly_specific":
        score += 3
    elif specificity == "moderately_specific":
        score += 1
    if n_tissues == 1:
        score += 1
    if has_tf:
        score += 1
    if score >= 5:
        return "Tier 1"
    elif score >= 3:
        return "Tier 2"
    elif score >= 1.5:
        return "Tier 3"
    return "Tier 4"


def build_catalog(species, max_targets=500):
    """Build a web catalog JSON for a species."""
    if species == "wheat":
        tsv_path = os.path.join(BASE, "results", "editability_catalog", "gene_editability_catalog.tsv")
    else:
        tsv_path = os.path.join(BASE, "results", f"{species}_editability", "gene_editability_catalog.tsv")

    print(f"\n{'='*60}")
    print(f"Building {species} catalog from {tsv_path}")

    df = pd.read_csv(tsv_path, sep="\t")
    print(f"  Raw rows: {len(df)}")

    # Filter to expressed genes with meaningful effects
    if "is_expressed" in df.columns:
        df = df[df["is_expressed"] == True]
    df = df[df["max_effect"].abs() > 0.1]
    print(f"  After filtering (expressed + effect > 0.1): {len(df)}")

    # Compute priority score
    spec_bonus = df["specificity_category"].map({
        "highly_specific": 3.0,
        "moderately_specific": 1.0,
        "constitutive": 0.0,
    }).fillna(0)
    df = df.copy()
    df["priority"] = df["max_effect"].abs() * 2 + spec_bonus
    if "n_tissues_affected" in df.columns:
        df["priority"] += (df["n_tissues_affected"] == 1).astype(float)

    # Sort and take top targets
    df = df.sort_values("priority", ascending=False).head(max_targets)
    print(f"  Selected top {len(df)} targets")

    # Load GO annotations
    go_map = load_go_annotations(species)
    print(f"  GO annotations loaded: {len(go_map)} genes")

    # Get tissue columns
    baseline_cols = [c for c in df.columns if c.startswith("baseline_")]
    delta_cols = [c for c in df.columns if c.startswith("delta_")]
    tissues = [c.replace("baseline_", "") for c in baseline_cols]

    catalog = []
    for _, row in df.iterrows():
        gene_id = str(row["gene_id"])
        # Strip version suffix for GO lookup
        gene_id_base = gene_id.split(".")[0]

        go_terms = go_map.get(gene_id, set()) | go_map.get(gene_id_base, set())
        go_cat = categorize_go(go_terms)
        go_desc = "; ".join(sorted(go_terms)[:8]) if go_terms else ""

        tissue = str(row.get("most_affected_tissue", ""))
        mark = str(row.get("dominant_mark", ""))
        mark_role = str(row.get("mark_role", ""))
        action = str(row.get("edit_action", ""))
        effect = float(row.get("max_effect", 0))
        specificity = str(row.get("specificity_category", "constitutive"))
        n_tissues = int(row.get("n_tissues_affected", 1)) if "n_tissues_affected" in row.index else 1

        tf = str(row.get("top_tf", "")) if pd.notna(row.get("top_tf")) else ""
        tf_imp = float(row.get("top_tf_importance", 0)) if pd.notna(row.get("top_tf_importance")) else 0
        n_reg = int(row.get("n_regulators", 0)) if "n_regulators" in row.index and pd.notna(row.get("n_regulators")) else 0

        mark_info = MARK_STRATEGY.get(mark, ("chromatin modification", "epigenome editing"))
        tool = mark_info[1]

        tier = compute_tier(effect, specificity, n_tissues, bool(tf))
        priority = float(row["priority"])

        phenotype = predict_phenotype(tissue, mark, action, effect, go_terms, go_cat, species)
        pheno_cats = assign_phenotype_categories(phenotype)

        entry = {
            "gene_id": gene_id,
            "tier": tier,
            "score": round(priority, 2),
            "tissue": tissue,
            "mark": mark,
            "mark_role": mark_role,
            "action": action,
            "effect": round(abs(effect), 3),
            "n_tissues": n_tissues,
            "specificity": specificity,
            "n_evidence": 1 + (1 if go_terms else 0) + (1 if tf else 0),
            "has_eqtl": False,
            "has_celltype": False,
            "coherence": "tf_active" if tf else "no_tf",
            "tf_motif": tf,
            "tf_tpm": None,
            "tf_expressed": bool(tf),
            "coexpr_rho": None,
            "go_category": go_cat,
            "description": go_desc[:300] if go_desc else "",
            "phenotype": phenotype,
            "tool": tool,
            "n_eqtl": 0,
            "celltype": "",
            "tau_celltype": None,
            "stage": "",
            "motif_pos": "",
            "n_regulators": n_reg,
            "phenotype_categories": pheno_cats,
        }
        catalog.append(entry)

    # Sort by score descending
    catalog.sort(key=lambda x: -x["score"])

    out_path = os.path.join(OUT, f"{species}_catalog.json")
    with open(out_path, "w") as f:
        json.dump(catalog, f, indent=None, separators=(",", ":"))
    print(f"  Written {len(catalog)} targets → {out_path}")
    sz = os.path.getsize(out_path)
    print(f"  File size: {sz/1024:.0f} KB")

    # Build phenotype index for this species
    pheno_counts = defaultdict(int)
    for t in catalog:
        for cat in t.get("phenotype_categories", []):
            pheno_counts[cat] += 1

    return catalog, dict(pheno_counts)


def main():
    all_pheno = defaultdict(lambda: {"count": 0, "top_targets": []})

    for species, max_t in [("rice", 500), ("arabidopsis", 500), ("tomato", 500), ("wheat", 500)]:
        catalog, pheno_counts = build_catalog(species, max_targets=max_t)
        for cat, count in pheno_counts.items():
            all_pheno[cat]["count"] += count

    # Also rebuild combined phenotype index from all catalogs
    # (strawberry already has its own)
    print("\nDone. Catalogs written to:", OUT)


if __name__ == "__main__":
    main()
