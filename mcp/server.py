#!/usr/bin/env python3
"""PLANTED MCP Server — Plant Editing Target Atlas.

Exposes the PLANTED editing catalog and locus data as MCP tools
that Claude can call mid-conversation.

Tools:
  planted_search   — find editing targets by gene, TF, tissue, phenotype
  planted_target   — get full detail for a specific gene
  planted_species  — species summary and metrics
  planted_primer   — get CRISPR guide RNAs for a target
  planted_explain  — explain why a target was prioritized
"""

import json
import os
from pathlib import Path
from mcp.server.mcpserver import MCPServer

# ─── Data paths ───────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent  # planted/
ATLAS_DATA = BASE / "atlas" / "public" / "data"
LOCUS_DIR = ATLAS_DATA / "locus"

# ─── Load data ────────────────────────────────────────────
def load_json(path):
    with open(path) as f:
        return json.load(f)

species_data = load_json(ATLAS_DATA / "species_summary.json")
catalog_data = load_json(ATLAS_DATA / "strawberry_catalog.json")
showcases_data = load_json(ATLAS_DATA / "showcases.json")

# Build lookup index
catalog_by_gene = {t["gene_id"]: t for t in catalog_data}

# ─── MCP Server ───────────────────────────────────────────
mcp = MCPServer(
    "planted",
    instructions="""PLANTED Atlas: chromatin-informed gene editing targets across 5 plant species.
Use these tools to search editing targets, get target details, design CRISPR guides,
and understand why targets were prioritized. Data covers wheat, rice, arabidopsis,
strawberry, and tomato — with the strawberry catalog having the most detail (372 targets)."""
)


@mcp.tool()
def planted_search(
    query: str = "",
    tissue: str = "",
    action: str = "",
    tier: str = "",
    mark: str = "",
    limit: int = 10,
) -> str:
    """Search PLANTED editing targets by gene ID, TF motif, tissue, action, tier, or mark.

    Args:
        query: Free text — matches gene_id, tf_motif, go_category, phenotype, description
        tissue: Filter by most affected tissue (e.g. 'late-fruit', 'leaf', 'root', 'seed')
        action: 'activate' or 'silence'
        tier: 'Tier 1', 'Tier 2', 'Tier 3', or 'Tier 4'
        mark: Dominant histone mark (e.g. 'H3K4me1', 'H3K27me3')
        limit: Max results (default 10)
    """
    results = []
    q = query.lower()

    for t in catalog_data:
        if q:
            searchable = " ".join([
                t.get("gene_id", ""), t.get("tf_motif", ""),
                t.get("go_category", ""), t.get("phenotype", ""),
                t.get("description", ""), str(t.get("tissue", "")),
            ]).lower()
            if q not in searchable:
                continue
        if tissue and t.get("tissue", "").lower() != tissue.lower():
            continue
        if action and t.get("action", "").lower() != action.lower():
            continue
        if tier and tier.lower() not in t.get("tier", "").lower():
            continue
        if mark and t.get("mark", "").lower() != mark.lower():
            continue
        results.append(t)

    results.sort(key=lambda x: -x.get("score", 0))
    results = results[:limit]

    if not results:
        return "No targets found matching your criteria."

    lines = [f"Found {len(results)} targets:\n"]
    for t in results:
        lines.append(
            f"- **{t['gene_id']}** ({t.get('tier', '?')}, score {t.get('score', 0):.2f})\n"
            f"  {t.get('tissue', '?')} | {t.get('mark', '?')} | {t.get('action', '?')} | effect {t.get('effect', 0):.3f}\n"
            f"  TF: {t.get('tf_motif', 'none')} | {t.get('coherence', '?').replace('_', ' ')}\n"
            f"  {t.get('phenotype', 'No phenotype prediction')[:120]}"
        )
    return "\n".join(lines)


@mcp.tool()
def planted_target(gene_id: str) -> str:
    """Get full details for a specific editing target.

    Args:
        gene_id: The gene identifier (e.g. 'FvesChr1G00284950')
    """
    t = catalog_by_gene.get(gene_id)
    if not t:
        # Try partial match
        matches = [g for g in catalog_by_gene if gene_id.lower() in g.lower()]
        if matches:
            return f"Gene '{gene_id}' not found. Did you mean: {', '.join(matches[:5])}?"
        return f"Gene '{gene_id}' not found in the catalog ({len(catalog_data)} targets available)."

    return f"""## {t['gene_id']}

**{t.get('tier', '?')}** | Priority score: {t.get('score', 0):.2f}

### Editing Strategy
- Tissue: {t.get('tissue', '?')}
- Dominant mark: {t.get('mark', '?')} ({t.get('mark_role', '?')})
- Action: {t.get('action', '?')}
- Effect size: {t.get('effect', 0):.3f}
- Specificity: {t.get('specificity', '?')}
- Tissues affected: {t.get('n_tissues', '?')}
- CRISPR tool: {t.get('tool', '?')}

### Transcription Factor
- TF motif: {t.get('tf_motif', 'none')}
- TF expression: {t.get('tf_tpm', '?')} TPM
- TF expressed: {t.get('tf_expressed', '?')}
- Co-expression rho: {t.get('coexpr_rho', '?')}
- Coherence: {t.get('coherence', '?').replace('_', ' ')}
- Motif position: {t.get('motif_pos', '?')}
- Upstream regulators: {t.get('n_regulators', '?')}

### Multi-Omics Evidence
- Evidence layers: {t.get('n_evidence', '?')}
- eQTL support: {t.get('has_eqtl', False)} ({t.get('n_eqtl', 0)} eQTLs)
- Cell-type: {t.get('celltype', 'none')} (tau={t.get('tau_celltype', '?')})
- Stage: {t.get('stage', '?')}

### Annotation
- GO category: {t.get('go_category', 'unknown')}
- Description: {t.get('description', 'none')}

### Predicted Phenotype
{t.get('phenotype', 'No phenotype prediction')}"""


@mcp.tool()
def planted_species(species_name: str = "") -> str:
    """Get species summary and metrics from the PLANTED atlas.

    Args:
        species_name: Optional — 'wheat', 'rice', 'arabidopsis', 'strawberry', or 'tomato'.
                      If empty, returns overview of all species.
    """
    if not species_name:
        lines = ["## PLANTED Atlas — 5 Species\n"]
        for key, sp in species_data.items():
            lines.append(
                f"### {sp['icon']} {sp['name']} ({sp['latin']})\n"
                f"- Tier {sp['tier']} | Model r = {sp['model_r']}\n"
                f"- {sp['genes']:,} genes assessed, {sp['editable']:,} editable, {sp['tissue_specific']:,} tissue-specific\n"
                f"- {sp['tissues']} tissues, {sp['marks']} marks"
                + (f", {sp['eqtl_n']:,} eQTL-supported ({sp['eqtl_pct']}%)" if sp['eqtl_n'] else "")
                + "\n"
            )
        return "\n".join(lines)

    sp = species_data.get(species_name.lower())
    if not sp:
        return f"Species '{species_name}' not found. Available: {', '.join(species_data.keys())}"

    return f"""## {sp['icon']} {sp['name']} — {sp['latin']}

**Tier {sp['tier']}** | Model Pearson r = {sp['model_r']}

| Metric | Value |
|--------|-------|
| Genes assessed | {sp['genes']:,} |
| Editable | {sp['editable']:,} ({sp['editable']/sp['genes']*100:.1f}%) |
| Tissue-specific | {sp['tissue_specific']:,} |
| Tissues profiled | {sp['tissues']} |
| Histone marks | {sp['marks']} |
| eQTL-supported | {sp['eqtl_n']:,} ({sp['eqtl_pct']}%) |"""


@mcp.tool()
def planted_primer(gene_id: str) -> str:
    """Get pre-designed CRISPR guide RNAs for epigenome editing of a target gene.

    Returns top 3 SpCas9 guides with sequence, PAM, GC content, and recommended
    editing tool (e.g. dCas9-LSD1 for H3K4me1 removal).

    Args:
        gene_id: The gene identifier (e.g. 'FvesChr1G00284950')
    """
    locus_file = LOCUS_DIR / f"{gene_id}.json"
    if not locus_file.exists():
        return f"No locus data for '{gene_id}'. Available for {len(list(LOCUS_DIR.glob('Fves*.json')))} strawberry targets."

    locus = load_json(locus_file)
    guides = locus.get("guides", [])

    if not guides:
        return f"No CRISPR guides could be designed for {gene_id}."

    lines = [
        f"## CRISPR Guides for {gene_id}\n",
        f"Target: {locus['dominant_mark']} at {locus['chrom']}:{locus['tss']:,} ({locus['strand']} strand)\n",
        f"Action: {locus['edit_action']} (predicted effect: {locus['effect']:.3f})\n",
        f"Recommended tool: **{guides[0]['tool']}** — {guides[0]['mechanism']}\n",
        "### Guide sequences (SpCas9, NGG PAM)\n",
        "| # | Sequence + PAM | Strand | GC% | Score |",
        "|---|----------------|--------|-----|-------|",
    ]

    for i, g in enumerate(guides):
        lines.append(
            f"| {i+1} | `{g['sequence']}`**`{g['pam']}`** | {g['strand']} | {g['gc_content']*100:.0f}% | {g['score']:.2f} |"
        )

    lines.append(f"\nPromoter region: {locus['chrom']}:{locus['window']['start']:,}-{locus['window']['end']:,}")
    lines.append(f"Sequence (first 200bp): `{locus.get('promoter_seq', 'N/A')[:100]}...`")

    return "\n".join(lines)


@mcp.tool()
def planted_explain(gene_id: str) -> str:
    """Explain why a target was prioritized — break down the evidence layers and scoring.

    Args:
        gene_id: The gene identifier (e.g. 'FvesChr1G00284950')
    """
    t = catalog_by_gene.get(gene_id)
    if not t:
        return f"Gene '{gene_id}' not found in catalog."

    # Break down priority score components
    has_edit = t.get('has_editability', False)
    has_eqtl = t.get('has_eqtl', False)
    has_cell = t.get('has_celltype', False)
    coherence = t.get('coherence', 'unknown')
    effect = t.get('effect', 0)
    n_evidence = t.get('n_evidence', 0)

    lines = [
        f"## Why {gene_id} is prioritized\n",
        f"**{t.get('tier', '?')}** — Priority score: {t.get('score', 0):.2f}\n",
        "### Evidence layers\n",
    ]

    if has_edit:
        lines.append(f"1. **Chromatin editability** (ChromatinFiLM model)\n"
                     f"   - Dominant mark: {t.get('mark', '?')} ({t.get('mark_role', '?')})\n"
                     f"   - Predicted effect: {effect:.3f} (expression change when mark is perturbed)\n"
                     f"   - Tissue: {t.get('tissue', '?')} | Specificity: {t.get('specificity', '?')}\n")

    if has_eqtl:
        lines.append(f"2. **eQTL support** (Fan et al. 2022)\n"
                     f"   - {t.get('n_eqtl', 0)} eQTLs mapped to this locus\n"
                     f"   - Natural genetic variation confirms this gene's expression is regulatable\n")

    if has_cell:
        lines.append(f"3. **Cell-type specificity** (snRNA-seq)\n"
                     f"   - Dominant cell type: {t.get('celltype', '?')}\n"
                     f"   - Cell-type tau: {t.get('tau_celltype', '?')}\n"
                     f"   - Stage: {t.get('stage', '?')}\n")

    lines.append(f"### TF coherence: {coherence.replace('_', ' ')}\n")
    if coherence == 'tf_active':
        lines.append("The TF motif in the promoter belongs to a transcription factor that is "
                     "actively expressed in the target tissue AND whose expression correlates "
                     "with the target gene. This is the strongest evidence for functional regulation.\n")
    elif coherence == 'tf_present':
        lines.append("The TF is expressed in the tissue but co-expression with the target is weak. "
                     "The motif may be functional but the regulatory relationship is indirect.\n")
    elif coherence == 'tf_absent':
        lines.append("The TF is not expressed in the target tissue. The motif may be non-functional "
                     "or may reflect regulation in a different developmental context.\n")

    lines.append(f"### Tier assignment\n")
    tier = t.get('tier', '')
    if 'Tier 1' in tier:
        lines.append("**Tier 1 (Highest confidence)**: All three evidence layers present "
                     "(chromatin editability + eQTL + cell-type specificity). "
                     "These are the most actionable targets.\n")
    elif 'Tier 2' in tier:
        lines.append("**Tier 2 (High confidence)**: Two of three evidence layers present. "
                     "Strong candidates for experimental validation.\n")
    elif 'Tier 3' in tier:
        lines.append("**Tier 3 (Moderate)**: Single evidence layer, typically chromatin-only. "
                     "Useful for exploratory editing experiments.\n")

    return "\n".join(lines)


# Load phenotype index
pheno_index_path = ATLAS_DATA / "phenotype_index.json"
phenotype_index = load_json(pheno_index_path) if pheno_index_path.exists() else {}

# Build phenotype categories per target
PHENO_CATEGORIES = {
    'Fruit quality': ['fruit', 'ripen', 'flavor', 'sugar', 'color', 'softening', 'anthocyanin', 'aroma'],
    'Stress tolerance': ['stress', 'drought', 'salt', 'cold', 'heat', 'defense', 'pathogen', 'disease', 'resistance'],
    'Nutrition': ['nutrient', 'vitamin', 'lipid', 'protein', 'amino', 'nitrogen', 'mineral', 'antioxidant'],
    'Shelf life': ['shelf', 'storage', 'senescence', 'decay', 'postharvest', 'firmness'],
    'Yield & growth': ['yield', 'growth', 'biomass', 'size', 'flower', 'pollen', 'seed', 'fertility', 'root'],
    'Hormone signaling': ['hormone', 'ethylene', 'auxin', 'gibberellin', 'abscisic', 'jasmonate'],
    'Cell wall & structure': ['cell wall', 'lignin', 'cellulose', 'pectin'],
    'Signal transduction': ['signal', 'kinase', 'receptor', 'calcium'],
    'Transport': ['transport', 'allocation', 'loading', 'channel', 'transporter'],
}


@mcp.tool()
def planted_phenotypes(
    category: str = "",
    query: str = "",
    limit: int = 10,
) -> str:
    """Browse predicted phenotypes and find editing targets by trait.

    Lists phenotype categories with target counts, or searches phenotype
    predictions across all targets.

    Args:
        category: Filter by phenotype category (e.g. 'Fruit quality', 'Stress tolerance',
                  'Nutrition', 'Shelf life', 'Yield & growth', 'Hormone signaling').
                  Leave empty to see all categories.
        query: Free text search across phenotype predictions (e.g. 'sugar', 'drought',
               'anthocyanin', 'ethylene'). Searches the full phenotype text of all targets.
        limit: Max results for query search (default 10)
    """
    # If query given, search phenotype text
    if query:
        q = query.lower()
        results = [t for t in catalog_data if q in t.get('phenotype', '').lower()]
        results.sort(key=lambda x: -x.get('score', 0))
        results = results[:limit]

        if not results:
            return f"No targets found with phenotype matching '{query}'."

        lines = [f"Found {len(results)} targets matching '{query}':\n"]
        for t in results:
            lines.append(
                f"- **{t['gene_id']}** ({t.get('tier', '?')}, score {t.get('score', 0):.2f})\n"
                f"  {t.get('tissue', '?')} | {t.get('action', '?')} | {t.get('mark', '?')}\n"
                f"  Phenotype: {t.get('phenotype', '')[:150]}"
            )
        return "\n".join(lines)

    # If category given, show targets in that category
    if category:
        cat_data = phenotype_index.get(category)
        if not cat_data:
            available = ", ".join(phenotype_index.keys())
            return f"Category '{category}' not found. Available: {available}"

        lines = [
            f"## {category} — {cat_data['count']} targets\n",
            "### Top targets\n",
        ]
        for t in cat_data['top_targets']:
            lines.append(
                f"- **{t['gene_id']}** ({t.get('tier', '?')}, score {t.get('score', 0):.2f})\n"
                f"  {t.get('tissue', '?')} | {t.get('action', '?')} | {t.get('mark', '')} | {t.get('tool', '')}\n"
                f"  {t.get('phenotype', '')[:150]}"
            )
        return "\n".join(lines)

    # Default: show all categories overview
    lines = ["## Phenotype Categories\n",
             "All 372 strawberry targets have predicted phenotypes. "
             "Targets can belong to multiple categories.\n",
             "| Category | Targets | Top gene |",
             "|----------|---------|----------|"]

    for cat, data in sorted(phenotype_index.items(), key=lambda x: -x[1]['count']):
        top = data['top_targets'][0] if data['top_targets'] else {}
        lines.append(f"| {cat} | {data['count']} | {top.get('gene_id', '-')} ({top.get('tier', '?')}) |")

    lines.append(f"\nUse `planted_phenotypes(category='Fruit quality')` to see top targets in a category,")
    lines.append(f"or `planted_phenotypes(query='sugar')` to search phenotype predictions.")

    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
