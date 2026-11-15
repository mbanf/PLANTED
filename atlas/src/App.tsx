import { useState, useEffect, useRef } from 'react'
import './App.css'
import { LocusBrowser } from './LocusBrowser'

// Types
interface Species {
  name: string; latin: string; tier: number; model_r: number
  genes: number; editable: number; tissue_specific: number
  tissues: number; marks: number; eqtl_n: number; eqtl_pct: number; icon: string
}

interface Target {
  gene_id: string; tier: string; score: number; tissue: string
  mark: string; mark_role: string; action: string; effect: number
  n_tissues: number; specificity: string; n_evidence: number
  has_eqtl: boolean; has_celltype: boolean; coherence: string
  tf_motif: string; tf_tpm: number | null; tf_expressed: boolean
  coexpr_rho: number | null; go_category: string; description: string
  phenotype: string; tool: string; n_eqtl: number; celltype: string
  tau_celltype: number | null; stage: string; motif_pos: string
  n_regulators: number; phenotype_categories?: string[]
}

interface PhenoCategory {
  count: number
  top_targets: { gene_id: string; tier: string; score: number; tissue: string; action: string; effect: number; phenotype: string; mark: string; tool: string }[]
}

interface Showcase {
  id: string; species: string; title: string; tier: string
  tissue: string; effect: number; mark: string; action: string
  tf: string; tf_tpm: number; coexpr: number; n_eqtl: number
  tool: string; summary: string; use_case: string
}

type View = 'home' | 'species' | 'catalog' | 'target' | 'phenotypes' | 'methods'

const SPECIES_CATALOG_FILES: Record<string, string> = {
  strawberry: '/data/strawberry_catalog.json',
  rice: '/data/rice_catalog.json',
  arabidopsis: '/data/arabidopsis_catalog.json',
  tomato: '/data/tomato_catalog.json',
  wheat: '/data/wheat_catalog.json',
}

const SPECIES_LABELS: Record<string, { name: string; icon: string }> = {
  strawberry: { name: 'Strawberry', icon: '\uD83C\uDF53' },
  wheat: { name: 'Wheat', icon: '\uD83C\uDF3E' },
  rice: { name: 'Rice', icon: '\uD83C\uDF3E' },
  arabidopsis: { name: 'Arabidopsis', icon: '\uD83C\uDF3F' },
  tomato: { name: 'Tomato', icon: '\uD83C\uDF45' },
}

function App() {
  const [view, setView] = useState<View>('home')
  const [species, setSpecies] = useState<Record<string, Species>>({})
  const [catalogs, setCatalogs] = useState<Record<string, Target[]>>({})
  const [catalogSpecies, setCatalogSpecies] = useState<string>('strawberry')
  const [showcases, setShowcases] = useState<Showcase[]>([])
  const [selectedSpecies, setSelectedSpecies] = useState<string | null>(null)
  const [selectedTarget, setSelectedTarget] = useState<Target | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [filterTier, setFilterTier] = useState<string>('all')
  const [filterTissue, setFilterTissue] = useState<string>('all')
  const [filterAction, setFilterAction] = useState<string>('all')
  const [filterUseCase, setFilterUseCase] = useState<string>('all')
  const [phenoIndex, setPhenoIndex] = useState<Record<string, PhenoCategory>>({})
  const [scrolled, setScrolled] = useState(false)
  const mainRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    fetch('/data/species_summary.json').then(r => r.json()).then(setSpecies)
    fetch('/data/showcases.json').then(r => r.json()).then(setShowcases)
    fetch('/data/phenotype_index.json').then(r => r.json()).then(setPhenoIndex)
    // Load all species catalogs
    for (const [sp, url] of Object.entries(SPECIES_CATALOG_FILES)) {
      fetch(url).then(r => r.json()).then(data => {
        setCatalogs(prev => ({ ...prev, [sp]: data }))
      })
    }
    const onScroll = () => setScrolled(window.scrollY > 20)
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  // scroll to top on view change
  useEffect(() => { window.scrollTo(0, 0) }, [view])

  const catalog = catalogs[catalogSpecies] || []
  const allCatalogTargets = Object.values(catalogs).flat()

  const navigate = (v: View, sp?: string) => {
    setView(v)
    if (sp) setSelectedSpecies(sp)
  }

  const openCatalog = (sp: string) => {
    setCatalogSpecies(sp)
    setSearchQuery('')
    setFilterTier('all')
    setFilterTissue('all')
    setFilterAction('all')
    setView('catalog')
  }

  const openTarget = (t: Target) => {
    setSelectedTarget(t)
    setView('target')
  }

  const filteredCatalog = catalog.filter(t => {
    if (searchQuery && !t.gene_id.toLowerCase().includes(searchQuery.toLowerCase()) &&
        !t.tf_motif.toLowerCase().includes(searchQuery.toLowerCase()) &&
        !t.go_category.toLowerCase().includes(searchQuery.toLowerCase())) return false
    if (filterTier !== 'all' && !t.tier.includes(filterTier)) return false
    if (filterTissue !== 'all' && t.tissue !== filterTissue) return false
    if (filterAction !== 'all' && t.action !== filterAction) return false
    return true
  })

  const tissues = [...new Set(catalog.map(t => t.tissue))].sort()
  const useCaseShowcases = filterUseCase === 'all' ? showcases : showcases.filter(s => s.use_case === filterUseCase)
  const totalEditable = Object.values(species).reduce((s, sp) => s + sp.editable, 0)
  const totalGenes = Object.values(species).reduce((s, sp) => s + sp.genes, 0)

  return (
    <div className="app" ref={mainRef}>
      {/* Topbar */}
      <header className={`topbar${scrolled ? ' scrolled' : ''}`}>
        <div className="brand" onClick={() => setView('home')}>
          PLANTED <span>atlas</span>
        </div>
        <nav>
          <button className={view === 'home' ? 'active' : ''} onClick={() => setView('home')}>Home</button>
          <button className={view === 'species' ? 'active' : ''} onClick={() => navigate('species')}>Species</button>
          <button className={view === 'catalog' ? 'active' : ''} onClick={() => navigate('catalog')}>Catalog</button>
          <button className={view === 'phenotypes' ? 'active' : ''} onClick={() => setView('phenotypes')}>Phenotypes</button>
          <button className={view === 'methods' ? 'active' : ''} onClick={() => setView('methods')}>Methods</button>
        </nav>
        <div className="right">
          <span className="eyebrow"><span className="status-pip" /> 5 species</span>
        </div>
      </header>

      {/* Content */}
      {view === 'home' && <HomeView species={species} showcases={useCaseShowcases}
        totalEditable={totalEditable} totalGenes={totalGenes}
        filterUseCase={filterUseCase} setFilterUseCase={setFilterUseCase}
        onSpecies={(sp) => navigate('species', sp)} onCatalog={() => openCatalog('strawberry')}
        onShowcase={(s) => { const t = allCatalogTargets.find(c => c.gene_id === s.id); if (t) openTarget(t) }} />}
      {view === 'species' && <SpeciesView species={species} selected={selectedSpecies}
        onSelect={(sp) => setSelectedSpecies(sp)} onCatalog={openCatalog} />}
      {view === 'catalog' && <CatalogView targets={filteredCatalog} search={searchQuery}
        onSearch={setSearchQuery} filterTier={filterTier} setFilterTier={setFilterTier}
        filterTissue={filterTissue} setFilterTissue={setFilterTissue}
        filterAction={filterAction} setFilterAction={setFilterAction}
        tissues={tissues} onSelect={openTarget}
        catalogSpecies={catalogSpecies} onSwitchSpecies={openCatalog} speciesInfo={species} />}
      {view === 'phenotypes' && <PhenotypesView phenoIndex={phenoIndex} catalogs={catalogs}
        onTarget={openTarget} />}
      {view === 'methods' && <MethodsView species={species} />}
      {view === 'target' && selectedTarget && <TargetView target={selectedTarget}
        onBack={() => setView('catalog')} />}

      {/* Footer */}
      <footer className="footer">
        <div className="inner">
          <span className="codeline">PLANTED Atlas &middot; Chromatin-informed gene editing targets &middot; Banf &amp; Hartwig 2026</span>
          <span className="codeline">{Object.keys(species).length} species &middot; {totalEditable.toLocaleString()} editable genes</span>
        </div>
      </footer>
    </div>
  )
}

// ─── Home View ───────────────────────────────────────────
function HomeView({ species, showcases, totalEditable, totalGenes, filterUseCase, setFilterUseCase, onSpecies, onCatalog, onShowcase }: {
  species: Record<string, Species>; showcases: Showcase[]
  totalEditable: number; totalGenes: number
  filterUseCase: string; setFilterUseCase: (v: string) => void
  onSpecies: (sp: string) => void; onCatalog: () => void
  onShowcase: (s: Showcase) => void
}) {
  return (
    <>
      <section className="fade">
        <div className="wrap">
          <span className="eyebrow">Plant Editing Target Atlas</span>
          <h1 style={{ marginTop: 16 }}>Chromatin-informed<br /><em>gene editing</em> targets</h1>
          <p className="lead" style={{ marginTop: 20 }}>
            Multi-species predictions from the PLANTED platform. Browse {totalEditable.toLocaleString()} editable
            genes across {Object.keys(species).length} plant species, ranked by tissue-specific chromatin signals,
            transcription factor co-expression, and eQTL support.
          </p>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginTop: 32 }}>
            <button className="btn" onClick={onCatalog}>Browse editing catalogs</button>
            <button className="btn sec" onClick={() => onSpecies('wheat')}>Explore species</button>
          </div>
          <div className="stats">
            {[
              [Object.keys(species).length, 'Species'],
              [totalGenes.toLocaleString(), 'Genes assessed'],
              [totalEditable.toLocaleString(), 'Editable'],
              ['5', 'Editing catalogs'],
            ].map(([n, l]) => (
              <div className="stat" key={String(l)}>
                <div className="big">{n}</div>
                <span className="eyebrow">{l}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="tight">
        <div className="wrap">
          <span className="eyebrow">Species</span>
          <h2 style={{ marginTop: 12 }}>Five crop genomes</h2>
          <div className="grid3" style={{ marginTop: 32 }}>
            {Object.entries(species).map(([key, sp]) => (
              <div key={key} className="species-card" onClick={() => onSpecies(key)}>
                <div className="sp-top">
                  <span className="icon">{sp.icon}</span>
                  <div>
                    <div className="name">{sp.name}</div>
                    <div className="latin">{sp.latin}</div>
                  </div>
                  <span className={`tier tier-${sp.tier}`} style={{ marginLeft: 'auto' }}>Tier {sp.tier}</span>
                </div>
                <div className="sp-metrics">
                  <div><span className="num">{sp.editable.toLocaleString()}</span> editable genes</div>
                  <div><span className="num">{sp.tissue_specific.toLocaleString()}</span> tissue-specific</div>
                  <div>Model r = <span className="num">{sp.model_r}</span></div>
                </div>
                <div className="sp-bar"><div className="sp-bar-fill" style={{ width: `${(sp.editable / sp.genes) * 100}%` }} /></div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="tight">
        <div className="wrap">
          <div className="band">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 24, flexWrap: 'wrap' }}>
              <div>
                <span className="eyebrow">Use cases</span>
                <h3 style={{ marginTop: 8 }}>Editing showcases</h3>
              </div>
              <div className="chips">
                {[['all', 'All'], ['fruit_quality', 'Fruit quality'], ['stress_tolerance', 'Stress'], ['nutrition', 'Nutrition'], ['shelf_life', 'Shelf life'], ['breeding', 'Breeding']].map(([val, label]) => (
                  <button key={val} className={`chip${filterUseCase === val ? ' active' : ''}`} onClick={() => setFilterUseCase(val)}>{label}</button>
                ))}
              </div>
            </div>
            <div className="grid2">
              {showcases.map(s => (
                <div key={s.id} className="showcase-card" onClick={() => onShowcase(s)}>
                  <div className="sc-badges">
                    <span className={`tier tier-${s.tier === 'Tier 1' ? 1 : 2}`}>{s.tier}</span>
                    <span className={`action ${s.action}`}>{s.action}</span>
                    <span className="quiet" style={{ fontSize: 12 }}>{s.tissue}</span>
                  </div>
                  <h3>{s.title}</h3>
                  <p className="sc-summary">{s.summary}</p>
                  <div className="sc-meta">
                    <span>TF <strong>{s.tf}</strong></span>
                    <span>{s.tf_tpm} TPM</span>
                    <span>Effect <strong>{s.effect}</strong></span>
                    <span>{s.tool}</span>
                    {s.n_eqtl > 0 && <span className="eqtl-chip">{s.n_eqtl} eQTL</span>}
                  </div>
                  <div className="sc-gene">{s.id}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      <section className="last" />
    </>
  )
}

// ─── Species View ────────────────────────────────────────
function SpeciesView({ species, selected, onSelect, onCatalog }: {
  species: Record<string, Species>; selected: string | null; onSelect: (sp: string) => void; onCatalog: (sp: string) => void
}) {
  const sp = selected ? species[selected] : null
  return (
    <>
      <section className="tight fade">
        <div className="wrap">
          <span className="eyebrow">Species</span>
          <h1 style={{ marginTop: 12 }}>Species overview</h1>
          <div className="species-tabs" style={{ marginTop: 32 }}>
            {Object.entries(species).map(([key, s]) => (
              <button key={key} className={selected === key ? 'active' : ''} onClick={() => onSelect(key)}>{s.icon} {s.name}</button>
            ))}
          </div>
          {sp && (
            <>
              <div className="sd-header">
                <h1>{sp.icon} {sp.name}</h1>
                <span className="sd-latin">{sp.latin}</span>
                <span className={`tier tier-${sp.tier}`}>Tier {sp.tier}</span>
              </div>
              <div className="tiles" style={{ marginTop: 32 }}>
                <div className="tile">
                  <div className="big">{sp.model_r}</div>
                  <span className="eyebrow">Model Pearson r</span>
                  <div className="bar-track"><div className="bar-fill" style={{ width: `${sp.model_r * 100}%` }} /></div>
                </div>
                <div className="tile">
                  <div className="big">{sp.genes.toLocaleString()}</div>
                  <span className="eyebrow">Genes assessed</span>
                </div>
                <div className="tile">
                  <div className="big">{sp.editable.toLocaleString()}</div>
                  <span className="eyebrow">Editable genes</span>
                  <div className="sub">{((sp.editable / sp.genes) * 100).toFixed(1)}% of total</div>
                </div>
                <div className="tile">
                  <div className="big">{sp.tissue_specific.toLocaleString()}</div>
                  <span className="eyebrow">Tissue-specific</span>
                </div>
              </div>
              <div className="grid3" style={{ marginTop: 16 }}>
                <div className="tile">
                  <div className="big">{sp.tissues}</div>
                  <span className="eyebrow">Tissues profiled</span>
                </div>
                <div className="tile">
                  <div className="big">{sp.marks}</div>
                  <span className="eyebrow">Histone marks</span>
                </div>
                <div className="tile">
                  <div className="big">{sp.eqtl_n > 0 ? sp.eqtl_n.toLocaleString() : '\u2014'}</div>
                  <span className="eyebrow">eQTL-supported</span>
                  {sp.eqtl_pct > 0 && <div className="sub">{sp.eqtl_pct}% overlap</div>}
                </div>
              </div>
              {selected && (
                <div style={{ marginTop: 40 }}>
                  <button className="btn" onClick={() => onCatalog(selected)}>Browse {sp.name} editing catalog</button>
                </div>
              )}
            </>
          )}
        </div>
      </section>
      <section className="last" />
    </>
  )
}

// ─── Catalog View ────────────────────────────────────────
function CatalogView({ targets, search, onSearch, filterTier, setFilterTier, filterTissue, setFilterTissue, filterAction, setFilterAction, tissues, onSelect, catalogSpecies, onSwitchSpecies, speciesInfo }: {
  targets: Target[]; search: string; onSearch: (q: string) => void
  filterTier: string; setFilterTier: (v: string) => void; filterTissue: string; setFilterTissue: (v: string) => void
  filterAction: string; setFilterAction: (v: string) => void; tissues: string[]; onSelect: (t: Target) => void
  catalogSpecies: string; onSwitchSpecies: (sp: string) => void; speciesInfo: Record<string, Species>
}) {
  const spLabel = SPECIES_LABELS[catalogSpecies] || { name: catalogSpecies, icon: '' }
  return (
    <>
      <section className="tight fade">
        <div className="wrap">
          <div className="catalog-head">
            <span className="eyebrow">Editing catalog</span>
            <h1>{spLabel.icon} {spLabel.name} targets</h1>
            <p className="lead">{targets.length} targets shown. Click a row to see the full editing strategy, TF information, and multi-omics evidence.</p>
          </div>
          <div className="chips" style={{ marginBottom: 20 }}>
            {Object.entries(SPECIES_LABELS).map(([key, sp]) => (
              <button key={key} className={`chip${catalogSpecies === key ? ' active' : ''}`}
                onClick={() => onSwitchSpecies(key)}>
                {sp.icon} {sp.name}
                {speciesInfo[key] && <span style={{ fontSize: 11, marginLeft: 4, opacity: 0.6 }}>{speciesInfo[key].editable.toLocaleString()}</span>}
              </button>
            ))}
          </div>
          <div className="catalog-filters">
            <input className="search-input" placeholder="Search gene ID, TF motif, GO category..." value={search} onChange={e => onSearch(e.target.value)} />
            <select value={filterTier} onChange={e => setFilterTier(e.target.value)}>
              <option value="all">All tiers</option><option value="Tier 1">Tier 1</option><option value="Tier 2">Tier 2</option><option value="Tier 3">Tier 3</option><option value="Tier 4">Tier 4</option>
            </select>
            <select value={filterTissue} onChange={e => setFilterTissue(e.target.value)}>
              <option value="all">All tissues</option>{tissues.map(t => <option key={t} value={t}>{t}</option>)}
            </select>
            <select value={filterAction} onChange={e => setFilterAction(e.target.value)}>
              <option value="all">All actions</option><option value="activate">Activate</option><option value="silence">Silence</option>
            </select>
          </div>
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th>Gene ID</th><th>Tier</th><th>Score</th><th>Tissue</th><th>Mark</th><th>Action</th><th>Effect</th><th>TF</th><th>Coherence</th><th>Evidence</th>
                </tr>
              </thead>
              <tbody>
                {targets.slice(0, 100).map(t => (
                  <tr key={t.gene_id} onClick={() => onSelect(t)}>
                    <td className="gene-id">{t.gene_id}</td>
                    <td><span className={`tier tier-${t.tier.replace('Tier ', '')}`}>{t.tier}</span></td>
                    <td className="num">{t.score.toFixed(2)}</td>
                    <td>{t.tissue}</td>
                    <td>{t.mark}</td>
                    <td><span className={`action ${t.action}`}>{t.action}</span></td>
                    <td className="effect-val">{t.effect.toFixed(3)}</td>
                    <td>{t.tf_motif || '\u2014'}</td>
                    <td><span className={`coherence ${t.coherence}`}>{t.coherence.replace('_', ' ')}</span></td>
                    <td>
                      <div className="ev-dots">
                        {t.has_eqtl && <span className="ev-dot eqtl" title="eQTL support">E</span>}
                        {t.has_celltype && <span className="ev-dot celltype" title="Cell-type">C</span>}
                        {t.tf_expressed && <span className="ev-dot tf" title="TF expressed">T</span>}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {targets.length > 100 && <p className="table-note">Showing first 100 of {targets.length}</p>}
          </div>
        </div>
      </section>
      <section className="last" />
    </>
  )
}

// ─── Target Detail View ──────────────────────────────────
function TargetView({ target: t, onBack }: { target: Target; onBack: () => void }) {
  return (
    <>
      <section className="tight fade">
        <div className="wrap">
          <button className="back-link" onClick={onBack}>Back to catalog</button>
          <div className="detail-head">
            <div>
              <h1>{t.gene_id}</h1>
              <div className="badges">
                <span className={`tier tier-${t.tier.replace('Tier ', '')}`}>{t.tier}</span>
                <span className={`action ${t.action}`}>{t.action}</span>
                <span className="quiet" style={{ fontSize: 13 }}>{t.tissue} &middot; {t.specificity}</span>
              </div>
            </div>
            <div style={{ textAlign: 'right' }}>
              <div className="score">{t.score.toFixed(2)}</div>
              <span className="eyebrow score-label">Priority score</span>
            </div>
          </div>

          <div className="grid3" style={{ marginTop: 48 }}>
            {/* Editing Strategy */}
            <div className="detail-card">
              <div className="card-title">Editing strategy</div>
              <dl className="kv">
                <dt>Tissue</dt><dd><strong>{t.tissue}</strong></dd>
                <dt>Mark</dt><dd><strong>{t.mark}</strong> ({t.mark_role})</dd>
                <dt>Action</dt><dd><span className={`action ${t.action}`}>{t.action}</span></dd>
                <dt>Effect</dt><dd><strong>{t.effect.toFixed(3)}</strong></dd>
                <dt>Specificity</dt><dd><strong>{t.specificity}</strong></dd>
                <dt>Tissues</dt><dd><strong>{t.n_tissues}</strong></dd>
                {t.tool && <><dt>CRISPR tool</dt><dd><strong>{t.tool}</strong></dd></>}
              </dl>
            </div>

            {/* Transcription Factor */}
            <div className="detail-card">
              <div className="card-title">Transcription factor</div>
              {t.tf_motif ? (
                <dl className="kv">
                  <dt>TF motif</dt><dd><strong>{t.tf_motif}</strong></dd>
                  <dt>Expression</dt><dd>{t.tf_tpm ? <strong>{t.tf_tpm} TPM</strong> : <span className="quiet">&mdash;</span>}</dd>
                  <dt>Expressed</dt><dd>{t.tf_expressed ? <span style={{ color: 'var(--green-text)' }}>Yes</span> : <span className="quiet">No</span>}</dd>
                  <dt>Co-expr &rho;</dt><dd>{t.coexpr_rho != null ? <strong>{t.coexpr_rho}</strong> : <span className="quiet">&mdash;</span>}</dd>
                  <dt>Coherence</dt><dd><span className={`coherence ${t.coherence}`}>{t.coherence.replace('_', ' ')}</span></dd>
                  <dt>Motif pos</dt><dd><strong>{t.motif_pos}</strong></dd>
                  <dt>Regulators</dt><dd><strong>{t.n_regulators}</strong></dd>
                </dl>
              ) : (
                <p className="quiet" style={{ fontStyle: 'italic' }}>No TF motif identified</p>
              )}
            </div>

            {/* Multi-Omics Evidence */}
            <div className="detail-card">
              <div className="card-title">Multi-omics evidence</div>
              <p style={{ fontSize: 13, color: 'var(--label)', lineHeight: 1.5, marginBottom: 16 }}>
                Each target is scored by independent evidence layers. More layers = higher confidence
                that editing this locus will produce the predicted effect.
              </p>
              <dl className="kv" style={{ marginBottom: 16 }}>
                <dt>Evidence layers</dt><dd><strong>{t.n_evidence}</strong> / 3</dd>
                {t.stage && <><dt>Dev. stage</dt><dd><strong>{t.stage}</strong></dd></>}
                {t.tau_celltype != null && <><dt>Cell-type specificity (&tau;)</dt><dd><strong>{t.tau_celltype.toFixed(3)}</strong></dd></>}
              </dl>
              <div className="evidence-list">
                <div className={`ev-flag ${t.has_eqtl ? 'on' : 'off'}`}>
                  <span className="pip" />
                  <div>
                    <strong>{t.has_eqtl ? `eQTL support (${t.n_eqtl})` : 'No eQTL data'}</strong>
                    <div style={{ fontSize: 12, color: 'var(--tert)', marginTop: 2 }}>
                      {t.has_eqtl ? 'Natural genetic variants near this gene affect its expression' : 'No known expression-associated variants at this locus'}
                    </div>
                  </div>
                </div>
                <div className={`ev-flag ${t.has_celltype ? 'on' : 'off'}`}>
                  <span className="pip" />
                  <div>
                    <strong>{t.has_celltype ? `Cell-type: ${t.celltype}` : 'No cell-type data'}</strong>
                    <div style={{ fontSize: 12, color: 'var(--tert)', marginTop: 2 }}>
                      {t.has_celltype ? 'Single-cell RNA-seq confirms expression in a specific cell population' : 'No single-cell resolution available for this species'}
                    </div>
                  </div>
                </div>
                <div className={`ev-flag ${t.tf_expressed ? 'on' : 'off'}`}>
                  <span className="pip" />
                  <div>
                    <strong>{t.tf_expressed ? 'TF expressed in tissue' : 'TF not expressed'}</strong>
                    <div style={{ fontSize: 12, color: 'var(--tert)', marginTop: 2 }}>
                      {t.tf_expressed
                        ? `${t.tf_motif} is actively transcribed, supporting a functional regulatory link`
                        : 'The predicted TF regulator is not expressed in the target tissue'}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="grid2" style={{ marginTop: 24 }}>
            <div className="detail-card">
              <div className="card-title">Functional annotation</div>
              <dl className="kv">
                <dt>GO category</dt><dd><strong>{t.go_category || 'unknown'}</strong></dd>
              </dl>
              {t.description && <p style={{ marginTop: 12, fontSize: 15, color: 'var(--body)' }}>{t.description}</p>}
            </div>
            <div className="detail-card">
              <div className="card-title">Predicted phenotype</div>
              {t.phenotype ? (
                <div className="pheno-block">{t.phenotype}</div>
              ) : (
                <p className="quiet" style={{ fontStyle: 'italic' }}>No phenotype prediction</p>
              )}
            </div>
          </div>

          {/* Locus Browser */}
          <LocusBrowser geneId={t.gene_id} />
        </div>
      </section>
      <section className="last" />
    </>
  )
}

// ─── Methods View ───────────────────────────────────────
const PIPELINE_STEPS = [
  {
    num: '01',
    title: 'Genome & annotation',
    detail: 'Reference genome assembly + gene models (GFF3) per species. We extract promoter regions (2 kb upstream of TSS) and gene body coordinates for all protein-coding genes.',
    data: 'Ensembl Plants, GDR, IWGSC',
  },
  {
    num: '02',
    title: 'Chromatin profiling',
    detail: 'Histone modification ChIP-seq or ChromHMM state maps across multiple tissues and developmental stages. Marks include H3K4me3, H3K27ac, H3K27me3, H3K36me3, and others depending on species data availability.',
    data: 'Public ChIP-seq, ChromHMM segmentations',
  },
  {
    num: '03',
    title: 'ChromatinFiLM model',
    detail: 'A Feature-wise Linear Modulation (FiLM) neural network that predicts tissue-specific gene expression from binned chromatin features. For each gene, 20 spatial bins per mark capture the positional distribution of histone modifications around the TSS. The model learns how chromatin state maps to expression across tissues simultaneously.',
    data: 'Binned mark signals + bulk RNA-seq TPM',
  },
  {
    num: '04',
    title: 'In-silico perturbation',
    detail: 'For every gene and every histone mark, we simulate removal of that mark and measure the predicted change in expression per tissue. This yields a per-gene, per-mark, per-tissue "effect size" \u2014 how much expression would change if that chromatin modification were edited away.',
    data: 'Model counterfactual predictions',
  },
  {
    num: '05',
    title: 'Gene regulatory network',
    detail: 'TF-target regulatory relationships inferred from motif scanning (JASPAR + DAP-seq PWMs), co-expression analysis, and/or pySCENIC regulon detection from single-cell RNA-seq where available. Identifies which transcription factors regulate each target gene.',
    data: 'Motif databases, scRNA-seq, bulk co-expression',
  },
  {
    num: '06',
    title: 'Multi-omics integration',
    detail: 'Up to three independent evidence layers are combined per target: (1) eQTL support \u2014 natural genetic variants at the locus that affect expression, (2) cell-type specificity \u2014 snRNA-seq confirming expression in a defined cell population, (3) TF coherence \u2014 the predicted regulator TF is expressed and its motif is present in the promoter.',
    data: 'eQTL catalogs, snRNA-seq, TF expression',
  },
  {
    num: '07',
    title: 'Target scoring & tiering',
    detail: 'Each gene receives a composite priority score based on chromatin effect magnitude, tissue specificity, number of evidence layers, and TF coherence. Targets are assigned tiers: Tier 1 (highest confidence, strong effect + multiple evidence layers) through Tier 4 (chromatin signal only).',
    data: 'Composite score algorithm',
  },
  {
    num: '08',
    title: 'Editing strategy',
    detail: 'For each target, the dominant chromatin mark determines the recommended epigenome editing tool. For example, H3K27me3-driven silencing suggests dCas9-EZH2 to deposit the mark, or dCas9-UTX to remove it. CRISPR guide RNAs are designed for the promoter region with GC content and PAM-site optimization.',
    data: 'Mark \u2192 editing tool mapping, guide RNA design',
  },
]

function MethodsView({ species }: { species: Record<string, Species> }) {
  return (
    <>
      <section className="fade">
        <div className="wrap">
          <span className="eyebrow">Methodology</span>
          <h1 style={{ marginTop: 16 }}>How PLANTED works</h1>
          <p className="lead" style={{ marginTop: 20, maxWidth: 720 }}>
            PLANTED identifies gene editing targets by combining chromatin-based expression models
            with transcription factor networks and multi-omics validation. Here is the full pipeline
            from raw genome data to actionable editing targets.
          </p>
        </div>
      </section>

      {/* Pipeline steps */}
      <section className="tight">
        <div className="wrap">
          <div className="band">
            <span className="eyebrow">Pipeline overview</span>
            <h2 style={{ marginTop: 8 }}>Eight-step workflow</h2>
            <div className="method-steps">
              {PIPELINE_STEPS.map(step => (
                <div key={step.num} className="method-step">
                  <div className="step-num">{step.num}</div>
                  <div className="step-body">
                    <h3>{step.title}</h3>
                    <p>{step.detail}</p>
                    <div className="step-data">{step.data}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Model details */}
      <section className="tight">
        <div className="wrap">
          <span className="eyebrow">Core model</span>
          <h2 style={{ marginTop: 8 }}>ChromatinFiLM architecture</h2>
          <div className="grid2" style={{ marginTop: 32 }}>
            <div className="detail-card">
              <div className="card-title">Input representation</div>
              <p style={{ fontSize: 15, color: 'var(--body)', lineHeight: 1.6 }}>
                Each gene is represented by its chromatin landscape: for every histone mark available
                in that species, we divide the 4 kb window around the TSS into 20 spatial bins (200 bp each).
                The signal intensity in each bin becomes a feature. With <em>M</em> marks, the input
                is an <em>M</em>&times;20 matrix per gene, plus GRN-derived features (TF connectivity,
                regulon membership).
              </p>
            </div>
            <div className="detail-card">
              <div className="card-title">FiLM conditioning</div>
              <p style={{ fontSize: 15, color: 'var(--body)', lineHeight: 1.6 }}>
                Feature-wise Linear Modulation (FiLM) allows a single model to predict expression
                across all tissues simultaneously. A tissue embedding generates per-layer scale and shift
                parameters (&gamma;, &beta;) that modulate the hidden representations. This means
                the model learns <em>one</em> shared chromatin grammar but produces tissue-specific predictions.
              </p>
            </div>
          </div>
          <div className="grid2" style={{ marginTop: 16 }}>
            <div className="detail-card">
              <div className="card-title">In-silico perturbation</div>
              <p style={{ fontSize: 15, color: 'var(--body)', lineHeight: 1.6 }}>
                To identify editing targets, we zero out each mark&rsquo;s bins one at a time and measure the
                predicted expression change per tissue. A gene whose expression drops sharply in one tissue
                when H3K27ac is removed, for instance, is a candidate for dCas9-HDAC silencing in that tissue.
                The &ldquo;effect size&rdquo; shown per target is this counterfactual &Delta;expression.
              </p>
            </div>
            <div className="detail-card">
              <div className="card-title">Validation & cross-species</div>
              <p style={{ fontSize: 15, color: 'var(--body)', lineHeight: 1.6 }}>
                Models are validated by 5-fold gene hold-out cross-validation (genes in the test set
                are never seen during training). Cross-species transfer of raw chromatin features shows
                limited portability, confirming that each species&rsquo; chromatin grammar is distinct.
                However, regulon-level conservation (via ortholog mapping) provides transferable priors
                for species with less data.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Evidence layers explanation */}
      <section className="tight">
        <div className="wrap">
          <span className="eyebrow">Confidence scoring</span>
          <h2 style={{ marginTop: 8 }}>Evidence layers</h2>
          <p style={{ color: 'var(--body)', fontSize: 16, lineHeight: 1.6, maxWidth: 700, marginTop: 16 }}>
            Each editing target is supported by up to three independent lines of evidence.
            Targets with more layers have higher confidence that the predicted editing effect is real.
          </p>
          <div className="grid3" style={{ marginTop: 32 }}>
            <div className="detail-card">
              <div className="card-title">Layer 1: eQTL support</div>
              <p style={{ fontSize: 14, color: 'var(--body)', lineHeight: 1.6 }}>
                Expression quantitative trait loci (eQTLs) are natural genetic variants that affect
                a gene&rsquo;s expression level. When an eQTL maps to our target gene, it provides
                orthogonal evidence that the locus is expression-sensitive &mdash; nature has already
                &ldquo;tested&rdquo; perturbation at this site.
              </p>
              <div className="step-data" style={{ marginTop: 12 }}>Available for: Strawberry, Wheat, Rice</div>
            </div>
            <div className="detail-card">
              <div className="card-title">Layer 2: Cell-type specificity</div>
              <p style={{ fontSize: 14, color: 'var(--body)', lineHeight: 1.6 }}>
                Single-cell / single-nucleus RNA-seq resolves expression to individual cell populations.
                When a target gene is specifically expressed in a defined cell type (e.g., vascular bundle,
                guard cells), it increases confidence that editing will produce a focused, predictable
                phenotype rather than pleiotropic effects.
              </p>
              <div className="step-data" style={{ marginTop: 12 }}>Available for: Strawberry (snRNA-seq)</div>
            </div>
            <div className="detail-card">
              <div className="card-title">Layer 3: TF coherence</div>
              <p style={{ fontSize: 14, color: 'var(--body)', lineHeight: 1.6 }}>
                A transcription factor motif in the promoter is only meaningful if the TF itself is
                expressed in the target tissue. TF coherence checks that: (1) a motif was found via
                PWM scanning, (2) the cognate TF is transcribed (TPM &gt; 1), and optionally (3)
                TF and target are co-expressed. This validates the regulatory logic behind the prediction.
              </p>
              <div className="step-data" style={{ marginTop: 12 }}>Available for: all 5 species</div>
            </div>
          </div>
        </div>
      </section>

      {/* Tier system */}
      <section className="tight">
        <div className="wrap">
          <span className="eyebrow">Target ranking</span>
          <h2 style={{ marginTop: 8 }}>Tier system</h2>
          <div className="rows" style={{ marginTop: 24, maxWidth: 700 }}>
            {[
              { tier: 1, label: 'Highest confidence', desc: 'Strong chromatin effect (>0.5), tissue-specific, multiple evidence layers, TF coherence confirmed. Ready for experimental validation.' },
              { tier: 2, label: 'High confidence', desc: 'Moderate-to-strong effect, at least one evidence layer beyond chromatin. Good candidates with additional support.' },
              { tier: 3, label: 'Moderate confidence', desc: 'Chromatin effect detected with some specificity. May lack eQTL or cell-type data but the signal is consistent.' },
              { tier: 4, label: 'Exploratory', desc: 'Chromatin signal only, no additional evidence layers. Suitable for hypothesis generation or bulk screening.' },
            ].map(t => (
              <div key={t.tier} className="row" style={{ cursor: 'default' }}>
                <div className="icon-col"><span className={`tier tier-${t.tier}`}>Tier {t.tier}</span></div>
                <div className="body">
                  <strong>{t.label}</strong>
                  <p style={{ fontSize: 14, color: 'var(--body)', marginTop: 4 }}>{t.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Per-species model performance */}
      <section className="tight">
        <div className="wrap">
          <span className="eyebrow">Performance</span>
          <h2 style={{ marginTop: 8 }}>Model accuracy by species</h2>
          <p style={{ color: 'var(--body)', fontSize: 16, lineHeight: 1.6, maxWidth: 700, marginTop: 16 }}>
            Cross-validated Pearson correlation between predicted and observed tissue-specific
            expression. Higher <em>r</em> means the chromatin features explain more expression variance.
          </p>
          <div className="grid3" style={{ marginTop: 32 }}>
            {Object.entries(species).map(([key, sp]) => (
              <div key={key} className="tile">
                <div style={{ fontSize: 28 }}>{sp.icon}</div>
                <div className="big" style={{ marginTop: 8 }}>{sp.model_r}</div>
                <span className="eyebrow">{sp.name} Pearson r</span>
                <div className="bar-track" style={{ marginTop: 8 }}><div className="bar-fill" style={{ width: `${sp.model_r * 100}%` }} /></div>
                <div className="sub" style={{ marginTop: 6 }}>{sp.marks} marks &middot; {sp.tissues} tissues &middot; {sp.genes.toLocaleString()} genes</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CRISPR tools */}
      <section className="tight">
        <div className="wrap">
          <span className="eyebrow">Editing tools</span>
          <h2 style={{ marginTop: 8 }}>Chromatin mark to CRISPR tool mapping</h2>
          <p style={{ color: 'var(--body)', fontSize: 16, lineHeight: 1.6, maxWidth: 700, marginTop: 16 }}>
            Each histone mark implies a specific epigenome editing strategy. PLANTED maps the dominant
            mark per target to the appropriate dCas9-effector fusion.
          </p>
          <div className="table-wrap" style={{ marginTop: 24 }}>
            <table className="data">
              <thead>
                <tr><th>Mark</th><th>Role</th><th>To silence</th><th>To activate</th></tr>
              </thead>
              <tbody>
                {[
                  ['H3K4me3', 'Promoter activation', 'dCas9-LSD1', 'dCas9-SET1'],
                  ['H3K27ac', 'Active enhancer', 'dCas9-HDAC', 'dCas9-p300'],
                  ['H3K27me3', 'Polycomb repression', 'dCas9-EZH2', 'dCas9-UTX'],
                  ['H3K36me3', 'Transcription elongation', '\u2014', 'dCas9-SETD2'],
                  ['H3K4me1', 'Enhancer priming', 'dCas9-LSD1', 'dCas9-MLL'],
                  ['H3K9me2', 'Heterochromatin', 'dCas9-SUV39H', 'dCas9-KDM4'],
                  ['ATAC', 'Open chromatin', 'dCas9-KRAB', 'CRISPRa (VPR)'],
                ].map(([mark, role, sil, act]) => (
                  <tr key={mark}>
                    <td><strong>{mark}</strong></td>
                    <td>{role}</td>
                    <td>{sil}</td>
                    <td>{act}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section className="tight">
        <div className="wrap">
          <div className="band" style={{ textAlign: 'center', padding: '48px 32px' }}>
            <span className="eyebrow">Reference</span>
            <h3 style={{ marginTop: 8 }}>Banf &amp; Hartwig (2026)</h3>
            <p style={{ fontSize: 15, color: 'var(--body)', maxWidth: 600, margin: '16px auto 0' }}>
              &ldquo;An agentic deep learning platform for predicting, explaining, and designing
              TF binding variation in maize&rdquo; &mdash; the foundational methodology behind PLANTED.
            </p>
          </div>
        </div>
      </section>

      <section className="last" />
    </>
  )
}

// ─── Phenotypes View ─────────────────────────────────────
const PHENO_ICONS: Record<string, string> = {
  'Fruit quality': '\uD83C\uDF53', 'Stress tolerance': '\uD83C\uDF21\uFE0F',
  'Nutrition': '\uD83E\uDD66', 'Shelf life': '\uD83D\uDCE6',
  'Yield & growth': '\uD83C\uDF31', 'Hormone signaling': '\u2697\uFE0F',
  'Cell wall & structure': '\uD83E\uDDF1', 'Signal transduction': '\u26A1',
  'Transport': '\uD83D\uDE9A', 'Other': '\uD83D\uDD2C',
}

const PHENO_KEYWORDS: Record<string, string[]> = {
  'Fruit quality': ['fruit', 'ripen', 'flavor', 'sugar', 'color', 'softening', 'anthocyanin', 'aroma', 'sweetness', 'acidity', 'taste'],
  'Stress tolerance': ['stress', 'drought', 'salt', 'cold', 'heat', 'defense', 'pathogen', 'disease', 'resistance', 'immune', 'oxidative'],
  'Nutrition': ['nutrient', 'vitamin', 'lipid', 'protein', 'amino', 'nitrogen', 'mineral', 'antioxidant', 'phenolic', 'carotenoid'],
  'Shelf life': ['shelf', 'storage', 'senescence', 'decay', 'postharvest', 'firmness', 'perishab'],
  'Yield & growth': ['yield', 'growth', 'biomass', 'size', 'flower', 'pollen', 'seed', 'fertility', 'root', 'vegetative', 'branching'],
  'Hormone signaling': ['hormone', 'ethylene', 'auxin', 'gibberellin', 'abscisic', 'jasmonate', 'cytokinin', 'brassinosteroid'],
  'Cell wall & structure': ['cell wall', 'lignin', 'cellulose', 'pectin', 'xylan', 'structural'],
  'Signal transduction': ['signal', 'kinase', 'receptor', 'calcium', 'phosphorylation'],
  'Transport': ['transport', 'allocation', 'loading', 'channel', 'transporter', 'efflux'],
}

const SPECIES_OPTS = [
  { key: 'all', label: 'All species' },
  { key: 'strawberry', label: 'Strawberry', icon: '\uD83C\uDF53' },
  { key: 'wheat', label: 'Wheat', icon: '\uD83C\uDF3E' },
  { key: 'rice', label: 'Rice', icon: '\uD83C\uDF3E' },
  { key: 'arabidopsis', label: 'Arabidopsis', icon: '\uD83C\uDF3F' },
  { key: 'tomato', label: 'Tomato', icon: '\uD83C\uDF45' },
]

function targetMatchesCategory(t: Target, cat: string): boolean {
  const kws = PHENO_KEYWORDS[cat]
  if (!kws) return false
  const pheno = (t.phenotype || '').toLowerCase()
  return kws.some(kw => pheno.includes(kw))
}

function PhenotypesView({ phenoIndex, catalogs, onTarget }: {
  phenoIndex: Record<string, PhenoCategory>; catalogs: Record<string, Target[]>
  onTarget: (t: Target) => void
}) {
  const [selectedCat, setSelectedCat] = useState<string | null>(null)
  const [searchQ, setSearchQ] = useState('')
  const [speciesFilter, setSpeciesFilter] = useState('all')
  const [showCount, setShowCount] = useState(20)

  // Species filter across all catalogs
  const allTargets = Object.values(catalogs).flat()
  const filteredCatalog = speciesFilter === 'all'
    ? allTargets
    : (catalogs[speciesFilter] || [])

  // Search across all phenotype text
  const searchResults = searchQ.length >= 3
    ? filteredCatalog
        .filter(t => t.phenotype?.toLowerCase().includes(searchQ.toLowerCase()))
        .sort((a, b) => b.score - a.score)
        .slice(0, 30)
    : []

  // Build live category counts from filtered catalog
  const categoryCounts: Record<string, number> = {}
  for (const cat of Object.keys(PHENO_KEYWORDS)) {
    categoryCounts[cat] = filteredCatalog.filter(t => targetMatchesCategory(t, cat)).length
  }

  // When a category is selected, get all matching targets
  const categoryTargets = selectedCat
    ? filteredCatalog
        .filter(t => targetMatchesCategory(t, selectedCat))
        .sort((a, b) => b.score - a.score)
    : []

  const categories = Object.entries(categoryCounts)
    .filter(([, count]) => count > 0)
    .sort((a, b) => b[1] - a[1])

  return (
    <>
      <section className="tight fade">
        <div className="wrap">
          <span className="eyebrow">Predicted phenotypes</span>
          <h1 style={{ marginTop: 12 }}>What can be edited?</h1>
          <p className="lead" style={{ marginTop: 16 }}>
            Every target has a predicted phenotype based on GO annotation, tissue context, and editing action.
            Browse by category or search for specific traits.
          </p>

          {/* Species filter */}
          <div className="chips" style={{ marginTop: 28 }}>
            {SPECIES_OPTS.map(sp => {
              const count = sp.key === 'all' ? allTargets.length : (catalogs[sp.key] || []).length
              return (
                <button key={sp.key}
                  className={`chip${speciesFilter === sp.key ? ' active' : ''}`}
                  onClick={() => { setSpeciesFilter(sp.key); setSelectedCat(null); setShowCount(20) }}>
                  {sp.icon ? `${sp.icon} ` : ''}{sp.label}
                  <span style={{ fontSize: 11, marginLeft: 4, opacity: 0.6 }}>{count}</span>
                </button>
              )
            })}
          </div>

          {/* Phenotype search */}
          <div style={{ marginTop: 20 }}>
            <input className="search-input" style={{ maxWidth: 500 }}
              placeholder="Search phenotypes: fruit quality, drought, sugar, shelf life..."
              value={searchQ} onChange={e => { setSearchQ(e.target.value); setSelectedCat(null) }} />
          </div>

          {/* Search results */}
          {searchResults.length > 0 && (
            <div style={{ marginTop: 24 }}>
              <span className="eyebrow">{searchResults.length} matches for &ldquo;{searchQ}&rdquo;</span>
              <div className="rows" style={{ marginTop: 12 }}>
                {searchResults.map(t => (
                  <div key={t.gene_id} className="row" onClick={() => onTarget(t)}>
                    <div className="icon-col" style={{ fontSize: 14, fontFamily: 'var(--mono)', color: 'var(--label)', paddingTop: 8 }}>
                      {t.score.toFixed(1)}
                    </div>
                    <div className="body">
                      <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
                        <span style={{ fontFamily: 'var(--mono)', fontSize: 14, color: 'var(--ink)', fontWeight: 400 }}>{t.gene_id}</span>
                        <span className={`tier tier-${t.tier.replace('Tier ', '')}`}>{t.tier}</span>
                        <span className={`action ${t.action}`}>{t.action}</span>
                        <span className="quiet" style={{ fontSize: 12 }}>{t.tissue}</span>
                      </div>
                      <p className="grey" style={{ fontSize: 14 }}>{t.phenotype?.slice(0, 200)}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Category grid (when no search and no category selected) */}
          {!searchQ && !selectedCat && (
            <div className="grid3" style={{ marginTop: 32 }}>
              {categories.map(([cat, count]) => (
                <div key={cat} className="card" style={{ cursor: 'pointer', transition: 'border-color .2s' }}
                  onClick={() => { setSelectedCat(cat); setShowCount(20) }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                    <div>
                      <span style={{ fontSize: 24 }}>{PHENO_ICONS[cat] || '\uD83D\uDD2C'}</span>
                      <h3 style={{ fontSize: 22, marginTop: 8 }}>{cat}</h3>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <div style={{ fontFamily: 'var(--display)', fontSize: 32, color: 'var(--ink)' }}>{count}</div>
                      <span className="eyebrow">targets</span>
                    </div>
                  </div>
                  <p className="quiet" style={{ marginTop: 12, fontSize: 13 }}>
                    Click to browse all {count} targets
                  </p>
                </div>
              ))}
            </div>
          )}

          {/* Full category view */}
          {!searchQ && selectedCat && (
            <div style={{ marginTop: 32 }}>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: 16, flexWrap: 'wrap' }}>
                <button className="back-link" onClick={() => setSelectedCat(null)}>All categories</button>
                <h2 style={{ fontSize: 'clamp(28px, 3.5vw, 40px)' }}>
                  {PHENO_ICONS[selectedCat]} {selectedCat}
                </h2>
                <span className="eyebrow" style={{ marginLeft: 'auto' }}>{categoryTargets.length} targets</span>
              </div>

              <div className="rows" style={{ marginTop: 24 }}>
                {categoryTargets.slice(0, showCount).map((t, i) => (
                  <div key={t.gene_id} className="row" onClick={() => onTarget(t)}>
                    <div className="icon-col" style={{ fontFamily: 'var(--mono)', fontSize: 12, color: 'var(--tert)', paddingTop: 6, textAlign: 'center' }}>
                      {i + 1}
                    </div>
                    <div className="body">
                      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
                        <span style={{ fontFamily: 'var(--mono)', fontSize: 14, color: 'var(--ink)', fontWeight: 400 }}>{t.gene_id}</span>
                        <span className={`tier tier-${t.tier.replace('Tier ', '')}`}>{t.tier}</span>
                        <span className={`action ${t.action}`}>{t.action}</span>
                        <span className="quiet" style={{ fontSize: 12 }}>{t.tissue}</span>
                        <span style={{ marginLeft: 'auto', fontFamily: 'var(--mono)', fontSize: 12, color: 'var(--label)' }}>
                          score {t.score.toFixed(2)}
                        </span>
                      </div>
                      <p style={{ fontSize: 14, color: 'var(--body)', lineHeight: 1.55, marginTop: 4 }}>
                        {t.phenotype?.slice(0, 220)}{(t.phenotype?.length || 0) > 220 ? '...' : ''}
                      </p>
                      <div style={{ display: 'flex', gap: 14, marginTop: 6, fontSize: 11, fontFamily: 'var(--mono)', color: 'var(--tert)', flexWrap: 'wrap' }}>
                        <span>{t.mark}</span>
                        <span>effect {t.effect.toFixed(3)}</span>
                        <span>{t.tool}</span>
                        {t.tf_motif && <span>TF: {t.tf_motif}</span>}
                        {t.has_eqtl && <span className="eqtl-chip">{t.n_eqtl} eQTL</span>}
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {categoryTargets.length > showCount && (
                <div style={{ textAlign: 'center', marginTop: 24 }}>
                  <button className="btn sec" onClick={() => setShowCount(s => s + 30)}>
                    Show more ({categoryTargets.length - showCount} remaining)
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </section>
      <section className="last" />
    </>
  )
}

export default App
