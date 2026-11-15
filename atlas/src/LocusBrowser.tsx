import { useState, useEffect, useRef } from 'react'

// ─── Types ───────────────────────────────────────────────
interface GeneFeature {
  type: string
  start: number
  end: number
}

interface ChromHMMInterval {
  start: number
  end: number
  state: number
}

interface MarkTracks {
  H3K4me3: number[]
  H3K27ac: number[]
  H3K9K14ac: number[]
  H3K36me3: number[]
  H3K4me1: number[]
  H3K27me3: number[]
  H3K9me2: number[]
}

interface MotifInfo {
  name: string
  position: number
  relative_to_tss: number
}

interface Guide {
  sequence: string
  pam: string
  strand: string
  position: number
  gc_content: number
  score: number
  tool: string
  mechanism: string
}

interface LocusData {
  gene_id: string
  chrom: string
  strand: string
  gene_start: number
  gene_end: number
  tss: number
  window: { start: number; end: number; size: number }
  gene_model: GeneFeature[]
  chromhmm: { red: ChromHMMInterval[]; pret: ChromHMMInterval[] }
  mark_tracks: { red: MarkTracks; pret: MarkTracks }
  motif: MotifInfo | null
  guides: Guide[]
  promoter_seq: string
  dominant_mark: string
  edit_action: string
  tissue: string
  effect: number
}

// ─── Constants ───────────────────────────────────────────
const MARKS = ['H3K4me3', 'H3K27ac', 'H3K9K14ac', 'H3K36me3', 'H3K4me1', 'H3K27me3', 'H3K9me2'] as const

const MARK_COLORS: Record<string, string> = {
  H3K4me3: '#DC2626', H3K27ac: '#EA580C', H3K9K14ac: '#CA8A04',
  H3K36me3: '#16A34A', H3K4me1: '#2563EB', H3K27me3: '#6B7280',
  H3K9me2: '#334155',
}

const MARK_LABELS: Record<string, string> = {
  H3K4me3: 'H3K4me3', H3K27ac: 'H3K27ac', H3K9K14ac: 'H3K9/14ac',
  H3K36me3: 'H3K36me3', H3K4me1: 'H3K4me1', H3K27me3: 'H3K27me3',
  H3K9me2: 'H3K9me2',
}

const STATE_COLORS: Record<number, string> = {
  1: '#FF0000', 2: '#FF6600', 3: '#008000', 4: '#006400',
  5: '#90EE90', 6: '#C0FFC0', 7: '#FFFF00', 8: '#FFFFCC',
  9: '#CC66CC', 10: '#CC99CC', 11: '#808080', 12: '#999999',
  13: '#666666', 14: '#404040', 15: '#65D1FF', 16: '#65D1FF',
  17: '#F5F5F5',
}

const STATE_NAMES: Record<number, string> = {
  1: 'Active TSS', 2: 'Flanking TSS', 3: "Txn (5')", 4: 'Strong Txn',
  5: 'Weak Txn', 6: 'Txn Transition', 7: 'Enhancer', 8: 'Weak Enhancer',
  9: 'Bivalent TSS', 10: 'Bivalent Enh.', 11: 'Polycomb Weak', 12: 'Polycomb',
  13: 'Polycomb Strong', 14: 'H3K9me2 Repr.', 15: 'Het. Low', 16: 'Het. H3K9me2',
  17: 'Quiescent',
}

// ─── Layout constants ────────────────────────────────────
const W = 800       // SVG width
const PAD_L = 90    // left label area
const PAD_R = 20
const TRACK_W = W - PAD_L - PAD_R
const ROW_H = 16    // height per mark track
const GAP = 2

// ─── Component ───────────────────────────────────────────
export function LocusBrowser({ geneId }: { geneId: string }) {
  const [data, setData] = useState<LocusData | null>(null)
  const [loading, setLoading] = useState(true)
  const [stage, setStage] = useState<'red' | 'pret'>('red')
  const [hoveredMark, setHoveredMark] = useState<string | null>(null)
  const [copiedIdx, setCopiedIdx] = useState<number | null>(null)
  const svgRef = useRef<SVGSVGElement>(null)

  useEffect(() => {
    setLoading(true)
    fetch(`/data/locus/${geneId}.json`)
      .then(r => r.json())
      .then(d => { setData(d); setLoading(false) })
      .catch(() => setLoading(false))
  }, [geneId])

  if (loading) return <div className="quiet" style={{ padding: 20, fontFamily: 'var(--mono)', fontSize: 12 }}>Loading locus data...</div>
  if (!data) return <div className="quiet" style={{ padding: 20 }}>No locus data available for {geneId}</div>

  const tracks = data.mark_tracks[stage]
  const chromhmm = data.chromhmm[stage]
  const bins = tracks[MARKS[0] as keyof MarkTracks].length

  // Compute total SVG height
  const coordH = 24        // coordinate axis
  const geneModelH = 28    // gene model track
  const chromhmmH = 18     // ChromHMM state bar
  const markSectionH = MARKS.length * (ROW_H + GAP)
  const motifH = data.motif ? 24 : 0
  const totalH = coordH + geneModelH + chromhmmH + markSectionH + motifH + 30

  const x = (frac: number) => PAD_L + frac * TRACK_W
  const bx = (bin: number) => x(bin / bins)
  const bw = TRACK_W / bins

  // Positions
  const tssX = x(data.strand === '+' ? 0.5 : 0.5)  // TSS is at center of window

  let yOff = 8

  const copyGuide = (idx: number, seq: string) => {
    navigator.clipboard.writeText(seq)
    setCopiedIdx(idx)
    setTimeout(() => setCopiedIdx(null), 2000)
  }

  return (
    <div style={{ marginTop: 32 }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 12, flexWrap: 'wrap', gap: 12 }}>
        <div>
          <span className="eyebrow">Locus browser</span>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 13, color: 'var(--tert)', marginTop: 4 }}>
            {data.chrom}:{data.window.start.toLocaleString()}-{data.window.end.toLocaleString()} ({data.strand} strand) &middot; {(data.window.size / 1000).toFixed(0)}kb window
          </div>
        </div>
        <div className="chips" style={{ gap: 4 }}>
          <button className={`chip${stage === 'red' ? ' active' : ''}`} onClick={() => setStage('red')} style={{ height: 26, fontSize: 12, padding: '0 10px' }}>
            Red (mature)
          </button>
          <button className={`chip${stage === 'pret' ? ' active' : ''}`} onClick={() => setStage('pret')} style={{ height: 26, fontSize: 12, padding: '0 10px' }}>
            PreT (immature)
          </button>
        </div>
      </div>

      {/* SVG Track View */}
      <div style={{ background: 'var(--card)', border: '1px solid var(--line)', borderRadius: 10, padding: '16px 12px', overflow: 'hidden' }}>
        <svg ref={svgRef} width="100%" viewBox={`0 0 ${W} ${totalH}`} style={{ display: 'block' }}>
          {/* Coordinate axis */}
          {(() => {
            const y = yOff
            yOff += coordH
            const ws = data.window.start
            const we = data.window.end
            return (
              <g>
                <line x1={PAD_L} y1={y + 18} x2={PAD_L + TRACK_W} y2={y + 18} stroke="var(--line-2)" strokeWidth={1} />
                {/* Tick marks */}
                {[0, 0.25, 0.5, 0.75, 1].map(f => {
                  const pos = Math.round(ws + f * (we - ws))
                  return (
                    <g key={f}>
                      <line x1={x(f)} y1={y + 15} x2={x(f)} y2={y + 21} stroke="var(--line-3)" strokeWidth={1} />
                      <text x={x(f)} y={y + 12} textAnchor="middle" fontSize={9} fontFamily="var(--mono)" fill="var(--tert)">
                        {f === 0.5 ? 'TSS' : `${(pos / 1000).toFixed(1)}kb`}
                      </text>
                    </g>
                  )
                })}
                {/* TSS arrow */}
                <line x1={x(0.5)} y1={y + 2} x2={x(0.5)} y2={y + 18} stroke="var(--coral)" strokeWidth={1.5} strokeDasharray="3,2" />
                <text x={PAD_L - 6} y={y + 14} textAnchor="end" fontSize={10} fontFamily="var(--mono)" fill="var(--label)">Position</text>
              </g>
            )
          })()}

          {/* Gene model */}
          {(() => {
            const y = yOff
            yOff += geneModelH
            return (
              <g>
                <text x={PAD_L - 6} y={y + 16} textAnchor="end" fontSize={10} fontFamily="var(--mono)" fill="var(--label)">Gene</text>
                {/* Gene body line */}
                <line x1={PAD_L} y1={y + 14} x2={PAD_L + TRACK_W} y2={y + 14} stroke="var(--line-2)" strokeWidth={1} />
                {/* Features */}
                {data.gene_model.map((feat, i) => {
                  const fx1 = x(feat.start / data.window.size)
                  const fx2 = x(feat.end / data.window.size)
                  const fw = Math.max(fx2 - fx1, 1)
                  let fill = 'var(--ink)'
                  let h = 12
                  if (feat.type === 'CDS') { fill = 'var(--ink)'; h = 12; }
                  else if (feat.type === 'exon') { fill = 'var(--body)'; h = 10; }
                  else if (feat.type.includes('UTR')) { fill = 'var(--tert)'; h = 6; }
                  return <rect key={i} x={fx1} y={y + 14 - h / 2} width={fw} height={h} fill={fill} rx={1} />
                })}
                {/* Strand arrow */}
                <text x={PAD_L + TRACK_W + 8} y={y + 18} fontSize={14} fill="var(--tert)">
                  {data.strand === '+' ? '\u2192' : '\u2190'}
                </text>
              </g>
            )
          })()}

          {/* ChromHMM state bar */}
          {(() => {
            const y = yOff
            yOff += chromhmmH + 4
            return (
              <g>
                <text x={PAD_L - 6} y={y + 12} textAnchor="end" fontSize={10} fontFamily="var(--mono)" fill="var(--label)">States</text>
                {chromhmm.map((interval, i) => {
                  const ix1 = x(interval.start / data.window.size)
                  const ix2 = x(interval.end / data.window.size)
                  return (
                    <rect key={i} x={ix1} y={y + 2} width={Math.max(ix2 - ix1, 1)} height={14}
                      fill={STATE_COLORS[interval.state] || '#eee'} stroke="var(--line)" strokeWidth={0.3} rx={1}>
                      <title>{STATE_NAMES[interval.state] || `State ${interval.state}`}</title>
                    </rect>
                  )
                })}
              </g>
            )
          })()}

          {/* Mark tracks (heatmap rows) */}
          {MARKS.map((mark, mi) => {
            const y = yOff + mi * (ROW_H + GAP)
            const values = tracks[mark as keyof MarkTracks]
            const color = MARK_COLORS[mark]
            const isHovered = hoveredMark === mark
            const isDominant = mark === data.dominant_mark

            return (
              <g key={mark}
                onMouseEnter={() => setHoveredMark(mark)}
                onMouseLeave={() => setHoveredMark(null)}
                style={{ cursor: 'pointer' }}
              >
                {/* Label */}
                <text x={PAD_L - 6} y={y + 12} textAnchor="end" fontSize={10}
                  fontFamily="var(--mono)" fill={isDominant ? 'var(--ink)' : 'var(--label)'}
                  fontWeight={isDominant ? 500 : 400}>
                  {MARK_LABELS[mark]}{isDominant ? ' *' : ''}
                </text>
                {/* Background */}
                {isHovered && (
                  <rect x={PAD_L - 2} y={y - 1} width={TRACK_W + 4} height={ROW_H + 2}
                    fill="var(--line)" rx={2} opacity={0.3} />
                )}
                {/* Heatmap bins */}
                {values.map((v, bi) => (
                  <rect key={bi} x={bx(bi)} y={y} width={bw + 0.5} height={ROW_H}
                    fill={color} opacity={v * 0.85 + 0.05} rx={0}>
                    <title>{MARK_LABELS[mark]}: {(v * 100).toFixed(0)}% ({stage === 'red' ? 'mature' : 'immature'})</title>
                  </rect>
                ))}
              </g>
            )
          })}
          {(() => { yOff += markSectionH + 4; return null })()}

          {/* Motif overlay */}
          {data.motif && (() => {
            const y = yOff
            yOff += motifH
            const mx = x(data.motif.position / data.window.size)
            return (
              <g>
                <text x={PAD_L - 6} y={y + 14} textAnchor="end" fontSize={10} fontFamily="var(--mono)" fill="var(--label)">Motif</text>
                {/* Motif position marker */}
                <line x1={mx} y1={y + 2} x2={mx} y2={y + 18} stroke="var(--coral)" strokeWidth={2} />
                <rect x={mx - 20} y={y + 2} width={40} height={16} fill="var(--coral)" opacity={0.12} rx={3} />
                <text x={mx} y={y + 14} textAnchor="middle" fontSize={9} fontFamily="var(--mono)" fill="var(--coral)" fontWeight={500}>
                  {data.motif.name}
                </text>
              </g>
            )
          })()}
        </svg>

        {/* Legend */}
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginTop: 10, paddingTop: 10, borderTop: '1px solid var(--line)' }}>
          {MARKS.map(mark => (
            <div key={mark} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, fontFamily: 'var(--mono)', color: mark === data.dominant_mark ? 'var(--ink)' : 'var(--tert)' }}>
              <span style={{ width: 10, height: 10, borderRadius: 2, background: MARK_COLORS[mark], display: 'inline-block' }} />
              {MARK_LABELS[mark]}
            </div>
          ))}
        </div>
      </div>

      {/* CRISPR Guide Design */}
      {data.guides.length > 0 && (
        <div style={{ marginTop: 24 }}>
          <span className="eyebrow">CRISPR guide design</span>
          <p style={{ fontSize: 14, color: 'var(--body)', marginTop: 6, maxWidth: '62ch' }}>
            Top {data.guides.length} SpCas9 guide RNAs for epigenome editing of the {data.dominant_mark} mark
            at the {data.gene_id} promoter. Tool: <strong style={{ fontWeight: 400, color: 'var(--ink)' }}>{data.guides[0].tool}</strong> &mdash; {data.guides[0].mechanism}.
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginTop: 16 }}>
            {data.guides.map((g, i) => (
              <div key={i} style={{
                display: 'flex', alignItems: 'center', gap: 8,
                border: '1px solid var(--line-3)', borderRadius: 10,
                background: i === 0 ? 'var(--card)' : 'var(--card-warm)',
                padding: '5px 5px 5px 14px',
              }}>
                <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--coral)', fontWeight: 500, minWidth: 14 }}>
                  {i + 1}
                </span>
                <code style={{
                  flex: 1, fontFamily: 'var(--mono)', fontSize: 13, color: 'var(--ink)',
                  letterSpacing: '0.06em', userSelect: 'all', WebkitUserSelect: 'all',
                  background: 'transparent', padding: 0, display: 'inline',
                }}>
                  {g.sequence}<span style={{ color: 'var(--coral)', fontWeight: 500 }}>{g.pam}</span>
                </code>
                <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--tert)', whiteSpace: 'nowrap' }}>
                  {g.strand} &middot; GC {(g.gc_content * 100).toFixed(0)}%
                </span>
                <button
                  onClick={() => copyGuide(i, g.sequence + g.pam)}
                  style={{
                    flex: 'none', height: 30, padding: '0 12px',
                    borderRadius: 999, border: 0,
                    background: copiedIdx === i ? 'var(--green-text)' : 'var(--ink)',
                    color: '#fff', font: '400 12.5px var(--sans)',
                    cursor: 'pointer', transition: 'background .2s',
                  }}
                >
                  {copiedIdx === i ? '\u2713 Copied' : 'Copy'}
                </button>
              </div>
            ))}
          </div>
          <p style={{ fontSize: 12, color: 'var(--tert)', marginTop: 10, fontFamily: 'var(--mono)' }}>
            Guides scored by GC content (optimal 40-60%), proximity to TSS, and poly-T avoidance. PAM highlighted in red.
          </p>
        </div>
      )}
    </div>
  )
}
