import { useAppStore } from '../../store/useAppStore.js'

export default function FunnelChart({ rows = [] }) {
  const setActivePage = useAppStore((state) => state.setActivePage)
  const displayRows = rows.length
    ? rows.slice(0, 4)
    : [
        { layer: 'L1', label: 'Automated Profile Scan', count_out: 12842 },
        { layer: 'L2', label: 'Technical Assessment Gate', count_out: 8733 },
        { layer: 'L3', label: 'Strategic Fit Review', count_out: 1921 },
        { layer: 'L7', label: 'Final Shortlist', count_out: 214 }
      ]

  return (
    <section className="panel funnel-panel">
      <div className="section-heading">
        <p>Pipeline Funnel</p>
        <h2>Visualization</h2>
      </div>

      <div className="funnel-stack">
        {displayRows.map((row, index) => {
          const width = [100, 86, 60, 30][index] || 24
          const eliminated = ['32% eliminated', '78% eliminated', '89% eliminated'][index]
          const final = index === displayRows.length - 1
          return (
            <div className="funnel-layer-wrap" key={`${row.layer}-${row.label}`}>
              <button
                className={`funnel-layer ${final ? 'final' : ''}`}
                style={{ width: `${width}%` }}
                type="button"
                onClick={() => setActivePage('candidates')}
              >
                <span>{final ? 'Final Shortlist' : `${row.layer}: ${row.label}`}</span>
                <strong>{row.count_out.toLocaleString()}</strong>
                {!final ? <em>Unit</em> : null}
              </button>
              {!final ? <div className="elimination-pill">down {eliminated}</div> : null}
            </div>
          )
        })}
      </div>
    </section>
  )
}
