import { Fragment } from 'react'
import { Download } from 'lucide-react'
import HorizontalLayerStack from '../components/overview/HorizontalLayerStack.jsx'
import { useCandidates } from '../hooks/useCandidates.js'
import { usePipelineRuns } from '../hooks/usePipeline.js'
import { useSystemHealth } from '../hooks/useSystemHealth.js'
import { useAppStore } from '../store/useAppStore.js'

const LOGIC_LABELS = { l2: 'Knowledge', l3: 'Reasoning', l4: 'Technical' }

export default function Overview() {
  const setActivePage = useAppStore((state) => state.setActivePage)
  const { data: candidates = [], isLoading: candidatesLoading } = useCandidates()
  const { data: runs = [], isLoading: runsLoading } = usePipelineRuns()
  const run = runs[0]
  const { data: health } = useSystemHealth()

  const pipelineRunning = run?.status === 'running'
  const statsLoading = runsLoading || pipelineRunning
  const candidatesUnready = candidatesLoading || pipelineRunning

  const topCandidate = candidates[0]

  const performerRows = candidates.slice(0, 5).map((candidate) => ({
    candidate,
    id: candidate.id,
    reasoning: candidate.evidence?.[0] || 'High-confidence heuristic match.'
  }))

  const total = run?.total_processed || 0
  const l1Rejects = run?.l1_rejects || 0
  const survivors = run?.survivors || 0
  const shortlist = run?.shortlist_count || 0
  const pct = (value) => (total > 0 ? `${((value / total) * 100).toFixed(1)}%` : '—')

  return (
    <div className="page-stack overview-page new-overview-page">
      <section className="new-overview-title">
        <div>
          <h1>Overview</h1>
          <p>
            Real-time intelligence dashboard monitoring candidate flow, heuristic survival rates, and risk
            distribution across the active ingestion pipeline.
          </p>
        </div>
        <div className="status-tile">
          <span>System Status: Optimal</span>
          <em>Latency: {health?.latency_ms || 42}ms</em>
        </div>
      </section>

      <section className="kpi-grid reference-kpis">
        <article className="metric-card">
          <p>Total Processed</p>
          {statsLoading ? (
            <span className="loading-pulse" aria-label="Calculating" />
          ) : (
            <>
              <strong>{total.toLocaleString()}</strong>
              <em>{pct(total)}</em>
            </>
          )}
        </article>
        <article className="metric-card danger">
          <p>L1 Rejects</p>
          {statsLoading ? (
            <span className="loading-pulse" aria-label="Calculating" />
          ) : (
            <>
              <strong>{l1Rejects.toLocaleString()}</strong>
              <em>{pct(l1Rejects)}</em>
            </>
          )}
        </article>
        <article className="metric-card">
          <p>Gate Survivors</p>
          {statsLoading ? (
            <span className="loading-pulse" aria-label="Calculating" />
          ) : (
            <>
              <strong>{survivors.toLocaleString()}</strong>
              <em>{pct(survivors)}</em>
            </>
          )}
        </article>
        <article className="metric-card highlight">
          <p>Final Shortlist</p>
          {statsLoading ? (
            <span className="loading-pulse" aria-label="Calculating" />
          ) : (
            <>
              <strong>{shortlist.toLocaleString()}</strong>
              <em>{pct(shortlist)} yield</em>
            </>
          )}
        </article>
      </section>

      <div className="overview-grid">
        <HorizontalLayerStack />
      </div>

      <section className="panel best-candidate-panel">
        {candidatesUnready || !topCandidate ? (
          <div className="candidate-feature-loading">
            <span className="spinner" aria-hidden="true" />
            <span>Waiting for pipeline results…</span>
          </div>
        ) : (
          <>
            <div className="candidate-feature">
              <div className="candidate-portrait">
                {topCandidate.initials || topCandidate.name?.slice(0, 2).toUpperCase()}
              </div>
              <div>
                <h2>{topCandidate.name}</h2>
                <p>ID: {topCandidate.id}</p>
              </div>
            </div>
            <div className="candidate-feature-grid">
              <div>
                <p className="micro-label">Core Strengths</p>
                <div className="domain-chips static">
                  {(topCandidate.required_skills || []).slice(0, 4).map((skill) => (
                    <span key={skill}>{skill}</span>
                  ))}
                </div>
                <div className="feature-bars">
                  {Object.entries(topCandidate.logic_scores || {})
                    .slice(0, 2)
                    .map(([layer, value]) => (
                      <Fragment key={layer}>
                        <label>
                          {LOGIC_LABELS[layer] || layer.toUpperCase()} <strong>{value}%</strong>
                        </label>
                        <span><i style={{ width: `${value}%` }} /></span>
                      </Fragment>
                    ))}
                </div>
              </div>
              <div>
                <p className="micro-label">Behavioral Signals</p>
                <p className="candidate-copy">
                  {topCandidate.evidence?.[0] || 'No behavioral evidence recorded for this candidate yet.'}
                </p>
                {topCandidate.evidence?.[1] ? <blockquote>"{topCandidate.evidence[1]}"</blockquote> : null}
              </div>
            </div>
            <button className="ghost-button" type="button" onClick={() => setActivePage('candidates')}>
              View all shortlisted
            </button>
          </>
        )}
      </section>

      <section className="panel top-candidates reference-performers-panel">
        <div className="section-heading inline performer-heading">
          <div>
            <h2>Top 10 Performers Preview</h2>
            <p>High-confidence heuristic matches</p>
          </div>
          <button className="ghost-button" type="button" onClick={() => setActivePage('candidates')}>
            <Download size={15} aria-hidden="true" />
            Download Top 100 CSV
          </button>
        </div>
        <div className="performer-table">
          <div className="performer-table-head">
            <span>Candidate ID</span>
            <span>Name</span>
            <span>Score</span>
            <span>Reasoning</span>
          </div>
          {performerRows.map(({ candidate, id, reasoning }) => (
            <button
              className="performer-table-row"
              key={candidate.id}
              type="button"
              onClick={() => {
                useAppStore.getState().setActiveCandidate(candidate.id)
                setActivePage('candidates')
              }}
            >
              <span>{id}</span>
              <strong>{candidate.name}</strong>
              <em>{candidate.match}</em>
              <p>{reasoning}</p>
            </button>
          ))}
        </div>
      </section>
    </div>
  )
}
