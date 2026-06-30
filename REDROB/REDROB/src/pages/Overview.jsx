import { Download } from 'lucide-react'
import HorizontalLayerStack from '../components/overview/HorizontalLayerStack.jsx'
import { useCandidates } from '../hooks/useCandidates.js'
import { usePipelineRuns } from '../hooks/usePipeline.js'
import { useSystemHealth } from '../hooks/useSystemHealth.js'
import { useAppStore } from '../store/useAppStore.js'

export default function Overview() {
  const setActivePage = useAppStore((state) => state.setActivePage)
  const { data: candidates = [] } = useCandidates()
  const { data: runs = [] } = usePipelineRuns()
  const run = runs[0]
  const { data: health } = useSystemHealth()
  const performerRows = candidates.slice(0, 5).map((candidate, index) => ({
    candidate,
    id: ['#RR-2024-901', '#RR-2024-842', '#RR-2024-715', '#RR-2024-603', '#RR-2024-004'][index] || `#RR-2024-${candidate.rank}`,
    reasoning:
      [
        'Exceptional low-level systems optimization and architectural foresight.',
        'Strong distributed systems background with proven scale experience.',
        'High collaborative entropy; excels in decentralized environments.',
        'Deep expertise in Rust/Go concurrency patterns and cloud-native infra.',
        'Clean identity signal with consistent evaluation-design evidence.'
      ][index] || candidate.evidence?.[0] || 'High-confidence heuristic match.'
  }))

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
          <strong>{run?.total_processed?.toLocaleString() || '12,842'}</strong>
          <em>+12.4%</em>
        </article>
        <article className="metric-card danger">
          <p>L1 Rejects</p>
          <strong>{run?.l1_rejects?.toLocaleString() || '4,109'}</strong>
          <em>32.0%</em>
        </article>
        <article className="metric-card">
          <p>Gate Survivors</p>
          <strong>{run?.survivors?.toLocaleString() || '8,733'}</strong>
          <em>68.0%</em>
        </article>
        <article className="metric-card highlight">
          <p>Final Shortlist</p>
          <strong>{run?.shortlist_count?.toLocaleString() || '214'}</strong>
          <em>1.6% yield</em>
        </article>
      </section>

      <div className="overview-grid">
        <HorizontalLayerStack />
      </div>

      <section className="panel best-candidate-panel">
        <div className="candidate-feature">
          <div className="candidate-portrait">AR</div>
          <div>
            <h2>Alex Rivera</h2>
            <p>ID: #RR-2024-901</p>
          </div>
        </div>
        <div className="candidate-feature-grid">
          <div>
            <p className="micro-label">Core Strengths</p>
            <div className="domain-chips static">
              <span>Distributed Systems</span>
              <span>Rust/Go</span>
              <span>Cloud Native</span>
              <span>High-Availability</span>
            </div>
            <div className="feature-bars">
              <label>System Design <strong>99%</strong></label>
              <span><i style={{ width: '99%' }} /></span>
              <label>Concurrency Patterns <strong>94%</strong></label>
              <span><i style={{ width: '94%' }} /></span>
            </div>
          </div>
          <div>
            <p className="micro-label">Behavioral Signals</p>
            <p className="candidate-copy">
              Exhibits high collaborative entropy; thrives in decentralized decision-making environments. Strong
              mentorship signals detected in peer-review simulations.
            </p>
            <blockquote>
              "Rivera demonstrates a rare synthesis of low-level systems optimization and high-level architectural
              foresight."
            </blockquote>
          </div>
        </div>
        <button className="ghost-button" type="button" onClick={() => setActivePage('candidates')}>
          View all shortlisted
        </button>
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
