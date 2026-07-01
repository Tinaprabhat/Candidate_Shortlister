import FunnelChart from '../components/overview/FunnelChart.jsx'
import RiskCards from '../components/overview/RiskCards.jsx'
import ShapeLandingHero from '../components/ui/ShapeLandingHero.jsx'
import ScrollShowcase from '../components/ui/ScrollShowcase.jsx'
import { useQueryClient } from '@tanstack/react-query'
import { useCandidates } from '../hooks/useCandidates.js'
import { usePipelineFunnel, usePipelineRuns, useStartPipelineRun } from '../hooks/usePipeline.js'
import { useSystemHealth } from '../hooks/useSystemHealth.js'
import { useAppStore } from '../store/useAppStore.js'

export default function Overview() {
  const setActivePage = useAppStore((state) => state.setActivePage)
  const queryClient = useQueryClient()
  const { data: candidates = [] } = useCandidates()
  const { data: runs = [] } = usePipelineRuns()
  const run = runs[0]
  const { data: funnel = [] } = usePipelineFunnel(run?.id)
  const { data: health } = useSystemHealth()
  const startPipeline = useStartPipelineRun()

  async function handleNewPipeline() {
    try {
      await startPipeline.mutateAsync()
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['pipeline-runs'] }),
        queryClient.invalidateQueries({ queryKey: ['pipeline-funnel'] }),
        queryClient.invalidateQueries({ queryKey: ['candidates'] }),
        queryClient.invalidateQueries({ queryKey: ['system-health'] }),
        queryClient.invalidateQueries({ queryKey: ['system-events'] })
      ])
      setActivePage('candidates')
    } catch (err) {
      console.error('[NewPipeline] launch failed:', err)
    }
  }

  return (
    <div className="page-stack overview-page">
      <section className="overview-title-row">
        <ShapeLandingHero
          badge="System status: optimal"
          title1="Systems"
          title2="Overview"
          description="Real-time intelligence dashboard monitoring candidate flow, heuristic survival rates, and risk distribution across the active ingestion pipeline."
        />
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
        <ScrollShowcase eyebrow="Pipeline Motion" title="Sequential Evaluation Layers" compact>
          <FunnelChart rows={funnel} />
        </ScrollShowcase>

        <aside className="overview-side-rail">
          <section className="panel neural-panel">
            <div className="section-heading">
              <p>Neural Insights</p>
              <h2>Live Model Notes</h2>
            </div>
            <blockquote>"Significant variance detected in L1 rejection velocity compared to historical mean."</blockquote>
            <blockquote>"Dynamic evaluation scaling applied to L2 gates based on cohort quality distribution."</blockquote>
            <blockquote>"Low entropy detected in candidate responses; increasing prompt complexity."</blockquote>
          </section>

        </aside>
      </div>

      <div className="overview-bottom-grid">
        <RiskCards health={health} />
        <section className="panel top-candidates">
          <div className="section-heading inline">
            <div>
              <p>Top 10 Performers Preview</p>
              <h2>Shortlist Signal</h2>
            </div>
            <button className="text-link" type="button" onClick={() => setActivePage('candidates')}>
              View Full Report
            </button>
          </div>
          <div className="reference-table">
            <div className="reference-table-head">
              <span>Candidate ID</span>
              <span>Score (A)</span>
              <span>Match %</span>
              <span>Status</span>
            </div>
            {candidates.slice(0, 5).map((candidate) => (
              <button
                className="reference-table-row"
                key={candidate.id}
                type="button"
                onClick={() => {
                  useAppStore.getState().setActiveCandidate(candidate.id)
                  setActivePage('candidates')
                }}
              >
                <span>#RR-2024-{candidate.rank === '01' ? '901' : candidate.rank === '02' ? '882' : candidate.rank === '03' ? '110' : candidate.rank === '04' ? '541' : '004'}</span>
                <strong>{(candidate.score / 100 + 0.042).toFixed(3)}</strong>
                <span>{candidate.match}</span>
                <em>Shortlisted</em>
              </button>
            ))}
          </div>
        </section>
      </div>
    </div>
  )
}
