import { ChevronDown, ChevronRight, Database, RotateCcw } from 'lucide-react'
import ProgressBar from '../components/ui/ProgressBar.jsx'
import { useSystemEvents, useSystemHealth } from '../hooks/useSystemHealth.js'

export default function SystemHealth() {
  const { data: health } = useSystemHealth()
  const { data: events = [] } = useSystemEvents()
  const executionRows = (events.length ? events : [
    { id: 'evt-1', type: 'patch', message: 'Ranking model patch v2.4.1 deployed', author: 'Admin_04', created_at: '14:22 UTC' },
    { id: 'evt-3', type: 'info', message: 'Supabase token refresh validated', author: 'Auth monitor', created_at: '10:18 UTC' },
    { id: 'evt-4', type: 'info', message: 'Candidate snapshot verified', author: 'Scheduler', created_at: '08:00 UTC' }
  ]).filter((event) => ['evt-1', 'evt-3', 'evt-4'].includes(event.id))

  return (
    <div className="page-stack health-page system-status-page">
      <section className="section-heading page-title">
        <h1>System Status</h1>
      </section>

      <section className="health-status-grid">
        <article className="status-strip">
          <span />
          <div>
            <p>Global Engine</p>
            <strong>Active - {health?.uptime || '99.98%'} Uptime</strong>
          </div>
        </article>
        <article className="status-strip amber">
          <span />
          <div>
            <p>Latency Threshold</p>
            <strong>Synced - {health?.latency_ms || 42}ms</strong>
          </div>
        </article>
      </section>

      <section className="system-diagnostics-heading">
        <p>Architecture Health</p>
        <h2>Core Diagnostics</h2>
      </section>

      <section className="panel integrity-report system-integrity">
        <div className="section-heading inline">
          <h2>Integrity Report</h2>
          <span>Generated 2m ago</span>
        </div>
        <blockquote>
          "The current architectural equilibrium remains within nominal parameters. We observe a minor deviation in
          ranking pipeline throughput, likely due to recent unstructured resume data. No manual intervention is required
          at this stage."
        </blockquote>
        <div className="diagnostic-grid">
          <article>
            <p>Neural Load</p>
            <strong>{health?.neural_load || 72}.8%</strong>
            <ProgressBar value={health?.neural_load || 72} />
          </article>
          <article>
            <p>Cache Memory</p>
            <strong>6.2 TB</strong>
            <ProgressBar value={68} />
          </article>
          <article>
            <p>Environment Threads</p>
            <strong>1,892</strong>
            <ProgressBar value={86} tone="tertiary" />
          </article>
        </div>
      </section>

      <section className="panel knowledge-base-panel">
        <h2><Database size={20} aria-hidden="true" /> Knowledge Base Status</h2>
        <div className="kb-status-grid">
          <div className="kb-row">
            <span>Index Syncing</span>
            <strong>Auto - 30m</strong>
          </div>
          <div className="kb-row">
            <span>Cache Accesses</span>
            <strong>1.2M</strong>
          </div>
          <div className="kb-row">
            <span>KB Total Volume</span>
            <strong>14.8 PB</strong>
          </div>
          <div className="kb-row">
            <span>Total Queries</span>
            <strong>850K</strong>
          </div>
        </div>
        <div className="data-files">
          <p>Data Files</p>
          <div className="data-file-grid">
            {['primary_node_01.dat', 'vector_index_main.db', 'neural_weights_v2.bin'].map((file) => (
              <button className="data-file-pill" type="button" key={file}>
                <span>{file}</span>
                <ChevronDown size={15} aria-hidden="true" />
              </button>
            ))}
          </div>
        </div>
      </section>

      <section className="panel system-health-card">
        <p>Operational Integrity</p>
        <h2>
          System Status:
          <span>Healthy</span>
        </h2>
        <strong>
          All architectural nodes are operating within optimal parameters. No anomalies detected in the current cycle.
        </strong>
      </section>

      <section className="panel execution-panel system-execution">
        <div className="section-heading">
          <h2>Execution Log</h2>
        </div>
        <div className="event-list">
          {executionRows.map((event) => {
            const Icon = event.type === 'patch' ? RotateCcw : ChevronRight
            return (
              <div className={`event-row ${event.type}`} key={event.id}>
                <Icon size={18} aria-hidden="true" />
                <div>
                  <strong>{event.message}</strong>
                  <span>{event.created_at} - {event.author}</span>
                </div>
              </div>
            )
          })}
        </div>
        <button className="ghost-button wide" type="button">View Detailed Logs</button>
      </section>

      <section className="panel topology-card system-topology">
        <p>Network Topology</p>
        <strong>Total Threads: {String(health?.active_nodes || '142 / 142').replace(' / ', ' / ')}</strong>
        <div className="topology-bars" aria-hidden="true">
          <span />
          <span />
          <span />
          <span />
          <span />
          <span />
          <span />
        </div>
      </section>
    </div>
  )
}
