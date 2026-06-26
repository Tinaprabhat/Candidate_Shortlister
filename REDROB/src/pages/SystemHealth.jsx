import { AlertTriangle, CheckCircle2, Database, RotateCcw, Shield, Terminal } from 'lucide-react'
import ProgressBar from '../components/ui/ProgressBar.jsx'
import ScrollShowcase from '../components/ui/ScrollShowcase.jsx'
import { useSystemEvents, useSystemHealth } from '../hooks/useSystemHealth.js'

export default function SystemHealth() {
  const { data: health } = useSystemHealth()
  const { data: events = [] } = useSystemEvents()

  return (
    <div className="page-stack health-page">
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

      <div className="health-layout">
        <div className="health-left">
          <ScrollShowcase eyebrow="Architecture Health" title="Core Diagnostics" compact>
            <section className="panel integrity-report">
              <div className="section-heading inline">
                <h2>Integrity Report</h2>
                <span>Generated 2m ago</span>
              </div>
              <blockquote>
                "The current architectural equilibrium remains within nominal parameters. We observe a minor deviation
                in ranking pipeline throughput, likely due to recent unstructured resume data. No manual intervention is
                required at this stage."
              </blockquote>
              <div className="diagnostic-grid">
                <article>
                  <p>Neural Load</p>
                  <strong>{health?.neural_load || 42}.8%</strong>
                  <ProgressBar value={health?.neural_load || 42} />
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
          </ScrollShowcase>

          <div className="health-config-grid">
            <section className="panel config-panel">
              <h2><Shield size={20} aria-hidden="true" /> Core Configuration</h2>
              <div className="config-row">
                <span>Enhanced Encryption</span>
                <button className="toggle on" type="button" aria-label="Enhanced encryption on" />
              </div>
              <div className="config-row">
                <span>Biometric Auth</span>
                <button className="toggle" type="button" aria-label="Biometric auth off" />
              </div>
              <div className="config-row">
                <span>Environment Diagnostics</span>
                <strong>Configure</strong>
              </div>
            </section>

            <section className="panel config-panel">
              <h2><Database size={20} aria-hidden="true" /> Knowledge Base Status</h2>
              <div className="config-row">
                <span>Index Syncing</span>
                <strong>Auto - 30m</strong>
              </div>
              <div className="config-row">
                <span>KB Total Volume</span>
                <strong>14.8 PB</strong>
              </div>
              <div className="config-row">
                <span>Regional Integrity</span>
                <strong>Optimized</strong>
              </div>
            </section>
          </div>
        </div>

        <aside className="health-right">
          <section className="panel efficiency-card">
            <CheckCircle2 size={42} aria-hidden="true" />
            <p>Core Efficiency Score</p>
            <strong>{health?.core_efficiency || 98}</strong>
            <span>System performance is exceeding the quarterly baseline by 12.4%.</span>
          </section>

          <section className="panel execution-panel">
            <div className="section-heading">
              <h2>Execution Log</h2>
            </div>
            <div className="event-list">
              {events.map((event) => {
                const Icon = event.type === 'warning' ? AlertTriangle : event.type === 'patch' ? RotateCcw : Terminal
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

          <section className="panel topology-card">
            <p>Network Topology</p>
            <strong>Active Nodes: {health?.active_nodes || '142/142'}</strong>
            <div className="topology-bars" aria-hidden="true">
              <span />
              <span />
              <span />
              <span />
              <span />
            </div>
          </section>
        </aside>
      </div>
    </div>
  )
}
