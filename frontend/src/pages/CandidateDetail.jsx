import { ArrowLeft, Braces, Calendar, Code2, Database, MapPin, RotateCcw, UserPlus, Zap } from 'lucide-react'
import Badge from '../components/ui/Badge.jsx'
import ProgressBar from '../components/ui/ProgressBar.jsx'
import ScrollShowcase from '../components/ui/ScrollShowcase.jsx'
import { useCandidate } from '../hooks/useCandidates.js'
import { useAppStore } from '../store/useAppStore.js'

export default function CandidateDetail() {
  const activeCandidateId = useAppStore((state) => state.activeCandidateId)
  const setActivePage = useAppStore((state) => state.setActivePage)
  const { data: candidate, isLoading } = useCandidate(activeCandidateId)

  if (isLoading || !candidate) {
    return <div className="table-empty">Loading dossier...</div>
  }

  return (
    <div className="page-stack detail-page">
      <button className="back-button" type="button" onClick={() => setActivePage('candidates')}>
        <ArrowLeft size={18} aria-hidden="true" />
        <span>Back to candidates</span>
      </button>

      <section className="detail-header">
        <div className="candidate-photo">
          <div className="photo-face">{candidate.initials}</div>
        </div>
        <div>
          <div className="detail-badges">
            <Badge tone="success">Top 1% Rank</Badge>
            <Badge tone={candidate.integrity_status === 'CLEAN' ? 'success' : 'warning'}>Verified</Badge>
          </div>
          <h1>{candidate.name}</h1>
          <p>{candidate.role}</p>
          <div className="detail-meta">
            <span>
              <MapPin size={15} aria-hidden="true" />
              {candidate.location}
            </span>
            <span>
              <Calendar size={15} aria-hidden="true" />
              Verified {new Date(candidate.verified_at).toLocaleDateString()}
            </span>
          </div>
          <div className="skill-chips">
            <span>Python_3.12</span>
            <span>PyTorch_2.0</span>
            <span>React_Native</span>
            <span>Kubernetes</span>
          </div>
        </div>
      </section>

      <div className="detail-grid">
        <section className="panel vetting-panel">
          <div className="section-heading">
            <h2>Vetting Velocity</h2>
            <p>Quantified funnel progression through proprietary neural filters.</p>
          </div>
          <div className="velocity-score">
            <strong>{candidate.score}.2</strong>
            <span>Composite Score</span>
          </div>
          <div className="score-lines">
            {[
              ['01. Technical Integrity Probe', 98, 'Match'],
              ['02. Behavioral Pattern Analysis', 92, 'Consistency'],
              ['03. Cultural Alignment Scale', 87, 'Affinity'],
              ['04. Lead Potential Quotient', 76, 'Conversion']
            ].map(([label, value, note]) => (
              <div className="score-line detail-line" key={label}>
                <span>{label}</span>
                <ProgressBar value={value} />
                <strong>{value}% {note}</strong>
              </div>
            ))}
          </div>
          <div className="legend-row">
            <span>Matching Ratio</span>
            <span>Historical Null</span>
            <span>Benchmark Zero</span>
          </div>
        </section>

        <aside className="detail-side">
          <section className="panel signature-panel">
            <div className="section-heading">
              <h2>Neural Signature</h2>
            </div>
            {[
              ['Logical Reasoning Engine', 98],
              ['Architectural Creative Syntax', 82],
              ['Inter-node Collaborative Index', 91],
              ['Domain Agility Pulse', 76]
            ].map(([label, value]) => (
              <div className="signature-line" key={label}>
                <span>{label}</span>
                <strong>{value} / 100</strong>
                <ProgressBar value={value} />
              </div>
            ))}
          </section>

          <section className="panel signal-panel">
            <div className="section-heading">
              <p>Atmospheric Signals</p>
            </div>
            {[
              [Zap, 'Execution Velocity', 'Top 0.5% in commit frequency and merge success rates.'],
              [Database, 'Deep Systems Modeler', 'Consistent identification of race conditions in early phases.'],
              [Code2, 'Low-Friction Architect', 'High peer-review acceptance with zero interpersonal conflicts logged.']
            ].map(([Icon, title, text]) => (
              <article className="signal-card" key={title}>
                <Icon size={18} aria-hidden="true" />
                <div>
                  <strong>{title}</strong>
                  <span>{text}</span>
                </div>
              </article>
            ))}
          </section>
        </aside>
      </div>

      <div className="detail-grid lower">
        <ScrollShowcase eyebrow="Verified Professional Evidence" title="Immutable Work History" compact>
          <section className="panel evidence-panel">
            {[
              [Braces, 'NeuralNet Optimizer v2', 'Pioneered a quantization-aware training pipeline that reduced LLM inference latency by 45% on edge hardware while maintaining 99.2% accuracy retention.'],
              [Database, 'Hydra-Sync Distributed DB', 'Architected a multi-region eventual consistency model resolving conflict resolution in under 12ms across global clusters.']
            ].map(([Icon, title, text]) => (
              <article className="evidence-entry" key={title}>
                <Icon size={20} aria-hidden="true" />
                <div>
                  <strong>{title}</strong>
                  <p>{text}</p>
                  <span>Github_Link &nbsp; Patent_Ref: 8,429,112</span>
                </div>
              </article>
            ))}
          </section>
        </ScrollShowcase>

        <aside className="detail-side">
          <section className="panel quote-panel">
            <blockquote>
              "{candidate.name.split(' ')[0]}'s evaluation profile reflects a rare convergence of raw algorithmic
              power and architectural foresight. The {candidate.score}.2 composite score is rooted in verified telemetry
              from high-stakes deployments."
            </blockquote>
            <p>Logic: Rev 4.2.1 - Bayesian Engine</p>
          </section>
          <button className="primary-action wide" type="button">
            <UserPlus size={18} aria-hidden="true" />
            <span>Shortlist Candidate</span>
          </button>
          <div className="detail-actions">
            <button className="ghost-button" type="button">
              <RotateCcw size={16} aria-hidden="true" />
              Audit Trace
            </button>
            <button className="ghost-button" type="button">
              <Braces size={16} aria-hidden="true" />
              Export JSON
            </button>
          </div>
        </aside>
      </div>

      <section className="panel mobile-timeline">
        <div className="section-heading">
          <p>Timeline</p>
          <h2>Screening Events</h2>
        </div>
        <div className="timeline-list horizontal">
          {candidate.timeline.map((event) => (
            <div className={`timeline-item ${event.type}`} key={`${event.time}-${event.label}`}>
              <span>{event.time}</span>
              <strong>{event.label}</strong>
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}
