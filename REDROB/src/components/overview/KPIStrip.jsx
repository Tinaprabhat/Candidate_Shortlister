import { Activity, Database, Server, Zap } from 'lucide-react'

const icons = {
  processed: Activity,
  survivors: Database,
  shortlist: Zap,
  nodes: Server
}

export default function KPIStrip({ run, health }) {
  const cards = [
    { key: 'processed', label: 'Processed', value: run?.total_processed?.toLocaleString() || '0' },
    { key: 'survivors', label: 'Survivors', value: run?.survivors?.toLocaleString() || '0' },
    { key: 'shortlist', label: 'Shortlist', value: run?.shortlist_count?.toLocaleString() || '0' },
    { key: 'nodes', label: 'Active Nodes', value: health?.active_nodes || '0 / 0' }
  ]

  return (
    <section className="kpi-grid" aria-label="Pipeline KPIs">
      {cards.map((card) => {
        const Icon = icons[card.key]
        return (
          <article className="metric-card" key={card.key}>
            <Icon size={18} aria-hidden="true" />
            <div>
              <p>{card.label}</p>
              <strong>{card.value}</strong>
            </div>
          </article>
        )
      })}
    </section>
  )
}
