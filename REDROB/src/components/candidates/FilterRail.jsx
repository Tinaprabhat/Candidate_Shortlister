import { RotateCcw } from 'lucide-react'
import { useAppStore } from '../../store/useAppStore.js'

export default function FilterRail() {
  const filters = useAppStore((state) => state.filters)
  const setFilters = useAppStore((state) => state.setFilters)
  const resetFilters = useAppStore((state) => state.resetFilters)

  return (
    <aside className="filter-rail">
      <div className="rail-heading">
        <span>Score Range</span>
      </div>

      <label className="field-stack">
        <input
          type="range"
          min="60"
          max="98"
          value={filters.score_min}
          onChange={(event) => setFilters({ score_min: Number(event.target.value) })}
        />
        <div className="range-labels">
          <strong>Min: {filters.score_min}</strong>
          <strong>Max: 100</strong>
        </div>
      </label>

      <div className="rail-heading">
        <span>Fraud Detection</span>
      </div>
      <label className="checkbox-line">
        <input
          type="checkbox"
          checked={filters.verified_only}
          onChange={(event) => setFilters({ verified_only: event.target.checked })}
        />
        <span>Verified only</span>
      </label>
      <label className="checkbox-line">
        <input type="checkbox" checked={false} readOnly />
        <span>Risk flagged</span>
      </label>

      <label className="field-stack">
        <span>Target Domains</span>
        <select value={filters.domain} onChange={(event) => setFilters({ domain: event.target.value })}>
          <option value="all">All domains</option>
          <option value="core-ml">Core ML</option>
          <option value="systems">Systems</option>
          <option value="alignment">Alignment</option>
        </select>
      </label>

      <div className="domain-chips">
        <button type="button" onClick={() => setFilters({ domain: 'core-ml' })}>LLM Architecture</button>
        <button type="button" onClick={() => setFilters({ domain: 'systems' })}>NLP Eng</button>
        <button type="button" onClick={() => setFilters({ domain: 'systems' })}>Cloud Infra</button>
        <button type="button" onClick={() => setFilters({ domain: 'alignment' })}>PyTorch</button>
      </div>

      <button className="ghost-button" type="button" onClick={resetFilters}>
        <RotateCcw size={16} aria-hidden="true" />
        <span>Reset</span>
      </button>

      <blockquote className="filter-quote">
        "Filtering for high-precision candidates reduces selection time by ~42% for this specific job profile."
      </blockquote>
    </aside>
  )
}
