import { RotateCcw } from 'lucide-react'
import { useCandidates } from '../../hooks/useCandidates.js'
import { useAppStore } from '../../store/useAppStore.js'

export default function FilterRail() {
  const filters = useAppStore((state) => state.filters)
  const setFilters = useAppStore((state) => state.setFilters)
  const resetFilters = useAppStore((state) => state.resetFilters)
  const { data: candidates = [] } = useCandidates()

  // Collect unique required and inferred skills from currently visible candidates
  const requiredSkills = [...new Set(candidates.flatMap((c) => c.required_skills || []))]
  const inferredSkills = [...new Set(candidates.flatMap((c) => c.inferred_skills || []))]

  function handleVerifiedOnly(checked) {
    setFilters({ verified_only: checked, ...(checked ? { risk_flagged: false } : {}) })
  }

  function handleRiskFlagged(checked) {
    setFilters({ risk_flagged: checked, ...(checked ? { verified_only: false } : {}) })
  }

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
          onChange={(event) => handleVerifiedOnly(event.target.checked)}
        />
        <span>Verified only</span>
      </label>
      <label className="checkbox-line">
        <input
          type="checkbox"
          checked={filters.risk_flagged}
          onChange={(event) => handleRiskFlagged(event.target.checked)}
        />
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

      {requiredSkills.length > 0 && (
        <>
          <div className="rail-heading">
            <span>Required Matched Skills</span>
          </div>
          <div className="skill-tag-rail">
            {requiredSkills.map((skill) => (
              <span key={skill} className="skill-tag skill-tag--required">{skill}</span>
            ))}
          </div>
        </>
      )}

      {inferredSkills.length > 0 && (
        <>
          <div className="rail-heading">
            <span>Inferred Matched Skills</span>
          </div>
          <div className="skill-tag-rail">
            {inferredSkills.map((skill) => (
              <span key={skill} className="skill-tag skill-tag--inferred">{skill}</span>
            ))}
          </div>
        </>
      )}

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
