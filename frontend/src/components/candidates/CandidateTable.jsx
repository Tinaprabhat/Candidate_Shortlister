import { useAppStore } from '../../store/useAppStore.js'

export default function CandidateTable({ candidates = [], isLoading }) {
  const activeCandidateId = useAppStore((state) => state.activeCandidateId)
  const setActiveCandidate = useAppStore((state) => state.setActiveCandidate)

  if (isLoading) {
    return <div className="table-empty">Loading candidates...</div>
  }

  if (!candidates.length) {
    return <div className="table-empty">No candidates match the current filters.</div>
  }

  return (
    <div className="ranked-stack">
      <div className="ranked-stack-head">
        <span>Rank</span>
        <span>Candidate</span>
      </div>
      {candidates.map((candidate) => {
        const active = activeCandidateId === candidate.id
        return (
          <button
            key={candidate.id}
            className={`rank-card ${active ? 'selected' : ''}`}
            type="button"
            onClick={() => setActiveCandidate(candidate.id)}
          >
            <span className="rank-cell">{candidate.rank}</span>
            <div className="candidate-cell">
              <div className="avatar">{candidate.initials}</div>
              <div>
                <strong>{candidate.name}</strong>
                <em>{candidate.score} / 100</em>
              </div>
            </div>
          </button>
        )
      })}
    </div>
  )
}
