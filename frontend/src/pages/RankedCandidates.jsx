import CandidateTable from '../components/candidates/CandidateTable.jsx'
import EvidenceDrawer from '../components/candidates/EvidenceDrawer.jsx'
import FilterRail from '../components/candidates/FilterRail.jsx'
import { useCandidates } from '../hooks/useCandidates.js'
import { useAppStore } from '../store/useAppStore.js'

export default function RankedCandidates() {
  const { data: candidates = [], isLoading, isFetching } = useCandidates()

  return (
    <div className="candidates-page">
      <FilterRail />
      <section className="candidates-main">
        <div className="section-heading candidate-heading">
          <div>
            <p>Ranked Candidates</p>
            <h1>Ranked Candidates</h1>
          </div>
          <span className="refresh-state">
            {isFetching ? 'Refreshing' : `Matched ${candidates.length} profiles - SR AI Researcher`}
          </span>
        </div>
        <CandidateTable candidates={candidates} isLoading={isLoading} />
      </section>
      <EvidenceDrawer />
    </div>
  )
}
