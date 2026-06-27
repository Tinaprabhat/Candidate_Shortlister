import { candidates, pipelineFunnel, pipelineRuns, systemEvents, systemHealth } from '../mocks/data.js'

function delay(value, ms = 240) {
  return new Promise((resolve) => {
    window.setTimeout(() => resolve(value), ms)
  })
}

export function listCandidates(filters = {}) {
  const scoreMin = Number(filters.score_min ?? 0)
  const verifiedOnly = filters.verified_only === true || filters.verified_only === 'true'
  const riskFlagged = filters.risk_flagged === true || filters.risk_flagged === 'true'
  const domain = filters.domain || 'all'

  const rows = candidates
    .filter((candidate) => candidate.score >= scoreMin)
    .filter((candidate) => !verifiedOnly || candidate.integrity_status === 'CLEAN')
    .filter((candidate) => !riskFlagged || candidate.integrity_status !== 'CLEAN')
    .filter((candidate) => domain === 'all' || candidate.domain === domain)
    .sort((a, b) => b.score - a.score)

  return delay(rows)
}

export function getCandidate(id) {
  return delay(candidates.find((candidate) => candidate.id === id) || candidates[0])
}

export function getPipelineRuns() {
  return delay(pipelineRuns)
}

export function getPipelineFunnel() {
  return delay(pipelineFunnel)
}

export function getSystemHealth() {
  return delay(systemHealth)
}

export function getSystemEvents() {
  return delay(systemEvents)
}

export function scheduleInterview(candidateId) {
  const candidate = candidates.find((row) => row.id === candidateId)
  return delay({
    ok: true,
    candidate_id: candidateId,
    message: `${candidate?.name || 'Candidate'} moved to interview queue.`
  })
}
