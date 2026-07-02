import api from './client.js'
import {
  getCandidate as getMockCandidate,
  getPipelineFunnel as getMockPipelineFunnel,
  getPipelineRuns as getMockPipelineRuns,
  getSystemEvents as getMockSystemEvents,
  getSystemHealth as getMockSystemHealth,
  listCandidates as listMockCandidates,
  scheduleInterview as mockScheduleInterview
} from './mockApi.js'

const useMockApi = import.meta.env.VITE_USE_MOCK_API !== 'false'

async function remoteOrMock(remoteCall, mockCall) {
  if (useMockApi) return mockCall()

  try {
    return await remoteCall()
  } catch (error) {
    if (error.response?.status === 401) throw error
    return mockCall()
  }
}

export function listCandidates(filters) {
  return remoteOrMock(
    () => api.get('/candidates', { params: filters }).then((res) => res.data),
    () => listMockCandidates(filters)
  )
}

export function getCandidate(id) {
  return remoteOrMock(
    () => api.get(`/candidates/${id}`).then((res) => res.data),
    () => getMockCandidate(id)
  )
}

export function getPipelineRuns() {
  return remoteOrMock(
    () => api.get('/pipeline/runs').then((res) => res.data),
    () => getMockPipelineRuns()
  )
}

export function getPipelineFunnel(runId) {
  return remoteOrMock(
    () => api.get(`/pipeline/runs/${runId}/funnel`).then((res) => res.data),
    () => getMockPipelineFunnel(runId)
  )
}

export function getSystemHealth() {
  return remoteOrMock(
    () => api.get('/system/health').then((res) => res.data),
    () => getMockSystemHealth()
  )
}

export function getSystemEvents() {
  return remoteOrMock(
    () => api.get('/system/events', { params: { limit: 10 } }).then((res) => res.data),
    () => getMockSystemEvents()
  )
}

export function scheduleInterview(candidateId) {
  return remoteOrMock(
    () => api.post(`/candidates/${candidateId}/schedule-interview`).then((res) => res.data),
    () => mockScheduleInterview(candidateId)
  )
}

export function uploadCandidates(file) {
  const upload = () => {
    const form = new FormData()
    form.append('candidates', file)
    return api.post('/pipeline/runs/upload', form).then((res) => res.data)
  }
  return remoteOrMock(upload, () => Promise.resolve({ ok: true, status: 'running', run_id: 'mock' }))
}
