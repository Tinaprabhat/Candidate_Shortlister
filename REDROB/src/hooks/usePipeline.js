import { useQuery } from '@tanstack/react-query'
import { getPipelineFunnel, getPipelineRuns } from '../api/redrobApi.js'

export function usePipelineRuns() {
  return useQuery({
    queryKey: ['pipeline-runs'],
    queryFn: getPipelineRuns
  })
}

export function usePipelineFunnel(runId) {
  return useQuery({
    queryKey: ['pipeline-funnel', runId],
    queryFn: () => getPipelineFunnel(runId),
    enabled: Boolean(runId)
  })
}
