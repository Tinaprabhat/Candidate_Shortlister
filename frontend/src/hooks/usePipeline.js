import { useMutation, useQuery } from '@tanstack/react-query'
import { getPipelineFunnel, getPipelineRuns, startPipelineRun } from '../api/redrobApi.js'

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

export function useStartPipelineRun() {
  return useMutation({
    mutationFn: startPipelineRun
  })
}
