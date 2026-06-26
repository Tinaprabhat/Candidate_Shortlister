import { useQuery } from '@tanstack/react-query'
import { getCandidate, listCandidates } from '../api/redrobApi.js'
import { useAppStore } from '../store/useAppStore.js'

export function useCandidates() {
  const filters = useAppStore((state) => state.filters)

  return useQuery({
    queryKey: ['candidates', filters],
    queryFn: () => listCandidates(filters),
    placeholderData: (previousData) => previousData
  })
}

export function useCandidate(candidateId) {
  return useQuery({
    queryKey: ['candidate', candidateId],
    queryFn: () => getCandidate(candidateId),
    enabled: Boolean(candidateId)
  })
}
