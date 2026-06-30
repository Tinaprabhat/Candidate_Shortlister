import { useQuery } from '@tanstack/react-query'
import { getSystemEvents, getSystemHealth } from '../api/redrobApi.js'

export function useSystemHealth() {
  return useQuery({
    queryKey: ['system-health'],
    queryFn: getSystemHealth
  })
}

export function useSystemEvents() {
  return useQuery({
    queryKey: ['system-events'],
    queryFn: getSystemEvents
  })
}
