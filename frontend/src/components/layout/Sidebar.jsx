import { LayoutGrid, Play, Settings, Users } from 'lucide-react'
import { useQueryClient } from '@tanstack/react-query'
import { useStartPipelineRun } from '../../hooks/usePipeline.js'
import { useAppStore } from '../../store/useAppStore.js'

const navItems = [
  { id: 'overview', label: 'Overview', icon: LayoutGrid, target: 'overview' },
  { id: 'candidates', label: 'Ranked Candidates', icon: Users, target: 'candidates' },
  { id: 'settings', label: 'Settings', icon: Settings, target: 'health' }
]

export default function Sidebar() {
  const activePage = useAppStore((state) => state.activePage)
  const setActivePage = useAppStore((state) => state.setActivePage)
  const queryClient = useQueryClient()
  const startPipeline = useStartPipelineRun()

  async function handleNewPipeline() {
    try {
      await startPipeline.mutateAsync()
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['pipeline-runs'] }),
        queryClient.invalidateQueries({ queryKey: ['pipeline-funnel'] }),
        queryClient.invalidateQueries({ queryKey: ['candidates'] }),
        queryClient.invalidateQueries({ queryKey: ['system-health'] }),
        queryClient.invalidateQueries({ queryKey: ['system-events'] })
      ])
      setActivePage('candidates')
    } catch (err) {
      console.error('[NewPipeline] launch failed:', err)
    }
  }

  return (
    <aside className="sidebar">
      <div className="brand-block">
        <div className="brand-name">RedRob</div>
        <div className="brand-subtitle">Command Center</div>
      </div>

      <nav className="sidebar-nav" aria-label="Primary navigation">
        {navItems.map((item) => {
          const active =
            (activePage === 'overview' && item.id === 'overview') ||
            (activePage === 'candidates' && item.id === 'candidates') ||
            (activePage === 'detail' && item.id === 'candidates') ||
            (activePage === 'health' && item.id === 'settings')
          const Icon = item.icon
          return (
            <button
              key={item.id}
              className={`nav-button ${active ? 'active' : ''}`}
              type="button"
              onClick={() => setActivePage(item.target)}
            >
              <Icon size={22} aria-hidden="true" />
              <span>{item.label}</span>
            </button>
          )
        })}
      </nav>

      <button className="primary-action" type="button" onClick={handleNewPipeline} disabled={startPipeline.isPending}>
        <Play size={16} aria-hidden="true" />
        <span>{startPipeline.isPending ? 'Starting...' : 'New Pipeline'}</span>
      </button>
    </aside>
  )
}
