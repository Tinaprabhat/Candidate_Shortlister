import { Bell, LogOut, Moon, Search, Sun, UserCircle } from 'lucide-react'
import { hasSupabaseConfig, signOut } from '../../api/supabase.js'
import { useAppStore } from '../../store/useAppStore.js'

const titles = {
  overview: 'Command Center',
  candidates: 'Pipeline Analysis',
  detail: 'Candidate Dossier: 104-B',
  health: 'Command Center'
}

export default function TopBar() {
  const activePage = useAppStore((state) => state.activePage)
  const darkMode = useAppStore((state) => state.darkMode)
  const toggleDark = useAppStore((state) => state.toggleDark)
  const logout = useAppStore((state) => state.logout)

  async function handleLogout() {
    await signOut()
    logout()
  }

  return (
    <header className="topbar">
      <div className="topbar-title">{titles[activePage] || 'Command Center'}</div>
      <label className="search-box">
        <Search size={18} aria-hidden="true" />
        <input type="search" placeholder={activePage === 'health' ? 'Search parameters...' : 'Search pipelines, candidates...'} />
      </label>

      <div className="topbar-actions">
        <button className="icon-button" type="button" aria-label="Notifications">
          <Bell size={19} aria-hidden="true" />
        </button>
        <button className="icon-button" type="button" onClick={toggleDark} aria-label="Toggle theme">
          {darkMode ? <Sun size={18} aria-hidden="true" /> : <Moon size={18} aria-hidden="true" />}
        </button>
        <div className="admin-pill">
          <span>{hasSupabaseConfig ? 'Admin_91' : 'Preview'}</span>
          <UserCircle size={25} aria-hidden="true" />
        </div>
        <button className="icon-button" type="button" onClick={handleLogout} aria-label="Sign out">
          <LogOut size={18} aria-hidden="true" />
        </button>
      </div>
    </header>
  )
}
