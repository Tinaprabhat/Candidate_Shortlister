import Sidebar from './Sidebar.jsx'
import TopBar from './TopBar.jsx'
import PageWrapper from './PageWrapper.jsx'
import Overview from '../../pages/Overview.jsx'
import RankedCandidates from '../../pages/RankedCandidates.jsx'
import CandidateDetail from '../../pages/CandidateDetail.jsx'
import SystemHealth from '../../pages/SystemHealth.jsx'
import { useAppStore } from '../../store/useAppStore.js'

const pages = {
  overview: <Overview />,
  candidates: <RankedCandidates />,
  detail: <CandidateDetail />,
  health: <SystemHealth />
}

export default function AppShell() {
  const activePage = useAppStore((state) => state.activePage)

  return (
    <div className="app-shell">
      <Sidebar />
      <TopBar />
      <main className="app-main">
        <PageWrapper pageKey={activePage}>{pages[activePage]}</PageWrapper>
      </main>
    </div>
  )
}
