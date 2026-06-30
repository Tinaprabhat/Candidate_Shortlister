import { useEffect } from 'react'
import AppShell from './components/layout/AppShell.jsx'
import InputPage from './pages/InputPage.jsx'
import ProcessingPage from './pages/ProcessingPage.jsx'
import { signOut } from './api/supabase.js'
import { useAppStore } from './store/useAppStore.js'

export default function App() {
  const flowStage = useAppStore((state) => state.flowStage)
  const darkMode = useAppStore((state) => state.darkMode)
  const hydrate = useAppStore((state) => state.hydrate)
  const logout = useAppStore((state) => state.logout)

  useEffect(() => {
    hydrate()
  }, [hydrate])

  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode)
  }, [darkMode])

  useEffect(() => {
    const onUnauthorized = () => {
      signOut()
      logout()
    }
    window.addEventListener('redrob:unauthorized', onUnauthorized)
    return () => window.removeEventListener('redrob:unauthorized', onUnauthorized)
  }, [logout])

  if (flowStage === 'input') {
    return <InputPage />
  }

  if (flowStage === 'processing') {
    return <ProcessingPage />
  }

  return <AppShell />
}
