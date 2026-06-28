import { useEffect } from 'react'
import AppShell from './components/layout/AppShell.jsx'
import LoginPage from './pages/LoginPage.jsx'
import { signOut } from './api/supabase.js'
import { useAppStore } from './store/useAppStore.js'

export default function App() {
  const authToken = useAppStore((state) => state.authToken)
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

  if (!authToken) {
    return <LoginPage />
  }

  return <AppShell />
}
