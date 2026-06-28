import { create } from 'zustand'

const defaultFilters = {
  score_min: 70,
  verified_only: true,
  risk_flagged: false,
  domain: 'all',
  page: 1,
  limit: 100
}

function readStoredBoolean(key, fallback) {
  const stored = localStorage.getItem(key)
  if (stored === null) return fallback
  return stored === 'true'
}

export const useAppStore = create((set) => ({
  activePage: 'overview',
  darkMode: readStoredBoolean('redrob_dark_mode', false),
  authToken: localStorage.getItem('supabase_token'),
  user: null,
  activeCandidateId: null,
  drawerOpen: false,
  filters: defaultFilters,

  hydrate: () => {
    const darkMode = readStoredBoolean('redrob_dark_mode', false)
    const authToken = localStorage.getItem('supabase_token')
    document.documentElement.classList.toggle('dark', darkMode)
    set({ darkMode, authToken })
  },

  setActivePage: (activePage) => set({ activePage }),

  toggleDark: () =>
    set((state) => {
      const darkMode = !state.darkMode
      localStorage.setItem('redrob_dark_mode', String(darkMode))
      document.documentElement.classList.toggle('dark', darkMode)
      return { darkMode }
    }),

  login: ({ token, user }) => {
    localStorage.setItem('supabase_token', token)
    set({ authToken: token, user })
  },

  logout: () => {
    localStorage.removeItem('supabase_token')
    set({
      authToken: null,
      user: null,
      activePage: 'overview',
      drawerOpen: false,
      activeCandidateId: null
    })
  },

  setActiveCandidate: (activeCandidateId) =>
    set({ activeCandidateId, drawerOpen: true }),

  closeDrawer: () => set({ drawerOpen: false, activeCandidateId: null }),

  setFilters: (partial) =>
    set((state) => ({
      filters: {
        ...state.filters,
        ...partial,
        page: 1
      }
    })),

  resetFilters: () => set({ filters: defaultFilters })
}))
