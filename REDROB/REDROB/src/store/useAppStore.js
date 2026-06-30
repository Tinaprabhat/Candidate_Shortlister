import { create } from 'zustand'

const defaultFilters = {
  score_min: 70,
  verified_only: true,
  risk_flagged: false,
  domain: 'all',
  page: 1,
  limit: 20
}

function readStoredBoolean(key, fallback) {
  const stored = localStorage.getItem(key)
  if (stored === null) return fallback
  return stored === 'true'
}

export const useAppStore = create((set) => ({
  flowStage: 'input',
  activePage: 'overview',
  darkMode: readStoredBoolean('redrob_dark_mode', false),
  authToken: localStorage.getItem('supabase_token'),
  user: null,
  uploadedFiles: {
    candidates: null,
    jobDescription: null
  },
  processingProgress: 0,
  activeCandidateId: null,
  drawerOpen: false,
  filters: defaultFilters,

  hydrate: () => {
    const darkMode = readStoredBoolean('redrob_dark_mode', false)
    const authToken = localStorage.getItem('supabase_token')
    const previewPage = new URLSearchParams(window.location.search).get('preview')
    if (previewPage === 'overview' || previewPage === 'health') {
      localStorage.setItem('redrob_dark_mode', 'true')
      document.documentElement.classList.add('dark')
      set({
        darkMode: true,
        authToken,
        flowStage: 'app',
        activePage: previewPage
      })
      return
    }
    document.documentElement.classList.toggle('dark', darkMode)
    set({ darkMode, authToken, flowStage: 'input' })
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
    set({ authToken: token, user, flowStage: 'app' })
  },

  logout: () => {
    localStorage.removeItem('supabase_token')
    set({
      flowStage: 'input',
      authToken: null,
      user: null,
      uploadedFiles: {
        candidates: null,
        jobDescription: null
      },
      processingProgress: 0,
      activePage: 'overview',
      drawerOpen: false,
      activeCandidateId: null
    })
  },

  setUploadedFiles: (uploadedFiles) => set({ uploadedFiles }),

  startProcessing: (uploadedFiles) =>
    set(() => {
      localStorage.setItem('redrob_dark_mode', 'true')
      document.documentElement.classList.add('dark')
      return {
        uploadedFiles,
        processingProgress: 0,
        flowStage: 'processing',
        darkMode: true,
        activePage: 'overview',
        drawerOpen: false
      }
    }),

  setProcessingProgress: (processingProgress) => set({ processingProgress }),

  completeProcessing: () =>
    set({
      flowStage: 'app',
      processingProgress: 100,
      activePage: 'overview'
    }),

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
