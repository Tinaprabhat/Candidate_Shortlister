import React from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ParticlesProvider } from '@tsparticles/react'
import { loadSlim } from '@tsparticles/slim'
import App from './App.jsx'
import './styles.css'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 30_000
    }
  }
})

const initParticles = async (engine) => {
  await loadSlim(engine)
}

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ParticlesProvider init={initParticles}>
      <QueryClientProvider client={queryClient}>
        <App />
      </QueryClientProvider>
    </ParticlesProvider>
  </React.StrictMode>
)
