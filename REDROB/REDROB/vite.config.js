import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    chunkSizeWarningLimit: 900,
    rollupOptions: {
      output: {
        manualChunks: {
          react: ['react', 'react-dom'],
          three: ['three', '@react-three/fiber', '@react-three/drei', '@react-spring/three'],
          motion: ['framer-motion'],
          data: ['@tanstack/react-query', 'zustand', 'axios', '@supabase/supabase-js']
        }
      }
    }
  },
  server: {
    port: 5173,
    strictPort: false
  }
})
