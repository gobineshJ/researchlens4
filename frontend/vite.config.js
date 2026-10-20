import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // The app calls "/api/..." on its own origin; the dev server forwards
    // it to FastAPI. Same origin means the refresh-token cookie is
    // first-party and no CORS is involved (nginx does the same in Docker).
    proxy: {
      '/api': {
        target: process.env.API_TARGET || 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
