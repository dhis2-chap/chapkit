import path from 'node:path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// The console is served by chapkit itself as a static app mounted at `/`. We
// build straight into the package so the assets ship in the wheel, use relative
// asset paths (base: './') so the bundle is location-independent, and proxy API
// calls to a running chapkit service during development.
const target = process.env.VITE_CHAPKIT_TARGET ?? 'http://localhost:8000'
const proxy = Object.fromEntries(
  ['/api', '/health', '/openapi.json', '/docs', '/redoc'].map((p) => [
    p,
    { target, changeOrigin: true },
  ]),
)

// https://vite.dev/config/
export default defineConfig({
  base: './',
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: { proxy },
  build: {
    outDir: path.resolve(__dirname, '../src/chapkit/api/apps/console'),
    emptyOutDir: true,
  },
})
