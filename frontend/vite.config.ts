import fs from 'node:fs'
import path from 'node:path'
import { defineConfig } from 'vite'
import type { Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// The console is served by chapkit itself as a static app mounted at `/`. We
// build straight into the package so the assets ship in the wheel, use relative
// asset paths (base: './') so the bundle is location-independent, and proxy API
// calls to a running chapkit service during development.
const target = process.env.VITE_CHAPKIT_TARGET ?? 'http://localhost:8000'
const proxy = Object.fromEntries(
  ['/api', '/health', '/openapi.json', '/docs', '/redoc', '/metrics'].map((p) => [
    p,
    { target, changeOrigin: true },
  ]),
)

const outDir = path.resolve(__dirname, '../src/chapkit/api/apps/console')

// The console ships inside the chapkit wheel, so its manifest version is the chapkit
// version: read it from pyproject.toml at build time and stamp it into the copied
// public/manifest.json (which carries a 0.0.0 placeholder).
function manifestVersion(): Plugin {
  return {
    name: 'chapkit-manifest-version',
    apply: 'build',
    closeBundle() {
      const pyproject = fs.readFileSync(path.resolve(__dirname, '../pyproject.toml'), 'utf8')
      const version = /^version = "([^"]+)"/m.exec(pyproject)?.[1]
      if (!version) throw new Error('chapkit version not found in ../pyproject.toml')
      const manifestPath = path.join(outDir, 'manifest.json')
      const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'))
      manifest.version = version
      fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + '\n')
    },
  }
}

// https://vite.dev/config/
export default defineConfig({
  base: './',
  plugins: [react(), tailwindcss(), manifestVersion()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: { proxy },
  build: {
    outDir,
    emptyOutDir: true,
    // Emit source maps only for coverage runs so V8 coverage can be remapped to
    // source; the shipped (committed) bundle stays map-free.
    sourcemap: !!process.env.COVERAGE,
  },
})
