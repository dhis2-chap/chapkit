import { defineConfig, devices } from '@playwright/test'

// The console is served by a chapkit service, so the e2e fixture is a real
// chapkit ML service (frontend/e2e/fixture_service.py). Playwright boots it via
// `uv run` from the repo root and drives the committed/built console bundle.
const PORT = 9099
const baseURL = `http://127.0.0.1:${PORT}`

export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  workers: 1,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? 'github' : 'list',
  use: {
    baseURL,
    trace: 'on-first-retry',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: {
    command: `uv run uvicorn fixture_service:app --app-dir frontend/e2e --host 127.0.0.1 --port ${PORT} --log-level warning`,
    cwd: '..',
    url: `${baseURL}/health`,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
})
