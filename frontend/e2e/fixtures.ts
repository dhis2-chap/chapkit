import { test as base, expect } from '@playwright/test'
import { CoverageReport } from 'monocart-coverage-reports'
import { coverageOptions } from './coverage-options'

const collect = !!process.env.COVERAGE

// Auto fixture: when COVERAGE is set, capture V8 JS coverage per test and feed it
// to monocart's cache; the global teardown remaps + reports it. No-op otherwise.
export const test = base.extend<{ _autoCoverage: void }>({
  _autoCoverage: [
    async ({ page, browserName }, use) => {
      const on = collect && browserName === 'chromium'
      if (on) await page.coverage.startJSCoverage({ resetOnNavigation: false })
      await use()
      if (on) {
        const entries = await page.coverage.stopJSCoverage()
        await new CoverageReport(coverageOptions).add(entries)
      }
    },
    { auto: true },
  ],
})

export { expect }
