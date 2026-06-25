import { CoverageReport } from 'monocart-coverage-reports'
import { coverageOptions } from './coverage-options'

/** Clear any stale coverage cache before a coverage run. */
export default async function globalSetup(): Promise<void> {
  if (process.env.COVERAGE) await new CoverageReport(coverageOptions).cleanCache()
}
