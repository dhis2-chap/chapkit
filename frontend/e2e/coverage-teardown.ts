import { CoverageReport } from 'monocart-coverage-reports'
import { coverageOptions } from './coverage-options'

/** Remap accumulated V8 coverage to source and emit the reports. */
export default async function globalTeardown(): Promise<void> {
  if (process.env.COVERAGE) await new CoverageReport(coverageOptions).generate()
}
