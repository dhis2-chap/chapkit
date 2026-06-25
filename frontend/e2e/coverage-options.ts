// Shared monocart-coverage-reports options. V8 coverage is captured on the built
// bundle and remapped to source via the build's source maps; we keep only our own
// src/ files in the report.
export const coverageOptions = {
  name: 'Chapkit Console Coverage',
  outputDir: './coverage',
  reports: ['v8', 'console-summary', 'lcovonly'],
  sourceFilter: (sourcePath: string) =>
    sourcePath.includes('src/') && !sourcePath.includes('node_modules'),
}
