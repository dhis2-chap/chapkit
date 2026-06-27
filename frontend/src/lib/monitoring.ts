// Discover whether the service exposes Prometheus monitoring, from the OpenAPI
// spec the console already loads. (A future servicekit `/system` capability flag
// could replace this — see WEB_CONSOLE_ROADMAP.md.)
import type { OpenApiSpec } from '@/lib/types'

/** The metrics endpoint path if the service exposes one, else null. */
export function findMetricsPath(spec: OpenApiSpec | undefined): string | null {
  if (!spec?.paths) return null
  for (const [path, operations] of Object.entries(spec.paths)) {
    const get = operations.get
    if (!get) continue
    if (path === '/metrics' || path.endsWith('/metrics')) return path
    if (get.tags?.includes('Observability') && /metric|prometheus/i.test(get.summary ?? '')) {
      return path
    }
  }
  return null
}

/** Fetch the raw Prometheus exposition text from the metrics endpoint. */
export async function fetchMetricsText(path: string): Promise<string> {
  const response = await fetch(path, { headers: { Accept: 'text/plain' } })
  if (!response.ok) throw new Error(`Metrics endpoint returned HTTP ${response.status}`)
  return response.text()
}
