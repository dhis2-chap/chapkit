// Poll the /metrics endpoint on an interval and keep a short rolling history in
// memory (no server-side storage). Counters become rates via consecutive deltas.
import { useEffect, useState } from 'react'

import { fetchMetricsText } from '@/lib/monitoring'
import { parsePrometheus } from '@/lib/prometheus'
import type { FamilyMeta, MetricSample } from '@/lib/prometheus'

const MAX_SNAPSHOTS = 120

export interface MetricsSnapshot {
  t: number
  values: Map<string, number> // metric name -> value summed across label sets
  meta: Record<string, FamilyMeta>
  samples: MetricSample[] // raw samples (labels preserved) for the explorer
}

export type MetricsStatus = 'idle' | 'live' | 'error'

/** Poll metrics at `intervalMs` (null = paused) and accumulate a rolling buffer. */
export function useMetrics(
  metricsPath: string | null,
  intervalMs: number | null,
): { snapshots: MetricsSnapshot[]; status: MetricsStatus; error: string | null } {
  const [snapshots, setSnapshots] = useState<MetricsSnapshot[]>([])
  const [status, setStatus] = useState<MetricsStatus>('idle')
  const [error, setError] = useState<string | null>(null)

  // Reset the history only when the metrics source changes (not on pause/resume).
  useEffect(() => {
    setSnapshots([])
    setStatus('idle')
  }, [metricsPath])

  useEffect(() => {
    if (!metricsPath || intervalMs == null) return
    let cancelled = false
    let timer = 0

    const tick = async () => {
      // Skip work while the tab is hidden, but keep the loop alive.
      if (typeof document !== 'undefined' && document.hidden) {
        schedule()
        return
      }
      try {
        const text = await fetchMetricsText(metricsPath)
        if (cancelled) return
        const parsed = parsePrometheus(text)
        const values = new Map<string, number>()
        for (const sample of parsed.samples) {
          values.set(sample.name, (values.get(sample.name) ?? 0) + sample.value)
        }
        setSnapshots((prev) =>
          [...prev, { t: Date.now(), values, meta: parsed.meta, samples: parsed.samples }].slice(
            -MAX_SNAPSHOTS,
          ),
        )
        setStatus('live')
        setError(null)
      } catch (err) {
        if (!cancelled) {
          setStatus('error')
          setError(err instanceof Error ? err.message : String(err))
        }
      }
      schedule()
    }

    const schedule = () => {
      if (!cancelled) timer = window.setTimeout(tick, intervalMs)
    }

    void tick()
    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [metricsPath, intervalMs])

  return { snapshots, status, error }
}
