// Monitoring screen: live, in-memory view of the service's Prometheus /metrics.
// No server-side storage — the browser polls and keeps a short rolling history.
import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Activity } from 'lucide-react'
import { Area, AreaChart, ResponsiveContainer, Tooltip, YAxis } from 'recharts'

import { api } from '@/lib/api'
import { findMetricsPath } from '@/lib/monitoring'
import { useMetrics } from '@/hooks/use-metrics'
import type { MetricsSnapshot } from '@/hooks/use-metrics'
import { EmptyState } from '@/components/console/common'
import { PageBody, PageHeader } from '@/components/console/page'
import { Badge } from '@/components/ui/badge'
import { Card } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'

const INTERVALS: { label: string; value: string; ms: number | null }[] = [
  { label: 'Paused', value: 'paused', ms: null },
  { label: 'Every 2s', value: '2', ms: 2000 },
  { label: 'Every 5s', value: '5', ms: 5000 },
  { label: 'Every 10s', value: '10', ms: 10000 },
]

type Point = { t: number; v: number }

/** Format a byte count compactly. */
function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  const units = ['KB', 'MB', 'GB', 'TB']
  let value = bytes / 1024
  let unit = 0
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024
    unit++
  }
  return `${value.toFixed(1)} ${units[unit]}`
}

/** Compact number formatting. */
function formatNumber(value: number, digits = 1): string {
  return value.toLocaleString(undefined, { maximumFractionDigits: digits })
}

/** Human uptime from seconds. */
function formatUptime(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds))
  const d = Math.floor(s / 86400)
  const h = Math.floor((s % 86400) / 3600)
  const m = Math.floor((s % 3600) / 60)
  if (d > 0) return `${d}d ${h}h`
  if (h > 0) return `${h}h ${m}m`
  return `${m}m ${s % 60}s`
}

/** Hover tooltip for a sparkline point: the formatted value and its timestamp. */
function SparkTooltip({
  active,
  payload,
  format,
  unit,
}: {
  active?: boolean
  payload?: readonly { value?: unknown; payload?: Point }[]
  format: (value: number) => string
  unit?: string
}) {
  const point = payload?.[0]
  if (!active || !point || typeof point.value !== 'number' || !point.payload) return null
  return (
    <div className="rounded-md border bg-popover px-2 py-1 text-xs shadow-sm">
      <div className="font-medium tabular-nums">
        {format(point.value)}
        {unit ? ` ${unit}` : ''}
      </div>
      <div className="tabular-nums text-muted-foreground">
        {new Date(point.payload.t).toLocaleTimeString()}
      </div>
    </div>
  )
}

/** Tiny sparkline over a rolling series, with a value tooltip on hover. */
function Sparkline({
  data,
  format,
  unit,
}: {
  data: Point[]
  format: (value: number) => string
  unit?: string
}) {
  if (data.length < 2) return <div className="h-10" />
  return (
    <ResponsiveContainer width="100%" height={40}>
      <AreaChart data={data} margin={{ top: 2, right: 0, bottom: 0, left: 0 }}>
        <YAxis hide domain={['dataMin', 'dataMax']} />
        <Tooltip
          isAnimationActive={false}
          cursor={{ stroke: 'var(--border)' }}
          wrapperStyle={{ outline: 'none', zIndex: 50 }}
          content={({ active, payload }) => (
            <SparkTooltip active={active} payload={payload} format={format} unit={unit} />
          )}
        />
        <Area
          type="monotone"
          dataKey="v"
          stroke="var(--primary)"
          fill="var(--primary)"
          fillOpacity={0.15}
          strokeWidth={1.5}
          isAnimationActive={false}
          dot={false}
          activeDot={{ r: 3 }}
        />
      </AreaChart>
    </ResponsiveContainer>
  )
}

interface Tile {
  key: string
  label: string
  value: number | null
  unit?: string
  format: (value: number) => string
  series: Point[]
  note?: string
}

/** A single metric tile: current value plus a sparkline. */
function MetricTile({ tile }: { tile: Tile }) {
  return (
    <Card className="space-y-2 p-4">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-medium text-muted-foreground">{tile.label}</span>
        {tile.value == null && tile.note ? (
          <Badge variant="outline" className="text-[0.65rem]">
            {tile.note}
          </Badge>
        ) : null}
      </div>
      <div className="text-2xl font-semibold tabular-nums">
        {tile.value == null ? '—' : tile.format(tile.value)}
        {tile.value != null && tile.unit ? (
          <span className="ml-1 text-sm font-normal text-muted-foreground">{tile.unit}</span>
        ) : null}
      </div>
      <Sparkline data={tile.series} format={tile.format} unit={tile.unit} />
    </Card>
  )
}

/** Build the derived tiles from the rolling snapshot buffer. */
function buildTiles(snapshots: MetricsSnapshot[]): Tile[] {
  const latest = snapshots.at(-1)
  const prev = snapshots.at(-2)

  const gaugeNow = (name: string) => latest?.values.get(name) ?? null
  const gaugeSeries = (name: string): Point[] =>
    snapshots.flatMap((s) => {
      const v = s.values.get(name)
      return v == null ? [] : [{ t: s.t, v }]
    })
  const rateBetween = (a: MetricsSnapshot, b: MetricsSnapshot, name: string): number | null => {
    const va = a.values.get(name)
    const vb = b.values.get(name)
    const dt = (b.t - a.t) / 1000
    if (va == null || vb == null || dt <= 0) return null
    return Math.max(0, (vb - va) / dt)
  }
  const rateNow = (name: string) => (latest && prev ? rateBetween(prev, latest, name) : null)
  const rateSeries = (name: string): Point[] => {
    const out: Point[] = []
    for (let i = 1; i < snapshots.length; i++) {
      const r = rateBetween(snapshots[i - 1], snapshots[i], name)
      if (r != null) out.push({ t: snapshots[i].t, v: r })
    }
    return out
  }
  const avgLatency = (): number | null => {
    if (!latest || !prev) return null
    const dc =
      (latest.values.get('http_server_duration_milliseconds_count') ?? 0) -
      (prev.values.get('http_server_duration_milliseconds_count') ?? 0)
    const ds =
      (latest.values.get('http_server_duration_milliseconds_sum') ?? 0) -
      (prev.values.get('http_server_duration_milliseconds_sum') ?? 0)
    return dc > 0 ? ds / dc : null
  }

  return [
    {
      key: 'requests',
      label: 'Requests / sec',
      value: rateNow('http_server_duration_milliseconds_count'),
      format: (v) => formatNumber(v, 2),
      series: rateSeries('http_server_duration_milliseconds_count'),
    },
    {
      key: 'latency',
      label: 'Avg latency',
      value: avgLatency(),
      unit: 'ms',
      format: (v) => formatNumber(v, 1),
      series: [],
    },
    {
      key: 'active',
      label: 'Active requests',
      value: gaugeNow('http_server_active_requests'),
      format: (v) => formatNumber(v, 0),
      series: gaugeSeries('http_server_active_requests'),
    },
    {
      key: 'cpu',
      label: 'CPU',
      value: (() => {
        const r = rateNow('process_cpu_seconds_total')
        return r == null ? null : r * 100
      })(),
      unit: '%',
      format: (v) => formatNumber(v, 1),
      series: rateSeries('process_cpu_seconds_total').map((p) => ({ t: p.t, v: p.v * 100 })),
      note: 'Linux only',
    },
    {
      key: 'memory',
      label: 'Memory (RSS)',
      value: gaugeNow('process_resident_memory_bytes'),
      format: formatBytes,
      series: gaugeSeries('process_resident_memory_bytes'),
      note: 'Linux only',
    },
    {
      key: 'fds',
      label: 'Open file descriptors',
      value: gaugeNow('process_open_fds'),
      format: (v) => formatNumber(v, 0),
      series: gaugeSeries('process_open_fds'),
      note: 'Linux only',
    },
    {
      key: 'gc',
      label: 'GC collections / sec',
      value: rateNow('python_gc_collections_total'),
      format: (v) => formatNumber(v, 2),
      series: rateSeries('python_gc_collections_total'),
    },
    {
      key: 'train',
      label: 'Train jobs (total)',
      value: gaugeNow('ml_train_jobs_total'),
      format: (v) => formatNumber(v, 0),
      series: gaugeSeries('ml_train_jobs_total'),
      note: 'after first train',
    },
    {
      key: 'predict',
      label: 'Predict jobs (total)',
      value: gaugeNow('ml_predict_jobs_total'),
      format: (v) => formatNumber(v, 0),
      series: gaugeSeries('ml_predict_jobs_total'),
      note: 'after first predict',
    },
  ]
}

/** Monitoring screen. */
export function MonitoringPage() {
  const openapi = useQuery({ queryKey: ['openapi'], queryFn: api.openapi })
  const metricsPath = findMetricsPath(openapi.data)
  const [intervalValue, setIntervalValue] = useState('5')
  const interval = INTERVALS.find((option) => option.value === intervalValue) ?? INTERVALS[2]

  const { snapshots, status, error } = useMetrics(metricsPath, interval.ms)

  const tiles = useMemo(() => buildTiles(snapshots), [snapshots])
  const latest = snapshots.at(-1)
  const uptime = latest?.values.get('process_start_time_seconds')
  const families = useMemo(() => {
    if (!latest) return []
    return Object.entries(latest.meta)
      .map(([name, meta]) => {
        // For "_info" metrics the value is always 1; the payload is in the labels.
        const labeled = latest.samples.find(
          (sample) => sample.name === name && Object.keys(sample.labels).length > 0,
        )
        return {
          name,
          type: meta.type,
          help: meta.help,
          value: latest.values.get(name) ?? null,
          labels: name.endsWith('_info') ? (labeled?.labels ?? null) : null,
        }
      })
      .sort((a, b) => a.name.localeCompare(b.name))
  }, [latest])
  const [filter, setFilter] = useState('')
  const visibleFamilies = families.filter((f) => f.name.includes(filter.trim()))

  const lastUpdated = latest ? new Date(latest.t).toLocaleTimeString() : '—'

  const actions = (
    <div className="flex items-center gap-3">
      <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
        <span
          className={`size-2 rounded-full ${
            status === 'live' ? 'bg-emerald-500' : status === 'error' ? 'bg-destructive' : 'bg-muted-foreground'
          }`}
        />
        {status === 'error' ? 'error' : `updated ${lastUpdated}`}
      </span>
      <Select value={intervalValue} onValueChange={setIntervalValue}>
        <SelectTrigger size="sm" className="w-32">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {INTERVALS.map((option) => (
            <SelectItem key={option.value} value={option.value}>
              {option.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )

  return (
    <>
      <PageHeader
        title="Monitoring"
        description="Live, in-memory metrics polled from the service's Prometheus endpoint."
        actions={metricsPath ? actions : undefined}
      />
      {!openapi.isLoading && !metricsPath ? (
        <PageBody>
          <EmptyState
            title="Monitoring is not enabled"
            hint="Add .with_monitoring() to your service builder to expose a /metrics endpoint."
          />
        </PageBody>
      ) : (
        <div className="flex min-h-0 flex-1 flex-col gap-4 p-6">
          {uptime != null || error ? (
            <div className="flex shrink-0 flex-wrap items-center gap-x-6 gap-y-1 text-sm">
              {uptime != null ? (
                <span>
                  <span className="text-muted-foreground">Uptime</span>{' '}
                  <span className="font-medium tabular-nums">
                    {formatUptime(Date.now() / 1000 - uptime)}
                  </span>
                </span>
              ) : null}
              {error ? <span className="text-destructive">{error}</span> : null}
            </div>
          ) : null}

          <div className="grid shrink-0 grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
            {tiles.map((tile) => (
              <MetricTile key={tile.key} tile={tile} />
            ))}
          </div>

          <Card className="flex min-h-0 flex-1 flex-col overflow-hidden p-0">
            <div className="flex shrink-0 items-center justify-between gap-3 border-b px-4 py-3">
              <div className="flex items-center gap-2">
                <Activity className="size-4 text-muted-foreground" />
                <span className="text-sm font-medium">All metrics</span>
                <span className="text-xs text-muted-foreground">{families.length}</span>
              </div>
              <Input
                value={filter}
                onChange={(event) => setFilter(event.target.value)}
                placeholder="Filter…"
                className="h-7 w-48"
              />
            </div>
            <div className="min-h-0 flex-1 overflow-auto">
              <table className="w-full text-sm">
                <tbody>
                  {visibleFamilies.map((family) => (
                    <tr key={family.name} className="border-b align-top last:border-0">
                      <td className="px-4 py-1.5">
                        <div className="font-mono text-xs">{family.name}</div>
                        {family.help ? (
                          <div className="text-[0.7rem] text-muted-foreground">{family.help}</div>
                        ) : null}
                        {family.labels ? (
                          <div className="text-[0.7rem] text-muted-foreground">
                            {Object.entries(family.labels)
                              .map(([key, value]) => `${key}=${value}`)
                              .join(' · ')}
                          </div>
                        ) : null}
                      </td>
                      <td className="px-2 py-1.5 text-xs whitespace-nowrap text-muted-foreground">
                        {family.type}
                      </td>
                      <td className="px-4 py-1.5 text-right font-mono text-xs tabular-nums whitespace-nowrap">
                        {family.value == null ? '—' : formatNumber(family.value, 2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}
    </>
  )
}
