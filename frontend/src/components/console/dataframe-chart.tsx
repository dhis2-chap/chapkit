// Line chart for a chapkit dataframe: a numeric measure over time, one series
// per group (e.g. disease_cases per location).
import { useMemo, useState } from 'react'
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from 'recharts'
import type { DataFrameContent } from '@/lib/types'
import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
} from '@/components/ui/chart'
import type { ChartConfig } from '@/components/ui/chart'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { EmptyState } from '@/components/console/common'

// The shadcn Nova preset's --chart-* tokens are greyscale, so use an explicit
// vivid palette to keep series distinguishable.
const SERIES_COLORS = [
  'oklch(0.62 0.21 255)', // blue
  'oklch(0.7 0.17 145)', // green
  'oklch(0.72 0.18 60)', // amber
  'oklch(0.63 0.24 20)', // red
  'oklch(0.6 0.22 305)', // purple
  'oklch(0.7 0.13 195)', // teal
  'oklch(0.7 0.19 35)', // orange
  'oklch(0.6 0.2 280)', // indigo
  'oklch(0.72 0.18 110)', // lime
  'oklch(0.65 0.2 350)', // pink
  'oklch(0.6 0.1 230)', // slate-blue
  'oklch(0.7 0.16 165)', // emerald
]
const MAX_SERIES = 12

/** Pick a column index by preferred name, falling back to a predicate. */
function pickColumn(
  columns: string[],
  preferred: string[],
  predicate: (idx: number) => boolean,
): number {
  for (const name of preferred) {
    const i = columns.indexOf(name)
    if (i >= 0) return i
  }
  for (let i = 0; i < columns.length; i++) if (predicate(i)) return i
  return -1
}

function isNumericColumn(data: unknown[][], idx: number): boolean {
  return data.some((row) => typeof row[idx] === 'number')
}

export function DataFrameChart({ frame }: { frame: DataFrameContent }) {
  const { columns, data } = frame

  const xIdx = useMemo(
    () =>
      pickColumn(
        columns,
        ['time_period', 'period', 'time', 'date'],
        (i) => typeof data[0]?.[i] === 'string',
      ),
    [columns, data],
  )
  const groupIdx = useMemo(
    () => pickColumn(columns, ['location', 'region', 'org_unit', 'group'], () => false),
    [columns],
  )
  const numericColumns = useMemo(
    () => columns.map((c, i) => ({ name: c, idx: i })).filter(({ idx }) => isNumericColumn(data, idx)),
    [columns, data],
  )

  const [measure, setMeasure] = useState(
    () => numericColumns.find((c) => c.name === 'disease_cases')?.name ?? numericColumns[0]?.name ?? '',
  )

  const measureIdx = columns.indexOf(measure)

  const { rows, series } = useMemo(() => {
    if (xIdx < 0 || measureIdx < 0) return { rows: [], series: [] as string[] }
    const xOrder: string[] = []
    const seen = new Set<string>()
    for (const row of data) {
      const x = String(row[xIdx])
      if (!seen.has(x)) {
        seen.add(x)
        xOrder.push(x)
      }
    }
    if (groupIdx < 0) {
      const single = xOrder.map((x) => {
        const match = data.find((row) => String(row[xIdx]) === x)
        return { x, [measure]: match ? (match[measureIdx] as number) : null }
      })
      return { rows: single, series: [measure] }
    }
    const groups: string[] = []
    const groupSeen = new Set<string>()
    for (const row of data) {
      const g = String(row[groupIdx])
      if (!groupSeen.has(g)) {
        groupSeen.add(g)
        groups.push(g)
      }
    }
    const limited = groups.slice(0, MAX_SERIES)
    const pivoted = xOrder.map((x) => {
      const entry: Record<string, unknown> = { x }
      for (const g of limited) {
        const match = data.find(
          (row) => String(row[xIdx]) === x && String(row[groupIdx]) === g,
        )
        entry[g] = match ? (match[measureIdx] as number) : null
      }
      return entry
    })
    return { rows: pivoted, series: limited }
  }, [data, xIdx, groupIdx, measureIdx, measure])

  const config = useMemo<ChartConfig>(() => {
    const c: ChartConfig = {}
    series.forEach((s, i) => {
      c[s] = { label: s, color: SERIES_COLORS[i % SERIES_COLORS.length] }
    })
    return c
  }, [series])

  if (numericColumns.length === 0 || xIdx < 0) {
    return <EmptyState title="Not chartable" hint="No time axis or numeric column found." />
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <span className="text-xs text-muted-foreground">Measure</span>
        <Select value={measure} onValueChange={setMeasure}>
          <SelectTrigger className="h-7 w-[14rem]" size="sm">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {numericColumns.map((c) => (
              <SelectItem key={c.name} value={c.name}>
                {c.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <span className="text-xs text-muted-foreground">
          over {columns[xIdx]}
          {groupIdx >= 0 ? ` · by ${columns[groupIdx]}` : ''}
        </span>
      </div>

      <ChartContainer config={config} className="h-[24rem] w-full">
        <LineChart data={rows} margin={{ left: 8, right: 16, top: 8 }}>
          <CartesianGrid vertical={false} />
          <XAxis dataKey="x" tickLine={false} axisLine={false} minTickGap={24} />
          <YAxis tickLine={false} axisLine={false} width={48} />
          <ChartTooltip content={<ChartTooltipContent />} />
          <ChartLegend content={<ChartLegendContent />} />
          {series.map((s) => (
            <Line
              key={s}
              type="monotone"
              dataKey={s}
              stroke={`var(--color-${s})`}
              dot={false}
              strokeWidth={2}
              connectNulls
            />
          ))}
        </LineChart>
      </ChartContainer>
    </div>
  )
}
