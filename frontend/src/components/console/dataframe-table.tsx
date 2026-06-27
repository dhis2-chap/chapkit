// Queryable table for a chapkit DataFrame payload: search, sort, and paginate.
import { useMemo, useState } from 'react'
import { ArrowDown, ArrowUp, ChevronsUpDown, Download, Search } from 'lucide-react'
import type { DataFrameContent } from '@/lib/types'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { cn } from '@/lib/utils'

type SortDir = 'asc' | 'desc' | null
const PAGE_SIZES = [25, 50, 100, 250]

/** Render a cell value compactly; round long floats for readability. */
function formatCell(value: unknown): string {
  if (value === null || value === undefined) return '—'
  if (typeof value === 'number') {
    return Number.isInteger(value) ? String(value) : value.toFixed(4).replace(/0+$/, '').replace(/\.$/, '')
  }
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

function compare(a: unknown, b: unknown): number {
  const an = typeof a === 'number'
  const bn = typeof b === 'number'
  if (an && bn) return (a as number) - (b as number)
  if (a === null || a === undefined) return -1
  if (b === null || b === undefined) return 1
  return String(a).localeCompare(String(b), undefined, { numeric: true })
}

type FieldType = 'integer' | 'number' | 'boolean' | 'string' | 'any'

/** Infer a Table-Schema-style type for a column from its non-null values. */
function inferFieldType(values: unknown[]): FieldType {
  const present = values.filter((v) => v !== null && v !== undefined)
  if (present.length === 0) return 'any'
  if (present.every((v) => typeof v === 'boolean')) return 'boolean'
  if (present.every((v) => typeof v === 'number')) {
    return present.every((v) => Number.isInteger(v as number)) ? 'integer' : 'number'
  }
  return 'string'
}

// Self-describing wrapper for a chapkit dataframe export: a $schema URN aware
// consumers can recognize, a `type` discriminator, and per-column inferred types.
// columns/data stay at the top level so lenient parsers still read the frame.
// URN form matches the convention used elsewhere (e.g. urn:servicekit:error:*).
const DATAFRAME_SCHEMA_ID = 'urn:chapkit:dataframe:1'
const DATAFRAME_TYPE = 'chapkit.dataframe'

export function DataFrameTable({ frame, fill = false }: { frame: DataFrameContent; fill?: boolean }) {
  const { columns, data } = frame
  const [query, setQuery] = useState('')
  const [sortCol, setSortCol] = useState<number | null>(null)
  const [sortDir, setSortDir] = useState<SortDir>(null)
  const [pageSize, setPageSize] = useState(50)
  const [page, setPage] = useState(0)

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return data
    return data.filter((row) => row.some((cell) => formatCell(cell).toLowerCase().includes(q)))
  }, [data, query])

  const sorted = useMemo(() => {
    if (sortCol === null || sortDir === null) return filtered
    const factor = sortDir === 'asc' ? 1 : -1
    return [...filtered].sort((ra, rb) => factor * compare(ra[sortCol], rb[sortCol]))
  }, [filtered, sortCol, sortDir])

  const pageCount = Math.max(1, Math.ceil(sorted.length / pageSize))
  const current = Math.min(page, pageCount - 1)
  const rows = sorted.slice(current * pageSize, current * pageSize + pageSize)

  function toggleSort(col: number) {
    if (sortCol !== col) {
      setSortCol(col)
      setSortDir('asc')
    } else if (sortDir === 'asc') {
      setSortDir('desc')
    } else {
      setSortCol(null)
      setSortDir(null)
    }
    setPage(0)
  }

  function triggerDownload(content: string, type: string, filename: string) {
    const url = URL.createObjectURL(new Blob([content], { type }))
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    a.click()
    URL.revokeObjectURL(url)
  }

  // Exports respect the current sort/filter so you download exactly what you see.
  function downloadCsv() {
    const escape = (v: unknown) => {
      const s = formatCell(v)
      return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
    }
    const csv = [columns.join(','), ...sorted.map((r) => r.map(escape).join(','))].join('\n')
    triggerDownload(csv, 'text/csv', 'dataframe.csv')
  }

  function downloadJson() {
    const json = JSON.stringify({ columns, data: sorted }, null, 2)
    triggerDownload(json, 'application/json', 'dataframe.json')
  }

  function downloadJsonWithSchema() {
    const fields = columns.map((name, i) => ({
      name,
      type: inferFieldType(sorted.map((row) => row[i])),
    }))
    const payload = {
      $schema: DATAFRAME_SCHEMA_ID,
      type: DATAFRAME_TYPE,
      schema: { fields },
      columns,
      data: sorted,
    }
    triggerDownload(
      JSON.stringify(payload, null, 2),
      'application/json',
      'dataframe.schema.json',
    )
  }

  return (
    <div className={cn('space-y-3', fill && 'flex h-full min-h-0 flex-col gap-3 space-y-0')}>
      <div className="flex flex-wrap items-center gap-2">
        <div className="relative min-w-[14rem] flex-1">
          <Search className="absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Filter rows…"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value)
              setPage(0)
            }}
            className="pl-8"
          />
        </div>
        <span className="text-xs text-muted-foreground">
          {sorted.length === data.length
            ? `${data.length} rows`
            : `${sorted.length} of ${data.length} rows`}{' '}
          · {columns.length} cols
        </span>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="outline" size="sm">
              <Download className="size-3.5" /> Download
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem onClick={downloadCsv}>CSV</DropdownMenuItem>
            <DropdownMenuItem onClick={downloadJson}>
              JSON (chapkit dataframe)
            </DropdownMenuItem>
            <DropdownMenuItem onClick={downloadJsonWithSchema}>
              JSON + $schema
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {/* Single scroll container (both axes) so the sticky header sticks to the
          visible top and one always-visible scrollbar indicates h-scroll. Using a
          raw <table> avoids shadcn Table's inner overflow-x wrapper, which would
          otherwise become the sticky scroll context and hide the h-scrollbar. */}
      <div
        className={cn(
          'show-scrollbars overflow-auto rounded-md border',
          fill ? 'min-h-0 flex-1' : 'max-h-[32rem]',
        )}
      >
        <table className="w-full caption-bottom text-sm">
          <TableHeader className="sticky top-0 z-10 bg-muted">
            <TableRow>
              {columns.map((col, i) => (
                <TableHead
                  key={col}
                  onClick={() => toggleSort(i)}
                  className="cursor-pointer whitespace-nowrap select-none hover:text-foreground"
                >
                  <span className="inline-flex items-center gap-1">
                    {col}
                    {sortCol === i ? (
                      sortDir === 'asc' ? (
                        <ArrowUp className="size-3" />
                      ) : (
                        <ArrowDown className="size-3" />
                      )
                    ) : (
                      <ChevronsUpDown className="size-3 opacity-30" />
                    )}
                  </span>
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((row, ri) => (
              <TableRow key={current * pageSize + ri}>
                {row.map((cell, ci) => (
                  <TableCell
                    key={ci}
                    className={cn(
                      'whitespace-nowrap font-mono text-xs',
                      typeof cell === 'number' && 'text-right tabular-nums',
                    )}
                  >
                    {formatCell(cell)}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </table>
      </div>

      <div className="flex shrink-0 flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
        <div className="flex items-center gap-2">
          <span>Rows per page</span>
          <Select
            value={String(pageSize)}
            onValueChange={(v) => {
              setPageSize(Number(v))
              setPage(0)
            }}
          >
            <SelectTrigger className="h-7 w-[5rem]" size="sm">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {PAGE_SIZES.map((s) => (
                <SelectItem key={s} value={String(s)}>
                  {s}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="flex items-center gap-2">
          <span>
            Page {current + 1} of {pageCount}
          </span>
          <Button
            variant="outline"
            size="sm"
            disabled={current === 0}
            onClick={() => setPage(current - 1)}
          >
            Prev
          </Button>
          <Button
            variant="outline"
            size="sm"
            disabled={current >= pageCount - 1}
            onClick={() => setPage(current + 1)}
          >
            Next
          </Button>
        </div>
      </div>
    </div>
  )
}
