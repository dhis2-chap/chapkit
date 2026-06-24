// Endpoint browser: lists every operation from the service's OpenAPI document.
import { useMemo, useState } from 'react'
import { ExternalLink, Search } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'
import type { OpenApiOperation } from '@/lib/types'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { ErrorState, Loading } from '@/components/console/common'
import { PageBody, PageHeader } from '@/components/console/page'
import { cn } from '@/lib/utils'

interface Endpoint {
  method: string
  path: string
  op: OpenApiOperation
}

const METHOD_COLORS: Record<string, string> = {
  get: 'bg-blue-500/15 text-blue-700 dark:text-blue-400',
  post: 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-400',
  put: 'bg-amber-500/15 text-amber-700 dark:text-amber-400',
  patch: 'bg-orange-500/15 text-orange-700 dark:text-orange-400',
  delete: 'bg-destructive/15 text-destructive',
}

const METHODS = new Set(['get', 'post', 'put', 'patch', 'delete'])

export function EndpointsPage() {
  const spec = useQuery({ queryKey: ['openapi'], queryFn: api.openapi })
  const [filter, setFilter] = useState('')

  const groups = useMemo(() => {
    if (!spec.data) return []
    const endpoints: Endpoint[] = []
    for (const [path, methods] of Object.entries(spec.data.paths)) {
      for (const [method, op] of Object.entries(methods)) {
        if (METHODS.has(method)) endpoints.push({ method, path, op })
      }
    }
    const q = filter.trim().toLowerCase()
    const filtered = q
      ? endpoints.filter(
          (e) =>
            e.path.toLowerCase().includes(q) ||
            e.method.includes(q) ||
            (e.op.summary ?? '').toLowerCase().includes(q) ||
            (e.op.tags ?? []).some((t) => t.toLowerCase().includes(q)),
        )
      : endpoints
    const byTag = new Map<string, Endpoint[]>()
    for (const e of filtered) {
      const tag = e.op.tags?.[0] ?? 'default'
      const list = byTag.get(tag) ?? []
      list.push(e)
      byTag.set(tag, list)
    }
    return [...byTag.entries()].sort((a, b) => a[0].localeCompare(b[0]))
  }, [spec.data, filter])

  return (
    <>
      <PageHeader
        title="Endpoints"
        description={
          spec.data
            ? `${spec.data.info.title} · OpenAPI ${spec.data.openapi}`
            : 'All HTTP operations exposed by this service'
        }
        actions={
          <Button variant="outline" size="sm" asChild>
            <a href="docs" target="_blank" rel="noreferrer">
              Open Swagger <ExternalLink className="size-3.5" />
            </a>
          </Button>
        }
      />
      <PageBody>
        {spec.isLoading ? (
          <Loading />
        ) : spec.error ? (
          <ErrorState error={spec.error} />
        ) : (
          <div className="space-y-6">
            <div className="relative max-w-sm">
              <Search className="absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                placeholder="Filter by path, method, tag…"
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                className="pl-8"
              />
            </div>

            {groups.map(([tag, endpoints]) => (
              <div key={tag} className="space-y-2">
                <h2 className="text-sm font-semibold text-muted-foreground">
                  {tag}
                </h2>
                <div className="divide-y rounded-md border">
                  {endpoints.map((e) => (
                    <div
                      key={`${e.method}-${e.path}`}
                      className="flex items-center gap-3 px-3 py-2 text-sm"
                    >
                      <Badge
                        variant="secondary"
                        className={cn(
                          'w-16 justify-center font-mono uppercase',
                          METHOD_COLORS[e.method] ?? '',
                        )}
                      >
                        {e.method}
                      </Badge>
                      <code className="shrink-0 font-mono text-xs">{e.path}</code>
                      <span className="truncate text-muted-foreground">
                        {e.op.summary ?? ''}
                      </span>
                      {e.op.deprecated ? (
                        <Badge variant="outline" className="ml-auto">
                          deprecated
                        </Badge>
                      ) : null}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </PageBody>
    </>
  )
}
