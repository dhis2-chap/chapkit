// Two-pane artifact browser: hierarchy tree on the left, selected artifact detail on the right.
import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import type { ReactNode } from 'react'
import {
  ChevronRight,
  Download,
  FileBox,
  RefreshCw,
  Trash2,
} from 'lucide-react'
import {
  useMutation,
  useQuery,
  useQueryClient,
} from '@tanstack/react-query'

import { api } from '@/lib/api'
import type { Artifact, DataFrameContent } from '@/lib/types'
import {
  formatBytes,
  formatDateTime,
  formatDuration,
} from '@/lib/format'
import {
  EmptyState,
  ErrorState,
  JsonView,
  Loading,
} from '@/components/console/common'
import { PageBody, PageHeader } from '@/components/console/page'
import { toast } from 'sonner'

import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { DataFrameTable } from '@/components/console/dataframe-table'
import { DataFrameChart } from '@/components/console/dataframe-chart'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog'
import { Separator } from '@/components/ui/separator'
import { ScrollArea } from '@/components/ui/scroll-area'

/** Short trailing id fragment for compact labels. */
function shortId(id: string): string {
  return id.length > 8 ? id.slice(-8) : id
}

/** Human label for an artifact node. */
function nodeLabel(node: Artifact): string {
  return node.level_label ?? node.data.type ?? 'artifact'
}

/** Type guard: detect a DataFrame-shaped content payload. */
function isDataFrame(value: unknown): value is DataFrameContent {
  if (typeof value !== 'object' || value === null) return false
  const candidate = value as Record<string, unknown>
  return (
    Array.isArray(candidate.columns) &&
    candidate.columns.every((column) => typeof column === 'string') &&
    Array.isArray(candidate.data)
  )
}

/** Type guard: detect a binary/placeholder content object. */
function isBinaryPlaceholder(value: unknown): boolean {
  if (typeof value !== 'object' || value === null) return false
  const candidate = value as Record<string, unknown>
  return candidate._type === 'bytes' || '_serialization_error' in candidate
}

interface TreeNode extends Artifact {
  childNodes: TreeNode[]
}

/** Build the artifact forest from a flat list using parent_id. */
function buildForest(artifacts: Artifact[]): TreeNode[] {
  const byId = new Map<string, TreeNode>()
  for (const artifact of artifacts) {
    byId.set(artifact.id, { ...artifact, childNodes: [] })
  }
  const roots: TreeNode[] = []
  for (const node of byId.values()) {
    const parent = node.parent_id ? byId.get(node.parent_id) : undefined
    if (parent) {
      parent.childNodes.push(node)
    } else {
      roots.push(node)
    }
  }
  return roots
}

function TreeRow({
  node,
  depth,
  expanded,
  onToggle,
  selectedId,
  onSelect,
}: {
  node: TreeNode
  depth: number
  expanded: Set<string>
  onToggle: (id: string) => void
  selectedId: string | null
  onSelect: (id: string) => void
}) {
  const hasChildren = node.childNodes.length > 0
  const isExpanded = expanded.has(node.id)
  const isSelected = selectedId === node.id

  return (
    <div>
      <div
        role="button"
        tabIndex={0}
        onClick={() => onSelect(node.id)}
        onKeyDown={(event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault()
            onSelect(node.id)
          }
        }}
        className={`flex cursor-pointer items-center gap-1 rounded-md px-2 py-1.5 text-sm hover:bg-muted ${
          isSelected ? 'bg-muted font-medium' : ''
        }`}
        style={{ paddingLeft: `${depth * 14 + 8}px` }}
      >
        <button
          type="button"
          aria-label={isExpanded ? 'Collapse' : 'Expand'}
          onClick={(event) => {
            event.stopPropagation()
            if (hasChildren) onToggle(node.id)
          }}
          className={`flex size-4 shrink-0 items-center justify-center ${
            hasChildren ? '' : 'invisible'
          }`}
        >
          <ChevronRight
            className={`size-3.5 transition-transform ${
              isExpanded ? 'rotate-90' : ''
            }`}
          />
        </button>
        <FileBox className="size-3.5 shrink-0 text-muted-foreground" />
        <span className="truncate">{nodeLabel(node)}</span>
        <span className="ml-auto pl-2 font-mono text-xs text-muted-foreground">
          {shortId(node.id)}
        </span>
      </div>
      {hasChildren && isExpanded ? (
        <div>
          {node.childNodes.map((child) => (
            <TreeRow
              key={child.id}
              node={child}
              depth={depth + 1}
              expanded={expanded}
              onToggle={onToggle}
              selectedId={selectedId}
              onSelect={onSelect}
            />
          ))}
        </div>
      ) : null}
    </div>
  )
}

function ContentTab({ artifact }: { artifact: Artifact }) {
  const { content, content_type, content_size } = artifact.data

  let body: ReactNode
  if (content === null || content === undefined) {
    body = <p className="text-sm text-muted-foreground">No content.</p>
  } else if (isDataFrame(content)) {
    body = (
      <Tabs defaultValue="table">
        <TabsList>
          <TabsTrigger value="table">Table</TabsTrigger>
          <TabsTrigger value="chart">Chart</TabsTrigger>
        </TabsList>
        <TabsContent value="table" className="pt-3">
          <DataFrameTable frame={content} />
        </TabsContent>
        <TabsContent value="chart" className="pt-3">
          <DataFrameChart frame={content} />
        </TabsContent>
      </Tabs>
    )
  } else if (isBinaryPlaceholder(content)) {
    body = (
      <div className="space-y-3">
        <p className="text-sm text-muted-foreground">
          Binary content is not displayed inline. Download it to inspect.
        </p>
        <Button asChild variant="outline" size="sm">
          <a href={api.artifactDownloadUrl(artifact.id)} download>
            <Download className="size-4" />
            Download
          </a>
        </Button>
      </div>
    )
  } else {
    body = <JsonView value={content} />
  }

  return (
    <div className="space-y-3">
      {content_type || content_size != null ? (
        <p className="text-xs text-muted-foreground">
          {content_type ? <span>{content_type}</span> : null}
          {content_type && content_size != null ? <span> · </span> : null}
          {content_size != null ? (
            <span>{formatBytes(content_size)}</span>
          ) : null}
        </p>
      ) : null}
      {body}
    </div>
  )
}

function MetadataTab({ artifact }: { artifact: Artifact }) {
  const metadata = artifact.data.metadata
  const summary: { label: string; value: string }[] = []
  if (metadata) {
    if (metadata.status)
      summary.push({ label: 'Status', value: metadata.status })
    if (metadata.config_id)
      summary.push({ label: 'Config', value: metadata.config_id })
    if (metadata.duration_seconds != null)
      summary.push({
        label: 'Duration',
        value: formatDuration(metadata.duration_seconds),
      })
    if (metadata.exit_code != null)
      summary.push({ label: 'Exit code', value: String(metadata.exit_code) })
  }

  // Remaining metadata fields shown as a clean key-value list (raw JSON lives in
  // the Raw tab). Skip the ones already surfaced as summary cards and empties.
  const summaryKeys = new Set(['status', 'config_id', 'duration_seconds', 'exit_code'])
  const rest = metadata
    ? Object.entries(metadata).filter(
        ([key, value]) => !summaryKeys.has(key) && value != null && value !== '',
      )
    : []

  return (
    <div className="space-y-3">
      {summary.length > 0 ? (
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          {summary.map((item) => (
            <div key={item.label} className="rounded-md border p-2">
              <p className="text-xs text-muted-foreground">{item.label}</p>
              <p className="truncate text-sm font-medium">{item.value}</p>
            </div>
          ))}
        </div>
      ) : null}
      {rest.length > 0 ? (
        <dl className="divide-y rounded-md border text-sm">
          {rest.map(([key, value]) => (
            <div key={key} className="grid grid-cols-[11rem_1fr] gap-2 px-3 py-1.5">
              <dt className="text-muted-foreground">{key}</dt>
              <dd className="min-w-0 break-words font-mono text-xs">
                {typeof value === 'object' ? JSON.stringify(value) : String(value)}
              </dd>
            </div>
          ))}
        </dl>
      ) : null}
      {!metadata ? (
        <p className="text-sm text-muted-foreground">No metadata.</p>
      ) : null}
    </div>
  )
}

function ArtifactDetail({
  id,
  onDeleted,
}: {
  id: string
  onDeleted: () => void
}) {
  const queryClient = useQueryClient()
  const { data, isLoading, error } = useQuery({
    queryKey: ['artifact', id],
    queryFn: () => api.artifact(id),
    enabled: Boolean(id),
  })

  const deleteMutation = useMutation({
    mutationFn: () => api.deleteArtifact(id),
    onSuccess: () => {
      toast.success('Artifact deleted')
      void queryClient.invalidateQueries({ queryKey: ['artifacts'] })
      onDeleted()
    },
    onError: (err: unknown) => {
      toast.error(err instanceof Error ? err.message : 'Failed to delete')
    },
  })

  if (isLoading) return <Loading label="Loading artifact…" />
  if (error) return <ErrorState error={error} />
  if (!data) return <EmptyState title="Artifact not found" />

  return (
    <Card className="border-0 shadow-none">
      <CardHeader>
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="space-y-1">
            <CardTitle className="font-mono text-base">{data.id}</CardTitle>
            <CardDescription>
              {nodeLabel(data)} · level {data.level}
              {data.hierarchy ? ` · ${data.hierarchy}` : ''}
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Button asChild variant="outline" size="sm">
              <a href={api.artifactDownloadUrl(data.id)} download>
                <Download className="size-4" />
                Download
              </a>
            </Button>
            <Dialog>
              <DialogTrigger asChild>
                <Button variant="destructive" size="sm">
                  <Trash2 className="size-4" />
                  Delete
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Delete artifact?</DialogTitle>
                  <DialogDescription>
                    This permanently removes artifact{' '}
                    <span className="font-mono">{shortId(data.id)}</span> and its
                    content. This cannot be undone.
                  </DialogDescription>
                </DialogHeader>
                <DialogFooter>
                  <DialogClose asChild>
                    <Button variant="outline">Cancel</Button>
                  </DialogClose>
                  <Button
                    variant="destructive"
                    disabled={deleteMutation.isPending}
                    onClick={() => deleteMutation.mutate()}
                  >
                    {deleteMutation.isPending ? 'Deleting…' : 'Delete'}
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>
          </div>
        </div>
        <Separator />
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div>
            <p className="text-xs text-muted-foreground">Created</p>
            <p>{formatDateTime(data.created_at)}</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Updated</p>
            <p>{formatDateTime(data.updated_at)}</p>
          </div>
        </div>
        {data.tags.length > 0 ? (
          <div className="flex flex-wrap gap-1">
            {data.tags.map((tag) => (
              <Badge key={tag} variant="secondary">
                {tag}
              </Badge>
            ))}
          </div>
        ) : null}
      </CardHeader>
      <CardContent>
        {isDataFrame(data.data.content) ? (
          <Tabs defaultValue="chart">
            <TabsList>
              <TabsTrigger value="chart">Chart</TabsTrigger>
              <TabsTrigger value="table">Table</TabsTrigger>
              <TabsTrigger value="metadata">Metadata</TabsTrigger>
              <TabsTrigger value="raw">Raw</TabsTrigger>
            </TabsList>
            <TabsContent value="chart" className="pt-3">
              <DataFrameChart frame={data.data.content} />
            </TabsContent>
            <TabsContent value="table" className="pt-3">
              <DataFrameTable frame={data.data.content} />
            </TabsContent>
            <TabsContent value="metadata" className="pt-3">
              <MetadataTab artifact={data} />
            </TabsContent>
            <TabsContent value="raw" className="pt-3">
              <JsonView value={data} />
            </TabsContent>
          </Tabs>
        ) : (
          <Tabs defaultValue="metadata">
            <TabsList>
              <TabsTrigger value="metadata">Metadata</TabsTrigger>
              <TabsTrigger value="content">Content</TabsTrigger>
              <TabsTrigger value="raw">Raw</TabsTrigger>
            </TabsList>
            <TabsContent value="metadata" className="pt-3">
              <MetadataTab artifact={data} />
            </TabsContent>
            <TabsContent value="content" className="pt-3">
              <ContentTab artifact={data} />
            </TabsContent>
            <TabsContent value="raw" className="pt-3">
              <JsonView value={data} />
            </TabsContent>
          </Tabs>
        )}
      </CardContent>
    </Card>
  )
}

export function ArtifactsPage() {
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  // Selection lives in the URL (#/artifacts/:artifactId) so it deep-links and
  // survives a refresh.
  const { artifactId } = useParams()
  const selectedId = artifactId ?? null
  const select = (id: string) => navigate(`/artifacts/${id}`)
  const [expanded, setExpanded] = useState<Set<string>>(new Set())

  const { data, isLoading, error } = useQuery({
    queryKey: ['artifacts'],
    queryFn: () => api.artifacts(),
  })

  const forest = useMemo(() => buildForest(data ?? []), [data])

  // Auto-expand the ancestors of a deep-linked selection so it is visible.
  useEffect(() => {
    if (!selectedId || !data) return
    const byId = new Map(data.map((a) => [a.id, a]))
    const ancestors = new Set<string>()
    let current = byId.get(selectedId)?.parent_id
    while (current) {
      ancestors.add(current)
      current = byId.get(current)?.parent_id ?? null
    }
    if (ancestors.size > 0) {
      setExpanded((prev) => {
        const next = new Set(prev)
        for (const id of ancestors) next.add(id)
        return next
      })
    }
  }, [selectedId, data])

  // The hierarchy name lives on $expand, not the flat list; fetch it from the
  // first artifact to title the tree panel.
  const firstId = data?.[0]?.id
  const { data: rootExpand } = useQuery({
    queryKey: ['artifact-hierarchy', firstId],
    queryFn: () => api.artifactExpand(firstId as string),
    enabled: Boolean(firstId),
  })
  const hierarchyName = rootExpand?.hierarchy

  const toggle = (id: string) => {
    setExpanded((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const refresh = () => {
    void queryClient.invalidateQueries({ queryKey: ['artifacts'] })
    if (selectedId)
      void queryClient.invalidateQueries({ queryKey: ['artifact', selectedId] })
  }

  return (
    <>
      <PageHeader
        title="Artifacts"
        description="Browse the artifact hierarchy produced by training and prediction runs."
        actions={
          <Button variant="outline" size="sm" onClick={refresh}>
            <RefreshCw className="size-4" />
            Refresh
          </Button>
        }
      />
      <PageBody>
        {isLoading ? (
          <Loading label="Loading artifacts…" />
        ) : error ? (
          <ErrorState error={error} />
        ) : forest.length === 0 ? (
          <EmptyState
            title="No artifacts yet"
            hint="Run a training or `chapkit test` to produce artifacts."
          />
        ) : (
          <div className="grid gap-4 lg:grid-cols-[20rem_1fr]">
            <Card className="overflow-hidden">
              <CardHeader className="py-3">
                <CardTitle className="text-sm">
                  {hierarchyName ?? 'Hierarchy'}
                </CardTitle>
                {hierarchyName ? (
                  <p className="text-xs text-muted-foreground">Artifact hierarchy</p>
                ) : null}
              </CardHeader>
              <CardContent className="p-0">
                <ScrollArea className="h-[calc(100vh-16rem)]">
                  <div className="p-2">
                    {forest.map((node) => (
                      <TreeRow
                        key={node.id}
                        node={node}
                        depth={0}
                        expanded={expanded}
                        onToggle={toggle}
                        selectedId={selectedId}
                        onSelect={select}
                      />
                    ))}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
            <div>
              {selectedId ? (
                <ArtifactDetail
                  id={selectedId}
                  onDeleted={() => navigate('/artifacts')}
                />
              ) : (
                <EmptyState
                  title="Select an artifact"
                  hint="Choose a node from the hierarchy to inspect its metadata and content."
                />
              )}
            </div>
          </div>
        )}
      </PageBody>
    </>
  )
}
