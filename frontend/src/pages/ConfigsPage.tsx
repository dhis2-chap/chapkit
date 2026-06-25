// Configs screen: master/detail browser with create/edit/delete for chapkit configs.
import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { Pencil, Plus, RefreshCw, Trash2 } from 'lucide-react'
import {
  useMutation,
  useQuery,
  useQueryClient,
} from '@tanstack/react-query'
import { toast } from 'sonner'

import { api, ApiError } from '@/lib/api'
import type { ConfigInput, ConfigItem } from '@/lib/types'
import { formatDateTime, formatRelative } from '@/lib/format'
import { cn } from '@/lib/utils'
import { EmptyState, ErrorState, Loading } from '@/components/console/common'
import { JsonEditor } from '@/components/console/json-editor'
import { PageBody, PageHeader } from '@/components/console/page'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'

const CONFIGS_KEY = ['configs'] as const

/** Best-effort zero value for a JSON Schema property by declared type. */
function zeroForType(prop: Record<string, unknown>): unknown {
  const type = prop.type
  const declared = Array.isArray(type)
    ? type.find((t) => t !== 'null')
    : type
  switch (declared) {
    case 'string':
      return ''
    case 'integer':
    case 'number':
      return 0
    case 'boolean':
      return false
    case 'array':
      return []
    case 'object':
      return {}
    default:
      return null
  }
}

/** Build a skeleton config-data object from a JSON Schema's properties. */
function skeletonFromSchema(
  schema: Record<string, unknown> | undefined,
): Record<string, unknown> {
  const properties = schema?.properties
  if (!properties || typeof properties !== 'object') return {}
  const result: Record<string, unknown> = {}
  for (const [key, raw] of Object.entries(
    properties as Record<string, unknown>,
  )) {
    if (key === 'id') continue
    const prop =
      raw && typeof raw === 'object' ? (raw as Record<string, unknown>) : {}
    result[key] = 'default' in prop ? prop.default : zeroForType(prop)
  }
  return result
}

/** Read a human-readable message from a thrown error. */
function errorMessage(err: unknown): string {
  return err instanceof ApiError ? err.message : String(err)
}

/** What the right-hand panel shows: a read-only view, the edit form, or a create form. */
type PanelMode = 'view' | 'edit' | 'create'

export function ConfigsPage() {
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  // Selection lives in the URL (#/configs/:configId) so it deep-links and
  // survives a refresh.
  const { configId } = useParams()
  const selectedId = configId ?? null
  const [mode, setMode] = useState<PanelMode>('view')
  // Selecting a config always returns the panel to read-only view.
  const select = (id: string) => {
    setMode('view')
    navigate(`/configs/${id}`)
  }
  const [deleteTarget, setDeleteTarget] = useState<ConfigItem | null>(null)

  const configsQuery = useQuery({
    queryKey: CONFIGS_KEY,
    queryFn: () => api.configs(),
  })

  const configs = useMemo(() => configsQuery.data ?? [], [configsQuery.data])
  const selected = useMemo(
    () => configs.find((config) => config.id === selectedId) ?? null,
    [configs, selectedId],
  )

  const invalidate = () =>
    queryClient.invalidateQueries({ queryKey: CONFIGS_KEY })

  const deleteMutation = useMutation({
    mutationFn: (id: string) => api.deleteConfig(id),
    onSuccess: (_data, id) => {
      toast.success('Config deleted')
      if (selectedId === id) navigate('/configs')
      setDeleteTarget(null)
      void invalidate()
    },
    onError: (err: unknown) => toast.error(errorMessage(err)),
  })

  return (
    <>
      <PageHeader
        title="Configs"
        description="Key-value configuration objects served by this chapkit instance."
        actions={
          <>
            <Button
              variant="outline"
              size="sm"
              onClick={() => void invalidate()}
              disabled={configsQuery.isFetching}
            >
              <RefreshCw
                className={cn(
                  'size-4',
                  configsQuery.isFetching && 'animate-spin',
                )}
              />
              Refresh
            </Button>
            <Button size="sm" onClick={() => setMode('create')}>
              <Plus className="size-4" />
              New config
            </Button>
          </>
        }
      />
      <PageBody>
        {configsQuery.isLoading ? (
          <Loading />
        ) : configsQuery.isError ? (
          <ErrorState error={configsQuery.error} />
        ) : configs.length === 0 ? (
          <EmptyState
            title="No configs yet"
            hint="Create one to get started, or run `chapkit test` to seed the service."
          />
        ) : (
          <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)]">
            <Card className="overflow-hidden">
              <CardHeader>
                <CardTitle>Configs</CardTitle>
                <CardDescription>
                  {configs.length} config{configs.length === 1 ? '' : 's'}
                </CardDescription>
              </CardHeader>
              <CardContent className="px-0">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Name</TableHead>
                      <TableHead className="text-right">Created</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {configs.map((config) => (
                      <TableRow
                        key={config.id}
                        data-state={
                          config.id === selectedId ? 'selected' : undefined
                        }
                        className="cursor-pointer"
                        onClick={() => select(config.id)}
                      >
                        <TableCell className="max-w-[14rem] truncate font-medium">
                          {config.name}
                        </TableCell>
                        <TableCell className="text-right text-muted-foreground">
                          {formatRelative(config.created_at)}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>

            {mode === 'create' ? (
              <ConfigForm
                key="create"
                mode="create"
                onCancel={() => setMode('view')}
                onSaved={(config) => {
                  void invalidate()
                  select(config.id)
                }}
              />
            ) : selected && mode === 'edit' ? (
              <ConfigForm
                key={`edit:${selected.id}`}
                mode="edit"
                config={selected}
                onCancel={() => setMode('view')}
                onSaved={() => {
                  setMode('view')
                  void invalidate()
                }}
              />
            ) : selected ? (
              <ConfigDetail
                config={selected}
                onEdit={() => setMode('edit')}
                onDelete={() => setDeleteTarget(selected)}
              />
            ) : (
              <Card>
                <CardContent className="py-16">
                  <EmptyState
                    title="No config selected"
                    hint="Select a config from the list, or create a new one."
                  />
                </CardContent>
              </Card>
            )}
          </div>
        )}
      </PageBody>

      <Dialog
        open={deleteTarget !== null}
        onOpenChange={(open) => {
          if (!open) setDeleteTarget(null)
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete config</DialogTitle>
            <DialogDescription>
              This permanently removes{' '}
              <span className="font-medium">{deleteTarget?.name}</span>. This
              action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <DialogClose asChild>
              <Button variant="outline">Cancel</Button>
            </DialogClose>
            <Button
              variant="destructive"
              disabled={deleteMutation.isPending}
              onClick={() => {
                if (deleteTarget) deleteMutation.mutate(deleteTarget.id)
              }}
            >
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}

/** Detail panel for a single selected config. */
function ConfigDetail({
  config,
  onEdit,
  onDelete,
}: {
  config: ConfigItem
  onEdit: () => void
  onDelete: () => void
}) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-start justify-between gap-4">
        <div className="space-y-1">
          <CardTitle>{config.name}</CardTitle>
          <CardDescription className="font-mono text-xs">
            {config.id}
          </CardDescription>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={onEdit}>
            <Pencil className="size-4" />
            Edit
          </Button>
          <Button variant="outline" size="sm" onClick={onDelete}>
            <Trash2 className="size-4" />
            Delete
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <dl className="grid grid-cols-2 gap-3 text-sm">
          <div className="space-y-0.5">
            <dt className="text-xs text-muted-foreground">Created</dt>
            <dd>{formatDateTime(config.created_at)}</dd>
          </div>
          <div className="space-y-0.5">
            <dt className="text-xs text-muted-foreground">Updated</dt>
            <dd>{formatDateTime(config.updated_at)}</dd>
          </div>
        </dl>

        <div className="space-y-1.5">
          <span className="text-xs text-muted-foreground">Tags</span>
          <div className="flex flex-wrap gap-1.5">
            {config.tags.length === 0 ? (
              <span className="text-sm text-muted-foreground">—</span>
            ) : (
              config.tags.map((tag) => (
                <Badge key={tag} variant="secondary">
                  {tag}
                </Badge>
              ))
            )}
          </div>
        </div>

        <Separator />

        <div className="space-y-1.5">
          <span className="text-xs text-muted-foreground">Data</span>
          <JsonEditor
            value={JSON.stringify(config.data, null, 2)}
            readOnly
            ariaLabel="Config data"
          />
        </div>
      </CardContent>
    </Card>
  )
}

/** Inline create/edit form rendered in the detail panel (no modal). */
function ConfigForm({
  mode,
  config,
  onSaved,
  onCancel,
}: {
  mode: 'create' | 'edit'
  config?: ConfigItem
  onSaved: (config: ConfigItem) => void
  onCancel: () => void
}) {
  const editing = mode === 'edit' ? (config ?? null) : null

  const schemaQuery = useQuery({
    queryKey: ['config-schema'],
    queryFn: () => api.configSchema(),
    enabled: mode === 'create',
  })

  const [name, setName] = useState(editing?.name ?? '')
  const [dataText, setDataText] = useState(
    editing ? JSON.stringify(editing.data, null, 2) : '',
  )
  const [parseError, setParseError] = useState<string | null>(null)

  // Seed the create form from the schema skeleton once it loads (only while untouched).
  useEffect(() => {
    if (mode === 'create' && schemaQuery.data && dataText === '') {
      setDataText(JSON.stringify(skeletonFromSchema(schemaQuery.data), null, 2))
    }
  }, [mode, schemaQuery.data, dataText])

  const saveMutation = useMutation({
    mutationFn: (input: { id: string | null; body: ConfigInput }) =>
      input.id
        ? api.updateConfig(input.id, input.body)
        : api.createConfig(input.body),
    onSuccess: (saved) => {
      toast.success(editing ? 'Config updated' : 'Config created')
      onSaved(saved)
    },
    onError: (err: unknown) => toast.error(errorMessage(err)),
  })

  const handleSubmit = () => {
    let parsed: unknown
    try {
      parsed = JSON.parse(dataText)
    } catch (err) {
      setParseError(err instanceof Error ? err.message : 'Invalid JSON')
      return
    }
    if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
      setParseError('Data must be a JSON object.')
      return
    }
    setParseError(null)
    saveMutation.mutate({
      id: editing?.id ?? null,
      body: { name, data: parsed as Record<string, unknown> },
    })
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between gap-4">
        <CardTitle>{editing ? 'Edit config' : 'New config'}</CardTitle>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={onCancel}>
            Cancel
          </Button>
          <Button
            size="sm"
            disabled={saveMutation.isPending || name.trim().length === 0}
            onClick={handleSubmit}
          >
            {editing ? 'Save changes' : 'Create config'}
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-1.5">
          <Label htmlFor="config-name">Name</Label>
          <Input
            id="config-name"
            value={name}
            onChange={(event) => setName(event.target.value)}
            placeholder="my-config"
          />
        </div>
        <div className="space-y-1.5">
          <Label>Data (JSON)</Label>
          <JsonEditor
            value={dataText}
            onChange={(next) => {
              setDataText(next)
              if (parseError) setParseError(null)
            }}
            ariaLabel="Config data"
            minHeight="16rem"
          />
          {parseError ? (
            <p className="text-xs text-destructive">{parseError}</p>
          ) : null}
        </div>
      </CardContent>
    </Card>
  )
}
