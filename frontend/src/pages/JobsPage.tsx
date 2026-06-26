// Live job monitor: polls the jobs list, shows a detail dialog, and cancels/removes jobs.
import { useMemo, useState } from 'react'
import { Ban, ExternalLink, RefreshCw, Trash2, X } from 'lucide-react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate, useParams } from 'react-router-dom'
import { toast } from 'sonner'

import { api, readJobArtifactId } from '@/lib/api'
import type { Job, JobStatus } from '@/lib/types'
import {
  CodeBlock,
  EmptyState,
  ErrorState,
  JobStatusBadge,
  Loading,
} from '@/components/console/common'
import { PageBody, PageHeader } from '@/components/console/page'
import { formatDateTime, formatDuration } from '@/lib/format'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Separator } from '@/components/ui/separator'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'

const STATUS_OPTIONS: JobStatus[] = [
  'pending',
  'running',
  'completed',
  'failed',
  'canceled',
]

const TERMINAL_STATES = new Set<JobStatus>(['completed', 'failed', 'canceled'])
const ACTIVE_STATES = new Set<JobStatus>(['pending', 'running'])

type StatusFilter = 'all' | JobStatus

/** Short, readable form of a job id. */
function shortId(id: string): string {
  return id.length > 12 ? `${id.slice(0, 8)}…${id.slice(-4)}` : id
}

/** Duration in seconds between two timestamps, when both are present. */
function jobDurationSeconds(job: Job): number | null {
  if (!job.started_at || !job.finished_at) return null
  const start = new Date(job.started_at).getTime()
  const end = new Date(job.finished_at).getTime()
  if (Number.isNaN(start) || Number.isNaN(end)) return null
  return (end - start) / 1000
}

/** Live monitor for background jobs. */
export function JobsPage() {
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  // Selection lives in the URL (#/jobs/:jobId) so a job detail deep-links and
  // survives a refresh, matching Configs and Artifacts.
  const { jobId } = useParams()
  const [filter, setFilter] = useState<StatusFilter>('all')

  const jobsQuery = useQuery({
    queryKey: ['jobs', filter],
    queryFn: () => api.jobs(filter === 'all' ? undefined : filter),
    refetchInterval: (query) => {
      const data = query.state.data
      if (data?.some((job) => ACTIVE_STATES.has(job.status))) return 1500
      return 8000
    },
  })

  const jobs = useMemo(() => jobsQuery.data ?? [], [jobsQuery.data])
  const selected = useMemo(
    () => jobs.find((job) => job.id === jobId) ?? null,
    [jobs, jobId],
  )

  const cancelMutation = useMutation({
    mutationFn: (id: string) => api.cancelJob(id),
    onSuccess: (_data, id) => {
      toast.success(`Job ${shortId(id)} updated`)
      // The job is gone; close its detail view if it was the one being shown.
      if (id === jobId) navigate('/jobs')
      void queryClient.invalidateQueries({ queryKey: ['jobs'] })
    },
    onError: (error: unknown) => {
      toast.error(error instanceof Error ? error.message : String(error))
    },
  })

  const handleCancel = (job: Job) => {
    if (!window.confirm(`Cancel job ${shortId(job.id)}?`)) return
    cancelMutation.mutate(job.id)
  }

  const handleRemove = (job: Job) => {
    if (!window.confirm(`Remove job ${shortId(job.id)}?`)) return
    cancelMutation.mutate(job.id)
  }

  const handleViewArtifact = async (job: Job) => {
    try {
      const artifactId = await readJobArtifactId(job.id)
      if (artifactId) {
        toast.success(`Result artifact ${shortId(artifactId)}`)
        navigate('/artifacts')
      } else {
        toast.info('This job produced no artifact.')
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : String(error))
    }
  }

  const actions = (
    <>
      <Select
        value={filter}
        onValueChange={(value) => setFilter(value as StatusFilter)}
      >
        <SelectTrigger className="w-40">
          <SelectValue placeholder="All statuses" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="all">All statuses</SelectItem>
          {STATUS_OPTIONS.map((status) => (
            <SelectItem key={status} value={status} className="capitalize">
              {status}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <Button
        variant="outline"
        size="sm"
        onClick={() => jobsQuery.refetch()}
        disabled={jobsQuery.isFetching}
      >
        <RefreshCw
          className={jobsQuery.isFetching ? 'animate-spin' : undefined}
        />
        Refresh
      </Button>
    </>
  )

  return (
    <>
      <PageHeader
        title="Jobs"
        description="Live monitor for training, prediction, and other background jobs."
        actions={actions}
      />
      <PageBody>
        {jobsQuery.isPending ? (
          <Loading label="Loading jobs…" />
        ) : jobsQuery.isError ? (
          <ErrorState error={jobsQuery.error} />
        ) : jobs.length === 0 ? (
          <EmptyState
            title="No jobs yet"
            hint="Submit a training or prediction, or run `chapkit test`."
          />
        ) : (
          <Card className="overflow-hidden p-0">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Status</TableHead>
                  <TableHead>Job id</TableHead>
                  <TableHead className="hidden md:table-cell">Submitted</TableHead>
                  <TableHead className="hidden md:table-cell">Started</TableHead>
                  <TableHead className="hidden md:table-cell">Finished</TableHead>
                  <TableHead className="hidden md:table-cell">Duration</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {jobs.map((job) => {
                  const isActive = ACTIVE_STATES.has(job.status)
                  return (
                    <TableRow
                      key={job.id}
                      data-state={job.id === jobId ? 'selected' : undefined}
                      className="cursor-pointer"
                      onClick={() => navigate(`/jobs/${job.id}`)}
                    >
                      <TableCell>
                        <JobStatusBadge status={job.status} />
                      </TableCell>
                      <TableCell className="font-mono text-xs whitespace-nowrap">
                        {job.id}
                      </TableCell>
                      <TableCell className="hidden text-xs text-muted-foreground md:table-cell">
                        {formatDateTime(job.submitted_at)}
                      </TableCell>
                      <TableCell className="hidden text-xs text-muted-foreground md:table-cell">
                        {formatDateTime(job.started_at)}
                      </TableCell>
                      <TableCell className="hidden text-xs text-muted-foreground md:table-cell">
                        {formatDateTime(job.finished_at)}
                      </TableCell>
                      <TableCell className="hidden text-xs text-muted-foreground md:table-cell">
                        {formatDuration(jobDurationSeconds(job))}
                      </TableCell>
                      <TableCell className="text-right">
                        {isActive ? (
                          <Button
                            variant="destructive"
                            size="xs"
                            disabled={cancelMutation.isPending}
                            onClick={(event) => {
                              event.stopPropagation()
                              handleCancel(job)
                            }}
                          >
                            <Ban />
                            Cancel
                          </Button>
                        ) : (
                          <Button
                            variant="ghost"
                            size="xs"
                            disabled={cancelMutation.isPending}
                            onClick={(event) => {
                              event.stopPropagation()
                              handleRemove(job)
                            }}
                          >
                            <Trash2 />
                            Remove
                          </Button>
                        )}
                      </TableCell>
                    </TableRow>
                  )
                })}
              </TableBody>
            </Table>
          </Card>
        )}
      </PageBody>

      <Dialog
        open={selected !== null}
        onOpenChange={(open) => {
          if (!open) navigate('/jobs')
        }}
      >
        <DialogContent className="max-w-2xl">
          {selected ? (
            <>
              <DialogHeader>
                <DialogTitle className="flex items-center gap-2">
                  <span className="font-mono text-sm break-all">
                    {selected.id}
                  </span>
                  <JobStatusBadge status={selected.status} />
                </DialogTitle>
                <DialogDescription>Background job details.</DialogDescription>
              </DialogHeader>

              <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
                <dt className="text-muted-foreground">Submitted</dt>
                <dd>{formatDateTime(selected.submitted_at)}</dd>
                <dt className="text-muted-foreground">Started</dt>
                <dd>{formatDateTime(selected.started_at)}</dd>
                <dt className="text-muted-foreground">Finished</dt>
                <dd>{formatDateTime(selected.finished_at)}</dd>
                <dt className="text-muted-foreground">Duration</dt>
                <dd>{formatDuration(jobDurationSeconds(selected))}</dd>
              </dl>

              {selected.error || selected.error_traceback ? (
                <>
                  <Separator />
                  {selected.error ? (
                    <div className="space-y-1">
                      <p className="text-sm font-medium text-destructive">
                        Error
                      </p>
                      <CodeBlock className="text-destructive">
                        {selected.error}
                      </CodeBlock>
                    </div>
                  ) : null}
                  {selected.error_traceback ? (
                    <div className="space-y-1">
                      <p className="text-sm font-medium">Traceback</p>
                      <CodeBlock>{selected.error_traceback}</CodeBlock>
                    </div>
                  ) : null}
                </>
              ) : null}

              <DialogFooter className="sm:justify-between">
                {selected.status === 'completed' ? (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => void handleViewArtifact(selected)}
                  >
                    <ExternalLink />
                    View result artifact
                  </Button>
                ) : (
                  <span />
                )}
                {ACTIVE_STATES.has(selected.status) ? (
                  <Button
                    variant="destructive"
                    size="sm"
                    disabled={cancelMutation.isPending}
                    onClick={() => handleCancel(selected)}
                  >
                    <Ban />
                    Cancel job
                  </Button>
                ) : TERMINAL_STATES.has(selected.status) ? (
                  <Button
                    variant="ghost"
                    size="sm"
                    disabled={cancelMutation.isPending}
                    onClick={() => handleRemove(selected)}
                  >
                    <X />
                    Remove job
                  </Button>
                ) : (
                  <span />
                )}
              </DialogFooter>
            </>
          ) : null}
        </DialogContent>
      </Dialog>
    </>
  )
}
