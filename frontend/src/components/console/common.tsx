// Shared presentational helpers used across console screens.
import { AlertCircle, Inbox, Loader2 } from 'lucide-react'
import type { ReactNode } from 'react'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'
import type { JobStatus } from '@/lib/types'

/** Centered spinner for loading states. */
export function Loading({ label = 'Loading…' }: { label?: string }) {
  return (
    <div className="flex items-center justify-center gap-2 py-16 text-muted-foreground">
      <Loader2 className="size-4 animate-spin" />
      <span className="text-sm">{label}</span>
    </div>
  )
}

/** Inline error panel. */
export function ErrorState({ error }: { error: unknown }) {
  const message = error instanceof Error ? error.message : String(error)
  return (
    <div className="flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/5 p-4 text-sm text-destructive">
      <AlertCircle className="mt-0.5 size-4 shrink-0" />
      <span>{message}</span>
    </div>
  )
}

/** Empty-state placeholder. */
export function EmptyState({
  title,
  hint,
}: {
  title: string
  hint?: ReactNode
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 py-16 text-center text-muted-foreground">
      <Inbox className="size-8" />
      <p className="text-sm font-medium">{title}</p>
      {hint ? <p className="max-w-sm text-xs">{hint}</p> : null}
    </div>
  )
}

const JOB_VARIANTS: Record<string, string> = {
  completed: 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-400',
  running: 'bg-blue-500/15 text-blue-700 dark:text-blue-400',
  pending: 'bg-amber-500/15 text-amber-700 dark:text-amber-400',
  failed: 'bg-destructive/15 text-destructive',
  canceled: 'bg-muted text-muted-foreground',
}

/** Color-coded badge for a job status. */
export function JobStatusBadge({ status }: { status: JobStatus }) {
  return (
    <Badge
      variant="secondary"
      className={cn('font-medium capitalize', JOB_VARIANTS[status] ?? '')}
    >
      {status}
    </Badge>
  )
}

/** Monospace block for JSON / code / tracebacks. */
export function CodeBlock({
  children,
  className,
}: {
  children: ReactNode
  className?: string
}) {
  return (
    <pre
      className={cn(
        'max-h-[28rem] overflow-auto rounded-md bg-muted p-3 font-mono text-xs leading-relaxed',
        className,
      )}
    >
      <code>{children}</code>
    </pre>
  )
}

/** Render any JSON value as a formatted code block. */
export function JsonView({ value }: { value: unknown }) {
  return <CodeBlock>{JSON.stringify(value, null, 2)}</CodeBlock>
}
