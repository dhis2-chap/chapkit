// Overview screen: service identity, model metadata, health, and quick links.
import {
  Activity,
  BookOpen,
  Boxes,
  ExternalLink,
  ListTree,
  SlidersHorizontal,
} from 'lucide-react'
import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'
import type { AssessedStatus } from '@/lib/types'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { ErrorState, Loading } from '@/components/console/common'
import { PageBody, PageHeader } from '@/components/console/page'

const ASSESSED_COLORS: Record<AssessedStatus, string> = {
  green: 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-400',
  yellow: 'bg-amber-500/15 text-amber-700 dark:text-amber-400',
  orange: 'bg-orange-500/15 text-orange-700 dark:text-orange-400',
  red: 'bg-destructive/15 text-destructive',
  gray: 'bg-muted text-muted-foreground',
}

function MetaRow({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="grid grid-cols-[10rem_1fr] gap-2 py-1.5 text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span className="min-w-0 break-words">{children}</span>
    </div>
  )
}

const QUICK_LINKS = [
  { to: '/configs', label: 'Configs', icon: SlidersHorizontal },
  { to: '/artifacts', label: 'Artifacts', icon: ListTree },
  { to: '/jobs', label: 'Jobs', icon: Boxes },
] as const

export function OverviewPage() {
  const info = useQuery({ queryKey: ['info'], queryFn: api.info })
  const health = useQuery({ queryKey: ['health'], queryFn: api.health })

  if (info.isLoading) return <Loading />
  if (info.error || !info.data) {
    return (
      <>
        <PageHeader title="Overview" />
        <PageBody>
          <ErrorState error={info.error ?? new Error('No service info')} />
        </PageBody>
      </>
    )
  }

  const svc = info.data
  const meta = svc.model_metadata
  const healthy = health.data?.status

  return (
    <>
      <PageHeader
        title={svc.display_name}
        description={svc.description ?? 'Chapkit service console'}
        actions={
          <Button variant="outline" size="sm" asChild>
            <a href="docs" target="_blank" rel="noreferrer">
              <BookOpen className="size-4" /> API Docs
              <ExternalLink className="size-3.5" />
            </a>
          </Button>
        }
      />
      <PageBody>
        <div className="grid gap-4 lg:grid-cols-3">
          <Card className="lg:col-span-2">
            <CardHeader>
              <CardTitle>Service</CardTitle>
              <CardDescription>Identity and capability contract</CardDescription>
            </CardHeader>
            <CardContent>
              <MetaRow label="Service ID">
                <code className="text-xs">{svc.id}</code>
              </MetaRow>
              <MetaRow label="Version">{svc.version}</MetaRow>
              {svc.period_type ? (
                <MetaRow label="Period type">{svc.period_type}</MetaRow>
              ) : null}
              {svc.min_prediction_periods != null ||
              svc.max_prediction_periods != null ? (
                <MetaRow label="Prediction periods">
                  {svc.min_prediction_periods ?? 0} – {svc.max_prediction_periods ?? '∞'}
                </MetaRow>
              ) : null}
              {svc.required_covariates && svc.required_covariates.length > 0 ? (
                <MetaRow label="Required covariates">
                  <span className="flex flex-wrap gap-1">
                    {svc.required_covariates.map((c) => (
                      <Badge key={c} variant="secondary">
                        {c}
                      </Badge>
                    ))}
                  </span>
                </MetaRow>
              ) : null}
              {svc.requires_geo != null ? (
                <MetaRow label="Requires geo">
                  {svc.requires_geo ? 'Yes' : 'No'}
                </MetaRow>
              ) : null}

              {meta ? (
                <>
                  <Separator className="my-3" />
                  {meta.author ? (
                    <MetaRow label="Author">{meta.author}</MetaRow>
                  ) : null}
                  {meta.organization ? (
                    <MetaRow label="Organization">{meta.organization}</MetaRow>
                  ) : null}
                  {meta.author_assessed_status ? (
                    <MetaRow label="Assessed status">
                      <Badge
                        variant="secondary"
                        className={ASSESSED_COLORS[meta.author_assessed_status]}
                      >
                        {meta.author_assessed_status}
                      </Badge>
                    </MetaRow>
                  ) : null}
                  {meta.author_note ? (
                    <MetaRow label="Note">{meta.author_note}</MetaRow>
                  ) : null}
                  {meta.repository_url ? (
                    <MetaRow label="Repository">
                      <a
                        className="text-primary underline-offset-4 hover:underline"
                        href={String(meta.repository_url)}
                        target="_blank"
                        rel="noreferrer"
                      >
                        {String(meta.repository_url)}
                      </a>
                    </MetaRow>
                  ) : null}
                  {meta.documentation_url ? (
                    <MetaRow label="Documentation">
                      <a
                        className="text-primary underline-offset-4 hover:underline"
                        href={String(meta.documentation_url)}
                        target="_blank"
                        rel="noreferrer"
                      >
                        {String(meta.documentation_url)}
                      </a>
                    </MetaRow>
                  ) : null}
                </>
              ) : null}
            </CardContent>
          </Card>

          <div className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Activity className="size-4" /> Health
                </CardTitle>
              </CardHeader>
              <CardContent>
                {health.isLoading ? (
                  <span className="text-sm text-muted-foreground">Checking…</span>
                ) : (
                  <Badge
                    variant="secondary"
                    className={
                      healthy === 'healthy'
                        ? 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-400'
                        : 'bg-amber-500/15 text-amber-700 dark:text-amber-400'
                    }
                  >
                    {healthy ?? 'unknown'}
                  </Badge>
                )}
                {health.data?.checks ? (
                  <div className="mt-3 space-y-1">
                    {Object.entries(health.data.checks).map(([name, c]) => (
                      <div key={name} className="flex justify-between text-xs">
                        <span className="text-muted-foreground">{name}</span>
                        <span>{c.state ?? '—'}</span>
                      </div>
                    ))}
                  </div>
                ) : null}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Quick links</CardTitle>
              </CardHeader>
              <CardContent className="grid gap-2">
                {QUICK_LINKS.map((l) => (
                  <Button
                    key={l.to}
                    variant="outline"
                    className="justify-start"
                    asChild
                  >
                    <Link to={l.to}>
                      <l.icon className="size-4" /> {l.label}
                    </Link>
                  </Button>
                ))}
              </CardContent>
            </Card>
          </div>
        </div>
      </PageBody>
    </>
  )
}
