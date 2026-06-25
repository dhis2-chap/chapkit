// System screen: runtime information and mounted static apps.
import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'
import { Badge } from '@/components/ui/badge'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { EmptyState, ErrorState, Loading } from '@/components/console/common'
import { PageBody, PageHeader } from '@/components/console/page'

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="grid grid-cols-[10rem_1fr] gap-2 py-1.5 text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span className="min-w-0 break-words font-mono text-xs">{value}</span>
    </div>
  )
}

export function SystemPage() {
  const system = useQuery({ queryKey: ['system'], queryFn: api.system })
  const apps = useQuery({ queryKey: ['apps'], queryFn: api.apps })

  return (
    <>
      <PageHeader title="System" description="Service runtime information" />
      <PageBody>
        {system.isLoading ? (
          <Loading />
        ) : system.error ? (
          <ErrorState error={system.error} />
        ) : (
          <div className="grid gap-4 lg:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Runtime</CardTitle>
                <CardDescription>Host and interpreter</CardDescription>
              </CardHeader>
              <CardContent>
                {system.data ? (
                  <>
                    <InfoRow label="Hostname" value={system.data.hostname} />
                    <InfoRow label="Platform" value={system.data.platform} />
                    <InfoRow
                      label="Python"
                      value={system.data.python_version}
                    />
                    <InfoRow label="Timezone" value={system.data.timezone} />
                    <InfoRow
                      label="Server time"
                      value={system.data.current_time}
                    />
                  </>
                ) : null}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Mounted apps</CardTitle>
                <CardDescription>Static apps served by this service</CardDescription>
              </CardHeader>
              <CardContent>
                {apps.isLoading ? (
                  <Loading />
                ) : apps.error ? (
                  <ErrorState error={apps.error} />
                ) : !apps.data || apps.data.length === 0 ? (
                  <EmptyState title="No mounted apps" />
                ) : (
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Name</TableHead>
                        <TableHead>Prefix</TableHead>
                        <TableHead>Version</TableHead>
                        <TableHead>Source</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {apps.data.map((app) => (
                        <TableRow key={`${app.name}-${app.prefix}`}>
                          <TableCell className="font-medium">{app.name}</TableCell>
                          <TableCell className="font-mono text-xs">
                            {app.prefix}
                          </TableCell>
                          <TableCell>{app.version}</TableCell>
                          <TableCell>
                            <Badge variant="secondary">
                              {app.is_package ? 'package' : 'filesystem'}
                            </Badge>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                )}
              </CardContent>
            </Card>
          </div>
        )}
      </PageBody>
    </>
  )
}
