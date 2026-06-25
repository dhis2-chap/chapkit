// Interactive Train screen — submit a $train job behind a successful $validate gate.
import { useEffect, useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { ExternalLink, Loader2, Play, Settings2, ShieldCheck, Sparkles } from 'lucide-react'
import { toast } from 'sonner'

import { api } from '@/lib/api'
import type { MLJobResponse, TrainPayload, ValidationResult } from '@/lib/types'

import { Loading, ErrorState, EmptyState, JobStatusBadge } from '@/components/console/common'
import { PageHeader } from '@/components/console/page'
import {
  GeneratorPanel,
  DiagnosticsView,
  DataFrameField,
  DEFAULT_GENERATOR_PARAMS,
  toSampleOptions,
  parseDataFrame,
  shortId,
} from '@/components/console/ml-shared'
import type { GeneratorParams } from '@/components/console/ml-shared'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter,
} from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from '@/components/ui/select'
import { Separator } from '@/components/ui/separator'
import {
  Sheet,
  SheetTrigger,
  SheetContent,
  SheetHeader,
  SheetFooter,
  SheetTitle,
  SheetDescription,
} from '@/components/ui/sheet'

/** Result card shown after a successful train submission. */
function JobResultCard({ job, onViewJobs }: { job: MLJobResponse; onViewJobs: () => void }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <JobStatusBadge status="pending" />
          Job submitted
        </CardTitle>
        <CardDescription>{job.message}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-1 text-sm">
        <div className="flex items-center gap-2">
          <span className="text-muted-foreground">Job</span>
          <span className="font-mono">{job.job_id}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-muted-foreground">Artifact</span>
          <span className="font-mono">{job.artifact_id}</span>
        </div>
      </CardContent>
      <CardFooter>
        <Button variant="outline" size="sm" onClick={onViewJobs}>
          <ExternalLink />
          View jobs
        </Button>
      </CardFooter>
    </Card>
  )
}

/** Train console page. */
export function TrainPage() {
  const navigate = useNavigate()
  const configsQuery = useQuery({ queryKey: ['configs'], queryFn: api.configs })

  const [configId, setConfigId] = useState<string>('')
  const [dataText, setDataText] = useState<string>('')
  const [geo, setGeo] = useState<unknown>(undefined)
  const [generator, setGenerator] = useState<GeneratorParams>(DEFAULT_GENERATOR_PARAMS)
  const [generatorOpen, setGeneratorOpen] = useState(false)
  const [validation, setValidation] = useState<ValidationResult | null>(null)
  const [validated, setValidated] = useState(false)
  const [result, setResult] = useState<MLJobResponse | null>(null)

  function invalidate() {
    setValidated(false)
    setValidation(null)
  }

  // Auto-select the first config once loaded so the form is actionable out of the box.
  useEffect(() => {
    const configs = configsQuery.data ?? []
    if (configs.length > 0 && configId === '') {
      setConfigId(configs[0].id)
    }
  }, [configsQuery.data, configId])

  /** Fetch a sample payload, push it into form state, and return the generated train payload. */
  async function fetchAndFillSample(): Promise<TrainPayload> {
    const payload = (await api.sampleData('train', {
      config_id: configId || undefined,
      ...toSampleOptions(generator),
    })) as TrainPayload
    setDataText(JSON.stringify(payload.data, null, 2))
    setGeo(payload.geo)
    if (payload.config_id) setConfigId(payload.config_id)
    invalidate()
    return payload
  }

  const sampleMutation = useMutation({
    mutationFn: fetchAndFillSample,
    onSuccess: () => toast.success('Sample training data loaded'),
    onError: (error: unknown) => toast.error(error instanceof Error ? error.message : String(error)),
  })

  const validateMutation = useMutation({
    mutationFn: (body: Record<string, unknown>) => api.validate(body),
    onSuccess: (res) => {
      setValidation(res)
      setValidated(res.valid)
      toast[res.valid ? 'success' : 'error'](
        res.valid ? 'Validation passed' : 'Validation found problems',
      )
    },
    onError: (error: unknown) => {
      setValidated(false)
      toast.error(error instanceof Error ? error.message : String(error))
    },
  })

  const trainMutation = useMutation({
    mutationFn: (body: TrainPayload) => api.train(body),
    onSuccess: (res) => {
      setResult(res)
      toast.success(`Training job ${shortId(res.job_id)} submitted`)
    },
    onError: (error: unknown) => toast.error(error instanceof Error ? error.message : String(error)),
  })

  /** Build the $validate request body from a concrete train payload. */
  function buildValidateBody(payload: TrainPayload): Record<string, unknown> {
    return {
      type: 'train',
      config_id: payload.config_id || configId,
      data: payload.data,
      ...(payload.geo ? { geo: payload.geo } : {}),
    }
  }

  function handleValidate() {
    if (!configId) {
      toast.error('Select a config first')
      return
    }
    const data = parseDataFrame(dataText, 'Training data')
    if (!data) {
      setValidated(false)
      return
    }
    validateMutation.mutate(buildValidateBody({ config_id: configId, data, geo }))
  }

  function handleSubmit() {
    if (!configId) return
    const data = parseDataFrame(dataText, 'Training data')
    if (!data) return
    trainMutation.mutate({ config_id: configId, data, ...(geo ? { geo } : {}) })
  }

  const configs = configsQuery.data ?? []
  const pending =
    sampleMutation.isPending || validateMutation.isPending || trainMutation.isPending

  return (
    <>
      <PageHeader
        title="Train"
        description="Submit a training job ($train) with $validate gating."
      />
      {configsQuery.isLoading ? (
        <div className="flex-1 overflow-auto p-6">
          <Loading label="Loading configs…" />
        </div>
      ) : configsQuery.isError ? (
        <div className="flex-1 overflow-auto p-6">
          <ErrorState error={configsQuery.error} />
        </div>
      ) : configs.length === 0 ? (
        <div className="flex-1 overflow-auto p-6">
          <EmptyState
            title="No configs yet"
            hint="Create a configuration on the Configs page before training a model."
          />
        </div>
      ) : (
        <div className="flex min-h-0 flex-1 flex-col">
          <div className="flex-1 space-y-4 overflow-auto p-6">
            <Card>
              <CardHeader>
                <CardTitle>Training input</CardTitle>
                <CardDescription>
                  Choose a config and provide the training DataFrame, then validate before
                  submitting.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="train-config">Config</Label>
                  <Select
                    value={configId}
                    onValueChange={(value) => {
                      setConfigId(value)
                      invalidate()
                    }}
                  >
                    <SelectTrigger id="train-config" className="w-full">
                      <SelectValue placeholder="Select a config…" />
                    </SelectTrigger>
                    <SelectContent>
                      {configs.map((config) => (
                        <SelectItem key={config.id} value={config.id}>
                          <span className="truncate">
                            {config.name}{' '}
                            <span className="text-muted-foreground">{config.id}</span>
                          </span>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <DataFrameField
                  id="train-data"
                  label="Training data"
                  placeholder='{ "columns": ["time_period", "value"], "data": [["2020-01", 12]] }'
                  value={dataText}
                  onChange={(next) => {
                    setDataText(next)
                    invalidate()
                  }}
                />
              </CardContent>
            </Card>

            {validation ? (
              <>
                <Separator />
                <DiagnosticsView result={validation} />
              </>
            ) : null}

            {result ? <JobResultCard job={result} onViewJobs={() => navigate('/jobs')} /> : null}
          </div>

          <div className="shrink-0 border-t bg-background px-6 py-3">
            <div className="flex flex-wrap items-center gap-2">
              <div className="inline-flex">
                <Button
                  variant="outline"
                  className="rounded-r-none"
                  onClick={() => sampleMutation.mutate()}
                  disabled={pending}
                >
                  {sampleMutation.isPending ? <Loader2 className="animate-spin" /> : <Sparkles />}
                  Generate
                </Button>
                <Sheet open={generatorOpen} onOpenChange={setGeneratorOpen}>
                  <SheetTrigger asChild>
                    <Button
                      variant="outline"
                      size="icon"
                      className="rounded-l-none border-l-0"
                      disabled={pending}
                      aria-label="Sample data generator options"
                    >
                      <Settings2 />
                    </Button>
                  </SheetTrigger>
                  <SheetContent side="right" className="w-full sm:max-w-md">
                    <SheetHeader>
                      <SheetTitle>Sample data generator</SheetTitle>
                      <SheetDescription>
                        Tune chapkit&apos;s synthetic data generator, then generate.
                      </SheetDescription>
                    </SheetHeader>
                    <div className="space-y-4 overflow-auto px-4">
                      <GeneratorPanel
                        params={generator}
                        onChange={setGenerator}
                        disabled={sampleMutation.isPending}
                      />
                    </div>
                    <SheetFooter>
                      <Button
                        onClick={() => {
                          setGeneratorOpen(false)
                          sampleMutation.mutate()
                        }}
                        disabled={sampleMutation.isPending}
                      >
                        {sampleMutation.isPending ? (
                          <Loader2 className="animate-spin" />
                        ) : (
                          <Sparkles />
                        )}
                        Generate
                      </Button>
                    </SheetFooter>
                  </SheetContent>
                </Sheet>
              </div>
              <Button
                variant="secondary"
                onClick={handleValidate}
                disabled={pending || !configId}
              >
                {validateMutation.isPending ? <Loader2 className="animate-spin" /> : <ShieldCheck />}
                Validate
              </Button>
              <Button onClick={handleSubmit} disabled={pending || !validated}>
                {trainMutation.isPending ? <Loader2 className="animate-spin" /> : <Play />}
                Train
              </Button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
