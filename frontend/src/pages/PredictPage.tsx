// Interactive Predict screen — run a $predict job from a trained model behind a $validate gate.
import { useEffect, useMemo, useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { ExternalLink, Loader2, Play, Settings2, ShieldCheck, Sparkles } from 'lucide-react'
import { toast } from 'sonner'

import { api } from '@/lib/api'
import type { DataFrameContent, MLJobResponse, PredictPayload, ValidationResult } from '@/lib/types'

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

/** Result card shown after a successful predict submission. */
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

/** Predict console page. */
export function PredictPage() {
  const navigate = useNavigate()
  const artifactsQuery = useQuery({ queryKey: ['artifacts'], queryFn: api.artifacts })

  const [artifactId, setArtifactId] = useState<string>('')
  const [historicText, setHistoricText] = useState<string>('')
  const [futureText, setFutureText] = useState<string>('')
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

  const trainingArtifacts = useMemo(
    () =>
      (artifactsQuery.data ?? []).filter(
        (artifact) => artifact.data?.type?.includes('training') || artifact.level === 0,
      ),
    [artifactsQuery.data],
  )

  // Auto-select the first trained model once loaded so the form is actionable out of the box.
  useEffect(() => {
    if (trainingArtifacts.length > 0 && artifactId === '') {
      setArtifactId(trainingArtifacts[0].id)
    }
  }, [trainingArtifacts, artifactId])

  /** Fetch a sample payload, push it into form state, and return the generated predict payload. */
  async function fetchAndFillSample(): Promise<PredictPayload> {
    const payload = (await api.sampleData('predict', toSampleOptions(generator))) as PredictPayload
    setHistoricText(JSON.stringify(payload.historic, null, 2))
    setFutureText(JSON.stringify(payload.future, null, 2))
    setGeo(payload.geo)
    if (payload.artifact_id) setArtifactId(payload.artifact_id)
    invalidate()
    return payload
  }

  const sampleMutation = useMutation({
    mutationFn: fetchAndFillSample,
    onSuccess: () => toast.success('Sample prediction data loaded'),
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

  const predictMutation = useMutation({
    mutationFn: (body: PredictPayload) => api.predict(body),
    onSuccess: (res) => {
      setResult(res)
      toast.success(`Prediction job ${shortId(res.job_id)} submitted`)
    },
    onError: (error: unknown) => toast.error(error instanceof Error ? error.message : String(error)),
  })

  function buildFrames(): { historic: DataFrameContent; future: DataFrameContent } | null {
    const historic = parseDataFrame(historicText, 'Historic data')
    if (!historic) return null
    const future = parseDataFrame(futureText, 'Future data')
    if (!future) return null
    return { historic, future }
  }

  /** Build the $validate request body from a concrete predict payload. */
  function buildValidateBody(payload: PredictPayload): Record<string, unknown> {
    return {
      type: 'predict',
      artifact_id: payload.artifact_id,
      historic: payload.historic,
      future: payload.future,
      ...(payload.geo ? { geo: payload.geo } : {}),
    }
  }

  function handleValidate() {
    if (!artifactId) {
      toast.error('Select a trained model artifact first')
      return
    }
    const frames = buildFrames()
    if (!frames) {
      setValidated(false)
      return
    }
    validateMutation.mutate(
      buildValidateBody({
        artifact_id: artifactId,
        historic: frames.historic,
        future: frames.future,
        geo,
      }),
    )
  }

  function handleSubmit() {
    if (!artifactId) return
    const frames = buildFrames()
    if (!frames) return
    predictMutation.mutate({
      artifact_id: artifactId,
      historic: frames.historic,
      future: frames.future,
      ...(geo ? { geo } : {}),
    })
  }

  const pending =
    sampleMutation.isPending || validateMutation.isPending || predictMutation.isPending

  return (
    <>
      <PageHeader
        title="Predict"
        description="Run predictions ($predict) from a trained model, with $validate gating."
      />
      {artifactsQuery.isLoading ? (
        <div className="flex-1 overflow-auto p-6">
          <Loading label="Loading artifacts…" />
        </div>
      ) : artifactsQuery.isError ? (
        <div className="flex-1 overflow-auto p-6">
          <ErrorState error={artifactsQuery.error} />
        </div>
      ) : trainingArtifacts.length === 0 ? (
        <div className="flex-1 overflow-auto p-6">
          <EmptyState
            title="No trained models yet"
            hint="Train a model on the Train page to produce an artifact you can predict with."
          />
        </div>
      ) : (
        <div className="flex min-h-0 flex-1 flex-col">
          <div className="flex-1 space-y-4 overflow-auto p-6">
            <Card>
              <CardHeader>
                <CardTitle>Prediction input</CardTitle>
                <CardDescription>
                  Choose a trained model and provide historic and future DataFrames, then validate
                  before submitting.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="predict-artifact">Trained model artifact</Label>
                  <Select
                    value={artifactId}
                    onValueChange={(value) => {
                      setArtifactId(value)
                      invalidate()
                    }}
                  >
                    <SelectTrigger id="predict-artifact" className="w-full">
                      <SelectValue placeholder="Select a trained model…" />
                    </SelectTrigger>
                    <SelectContent>
                      {trainingArtifacts.map((artifact) => (
                        <SelectItem key={artifact.id} value={artifact.id}>
                          <span className="truncate">
                            {artifact.id}
                            {artifact.data?.type ? (
                              <span className="text-muted-foreground">
                                {' '}
                                ({artifact.data.type})
                              </span>
                            ) : null}
                          </span>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <DataFrameField
                  id="predict-historic"
                  label="Historic data"
                  placeholder='{ "columns": ["time_period", "value"], "data": [["2020-01", 12]] }'
                  value={historicText}
                  onChange={(next) => {
                    setHistoricText(next)
                    invalidate()
                  }}
                />

                <DataFrameField
                  id="predict-future"
                  label="Future data"
                  placeholder='{ "columns": ["time_period"], "data": [["2021-01"]] }'
                  value={futureText}
                  onChange={(next) => {
                    setFutureText(next)
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
                disabled={pending || !artifactId}
              >
                {validateMutation.isPending ? <Loader2 className="animate-spin" /> : <ShieldCheck />}
                Validate
              </Button>
              <Button onClick={handleSubmit} disabled={pending || !validated}>
                {predictMutation.isPending ? <Loader2 className="animate-spin" /> : <Play />}
                Predict
              </Button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
