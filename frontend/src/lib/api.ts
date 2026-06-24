// Low-level HTTP helpers for the chapkit API. All paths are relative to the
// service root so the console works wherever it is mounted (default: `/`).

import type {
  Artifact,
  ConfigInput,
  ConfigItem,
  DataFrameContent,
  HealthStatus,
  Job,
  JobStatus,
  MLJobResponse,
  OpenApiSpec,
  PredictPayload,
  ServiceInfo,
  SystemApp,
  SystemInfo,
  TrainPayload,
  ValidationResult,
} from './types'

/** RFC 9457 problem-details error raised for non-2xx responses. */
export class ApiError extends Error {
  status: number
  detail?: string

  constructor(status: number, title: string, detail?: string) {
    super(detail ? `${title}: ${detail}` : title)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: {
      Accept: 'application/json',
      ...(init?.body ? { 'Content-Type': 'application/json' } : {}),
      ...init?.headers,
    },
  })
  if (!res.ok) {
    let title = res.statusText || `HTTP ${res.status}`
    let detail: string | undefined
    try {
      const problem = await res.json()
      title = problem.title ?? title
      detail = problem.detail
    } catch {
      // non-JSON error body; keep status text
    }
    throw new ApiError(res.status, title, detail)
  }
  if (res.status === 204) return undefined as T
  return (await res.json()) as T
}

export const api = {
  // --- service / system ---------------------------------------------------
  info: () => request<ServiceInfo>('api/v1/info'),
  system: () => request<SystemInfo>('api/v1/system'),
  apps: () => request<SystemApp[]>('api/v1/system/apps'),
  health: () => request<HealthStatus>('health'),
  openapi: () => request<OpenApiSpec>('openapi.json'),

  // --- configs ------------------------------------------------------------
  configs: () => request<ConfigItem[]>('api/v1/configs'),
  config: (id: string) => request<ConfigItem>(`api/v1/configs/${id}`),
  configSchema: () => request<Record<string, unknown>>('api/v1/configs/$schema'),
  createConfig: (body: ConfigInput) =>
    request<ConfigItem>('api/v1/configs', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  updateConfig: (id: string, body: ConfigInput) =>
    request<ConfigItem>(`api/v1/configs/${id}`, {
      method: 'PUT',
      body: JSON.stringify(body),
    }),
  deleteConfig: (id: string) =>
    request<void>(`api/v1/configs/${id}`, { method: 'DELETE' }),

  // --- artifacts ----------------------------------------------------------
  artifacts: () => request<Artifact[]>('api/v1/artifacts'),
  artifact: (id: string) => request<Artifact>(`api/v1/artifacts/${id}`),
  artifactTree: (id: string) =>
    request<Artifact>(`api/v1/artifacts/${id}/$tree`),
  deleteArtifact: (id: string) =>
    request<void>(`api/v1/artifacts/${id}`, { method: 'DELETE' }),
  /** Direct browser download URL for an artifact's content. */
  artifactDownloadUrl: (id: string) => `api/v1/artifacts/${id}/$download`,

  // --- jobs ---------------------------------------------------------------
  jobs: (status?: JobStatus) =>
    request<Job[]>(
      `api/v1/jobs${status ? `?status_filter=${encodeURIComponent(status)}` : ''}`,
    ),
  job: (id: string) => request<Job>(`api/v1/jobs/${id}`),
  cancelJob: (id: string) =>
    request<void>(`api/v1/jobs/${id}`, { method: 'DELETE' }),

  // --- ml -----------------------------------------------------------------
  train: (body: TrainPayload) =>
    request<MLJobResponse>('api/v1/ml/$train', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  predict: (body: PredictPayload) =>
    request<MLJobResponse>('api/v1/ml/$predict', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  validate: (body: Record<string, unknown>) =>
    request<ValidationResult>('api/v1/ml/$validate', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  /** Sample train/predict payloads generated server-side (chapkit extension). */
  sampleData: (
    kind: 'train' | 'predict',
    options: SampleDataOptions = {},
  ) => {
    const params = new URLSearchParams({ kind })
    for (const [key, value] of Object.entries(options)) {
      if (value !== undefined && value !== null && value !== '') {
        params.set(key, String(value))
      }
    }
    return request<TrainPayload | PredictPayload>(
      `api/v1/ml/$sample-data?${params.toString()}`,
    )
  },
}

/** Tunable parameters for the server-side sample-data generator. */
export interface SampleDataOptions {
  config_id?: string
  num_locations?: number
  num_periods?: number
  num_features?: number
  period_type?: 'monthly' | 'weekly'
  geo_type?: 'polygon' | 'point'
  include_geo?: boolean
  seed?: number
}

/**
 * Read a completed job's result-artifact id. The id is only carried on the
 * job's `$stream` SSE channel, so we read the first event and stop.
 */
export async function readJobArtifactId(jobId: string): Promise<string | null> {
  const controller = new AbortController()
  try {
    const res = await fetch(`api/v1/jobs/${jobId}/$stream`, {
      signal: controller.signal,
    })
    const reader = res.body?.getReader()
    if (!reader) return null
    const { value } = await reader.read()
    void reader.cancel()
    const match = new TextDecoder().decode(value).match(/data: (\{.*\})/)
    const record = match ? JSON.parse(match[1]) : {}
    return record.artifact_id ?? null
  } catch {
    return null
  } finally {
    controller.abort()
  }
}

export type { DataFrameContent }
