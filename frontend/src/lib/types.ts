// Type definitions mirroring chapkit's HTTP API (servicekit + chapkit modules).
// The console talks directly to the service's own /api/v1 surface; there is no
// DHIS2 proxy in the path.

export type AssessedStatus = 'gray' | 'red' | 'orange' | 'yellow' | 'green'

export interface ModelMetadata {
  author?: string | null
  author_note?: string | null
  author_assessed_status?: AssessedStatus | null
  contact_email?: string | null
  organization?: string | null
  organization_logo_url?: string | null
  citation_info?: string | null
  repository_url?: string | null
  documentation_url?: string | null
}

export interface ServiceInfo {
  id: string
  display_name: string
  version: string
  description?: string | null
  model_metadata?: ModelMetadata | null
  period_type?: string | null
  min_prediction_periods?: number | null
  max_prediction_periods?: number | null
  allow_free_additional_continuous_covariates?: boolean
  required_covariates?: string[]
  requires_geo?: boolean
}

export interface SystemInfo {
  current_time: string
  timezone: string
  python_version: string
  platform: string
  hostname: string
}

export interface SystemApp {
  name: string
  version: string
  prefix: string
  description?: string
  author?: string
  entry: string
  is_package: boolean
}

export interface HealthStatus {
  status?: string
  checks?: Record<string, { state?: string; message?: string }>
  [key: string]: unknown
}

export interface ConfigItem {
  id: string
  created_at: string
  updated_at: string
  tags: string[]
  name: string
  data: Record<string, unknown>
}

export interface ConfigInput {
  name: string
  data: Record<string, unknown>
  tags?: string[]
}

export interface ArtifactMetadata {
  status?: string
  config_id?: string | null
  started_at?: string | null
  completed_at?: string | null
  duration_seconds?: number | null
  exit_code?: number | null
  stdout?: string | null
  stderr?: string | null
  [key: string]: unknown
}

export interface ArtifactData {
  type?: string
  metadata?: ArtifactMetadata
  content?: unknown
  content_type?: string | null
  content_size?: number | null
}

export interface Artifact {
  id: string
  created_at: string
  updated_at: string
  tags: string[]
  data: ArtifactData
  parent_id: string | null
  level: number
  level_label?: string
  hierarchy?: string
  children?: Artifact[]
}

export interface DataFrameContent {
  columns: string[]
  data: unknown[][]
}

export type JobStatus =
  | 'pending'
  | 'running'
  | 'completed'
  | 'failed'
  | 'canceled'
  | string

export interface Job {
  id: string
  status: JobStatus
  submitted_at: string | null
  started_at: string | null
  finished_at: string | null
  error: string | null
  error_traceback: string | null
}

export type DiagnosticSeverity = 'error' | 'warning' | 'info'

export interface ValidationDiagnostic {
  severity: DiagnosticSeverity
  code: string
  message: string
  field?: string
}

export interface ValidationResult {
  valid: boolean
  diagnostics: ValidationDiagnostic[]
}

export interface MLJobResponse {
  job_id: string
  artifact_id: string
  message: string
}

export interface TrainPayload {
  config_id: string
  data: DataFrameContent
  geo?: unknown
}

export interface PredictPayload {
  artifact_id: string
  historic: DataFrameContent
  future: DataFrameContent
  geo?: unknown
}

// --- OpenAPI (subset used by the endpoint browser) ------------------------

export interface OpenApiOperation {
  summary?: string
  description?: string
  tags?: string[]
  operationId?: string
  deprecated?: boolean
}

export interface OpenApiSpec {
  openapi: string
  info: { title: string; version: string; description?: string }
  paths: Record<string, Record<string, OpenApiOperation>>
}
