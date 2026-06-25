import { request, expect } from '@playwright/test'
import type { APIRequestContext } from '@playwright/test'

export const baseURL = 'http://127.0.0.1:9099'

/** Fresh API context bound to the fixture service. */
export async function newApi(): Promise<APIRequestContext> {
  return request.newContext({ baseURL })
}

/** Poll a job until it reaches a terminal state. */
export async function pollJob(api: APIRequestContext, jobId: string): Promise<string> {
  for (let i = 0; i < 60; i++) {
    const job = await (await api.get(`/api/v1/jobs/${jobId}`)).json()
    if (['completed', 'failed', 'error'].includes(job.status)) return job.status
    await new Promise((r) => setTimeout(r, 250))
  }
  return 'timeout'
}

/** Create a config via the API and return it. */
export async function createConfig(
  api: APIRequestContext,
  name: string,
  data: Record<string, unknown> = {},
): Promise<{ id: string; name: string }> {
  return (await api.post('/api/v1/configs', { data: { name, data } })).json()
}

/** Train a model for a config via the API; returns the trained artifact id. */
export async function trainModel(api: APIRequestContext, configId: string, seed = 1): Promise<string> {
  const sample = await (
    await api.get(`/api/v1/ml/$generate-sample-data?kind=train&config_id=${configId}&num_locations=3&num_periods=12&seed=${seed}`)
  ).json()
  const job = await (await api.post('/api/v1/ml/$train', { data: { config_id: configId, data: sample.data } })).json()
  expect(await pollJob(api, job.job_id)).toBe('completed')
  return job.artifact_id
}
