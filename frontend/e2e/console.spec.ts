import { request as playwrightRequest } from '@playwright/test'
import type { APIRequestContext } from '@playwright/test'
import { test, expect } from './fixtures'

const baseURL = 'http://127.0.0.1:9099'

async function pollJob(api: APIRequestContext, jobId: string): Promise<string> {
  for (let i = 0; i < 60; i++) {
    const job = await (await api.get(`/api/v1/jobs/${jobId}`)).json()
    if (['completed', 'failed', 'error'].includes(job.status)) return job.status
    await new Promise((r) => setTimeout(r, 250))
  }
  return 'timeout'
}

// Seed one config -> trained model -> prediction via the API, so the read-heavy
// screens (Artifacts chart, Predict model picker) have content to render.
let predictionArtifactId = ''

test.beforeAll(async () => {
  const api = await playwrightRequest.newContext({ baseURL })
  const cfg = await (await api.post('/api/v1/configs', { data: { name: 'e2e-seed', data: {} } })).json()
  const trainSample = await (
    await api.get(`/api/v1/ml/$generate-sample-data?kind=train&config_id=${cfg.id}&num_locations=3&num_periods=12&seed=1`)
  ).json()
  const trainJob = await (await api.post('/api/v1/ml/$train', { data: { config_id: cfg.id, data: trainSample.data } })).json()
  expect(await pollJob(api, trainJob.job_id)).toBe('completed')

  const predictSample = await (await api.get('/api/v1/ml/$generate-sample-data?kind=predict&num_locations=3&num_periods=6&seed=2')).json()
  const predictJob = await (
    await api.post('/api/v1/ml/$predict', {
      data: { artifact_id: trainJob.artifact_id, historic: predictSample.historic, future: predictSample.future },
    })
  ).json()
  expect(await pollJob(api, predictJob.job_id)).toBe('completed')

  const artifacts = await (await api.get('/api/v1/artifacts')).json()
  predictionArtifactId = artifacts.find((a: { level: number }) => a.level === 1).id
  await api.dispose()
})

test('overview renders the live service identity and health', async ({ page }) => {
  await page.goto('/#/')
  await expect(page.getByRole('heading', { name: 'Chapkit e2e fixture' })).toBeVisible()
  await expect(page.getByText('healthy').first()).toBeVisible()
})

test('sidebar navigates to every screen', async ({ page }) => {
  await page.goto('/#/')
  // Scope to the sidebar: the Overview "Quick links" card also links to some pages.
  const sidebar = page.locator('[data-slot="sidebar"]')
  for (const label of ['Configs', 'Artifacts', 'Jobs', 'Train', 'Predict', 'Endpoints', 'System']) {
    await sidebar.getByRole('link', { name: label, exact: true }).click()
    await expect(page.getByRole('heading', { name: label, exact: true })).toBeVisible()
  }
})

test('create a config through the UI', async ({ page }) => {
  await page.goto('/#/configs')
  await page.getByRole('button', { name: 'New config' }).click()
  await page.locator('#config-name').fill('made-in-e2e')
  await page.getByRole('button', { name: 'Create config' }).click()
  await expect(page.getByText('made-in-e2e').first()).toBeVisible()
})

test('train: generate, validate, then submit a job', async ({ page }) => {
  await page.goto('/#/train')
  await page.getByRole('button', { name: 'Generate', exact: true }).click()
  await expect(page.locator('table tbody tr').first()).toBeVisible()
  await page.getByRole('button', { name: 'Validate' }).click()
  const train = page.getByRole('button', { name: 'Train', exact: true })
  await expect(train).toBeEnabled()
  await train.click()
  await expect(page.locator('[data-sonner-toast]').filter({ hasText: 'submitted' })).toBeVisible()
})

test('jobs lists submitted jobs with full ULIDs', async ({ page }) => {
  await page.goto('/#/jobs')
  const firstId = page.locator('table tbody td.font-mono').first()
  await expect(firstId).toBeVisible()
  expect((await firstId.innerText()).trim()).toMatch(/^01[0-9A-Z]{24}$/)
})

test('a prediction artifact renders its chart', async ({ page }) => {
  await page.goto(`/#/artifacts/${predictionArtifactId}`)
  await page.getByRole('tab', { name: 'Chart', exact: true }).click()
  await expect(page.locator('.recharts-surface').first()).toBeVisible()
})
