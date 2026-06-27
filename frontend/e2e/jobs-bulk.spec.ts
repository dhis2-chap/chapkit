import { test, expect } from './fixtures'
import { newApi, createConfig } from './helpers'
import type { APIRequestContext } from '@playwright/test'

/** Submit a $train job and return its job id (without waiting for completion). */
async function submitTrainJob(api: APIRequestContext, configId: string, seed: number): Promise<string> {
  const sample = await (
    await api.get(
      `/api/v1/ml/$generate-sample-data?kind=train&config_id=${configId}&num_locations=2&num_periods=12&seed=${seed}`,
    )
  ).json()
  const job = await (
    await api.post('/api/v1/ml/$train', { data: { config_id: configId, data: sample.data } })
  ).json()
  return job.job_id
}

let configId = ''

test.beforeAll(async () => {
  const api = await newApi()
  const cfg = await createConfig(api, 'jobs-bulk-config')
  configId = cfg.id
  await api.dispose()
})

test('jobs list has selection checkboxes and a group-by-status toggle', async ({ page }) => {
  const api = await newApi()
  await submitTrainJob(api, configId, 11)
  await api.dispose()

  await page.goto('/#/jobs')
  const selectAll = page.getByRole('checkbox', { name: 'Select all jobs' })
  await expect(selectAll).toBeVisible()
  await selectAll.check()
  await expect(page.getByText(/\d+ selected/)).toBeVisible()
  await expect(page.getByRole('button', { name: 'Remove selected' })).toBeVisible()

  // Grouping inserts a full-width status header row.
  await page.getByRole('button', { name: 'Group by status' }).click()
  await expect(page.locator('td[colspan]').first()).toBeVisible()
})

test('bulk remove deletes exactly the selected jobs', async ({ page }) => {
  const api = await newApi()
  const idA = await submitTrainJob(api, configId, 21)
  const idB = await submitTrainJob(api, configId, 22)
  await api.dispose()

  await page.goto('/#/jobs')
  await page.getByRole('row', { name: new RegExp(idA) }).getByRole('checkbox').check()
  await page.getByRole('row', { name: new RegExp(idB) }).getByRole('checkbox').check()
  await expect(page.getByText('2 selected')).toBeVisible()

  page.once('dialog', (dialog) => dialog.accept())
  await page.getByRole('button', { name: 'Remove selected' }).click()

  await expect(page.getByText(idA)).toHaveCount(0)
  await expect(page.getByText(idB)).toHaveCount(0)
})
