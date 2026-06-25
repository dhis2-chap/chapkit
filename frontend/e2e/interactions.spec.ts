import { test, expect } from './fixtures'
import { newApi, createConfig, trainModel, pollJob } from './helpers'

test.beforeAll(async () => {
  const api = await newApi()
  const cfg = await createConfig(api, 'interactions-config')
  await trainModel(api, cfg.id)
  await api.dispose()
})

test('generator options sheet tunes and generates data', async ({ page }) => {
  await page.goto('/#/train')
  await page.getByRole('button', { name: /generator options/i }).click()
  await expect(page.getByText('Panel shape')).toBeVisible()
  await page.locator('#gen-locations').fill('4')
  await page.locator('#gen-seed').fill('5')
  await page.locator('#gen-include-geo').check()
  await page.getByRole('dialog').getByRole('button', { name: 'Generate', exact: true }).click()
  await expect(page.locator('table tbody tr').first()).toBeVisible()
})

test('dataframe table downloads and paginates', async ({ page }) => {
  await page.goto('/#/train')
  await page.getByRole('button', { name: 'Generate', exact: true }).click()
  await expect(page.locator('table tbody tr').first()).toBeVisible()

  await page.getByRole('button', { name: 'Download' }).click()
  const csv = page.waitForEvent('download')
  await page.getByRole('menuitem', { name: 'CSV' }).click()
  expect((await csv).suggestedFilename()).toBe('dataframe.csv')

  await page.getByRole('button', { name: 'Download' }).click()
  const schemaJson = page.waitForEvent('download')
  await page.getByRole('menuitem', { name: 'JSON + $schema' }).click()
  await schemaJson

  await page.getByRole('button', { name: 'Next' }).click()
  await expect(page.getByText(/Page 2 of/)).toBeVisible()
})

test('jobs: open detail and remove a job', async ({ page }) => {
  const api = await newApi()
  const cfg = await createConfig(api, 'jobs-config')
  await trainModel(api, cfg.id)
  await trainModel(api, cfg.id, 2)
  await api.dispose()

  await page.goto('/#/jobs')
  await expect(page.locator('table tbody tr').first()).toBeVisible()

  await page.locator('table tbody tr').first().click()
  await expect(page.getByRole('dialog')).toBeVisible()
  await page.keyboard.press('Escape')

  const before = await page.locator('table tbody tr').count()
  page.once('dialog', (d) => d.accept())
  await page.locator('table tbody tr').first().getByRole('button', { name: 'Remove' }).click()
  await expect(page.locator('table tbody tr')).toHaveCount(before - 1)
})

test('artifacts: expand the tree, switch tabs, and delete', async ({ page }) => {
  const api = await newApi()
  const cfg = await createConfig(api, 'artifacts-config')
  const trainedId = await trainModel(api, cfg.id)
  const sample = await (await api.get('/api/v1/ml/$generate-sample-data?kind=predict&num_locations=3&num_periods=6&seed=3')).json()
  const predictJob = await (
    await api.post('/api/v1/ml/$predict', {
      data: { artifact_id: trainedId, historic: sample.historic, future: sample.future },
    })
  ).json()
  expect(await pollJob(api, predictJob.job_id)).toBe('completed')
  const predictionId = predictJob.artifact_id
  await api.dispose()

  await page.goto('/#/artifacts')
  // Clicking a workspace row toggles its expansion (covers the tree expand path).
  await page.locator('[role="button"]', { hasText: 'ml_training_workspace' }).first().click()

  // Deep-link to the prediction: it auto-selects and shows its detail tabs.
  await page.goto(`/#/artifacts/${predictionId}`)
  await page.getByRole('tab', { name: 'Metadata', exact: true }).click()
  await page.getByRole('tab', { name: 'Raw', exact: true }).click()
  await page.getByRole('tab', { name: 'Table', exact: true }).click()

  await page.getByRole('button', { name: 'Delete', exact: true }).click()
  await page.getByRole('dialog').getByRole('button', { name: 'Delete', exact: true }).click()
  await expect(page.getByText('Select an artifact')).toBeVisible()
})
