import { test, expect } from './fixtures'
import { newApi, createConfig } from './helpers'

// A submitted (eventually completed) job so the Jobs screen has a row to deep-link.
let jobId = ''

test.beforeAll(async () => {
  const api = await newApi()
  const cfg = await createConfig(api, 'jobs-spec-config')
  const sample = await (
    await api.get(
      `/api/v1/ml/$generate-sample-data?kind=train&config_id=${cfg.id}&num_locations=2&num_periods=12&seed=3`,
    )
  ).json()
  const job = await (
    await api.post('/api/v1/ml/$train', { data: { config_id: cfg.id, data: sample.data } })
  ).json()
  jobId = job.job_id
  await api.dispose()
})

test('deep-link opens the matching job detail dialog', async ({ page }) => {
  await page.goto(`/#/jobs/${jobId}`)
  const dialog = page.getByRole('dialog')
  await expect(dialog).toBeVisible()
  await expect(dialog).toContainText(jobId)
})

test('clicking a job row deep-links the URL and Escape closes it', async ({ page }) => {
  await page.goto('/#/jobs')
  await page.locator('table tbody tr').first().click()
  await expect(page).toHaveURL(/#\/jobs\/01[0-9A-Z]{24}$/)
  await expect(page.getByRole('dialog')).toBeVisible()

  await page.keyboard.press('Escape')
  await expect(page).toHaveURL(/#\/jobs$/)
  await expect(page.getByRole('dialog')).toHaveCount(0)
})
