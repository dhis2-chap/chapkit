import { test, expect } from './fixtures'
import { newApi } from './helpers'

test('create a config through the schema-driven form', async ({ page }) => {
  await page.goto('/#/configs')
  await page.getByRole('button', { name: 'New config' }).click()
  await page.locator('#config-name').fill('schema-form-e2e')

  // Typed inputs generated from the config JSON Schema.
  await page.getByRole('spinbutton', { name: 'Prediction Periods' }).fill('9')
  await page.getByRole('checkbox', { name: /Region Seasonal/ }).check()
  await page.getByRole('button', { name: 'Create config' }).click()

  await expect(page.getByText('schema-form-e2e').first()).toBeVisible()

  // The created config carries the values entered through the form.
  const api = await newApi()
  const configs: { name: string; data: Record<string, unknown> }[] = await (
    await api.get('/api/v1/configs')
  ).json()
  await api.dispose()
  const created = configs.find((config) => config.name === 'schema-form-e2e')
  expect(created?.data.prediction_periods).toBe(9)
  expect(created?.data.region_seasonal).toBe(true)
})

test('form edits stay in sync with the JSON tab', async ({ page }) => {
  await page.goto('/#/configs')
  await page.getByRole('button', { name: 'New config' }).click()
  await page.getByRole('spinbutton', { name: 'Prediction Periods' }).fill('12')

  await page.getByRole('tab', { name: 'JSON', exact: true }).click()
  await expect(page.locator('.cm-content')).toContainText(/"prediction_periods":\s*12/)
})
