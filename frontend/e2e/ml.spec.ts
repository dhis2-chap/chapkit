import { test, expect } from './fixtures'
import { newApi, createConfig, trainModel } from './helpers'

test.beforeAll(async () => {
  // Ensure a config and a trained model exist for the Train/Predict screens.
  const api = await newApi()
  const cfg = await createConfig(api, 'ml-spec-config')
  await trainModel(api, cfg.id)
  await api.dispose()
})

test('predict via the UI', async ({ page }) => {
  await page.goto('/#/predict')
  await page.getByRole('button', { name: 'Generate', exact: true }).click()
  await expect(page.locator('table tbody tr').first()).toBeVisible()
  await page.getByRole('button', { name: 'Validate' }).click()
  const predict = page.getByRole('button', { name: 'Predict', exact: true })
  await expect(predict).toBeEnabled()
  await predict.click()
  await expect(page.locator('[data-sonner-toast]').filter({ hasText: 'submitted' })).toBeVisible()
})

test('dataframe table: filter, sort, and download menu', async ({ page }) => {
  await page.goto('/#/train')
  await page.getByRole('button', { name: 'Generate', exact: true }).click()
  await expect(page.locator('table tbody tr').first()).toBeVisible()

  await page.getByPlaceholder(/Filter rows/).fill('location_0')
  await expect(page.locator('table tbody tr').first()).toContainText('location_0')
  await page.getByPlaceholder(/Filter rows/).fill('')

  await page.getByRole('columnheader', { name: /disease_cases/ }).click()

  await page.getByRole('button', { name: 'Download' }).click()
  await expect(page.getByRole('menuitem', { name: 'CSV' })).toBeVisible()
  await page.keyboard.press('Escape')
})

test('apply from dry run: Train now submits straight from the success alert', async ({ page }) => {
  await page.goto('/#/train')
  await page.getByRole('button', { name: 'Generate', exact: true }).click()
  await expect(page.locator('table tbody tr').first()).toBeVisible()
  await page.getByRole('button', { name: 'Validate' }).click()

  const trainNow = page.getByRole('button', { name: 'Train now', exact: true })
  await expect(trainNow).toBeVisible()
  await trainNow.click()
  await expect(page.locator('[data-sonner-toast]').filter({ hasText: 'submitted' })).toBeVisible()
})

test('Train JSON pane is read-only until Edit is toggled', async ({ page }) => {
  await page.goto('/#/train')
  await page.getByRole('button', { name: 'Generate', exact: true }).click()
  await expect(page.locator('table tbody tr').first()).toBeVisible()

  await page.getByRole('tab', { name: 'JSON', exact: true }).click()
  await expect(page.locator('.cm-editor')).toBeVisible()
  await expect(page.locator('.cm-content[contenteditable="true"]')).toHaveCount(0)

  await page.getByRole('button', { name: 'Edit', exact: true }).click()
  await expect(page.locator('.cm-content[contenteditable="true"]')).toHaveCount(1)
})
