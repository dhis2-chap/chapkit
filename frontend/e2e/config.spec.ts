import { test, expect } from './fixtures'
import { newApi, createConfig } from './helpers'

test('edit a config and save', async ({ page }) => {
  const api = await newApi()
  const cfg = await createConfig(api, 'edit-me', { prediction_periods: 3 })
  await api.dispose()

  await page.goto(`/#/configs/${cfg.id}`)
  await page.getByRole('button', { name: 'Edit', exact: true }).click()
  await page.locator('#config-name').fill('edited-config')
  await page.getByRole('button', { name: 'Save changes' }).click()

  // Back to the read-only view, showing the new name and the Edit button again.
  await expect(page.getByText('edited-config').first()).toBeVisible()
  await expect(page.getByRole('button', { name: 'Edit', exact: true })).toBeVisible()
})

test('delete a config', async ({ page }) => {
  const api = await newApi()
  await createConfig(api, 'keep-me')
  const cfg = await createConfig(api, 'delete-me')
  await api.dispose()

  await page.goto(`/#/configs/${cfg.id}`)
  await page.getByRole('button', { name: 'Delete', exact: true }).click()
  await page.getByRole('dialog').getByRole('button', { name: 'Delete', exact: true }).click()

  await expect(page.getByRole('cell', { name: 'delete-me' })).toHaveCount(0)
})

test('config editor offers schema-aware autocomplete', async ({ page }) => {
  const api = await newApi()
  const cfg = await createConfig(api, 'autocomplete-me')
  await api.dispose()

  await page.goto(`/#/configs/${cfg.id}`)
  await page.getByRole('button', { name: 'Edit', exact: true }).click()
  await page.getByRole('tab', { name: 'JSON', exact: true }).click()
  await expect(page.locator('.cm-content[contenteditable="true"]')).toBeVisible()
  await page.waitForTimeout(1000) // allow the lazy schema chunk to load

  const editor = page.locator('.cm-content[contenteditable="true"]').first()
  await editor.click()
  await page.keyboard.press('ControlOrMeta+a')
  await page.keyboard.type('{\n  "', { delay: 30 })
  await expect(page.locator('.cm-tooltip-autocomplete')).toContainText('prediction_periods')
})

test('config editor flags a value that violates the schema', async ({ page }) => {
  const api = await newApi()
  const cfg = await createConfig(api, 'validate-me')
  await api.dispose()

  await page.goto(`/#/configs/${cfg.id}`)
  await page.getByRole('button', { name: 'Edit', exact: true }).click()
  await page.getByRole('tab', { name: 'JSON', exact: true }).click()
  await expect(page.locator('.cm-content[contenteditable="true"]')).toBeVisible()
  await page.waitForTimeout(1000)

  const editor = page.locator('.cm-content[contenteditable="true"]').first()
  await editor.click()
  await page.keyboard.press('ControlOrMeta+a')
  await page.keyboard.type('{ "prediction_periods": "not-a-number" }', { delay: 10 })
  await expect(page.locator('.cm-lint-marker-error')).toBeVisible()
})

test('New config requires a name', async ({ page }) => {
  await page.goto('/#/configs')
  await page.getByRole('button', { name: 'New config' }).click()
  await page.locator('#config-name').fill('')
  await page.getByRole('button', { name: 'Create config' }).click()
  await expect(page.getByText('Name is required.')).toBeVisible()
})
