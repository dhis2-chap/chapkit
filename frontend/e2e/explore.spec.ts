import { test, expect } from '@playwright/test'

test('endpoints page lists operations parsed from openapi', async ({ page }) => {
  await page.goto('/#/api')
  await expect(page.getByRole('heading', { name: 'Endpoints', exact: true })).toBeVisible()
  await expect(page.getByText('/api/v1/configs').first()).toBeVisible()
})

test('system page renders runtime info', async ({ page }) => {
  await page.goto('/#/system')
  await expect(page.getByRole('heading', { name: 'System', exact: true })).toBeVisible()
})
