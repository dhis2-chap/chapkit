import { test, expect } from './fixtures'

test('monitoring appears in the sidebar when the service exposes /metrics', async ({ page }) => {
  await page.goto('/#/')
  const sidebar = page.locator('[data-slot="sidebar"]')
  await expect(sidebar.getByRole('link', { name: 'Monitoring', exact: true })).toBeVisible()
})

test('monitoring screen renders live tiles and the raw explorer', async ({ page }) => {
  await page.goto('/#/monitoring')
  await expect(page.getByRole('heading', { name: 'Monitoring' })).toBeVisible()
  // Cross-platform metric tiles.
  await expect(page.getByText('Requests / sec')).toBeVisible()
  await expect(page.getByText('Active requests')).toBeVisible()
  // The first poll has populated the raw metrics explorer.
  await expect(page.getByText('python_gc_collections_total')).toBeVisible()
})

test('the raw metrics explorer filters families', async ({ page }) => {
  await page.goto('/#/monitoring')
  await expect(page.getByText('python_gc_collections_total')).toBeVisible()

  await page.getByPlaceholder('Filter…').fill('http_server')
  await expect(page.getByText('python_gc_collections_total')).toHaveCount(0)
  await expect(page.getByText('http_server_active_requests')).toBeVisible()
})
