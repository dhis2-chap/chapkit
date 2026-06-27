import { test, expect } from './fixtures'

test('the primary sidebar is keyboard-resizable and persists its width', async ({ page }) => {
  await page.goto('/#/')
  const resizer = page.locator('[data-slot="sidebar-resizer"]')
  await expect(resizer).toBeVisible()
  const before = Number(await resizer.getAttribute('aria-valuenow'))

  await resizer.focus()
  await page.keyboard.press('ArrowRight')
  await page.keyboard.press('ArrowRight')
  const after = Number(await resizer.getAttribute('aria-valuenow'))
  expect(after).toBeGreaterThan(before)

  // Width is persisted and restored on reload.
  await page.reload()
  const reloaded = page.locator('[data-slot="sidebar-resizer"]')
  await expect(reloaded).toHaveAttribute('aria-valuenow', String(after))
})

test('the sidebar resizer hides when the sidebar is collapsed', async ({ page }) => {
  await page.goto('/#/')
  await expect(page.locator('[data-slot="sidebar-resizer"]')).toBeVisible()
  await page.getByRole('button', { name: 'Toggle Sidebar' }).click()
  await expect(page.locator('[data-slot="sidebar-resizer"]')).toHaveCount(0)
})
