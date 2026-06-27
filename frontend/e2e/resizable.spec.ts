import { test, expect } from './fixtures'
import { newApi, createConfig } from './helpers'

// The master/detail split only renders once a config exists.
test.beforeAll(async () => {
  const api = await newApi()
  await createConfig(api, 'resizable-spec-config')
  await api.dispose()
})

test('configs master/detail is drag/keyboard resizable and persists the split', async ({ page }) => {
  await page.goto('/#/configs')
  const handle = page.locator('[data-slot="resizable-handle"]')
  await expect(handle).toBeVisible()
  const before = Number(await handle.getAttribute('aria-valuenow'))

  await handle.focus()
  await page.keyboard.press('ArrowLeft')
  await page.keyboard.press('ArrowLeft')
  const after = Number(await handle.getAttribute('aria-valuenow'))
  expect(after).toBeLessThan(before)

  // The chosen split is saved (autoSaveId, debounced) and restored after a reload.
  await expect
    .poll(() =>
      page.evaluate(() =>
        localStorage.getItem('react-resizable-panels:console:configs-split'),
      ),
    )
    .toContain('"layout"')
  await page.reload()
  const reloaded = page.locator('[data-slot="resizable-handle"]')
  await expect(reloaded).toBeVisible()
  expect(Number(await reloaded.getAttribute('aria-valuenow'))).toBeLessThan(before)
})

test('master/detail stacks with no resize handle on narrow viewports', async ({ page }) => {
  await page.setViewportSize({ width: 600, height: 900 })
  await page.goto('/#/configs')
  await expect(page.locator('table tbody tr').first()).toBeVisible()
  await expect(page.locator('[data-slot="resizable-handle"]')).toHaveCount(0)
})
