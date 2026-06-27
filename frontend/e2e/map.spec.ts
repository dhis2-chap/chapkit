import { test, expect } from './fixtures'

test('map renders a MapLibre map with the generated geometry', async ({ page }) => {
  await page.goto('/#/map')
  await expect(page.locator('.maplibregl-map')).toBeVisible()
  // The value column defaults to disease_cases (driven by the self-describing schema).
  await expect(page.getByText('disease_cases').first()).toBeVisible()
})

test('scrubbing the time slider advances the period', async ({ page }) => {
  await page.goto('/#/map')
  await expect(page.locator('.maplibregl-map')).toBeVisible()

  const periodLabel = page.locator('span.tabular-nums').first()
  const before = await periodLabel.textContent()

  const slider = page.locator('input[type=range]')
  const max = Number(await slider.getAttribute('max'))
  await slider.fill(String(max))

  await expect(periodLabel).not.toHaveText(before ?? '')
})

test('the play control toggles to pause', async ({ page }) => {
  await page.goto('/#/map')
  await expect(page.locator('.maplibregl-map')).toBeVisible()

  await page.getByRole('button', { name: 'Play animation' }).click()
  await expect(page.getByRole('button', { name: 'Pause animation' })).toBeVisible()
})
