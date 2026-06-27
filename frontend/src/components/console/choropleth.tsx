// MapLibre choropleth: the service's GeoJSON regions over an OpenFreeMap basemap
// (matching dhis2/open-climate-service), colored by a per-location value.
import { useEffect, useRef } from 'react'
import maplibregl from 'maplibre-gl'
import type { ExpressionSpecification, GeoJSONSource } from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'

import type { BBox, FeatureCollection } from '@/lib/geo'
import { rampColor } from '@/lib/geo'

// Free, no-API-key vector basemap (same as dhis2/open-climate-service).
const BASEMAP_STYLE = 'https://tiles.openfreemap.org/styles/positron'
const NO_DATA_COLOR = '#cbd5e1'
const EMPTY: FeatureCollection = { type: 'FeatureCollection', features: [] }

export interface HoverInfo {
  id: string
  value: number | null
}

/** Data-driven fill color: a ramp over [min, max], grey where the value is absent. */
function fillColor(min: number, max: number): ExpressionSpecification {
  if (!(max > min)) return rampColor(0.5) as unknown as ExpressionSpecification
  const stops = [0, 0.25, 0.5, 0.75, 1]
  const interpolate: unknown[] = ['interpolate', ['linear'], ['get', 'value']]
  for (const t of stops) interpolate.push(min + (max - min) * t, rampColor(t))
  return ['case', ['has', 'value'], interpolate, NO_DATA_COLOR] as unknown as ExpressionSpecification
}

function fitTo(map: maplibregl.Map, bbox: BBox): void {
  const [west, south, east, north] = bbox
  map.fitBounds(
    [
      [west, south],
      [east, north],
    ],
    { padding: 48, duration: 0, maxZoom: 9 },
  )
}

/** Render features over a basemap, colored by feature.properties.value in [min, max]. */
export function Choropleth({
  data,
  bbox,
  min,
  max,
  onHover,
}: {
  data: FeatureCollection
  bbox: BBox
  min: number
  max: number
  onHover: (info: HoverInfo | null) => void
}) {
  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<maplibregl.Map | null>(null)
  const loadedRef = useRef(false)

  // Latest props for the async style-load handler to read.
  const stateRef = useRef({ data, bbox, min, max, onHover })
  stateRef.current = { data, bbox, min, max, onHover }

  // Initialize the map once.
  useEffect(() => {
    if (!containerRef.current) return
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: BASEMAP_STYLE,
      attributionControl: { compact: true },
    })
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right')
    mapRef.current = map

    // Keep the canvas sized to its container (sidebar/panel resizes, etc.).
    const resizeObserver = new ResizeObserver(() => map.resize())
    resizeObserver.observe(containerRef.current)

    map.on('load', () => {
      loadedRef.current = true
      const current = stateRef.current
      map.addSource('locations', { type: 'geojson', data: current.data })
      map.addLayer({
        id: 'locations-fill',
        type: 'fill',
        source: 'locations',
        paint: { 'fill-color': fillColor(current.min, current.max), 'fill-opacity': 0.78 },
      })
      map.addLayer({
        id: 'locations-outline',
        type: 'line',
        source: 'locations',
        paint: { 'line-color': '#475569', 'line-width': 0.6 },
      })
      fitTo(map, current.bbox)

      map.on('mousemove', 'locations-fill', (event) => {
        const feature = event.features?.[0]
        if (!feature) return
        map.getCanvas().style.cursor = 'pointer'
        const props = feature.properties ?? {}
        const value = typeof props.value === 'number' ? props.value : null
        stateRef.current.onHover({ id: String(props.id), value })
      })
      map.on('mouseleave', 'locations-fill', () => {
        map.getCanvas().style.cursor = ''
        stateRef.current.onHover(null)
      })
    })

    return () => {
      loadedRef.current = false
      resizeObserver.disconnect()
      map.remove()
      mapRef.current = null
    }
  }, [])

  // Push new per-period feature values.
  useEffect(() => {
    const map = mapRef.current
    if (!map || !loadedRef.current) return
    const source = map.getSource('locations') as GeoJSONSource | undefined
    if (source) source.setData(data as unknown as Parameters<GeoJSONSource['setData']>[0])
  }, [data])

  // Re-color when the value range changes (e.g. a different column).
  useEffect(() => {
    const map = mapRef.current
    if (!map || !loadedRef.current || !map.getLayer('locations-fill')) return
    map.setPaintProperty('locations-fill', 'fill-color', fillColor(min, max))
  }, [min, max])

  // Reframe when the geometry's extent changes.
  useEffect(() => {
    const map = mapRef.current
    if (!map || !loadedRef.current) return
    fitTo(map, bbox)
  }, [bbox])

  return <div ref={containerRef} className="h-full w-full" />
}

/** A horizontal gradient legend matching the choropleth ramp, with min/max labels. */
export function ChoroplethLegend({
  label,
  min,
  max,
}: {
  label: string
  min: number
  max: number
}) {
  const gradient = `linear-gradient(to right, ${[0, 0.25, 0.5, 0.75, 1]
    .map((t) => rampColor(t))
    .join(', ')})`
  const format = (value: number) =>
    Number.isFinite(value) ? value.toLocaleString(undefined, { maximumFractionDigits: 2 }) : '—'

  return (
    <div className="space-y-1">
      <p className="text-xs font-medium text-muted-foreground">{label}</p>
      <div className="h-3 w-full rounded-sm border" style={{ background: gradient }} />
      <div className="flex justify-between text-xs text-muted-foreground">
        <span>{format(min)}</span>
        <span>{format(max)}</span>
      </div>
    </div>
  )
}

export { EMPTY as EMPTY_FEATURE_COLLECTION }
