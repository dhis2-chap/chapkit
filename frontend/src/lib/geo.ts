// Minimal GeoJSON types, bbox derivation, and a sequential color ramp for the
// MapLibre choropleth. Geometry projection is handled by MapLibre itself.

export type BBox = [number, number, number, number] // [west, south, east, north]

export interface Geometry {
  type: string
  coordinates?: unknown
  geometries?: Geometry[]
}

export interface Feature {
  type: 'Feature'
  geometry: Geometry | null
  properties: Record<string, unknown> | null
}

export interface FeatureCollection {
  type: 'FeatureCollection'
  features: Feature[]
  bbox?: number[]
}

/** Narrow an unknown value to a FeatureCollection after a minimal shape check. */
export function asFeatureCollection(value: unknown): FeatureCollection | null {
  if (
    typeof value === 'object' &&
    value !== null &&
    (value as FeatureCollection).type === 'FeatureCollection' &&
    Array.isArray((value as FeatureCollection).features)
  ) {
    return value as FeatureCollection
  }
  return null
}

/** Yield every [lon, lat] position from an arbitrarily nested coordinate array. */
function* positions(coordinates: unknown): Generator<[number, number]> {
  if (
    Array.isArray(coordinates) &&
    coordinates.length >= 2 &&
    typeof coordinates[0] === 'number' &&
    typeof coordinates[1] === 'number'
  ) {
    yield [coordinates[0], coordinates[1]]
    return
  }
  if (Array.isArray(coordinates)) {
    for (const item of coordinates) yield* positions(item)
  }
}

/** Every position of a geometry, recursing into GeometryCollections. */
function* geometryPositions(geometry: Geometry | null): Generator<[number, number]> {
  if (!geometry) return
  if (geometry.type === 'GeometryCollection') {
    for (const sub of geometry.geometries ?? []) yield* geometryPositions(sub)
    return
  }
  yield* positions(geometry.coordinates)
}

/** Derive a bbox from features when the collection does not carry one. */
export function bboxOfFeatures(features: Feature[]): BBox | null {
  let west = Infinity
  let south = Infinity
  let east = -Infinity
  let north = -Infinity
  for (const feature of features) {
    for (const [lon, lat] of geometryPositions(feature.geometry)) {
      if (lon < west) west = lon
      if (lon > east) east = lon
      if (lat < south) south = lat
      if (lat > north) north = lat
    }
  }
  if (!Number.isFinite(west)) return null
  return [west, south, east, north]
}

// Sequential ColorBrewer-style OrRd ramp (light -> dark), interpolated in RGB.
const RAMP: [number, number, number][] = [
  [254, 240, 217],
  [253, 204, 138],
  [252, 141, 89],
  [227, 74, 51],
  [179, 0, 0],
]

/** Map a normalized value in [0, 1] to a hex color along the sequential ramp. */
export function rampColor(t: number): string {
  const clamped = Math.max(0, Math.min(1, Number.isFinite(t) ? t : 0))
  const scaled = clamped * (RAMP.length - 1)
  const index = Math.min(RAMP.length - 2, Math.floor(scaled))
  const frac = scaled - index
  const [r1, g1, b1] = RAMP[index]
  const [r2, g2, b2] = RAMP[index + 1]
  const mix = (a: number, b: number) => Math.round(a + (b - a) * frac)
  return `rgb(${mix(r1, r2)}, ${mix(g1, g2)}, ${mix(b1, b2)})`
}
