// Minimal parser for the Prometheus text exposition format served at /metrics.
// We only need scalar samples + family type/help; histograms surface as their
// derived _count/_sum/_bucket series (each its own metric name).

export interface MetricSample {
  name: string
  labels: Record<string, string>
  value: number
}

export interface FamilyMeta {
  type: string
  help: string
}

export interface ParsedMetrics {
  samples: MetricSample[]
  meta: Record<string, FamilyMeta>
}

/** Parse the label set inside `{...}` into a plain object. */
function parseLabels(inner: string): Record<string, string> {
  const labels: Record<string, string> = {}
  const re = /([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"/g
  let match: RegExpExecArray | null
  while ((match = re.exec(inner)) !== null) {
    labels[match[1]] = match[2].replace(/\\(["\\n])/g, (_, c) => (c === 'n' ? '\n' : c))
  }
  return labels
}

/** Parse Prometheus exposition text into samples plus per-family type/help. */
export function parsePrometheus(text: string): ParsedMetrics {
  const samples: MetricSample[] = []
  const meta: Record<string, FamilyMeta> = {}

  for (const raw of text.split('\n')) {
    const line = raw.trim()
    if (line === '') continue

    if (line.startsWith('#')) {
      const parts = line.split(/\s+/)
      const kind = parts[1]
      const name = parts[2]
      if (!name) continue
      const rest = line.slice(line.indexOf(name) + name.length).trim()
      if (kind === 'TYPE') meta[name] = { type: rest, help: meta[name]?.help ?? '' }
      else if (kind === 'HELP') meta[name] = { type: meta[name]?.type ?? 'untyped', help: rest }
      continue
    }

    const braceStart = line.indexOf('{')
    let name: string
    let labels: Record<string, string>
    let rest: string
    if (braceStart !== -1) {
      const braceEnd = line.lastIndexOf('}')
      name = line.slice(0, braceStart)
      labels = parseLabels(line.slice(braceStart + 1, braceEnd))
      rest = line.slice(braceEnd + 1).trim()
    } else {
      const space = line.indexOf(' ')
      if (space === -1) continue
      name = line.slice(0, space)
      labels = {}
      rest = line.slice(space + 1).trim()
    }

    const value = Number(rest.split(/\s+/)[0])
    if (Number.isFinite(value)) samples.push({ name, labels, value })
  }

  return { samples, meta }
}

/** Sum every sample with the given metric name (collapsing labels). */
export function sumByName(parsed: ParsedMetrics, name: string): number | null {
  let total = 0
  let found = false
  for (const sample of parsed.samples) {
    if (sample.name === name) {
      total += sample.value
      found = true
    }
  }
  return found ? total : null
}
