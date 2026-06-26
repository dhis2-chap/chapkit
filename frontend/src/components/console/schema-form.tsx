// Render a config-data form from a JSON Schema: typed inputs for scalars, enums,
// and scalar arrays, with a graceful fallback note for anything more complex.
import { Plus, X } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'

type JsonObject = Record<string, unknown>

/** A property schema normalized into the bits the form needs. */
interface NormalizedField {
  type?: string
  enum?: (string | number)[]
  items?: JsonObject
  title?: string
  description?: string
  default?: unknown
  nullable: boolean
}

/** Turn a snake/camel key into a readable label as a last resort when no title exists. */
function humanizeKey(key: string): string {
  return key
    .replace(/[_-]+/g, ' ')
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .replace(/\b\w/g, (c) => c.toUpperCase())
}

/** Resolve a `#/$defs/Name` ref against the root schema's $defs. */
function resolveRef(ref: string, defs: JsonObject): JsonObject {
  const name = ref.replace(/^#\/\$defs\//, '')
  const target = defs[name]
  return target && typeof target === 'object' ? (target as JsonObject) : {}
}

/** Normalize a property schema, resolving $ref and unwrapping nullable anyOf/allOf/oneOf. */
function normalizeProp(raw: JsonObject, defs: JsonObject): NormalizedField {
  let prop = raw
  if (typeof prop.$ref === 'string') {
    prop = { ...resolveRef(prop.$ref, defs), ...prop }
  }

  let nullable = false
  for (const key of ['anyOf', 'oneOf', 'allOf'] as const) {
    const branches = prop[key]
    if (Array.isArray(branches)) {
      const subs = branches.map((b) =>
        b && typeof b === 'object' ? (b as JsonObject) : {},
      )
      nullable = subs.some((s) => s.type === 'null')
      const chosen = subs.find((s) => s.type !== 'null') ?? {}
      const resolved =
        typeof chosen.$ref === 'string' ? resolveRef(chosen.$ref, defs) : chosen
      prop = { ...resolved, ...prop }
      delete prop[key]
    }
  }

  return {
    type: typeof prop.type === 'string' ? prop.type : undefined,
    enum: Array.isArray(prop.enum) ? (prop.enum as (string | number)[]) : undefined,
    items:
      prop.items && typeof prop.items === 'object'
        ? (prop.items as JsonObject)
        : undefined,
    title: typeof prop.title === 'string' ? prop.title : undefined,
    description: typeof prop.description === 'string' ? prop.description : undefined,
    default: 'default' in prop ? prop.default : undefined,
    nullable,
  }
}

type FieldKind = 'string' | 'number' | 'integer' | 'boolean' | 'enum' | 'array' | 'unsupported'

/** Decide how to render a normalized field. */
function fieldKind(field: NormalizedField): FieldKind {
  if (field.enum) return 'enum'
  switch (field.type) {
    case 'boolean':
      return 'boolean'
    case 'integer':
      return 'integer'
    case 'number':
      return 'number'
    case 'string':
      return 'string'
    case 'array': {
      const itemType = field.items?.type
      if (itemType === 'string' || itemType === 'number' || itemType === 'integer') {
        return 'array'
      }
      return 'unsupported'
    }
    default:
      return 'unsupported'
  }
}

/** Whether the schema has at least one object property the form can show. */
export function schemaHasProperties(schema: JsonObject | undefined): boolean {
  const properties = schema?.properties
  return Boolean(
    properties &&
      typeof properties === 'object' &&
      Object.keys(properties).some((key) => key !== 'id'),
  )
}

/** A single schema-driven field row. */
function SchemaField({
  name,
  field,
  value,
  required,
  disabled,
  onChange,
}: {
  name: string
  field: NormalizedField
  value: unknown
  required: boolean
  disabled?: boolean
  onChange: (next: unknown) => void
}) {
  const kind = fieldKind(field)
  const label = field.title ?? humanizeKey(name)
  const inputId = `cfg-field-${name}`

  const labelRow = (
    <Label htmlFor={inputId} className="flex items-center gap-1">
      {label}
      {required ? <span className="text-destructive">*</span> : null}
    </Label>
  )
  const description = field.description ? (
    <p className="text-xs text-muted-foreground">{field.description}</p>
  ) : null

  if (kind === 'boolean') {
    return (
      <label
        htmlFor={inputId}
        className="flex items-start gap-2 rounded-md border p-3 text-sm"
      >
        <input
          id={inputId}
          type="checkbox"
          className="mt-0.5 size-4 accent-primary"
          checked={Boolean(value)}
          disabled={disabled}
          onChange={(event) => onChange(event.target.checked)}
        />
        <span>
          <span className="font-medium">{label}</span>
          {field.description ? (
            <span className="block text-xs text-muted-foreground">
              {field.description}
            </span>
          ) : null}
        </span>
      </label>
    )
  }

  if (kind === 'enum') {
    const current = value === undefined || value === null ? '' : String(value)
    return (
      <div className="space-y-1.5">
        {labelRow}
        <Select
          value={current}
          disabled={disabled}
          onValueChange={(next) => onChange(next)}
        >
          <SelectTrigger id={inputId} className="w-full">
            <SelectValue placeholder="Select…" />
          </SelectTrigger>
          <SelectContent>
            {(field.enum ?? []).map((option) => (
              <SelectItem key={String(option)} value={String(option)}>
                {String(option)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {description}
      </div>
    )
  }

  if (kind === 'number' || kind === 'integer') {
    const current = value === undefined || value === null ? '' : String(value)
    return (
      <div className="space-y-1.5">
        {labelRow}
        <Input
          id={inputId}
          type="number"
          step={kind === 'integer' ? 1 : 'any'}
          value={current}
          disabled={disabled}
          placeholder={field.default != null ? String(field.default) : undefined}
          onChange={(event) => {
            const raw = event.target.value
            if (raw === '') {
              onChange(field.nullable ? null : undefined)
              return
            }
            const parsed = Number(raw)
            onChange(Number.isNaN(parsed) ? raw : parsed)
          }}
        />
        {description}
      </div>
    )
  }

  if (kind === 'array') {
    const items = Array.isArray(value) ? value : []
    const itemType = field.items?.type
    const isNumeric = itemType === 'number' || itemType === 'integer'
    const update = (next: unknown[]) => onChange(next)
    return (
      <div className="space-y-1.5">
        {labelRow}
        <div className="space-y-2">
          {items.map((item, index) => (
            <div key={index} className="flex items-center gap-2">
              <Input
                aria-label={`${label} ${index + 1}`}
                type={isNumeric ? 'number' : 'text'}
                value={item === null || item === undefined ? '' : String(item)}
                disabled={disabled}
                onChange={(event) => {
                  const raw = event.target.value
                  const parsed = isNumeric && raw !== '' ? Number(raw) : raw
                  update(items.map((existing, i) => (i === index ? parsed : existing)))
                }}
              />
              <Button
                type="button"
                variant="ghost"
                size="icon"
                disabled={disabled}
                aria-label={`Remove ${label} ${index + 1}`}
                onClick={() => update(items.filter((_, i) => i !== index))}
              >
                <X />
              </Button>
            </div>
          ))}
          <Button
            type="button"
            variant="outline"
            size="xs"
            disabled={disabled}
            onClick={() => update([...items, isNumeric ? 0 : ''])}
          >
            <Plus />
            Add item
          </Button>
        </div>
        {description}
      </div>
    )
  }

  if (kind === 'string') {
    const current = value === undefined || value === null ? '' : String(value)
    return (
      <div className="space-y-1.5">
        {labelRow}
        <Input
          id={inputId}
          value={current}
          disabled={disabled}
          placeholder={field.default != null ? String(field.default) : undefined}
          onChange={(event) =>
            onChange(event.target.value === '' && field.nullable ? null : event.target.value)
          }
        />
        {description}
      </div>
    )
  }

  // Unsupported (nested objects, mixed unions): point the user at the JSON tab.
  return (
    <div className="space-y-1.5">
      {labelRow}
      <p className="rounded-md border border-dashed p-3 text-xs text-muted-foreground">
        This field is too complex to edit here. Use the JSON tab.
      </p>
    </div>
  )
}

/** A form generated from a config JSON Schema. Unknown keys in `value` are preserved. */
export function SchemaForm({
  schema,
  value,
  disabled,
  onChange,
}: {
  schema: JsonObject | undefined
  value: JsonObject
  disabled?: boolean
  onChange: (next: JsonObject) => void
}) {
  const properties = (schema?.properties ?? {}) as JsonObject
  const defs = (schema?.$defs ?? {}) as JsonObject
  const required = Array.isArray(schema?.required) ? (schema?.required as string[]) : []
  const entries = Object.entries(properties).filter(([key]) => key !== 'id')

  return (
    <div className="space-y-4">
      {entries.map(([key, raw]) => {
        const field = normalizeProp(
          raw && typeof raw === 'object' ? (raw as JsonObject) : {},
          defs,
        )
        return (
          <SchemaField
            key={key}
            name={key}
            field={field}
            value={value[key]}
            required={required.includes(key)}
            disabled={disabled}
            onChange={(next) => {
              const updated = { ...value }
              if (next === undefined) delete updated[key]
              else updated[key] = next
              onChange(updated)
            }}
          />
        )
      })}
    </div>
  )
}
