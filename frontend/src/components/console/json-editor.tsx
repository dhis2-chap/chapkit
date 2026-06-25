// Shared JSON editor/viewer (CodeMirror) used for config data and ML DataFrames.
import { useEffect, useMemo, useState } from 'react'
import CodeMirror from '@uiw/react-codemirror'
import type { Extension } from '@codemirror/state'
import { EditorView } from '@codemirror/view'
import { json, jsonParseLinter } from '@codemirror/lang-json'
import { lintGutter, linter } from '@codemirror/lint'
import { useTheme } from 'next-themes'
import { cn } from '@/lib/utils'

// Make CodeMirror inherit our container's background/font so it reads as the same
// surface in both view and edit, instead of CodeMirror's own theme background.
const surfaceTheme = EditorView.theme({
  '&': { backgroundColor: 'transparent', fontSize: '0.75rem' },
  '.cm-gutters': { backgroundColor: 'transparent', border: 'none' },
  '.cm-content': { fontFamily: 'var(--font-mono)' },
  '.cm-activeLine': { backgroundColor: 'transparent' },
  '.cm-activeLineGutter': { backgroundColor: 'transparent' },
  '&.cm-focused': { outline: 'none' },
})

/** CodeMirror JSON editor; readOnly renders the same surface without editing or lint. */
export function JsonEditor({
  value,
  onChange,
  readOnly = false,
  minHeight = '8rem',
  maxHeight = '28rem',
  placeholder,
  schema,
  ariaLabel,
  className,
}: {
  value: string
  onChange?: (value: string) => void
  readOnly?: boolean
  minHeight?: string
  maxHeight?: string
  placeholder?: string
  // JSON Schema driving Ctrl-Space completion + inline validation (editable only).
  schema?: Record<string, unknown>
  ariaLabel?: string
  className?: string
}) {
  const { resolvedTheme } = useTheme()
  const isDark = resolvedTheme === 'dark'
  const schemaAware = Boolean(schema) && !readOnly

  // codemirror-json-schema pulls in a full JSON-Schema validator (~160KB gz), so
  // load it lazily and only when a schema editor is actually mounted — it stays
  // out of the main bundle and downloads on demand (e.g. opening the config form).
  const [schemaExt, setSchemaExt] = useState<Extension[] | null>(null)
  useEffect(() => {
    if (!schemaAware || !schema) {
      setSchemaExt(null)
      return
    }
    let cancelled = false
    void import('codemirror-json-schema').then(({ jsonSchema }) => {
      if (!cancelled) setSchemaExt(jsonSchema(schema as Parameters<typeof jsonSchema>[0]))
    })
    return () => {
      cancelled = true
    }
  }, [schemaAware, schema])

  const extensions = useMemo<Extension[]>(() => {
    const exts: Extension[] = [EditorView.lineWrapping, surfaceTheme]
    if (schemaAware && schemaExt) {
      // jsonSchema() bundles the JSON language + schema completion/validation/hover.
      exts.push(...schemaExt, lintGutter())
    } else {
      exts.push(json())
      // Inline parse lint only while editing, and not for an empty field
      // (an untouched editor shouldn't show a JSON error).
      if (!readOnly) {
        const jsonLint = jsonParseLinter()
        exts.push(
          lintGutter(),
          linter((view) => (view.state.doc.toString().trim() === '' ? [] : jsonLint(view))),
        )
      }
    }
    return exts
  }, [readOnly, schemaExt, schemaAware])

  return (
    <div
      className={cn(
        'overflow-hidden rounded-md border',
        readOnly ? 'bg-muted/40' : 'bg-background',
        className,
      )}
    >
      <CodeMirror
        value={value}
        onChange={onChange}
        readOnly={readOnly}
        editable={!readOnly}
        theme={isDark ? 'dark' : 'light'}
        extensions={extensions}
        minHeight={minHeight}
        maxHeight={maxHeight}
        placeholder={placeholder}
        basicSetup={{
          lineNumbers: !readOnly,
          foldGutter: !readOnly,
          highlightActiveLine: false,
          highlightActiveLineGutter: false,
          // Enable the autocomplete extension only when a schema drives it.
          autocompletion: schemaAware,
        }}
        aria-label={ariaLabel}
      />
    </div>
  )
}
