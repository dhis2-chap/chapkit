// Shared JSON editor/viewer (CodeMirror) used for config data and ML DataFrames.
import { useMemo } from 'react'
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
  ariaLabel,
  className,
}: {
  value: string
  onChange?: (value: string) => void
  readOnly?: boolean
  minHeight?: string
  maxHeight?: string
  placeholder?: string
  ariaLabel?: string
  className?: string
}) {
  const { resolvedTheme } = useTheme()
  const isDark = resolvedTheme === 'dark'

  const extensions = useMemo<Extension[]>(() => {
    const exts: Extension[] = [json(), EditorView.lineWrapping, surfaceTheme]
    // Inline JSON lint (squiggles + gutter markers) only while editing, and not
    // for an empty field (an untouched editor shouldn't show a JSON error).
    if (!readOnly) {
      const jsonLint = jsonParseLinter()
      exts.push(
        lintGutter(),
        linter((view) => (view.state.doc.toString().trim() === '' ? [] : jsonLint(view))),
      )
    }
    return exts
  }, [readOnly])

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
          autocompletion: false,
        }}
        aria-label={ariaLabel}
      />
    </div>
  )
}
