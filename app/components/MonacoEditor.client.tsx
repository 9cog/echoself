import Editor, { type Monaco, type OnMount } from "@monaco-editor/react";
import type { editor } from "monaco-editor";
import "~/styles/monaco.css";

type MonacoEditorInstance = Parameters<OnMount>[0];

export interface MonacoEditorProps {
  value: string;
  onChange: (value: string) => void;
  language?: string;
  theme?: string;
  options?: editor.IStandaloneEditorConstructionOptions;
  onMount?: (editor: MonacoEditorInstance, monaco: Monaco) => void;
}

const defaultOptions: editor.IStandaloneEditorConstructionOptions = {
  automaticLayout: true,
  fontFamily: "JetBrains Mono, Menlo, Monaco, Courier New, monospace",
  fontSize: 14,
  lineHeight: 21,
  minimap: { enabled: true },
  scrollBeyondLastLine: false,
  renderLineHighlight: "all",
  cursorBlinking: "smooth",
  cursorSmoothCaretAnimation: "on",
  smoothScrolling: true,
  bracketPairColorization: { enabled: true },
  padding: { top: 10 },
  folding: true,
  showFoldingControls: "always",
  wordWrap: "on",
  fixedOverflowWidgets: true,
};

/**
 * Loads Monaco through @monaco-editor/react's AMD loader.
 * A direct `import * as monaco from "monaco-editor"` breaks under Vite
 * (`CallbackIterable is not a constructor`) and never mounts a buffer.
 */
const MonacoEditor: React.FC<MonacoEditorProps> = ({
  value,
  onChange,
  language = "typescript",
  theme = "vs-dark",
  options = {},
  onMount,
}) => {
  return (
    <div className="h-full w-full" data-testid="monaco-editor-container">
      <Editor
        height="100%"
        language={language}
        theme={theme}
        value={value}
        options={{ ...defaultOptions, ...options }}
        onChange={next => onChange(next ?? "")}
        onMount={onMount}
        loading={
          <div className="h-full w-full flex items-center justify-center text-sm opacity-50">
            Loading editor...
          </div>
        }
      />
    </div>
  );
};

export default MonacoEditor;
