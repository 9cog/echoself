import { useEffect, useRef, useState, useCallback } from "react";
import { Terminal } from "xterm";
import { FitAddon } from "xterm-addon-fit";
import { WebLinksAddon } from "xterm-addon-web-links";
import "xterm/css/xterm.css";

interface TerminalComponentProps {
  onCommand?: (command: string) => void;
  initialOutput?: string[];
  commandHistory?: string[];
  isBusy?: boolean;
}

const TerminalComponent: React.FC<TerminalComponentProps> = ({
  onCommand,
  initialOutput = [],
  commandHistory = [],
  isBusy = false,
}) => {
  const terminalRef = useRef<HTMLDivElement>(null);
  const xtermRef = useRef<Terminal | null>(null);
  const fitAddonRef = useRef<FitAddon | null>(null);
  const [isTerminalReady, setIsTerminalReady] = useState<boolean>(false);
  const [historyIndex, setHistoryIndex] = useState<number>(-1);

  // Create stable function references using useCallback
  const inputBufferRef = useRef("");

  const writePrompt = useCallback(() => {
    if (!xtermRef.current) return;
    xtermRef.current.write("$ ");
  }, []);

  const handleEnterKey = useCallback(() => {
    const term = xtermRef.current;
    if (!term) return;

    term.write("\r\n");
    const command = inputBufferRef.current.trim();
    inputBufferRef.current = "";
    setHistoryIndex(-1);
    if (command && onCommand) {
      onCommand(command);
      return;
    }
    writePrompt();
  }, [onCommand, writePrompt]);

  const handleBackspace = useCallback(() => {
    if (!xtermRef.current || inputBufferRef.current.length === 0) return;

    xtermRef.current.write("\b \b");
    inputBufferRef.current = inputBufferRef.current.slice(0, -1);
  }, []);

  const handleTab = useCallback(() => {
    // Simple command completion
    if (!xtermRef.current || !inputBufferRef.current) return;

    const commands = [
      "help",
      "clear",
      "echo",
      "ls",
      "pwd",
      "cd",
      "node",
      "python",
      "npm",
      "version",
    ];

    const buffer = inputBufferRef.current;
    const matches = commands.filter(cmd => cmd.startsWith(buffer));

    if (matches.length === 1) {
      const completion = matches[0].slice(buffer.length);
      xtermRef.current.write(completion);
      inputBufferRef.current = matches[0];
    } else if (matches.length > 1) {
      xtermRef.current.write("\r\n");
      xtermRef.current.write(matches.join("  ") + "\r\n");
      writePrompt();
      xtermRef.current.write(buffer);
    }
  }, [writePrompt]);

  const clearBuffer = () => {
    const term = xtermRef.current;
    if (!term) return;
    while (inputBufferRef.current.length > 0) {
      term.write("\b \b");
      inputBufferRef.current = inputBufferRef.current.slice(0, -1);
    }
  };

  const handleArrowUp = useCallback(() => {
    if (!xtermRef.current || commandHistory.length === 0) return;

    const newIndex =
      historyIndex === -1
        ? commandHistory.length - 1
        : Math.max(0, historyIndex - 1);

    clearBuffer();
    const historyItem = commandHistory[newIndex];
    xtermRef.current.write(historyItem);
    inputBufferRef.current = historyItem;
    setHistoryIndex(newIndex);
  }, [commandHistory, historyIndex]);

  const handleArrowDown = useCallback(() => {
    if (!xtermRef.current || historyIndex === -1) return;

    const newIndex =
      historyIndex === commandHistory.length - 1 ? -1 : historyIndex + 1;

    clearBuffer();

    if (newIndex === -1) {
      setHistoryIndex(-1);
    } else {
      const historyItem = commandHistory[newIndex];
      xtermRef.current.write(historyItem);
      inputBufferRef.current = historyItem;
      setHistoryIndex(newIndex);
    }
  }, [commandHistory, historyIndex]);

  const handlersRef = useRef({
    isBusy,
    writePrompt,
    handleEnterKey,
    handleBackspace,
    handleTab,
    handleArrowUp,
    handleArrowDown,
  });
  handlersRef.current = {
    isBusy,
    writePrompt,
    handleEnterKey,
    handleBackspace,
    handleTab,
    handleArrowUp,
    handleArrowDown,
  };
  const linesWrittenRef = useRef(0);
  const wasBusyRef = useRef(false);

  // Create the terminal once. Output and key handlers update through refs.
  useEffect(() => {
    if (!terminalRef.current || xtermRef.current) return;

    try {
      const term = new Terminal({
        convertEol: true,
        fontFamily: "JetBrains Mono, Menlo, Monaco, Courier New, monospace",
        fontSize: 14,
        cursorBlink: true,
        cursorStyle: "block",
        theme: {
          background: "#1e1e2e",
          foreground: "#cdd6f4",
          cursor: "#f5e0dc",
          black: "#45475a",
          red: "#f38ba8",
          green: "#a6e3a1",
          yellow: "#f9e2af",
          blue: "#89b4fa",
          magenta: "#cba6f7",
          cyan: "#94e2d5",
          white: "#bac2de",
        },
      });

      // Load addons
      const fitAddon = new FitAddon();
      term.loadAddon(fitAddon);

      const webLinksAddon = new WebLinksAddon();
      term.loadAddon(webLinksAddon);

      // Open terminal in the container
      term.open(terminalRef.current);

      // Store references
      xtermRef.current = term;
      fitAddonRef.current = fitAddon;

      // Set up input handling
      term.onKey(({ key, domEvent }) => {
        const handlers = handlersRef.current;
        if (handlers.isBusy) {
          if (domEvent.ctrlKey && domEvent.key === "c") {
            term.write("^C\r\n");
            handlers.writePrompt();
          }
          return;
        }

        switch (domEvent.keyCode) {
          case 13: // Enter
            handlers.handleEnterKey();
            break;
          case 8: // Backspace
            handlers.handleBackspace();
            break;
          case 9: // Tab
            handlers.handleTab();
            break;
          case 38: // Arrow Up
            handlers.handleArrowUp();
            break;
          case 40: // Arrow Down
            handlers.handleArrowDown();
            break;
          case 67: // 'c' key
            if (domEvent.ctrlKey) {
              term.write("^C\r\n");
              inputBufferRef.current = "";
              handlers.writePrompt();
            } else {
              term.write(key);
              inputBufferRef.current += key;
            }
            break;
          case 76: // 'l' key
            if (domEvent.ctrlKey) {
              term.clear();
              handlers.writePrompt();
            } else {
              term.write(key);
              inputBufferRef.current += key;
            }
            break;
          default:
            if (!domEvent.ctrlKey && !domEvent.altKey && key) {
              term.write(key);
              inputBufferRef.current += key;
            }
        }
      });

      const fitTimer = globalThis.setTimeout(() => {
        try {
          fitAddon.fit();
        } catch (error) {
          console.error("Error fitting terminal:", error);
        }
        setIsTerminalReady(true);
      }, 100);
      return () => {
        globalThis.clearTimeout(fitTimer);
        term.dispose();
        xtermRef.current = null;
        fitAddonRef.current = null;
        linesWrittenRef.current = 0;
        wasBusyRef.current = false;
        inputBufferRef.current = "";
        setIsTerminalReady(false);
      };
    } catch (error) {
      console.error("Error initializing terminal:", error);
    }
  }, []);

  useEffect(() => {
    const term = xtermRef.current;
    if (!term || !isTerminalReady) return;
    const fresh = initialOutput.slice(linesWrittenRef.current);
    const isFirst = linesWrittenRef.current === 0;
    for (const line of fresh) {
      term.writeln(line);
    }
    linesWrittenRef.current = initialOutput.length;
    if (isFirst || (wasBusyRef.current && !isBusy)) {
      writePrompt();
    }
    wasBusyRef.current = isBusy;
  }, [initialOutput, isBusy, isTerminalReady, writePrompt]);

  // Handle terminal resize
  useEffect(() => {
    if (!isTerminalReady) return;

    const handleResize = () => {
      try {
        fitAddonRef.current?.fit();
      } catch (error) {
        console.error("Error fitting terminal:", error);
      }
    };

    globalThis.addEventListener("resize", handleResize);
    return () => globalThis.removeEventListener("resize", handleResize);
  }, [isTerminalReady]);

  return <div className="h-full w-full" ref={terminalRef} />;
};

export default TerminalComponent;
