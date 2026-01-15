import React, { createContext, useContext, useState, useRef } from "react";
import { useAppStore } from "../store/appStore";
import { useMem0AI } from "../services/mem0aiService";
import { supabase } from "../services/supabaseClient";
import { RoomType } from "../components/EchoHomeMap";
import { ToroidalDialogue } from "../types/ToroidalCognitive";

// Import Memory type but NOT useMemory
type Memory = {
  id: string;
  title: string;
  content: string;
  tags: string[];
  createdAt: string;
  updatedAt: string;
};

// Memory content interface
interface MemoryContent {
  title?: string;
  content: string;
  metadata?: Record<string, unknown>;
}

// General content type for transfers
type TransferableContent = string | MemoryContent | Record<string, unknown>;

interface OrchestratorState {
  // Current focus and context
  activeComponent:
    | "map"
    | "editor"
    | "terminal"
    | "file-explorer"
    | "chat"
    | "memory";
  activeRoom: RoomType;
  previousComponent:
    | "map"
    | "editor"
    | "terminal"
    | "file-explorer"
    | "chat"
    | "memory"
    | null;

  // Integration status
  memoryIntegrationStatus: "active" | "inactive" | "connecting";
  aiIntegrationStatus: "active" | "inactive" | "initializing";
  terminalStatus: "ready" | "busy" | "error";

  // Toroidal cognitive system status
  toroidalMode: "disabled" | "echo" | "marduk" | "synced";
  toroidalStatus: "active" | "inactive" | "error";
  hemisphereBalance: number; // -1 (left-heavy) to 1 (right-heavy)

  // System-wide context
  currentTheme: "light" | "dark";
  sessionHistory: ContextEvent[];
  systemHealthStatus: SystemHealth;

  // User activity
  lastUserAction: string;
  lastUserActionTimestamp: number;
}

interface SystemHealth {
  memoryUsage: number; // Simulated memory usage percentage
  operationalStatus: "optimal" | "degraded" | "impaired";
  lastHealthCheck: number;
  activeConnections: number;
}

interface ContextEvent {
  id: string;
  type:
    | "navigation"
    | "file_operation"
    | "memory_operation"
    | "ai_interaction"
    | "system_event";
  description: string;
  timestamp: number;
  component: string;
  data?: Record<string, unknown>;
}

interface OrchestratorContextType {
  state: OrchestratorState;
  // Navigation and focus
  setActiveComponent: (component: OrchestratorState["activeComponent"]) => void;
  navigateToRoom: (room: RoomType) => void;
  focusOnFile: (fileId: string | null) => void;
  focusOnMemory: (memoryId: string | null) => void;

  // Actions
  executeInTerminal: (command: string) => Promise<string>;
  saveToMemory: (
    content: MemoryContent | string,
    tags: string[]
  ) => Promise<boolean>;
  transferContentBetweenComponents: (
    from: string,
    to: string,
    content: TransferableContent
  ) => boolean;

  // System management
  performHealthCheck: () => SystemHealth;
  refreshSystemContext: () => void;

  // Toroidal cognitive system management
  setToroidalMode: (mode: OrchestratorState["toroidalMode"]) => void;
  getToroidalStatus: () => {
    mode: OrchestratorState["toroidalMode"];
    status: OrchestratorState["toroidalStatus"];
    hemisphereBalance: number;
  };
  generateToroidalResponse: (
    prompt: string,
    options?: {
      responseMode?: "dual" | "echo-only" | "marduk-only" | "synced";
      creativityLevel?:
        | "balanced"
        | "analytical"
        | "creative"
        | "philosophical";
    }
  ) => Promise<ToroidalDialogue>;

  // Event logging
  logEvent: (event: Omit<ContextEvent, "id" | "timestamp">) => void;
  getRecentEvents: (count?: number) => ContextEvent[];
}

const defaultState: OrchestratorState = {
  activeComponent: "map",
  activeRoom: "overview",
  previousComponent: null,
  memoryIntegrationStatus: "inactive",
  aiIntegrationStatus: "inactive",
  terminalStatus: "ready",

  // Toroidal cognitive system defaults
  toroidalMode: "disabled",
  toroidalStatus: "inactive",
  hemisphereBalance: 0,

  currentTheme: "dark",
  sessionHistory: [],
  systemHealthStatus: {
    memoryUsage: 0,
    operationalStatus: "optimal",
    lastHealthCheck: Date.now(),
    activeConnections: 0,
  },
  lastUserAction: "system_initialization",
  lastUserActionTimestamp: Date.now(),
};

const OrchestratorContext = createContext<OrchestratorContextType | undefined>(
  undefined
);

export const useOrchestrator = () => {
  const context = useContext(OrchestratorContext);
  if (context === undefined) {
    throw new Error(
      "useOrchestrator must be used within an OrchestratorProvider"
    );
  }
  return context;
};

// Interface for memory functions that will be set from outside
interface MemoryFunctions {
  addMemory?: (
    memory: Omit<Memory, "id" | "createdAt" | "updatedAt">
  ) => Promise<Memory>;
}

// Global ref to store memory functions that can be set from MemoryConnector
const memoryFunctionsRef = {
  current: {} as MemoryFunctions,
};

// Reference to terminal execution function (will be set by Terminal component)
let executeTerminalCommandRef: ((command: string) => Promise<string>) | null =
  null;

// Function to set terminal execution function from Terminal component
export const setTerminalExecutionFunction = (
  executeFunction: (command: string) => Promise<string>
) => {
  executeTerminalCommandRef = executeFunction;
};

export const OrchestratorProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [state, setState] = useState<OrchestratorState>(defaultState);
  const appStore = useAppStore();
  const mem0ai = useMem0AI();

  // Use useRef for initialization logic to avoid useEffect in the component body
  const initialized = useRef(false);

  if (!initialized.current) {
    initialized.current = true;

    // Initialize orchestrator
    const initializeServices = async () => {
      try {
        // Update theme from app store
        setState(prev => ({
          ...prev,
          currentTheme: appStore.theme as "light" | "dark",
        }));

        // Check for existing Supabase session
        const { data } = await supabase.auth.getSession();
        if (data.session) {
          // Initialize Mem0AI if API key is available
          const storedApiKey = localStorage.getItem("openai_api_key");
          if (storedApiKey) {
            try {
              mem0ai.initialize(storedApiKey, data.session.user.id);
              setState(prev => ({
                ...prev,
                memoryIntegrationStatus: "active",
                aiIntegrationStatus: "active",
              }));

              logSystemEvent({
                type: "system_event",
                description: "AI and Memory systems initialized successfully",
                component: "orchestrator",
              });
            } catch (error) {
              logSystemEvent({
                type: "system_event",
                description: "Error initializing AI systems",
                component: "orchestrator",
                data: { error },
              });
            }
          } else {
            setState(prev => ({
              ...prev,
              memoryIntegrationStatus: "inactive",
              aiIntegrationStatus: "inactive",
            }));
          }
        }

        // Start periodic health checks
        const healthCheckInterval = setInterval(() => {
          performHealthCheck();
        }, 60000); // Every minute

        return () => {
          clearInterval(healthCheckInterval);
        };
      } catch (error) {
        console.error("Error initializing orchestrator:", error);
      }
    };

    initializeServices();
  }

  // Log event with automatic ID and timestamp
  const logSystemEvent = (event: Omit<ContextEvent, "id" | "timestamp">) => {
    const newEvent: ContextEvent = {
      ...event,
      id: `event_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: Date.now(),
    };

    setState(prev => ({
      ...prev,
      sessionHistory: [...prev.sessionHistory, newEvent].slice(-100), // Keep last 100 events
    }));
  };

  // Set active component with context awareness
  const setActiveComponent = (
    component: OrchestratorState["activeComponent"]
  ) => {
    setState(prev => ({
      ...prev,
      previousComponent: prev.activeComponent,
      activeComponent: component,
      lastUserAction: `navigate_to_${component}`,
      lastUserActionTimestamp: Date.now(),
    }));

    logSystemEvent({
      type: "navigation",
      description: `Navigated to ${component}`,
      component,
    });
  };

  // Navigate to a specific room in the map
  const navigateToRoom = (room: RoomType) => {
    setState(prev => ({
      ...prev,
      activeRoom: room,
      activeComponent: "map",
      previousComponent:
        prev.activeComponent === "map"
          ? prev.previousComponent
          : prev.activeComponent,
      lastUserAction: `navigate_to_room_${room}`,
      lastUserActionTimestamp: Date.now(),
    }));

    logSystemEvent({
      type: "navigation",
      description: `Navigated to room: ${room}`,
      component: "map",
      data: { room },
    });

    // Auto-activate related components based on room
    if (room === "memory-library") {
      // Could trigger memory view to open
    } else if (room === "workshop") {
      // Could focus on editor
    } else if (room === "communications-hub") {
      // Could open chat
    }
  };

  // Focus on a specific file
  const focusOnFile = (fileId: string | null) => {
    if (fileId) {
      const file = appStore.files.find(f => f.id === fileId);
      if (file) {
        appStore.setCurrentFile(file);
        setActiveComponent("editor");

        logSystemEvent({
          type: "file_operation",
          description: `Focused on file: ${file.name}`,
          component: "editor",
          data: { fileId, fileName: file.name },
        });
      }
    } else {
      appStore.setCurrentFile(null);
    }
  };

  // Focus on a specific memory
  const focusOnMemory = (memoryId: string | null) => {
    setActiveComponent("memory");

    if (memoryId) {
      logSystemEvent({
        type: "memory_operation",
        description: `Focused on memory`,
        component: "memory",
        data: { memoryId },
      });
    }
  };

  // Execute a command in the terminal
  const executeInTerminal = async (command: string): Promise<string> => {
    setState(prev => ({
      ...prev,
      terminalStatus: "busy",
      lastUserAction: "terminal_command",
      lastUserActionTimestamp: Date.now(),
    }));

    logSystemEvent({
      type: "system_event",
      description: `Terminal command executed: ${command}`,
      component: "terminal",
      data: { command },
    });

    let result = "";

    try {
      // Use the registered terminal execution function if available
      if (executeTerminalCommandRef) {
        result = await executeTerminalCommandRef(command);
      } else {
        // Fallback to simulated response if no terminal function is registered
        await new Promise(resolve => setTimeout(resolve, 500));
        result = `Command simulation: ${command}\n\nNote: Terminal execution function not registered. Please make sure the terminal component is loaded.`;
      }
    } catch (error: unknown) {
      result = `Error executing command: ${error instanceof Error ? error.message : "Unknown error"}`;
      console.error("Error executing terminal command:", error);
    } finally {
      setState(prev => ({
        ...prev,
        terminalStatus: "ready",
      }));
    }

    return result;
  };

  // Save content to memory system
  const saveToMemory = async (
    content: MemoryContent | string,
    tags: string[]
  ): Promise<boolean> => {
    try {
      // Try to use the memory function if available
      if (memoryFunctionsRef.current.addMemory) {
        await memoryFunctionsRef.current.addMemory({
          title:
            (typeof content === "object" &&
            content !== null &&
            "title" in content
              ? content.title
              : undefined) || `Memory from ${state.activeComponent}`,
          content:
            typeof content === "string" ? content : JSON.stringify(content),
          tags,
        });

        logSystemEvent({
          type: "memory_operation",
          description: "Content saved to memory",
          component: state.activeComponent,
          data: {
            content:
              (typeof content === "object" &&
              content !== null &&
              "title" in content
                ? content.title
                : "content") || "content",
            tags,
          },
        });

        return true;
      } else {
        // If memory functions not available, try to use Mem0AI instead
        if (mem0ai.isInitialized()) {
          await mem0ai.addMemory({
            title:
              (typeof content === "object" &&
              content !== null &&
              "title" in content
                ? content.title
                : undefined) || `Memory from ${state.activeComponent}`,
            content:
              typeof content === "string" ? content : JSON.stringify(content),
            tags,
            type: "memory",
          });

          logSystemEvent({
            type: "memory_operation",
            description: "Content saved to Mem0AI memory",
            component: state.activeComponent,
            data: {
              content:
                (typeof content === "object" &&
                content !== null &&
                "title" in content
                  ? content.title
                  : "content") || "content",
              tags,
            },
          });

          return true;
        }

        console.error("No memory system available");
        logSystemEvent({
          type: "system_event",
          description: "Failed to save to memory: No memory system available",
          component: state.activeComponent,
        });
        return false;
      }
    } catch (error) {
      logSystemEvent({
        type: "system_event",
        description: "Error saving to memory",
        component: state.activeComponent,
        data: { error },
      });
      return false;
    }
  };

  // Transfer content between components
  const transferContentBetweenComponents = (
    from: string,
    to: string,
    content: TransferableContent
  ): boolean => {
    logSystemEvent({
      type: "system_event",
      description: `Content transferred from ${from} to ${to}`,
      component: "orchestrator",
      data: { from, to, contentType: typeof content },
    });

    // In a real implementation, this would handle the actual content transfer
    return true;
  };

  // Perform system health check
  const performHealthCheck = (): SystemHealth => {
    // Simulate health check
    const memoryUsage = Math.floor(Math.random() * 30) + 10; // 10-40%
    const healthStatus: SystemHealth = {
      memoryUsage,
      operationalStatus:
        memoryUsage > 80
          ? "impaired"
          : memoryUsage > 60
            ? "degraded"
            : "optimal",
      lastHealthCheck: Date.now(),
      activeConnections: state.sessionHistory.filter(
        e => e.timestamp > Date.now() - 300000
      ).length, // Events in last 5 minutes
    };

    setState(prev => ({
      ...prev,
      systemHealthStatus: healthStatus,
    }));

    return healthStatus;
  };

  // Refresh system context
  const refreshSystemContext = () => {
    logSystemEvent({
      type: "system_event",
      description: "System context refreshed",
      component: "orchestrator",
    });

    // Re-initialize services and update state
    performHealthCheck();
  };

  // Get recent events
  const getRecentEvents = (count: number = 10): ContextEvent[] => {
    return state.sessionHistory.slice(-count);
  };

  // Toroidal cognitive system methods
  const setToroidalMode = (mode: OrchestratorState["toroidalMode"]) => {
    setState(prev => ({
      ...prev,
      toroidalMode: mode,
      toroidalStatus: mode === "disabled" ? "inactive" : "active",
      lastUserAction: `toroidal_mode_${mode}`,
      lastUserActionTimestamp: Date.now(),
    }));

    logSystemEvent({
      type: "system_event",
      component: "toroidal",
      description: `Toroidal mode changed to: ${mode}`,
      data: { newMode: mode },
    });
  };

  const getToroidalStatus = () => ({
    mode: state.toroidalMode,
    status: state.toroidalStatus,
    hemisphereBalance: state.hemisphereBalance,
  });

  const generateToroidalResponse = async (
    prompt: string,
    _options: {
      responseMode?: "dual" | "echo-only" | "marduk-only" | "synced";
      creativityLevel?:
        | "balanced"
        | "analytical"
        | "creative"
        | "philosophical";
    } = {}
  ) => {
    // This would integrate with the ToroidalCognitiveService
    // For now, return a mock response indicating the feature is available
    const mockResponse: ToroidalDialogue = {
      deepTreeEchoResponse: {
        persona: "deepTreeEcho",
        content: `*"Your inquiry creates beautiful resonance patterns..."* - Echo response for: ${prompt}`,
        timestamp: new Date(),
        processingTime: 500,
      },
      mardukResponse: {
        persona: "marduk",
        content: `*"Excellent. Analyzing request through recursive processing..."* - Marduk response for: ${prompt}`,
        timestamp: new Date(),
        processingTime: 500,
      },
      reflection: {
        content: `## **Echo + Marduk (Reflection)**\n\nThe toroidal cognitive system is operational and ready to process: ${prompt}`,
        synergy: "complementary" as const,
        unified_answer:
          "The system demonstrates balanced cognitive processing capabilities.",
      },
      metadata: {
        queryId: Date.now().toString(),
        totalProcessingTime: 1000,
        contextType: "balanced" as const,
      },
    };

    // Update hemisphere balance in state based on response
    setState(prev => ({
      ...prev,
      hemisphereBalance: 0, // Use default value since metadata structure changed
      lastUserAction: "toroidal_response_generated",
      lastUserActionTimestamp: Date.now(),
    }));

    return mockResponse;
  };

  // Create context value object
  const contextValue: OrchestratorContextType = {
    state,
    setActiveComponent,
    navigateToRoom,
    focusOnFile,
    focusOnMemory,
    executeInTerminal,
    saveToMemory,
    transferContentBetweenComponents,
    performHealthCheck,
    refreshSystemContext,
    setToroidalMode,
    getToroidalStatus,
    generateToroidalResponse,
    logEvent: logSystemEvent,
    getRecentEvents,
  };

  return (
    <OrchestratorContext.Provider value={contextValue}>
      {children}
    </OrchestratorContext.Provider>
  );
};

// Function to set memory functions from outside (called by MemoryConnector)
export const setMemoryFunctions = (
  addMemoryFunc: MemoryFunctions["addMemory"]
) => {
  memoryFunctionsRef.current.addMemory = addMemoryFunc;
};
