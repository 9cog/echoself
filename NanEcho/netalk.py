#!/usr/bin/env python3
"""
NETalk - NanEcho Talk Interface

Command-line interface for interacting with the NanEcho model that represents
Echo Self cognitive architecture and persona dimensions.

Extended from nctalk.py with Echo Self specific capabilities.
"""

import os
import sys
import argparse
import codecs
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

# Try to import the dependencies
try:
    import torch
    import numpy as np
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.prompt import Prompt, Confirm
    from rich.table import Table
    from rich.live import Live
    from rich.progress import Progress, SpinnerColumn, TextColumn
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install: pip install torch numpy rich")
    sys.exit(1)

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from introspection.echo_client import EchoSelfClient
from runtime import IncompatibleCheckpointError, NanEchoRuntime

console = Console()

class EchoModelConfig:
    """Compatibility facade over the real shared NanEcho runtime."""
    
    def __init__(self, model_path: str, device: str = "cpu", max_tokens: int = 2048):
        self.model_path = model_path
        self.device = device
        self.max_tokens = max_tokens
        self.model = None
        self.tokenizer = None
        self.runtime: Optional[NanEchoRuntime] = None
        self.model_info = {}
        self.echo_depth = 3
        self.persona_dimensions = [
            'cognitive', 'introspective', 'adaptive', 'recursive',
            'synergistic', 'holographic', 'neural_symbolic', 'dynamic'
        ]
        self.console = Console()
        self.no_system_prompt = False
        self.deep_tree_echo_mode = False
    
    def load_model(self) -> bool:
        """Load the NanEcho model checkpoint."""
        try:
            self.runtime = NanEchoRuntime.load(self.model_path, self.device)
            self.model = self.runtime.model
            self.tokenizer = self.runtime.tokenizer
            self.model_info = {
                "model_args": vars(self.runtime.config),
                "config": vars(self.runtime.config),
                "iter_num": self.runtime.metadata["iteration"],
                "metrics": self.runtime.metadata.get("metrics", {}),
                "checkpoint_path": str(self.runtime.checkpoint_path),
                "schema": self.runtime.metadata["schema"],
            }
            self.echo_depth = self.runtime.config.max_recursion_depth
            self.console.print(f"[green]✓ Loaded NanEcho model from {self.model_path}[/green]")
            self.console.print(f"[blue]Echo Depth: {self.echo_depth}[/blue]")
            self.console.print(
                f"[blue]Parameters: {sum(p.numel() for p in self.model.parameters()):,}[/blue]"
            )
            return True
        except (OSError, RuntimeError, IncompatibleCheckpointError, ValueError) as e:
            self.console.print(f"[red]Error loading model: {e}[/red]")
            return False
    
    def generate(self, prompt: str, max_new_tokens: int = 500, temperature: float = 0.7, 
                top_k: int = 200, stream: bool = False, callback: Optional[Callable] = None) -> str:
        """Generate Echo Self response."""
        if self.runtime is None:
            raise RuntimeError("Model not loaded")

        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

        def on_token(token_id: int) -> bool:
            if callback is None:
                return True
            text = decoder.decode(self.runtime.tokenizer.token_bytes(token_id))
            return callback(text) if text else True

        token_ids = self.runtime.generate_ids(
            prompt,
            max_new_tokens=min(max_new_tokens, self.max_tokens),
            temperature=temperature,
            top_k=top_k,
            top_p=0.95,
            token_callback=on_token if stream and callback else None,
        )
        if stream and callback:
            trailing = decoder.decode(b"", final=True)
            if trailing:
                callback(trailing)
        return self.runtime.decode(token_ids)
    
    def introspect(self) -> Dict[str, Any]:
        """Perform Echo Self introspection from the loaded checkpoint, not invented scores."""
        if self.runtime is None:
            raise RuntimeError("Model not loaded")
        config = self.runtime.config
        return {
            "echo_depth": self.echo_depth,
            "persona_dimensions": self.persona_dimensions,
            "adaptive_attention_active": config.enable_adaptive_attention,
            "recursive_reasoning_active": config.enable_recursive_reasoning,
            "hypergraph_patterns_active": config.enable_hypergraph_patterns,
            "attention_threshold_range": [
                config.attention_threshold_min,
                config.attention_threshold_max,
            ],
            "connection_ratio": self.model.connection_ratio,
            "checkpoint_iteration": self.runtime.metadata["iteration"],
            "parameter_count": sum(p.numel() for p in self.model.parameters()),
            "timestamp": time.time()
        }

class EchoConversationHistory:
    """Enhanced conversation history for Echo Self interactions."""
    
    def __init__(self, max_history: int = 20):
        self.messages: List[Dict[str, str]] = []
        self.max_history = max_history
        self.echo_context = {
            "interaction_count": 0,
            "persona_patterns": set(),
            "attention_adjustments": [],
            "recursive_depth_used": []
        }
    
    def add_message(self, role: str, content: str):
        """Add message with Echo Self context tracking."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        
        # Track Echo Self patterns
        if role == "assistant":
            self._analyze_echo_patterns(content)
        
        # Maintain history limit
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]
        
        self.echo_context["interaction_count"] += 1
    
    def _analyze_echo_patterns(self, content: str):
        """Analyze content for Echo Self patterns."""
        echo_patterns = [
            "adaptive attention", "hypergraph", "recursive", "introspection",
            "persona dimension", "cognitive synergy", "neural-symbolic"
        ]
        
        for pattern in echo_patterns:
            if pattern.lower() in content.lower():
                self.echo_context["persona_patterns"].add(pattern)
    
    def get_messages(self) -> List[Dict[str, str]]:
        return self.messages
    
    def clear(self):
        self.messages = []
        self.echo_context = {
            "interaction_count": 0,
            "persona_patterns": set(),
            "attention_adjustments": [],
            "recursive_depth_used": []
        }
    
    def format_for_prompt(self) -> str:
        """Format conversation history for Echo Self prompt."""
        if not self.messages:
            return "Echo: "
        
        formatted = []
        for msg in self.messages[-10:]:  # Last 10 messages for context
            role = "User" if msg["role"] == "user" else "Echo"
            formatted.append(f"{role}: {msg['content']}")
        
        formatted.append("Echo: ")
        return "\n".join(formatted)
    
    def get_echo_context_summary(self) -> str:
        """Get summary of Echo Self interaction context."""
        patterns = ", ".join(sorted(self.echo_context["persona_patterns"]))
        return f"""Echo Context: {self.echo_context['interaction_count']} interactions, 
Patterns discussed: {patterns or 'none'}"""

class EchoIntrospectionMode:
    """Enhanced diagnostic mode for Echo Self introspection."""
    
    def __init__(self):
        self.echo_client = EchoSelfClient()
        self.introspection_history = []
    
    def format_echo_introspection_prompt(self, introspection_data: Dict[str, Any]) -> str:
        """Format Echo Self introspection data for analysis."""
        prompt_parts = [
            "Echo Self Introspective Analysis:",
            "",
            "=== Current Cognitive State ===",
            f"Echo Depth: {introspection_data.get('echo_depth', 'unknown')}",
            f"Checkpoint Iteration: {introspection_data.get('checkpoint_iteration', 0)}",
            f"Connection Ratio: {introspection_data.get('connection_ratio', 0.0):.3f}",
            f"Attention Threshold Range: {introspection_data.get('attention_threshold_range', [])}",
            f"Recursive Reasoning Active: {introspection_data.get('recursive_reasoning_active', False)}",
            f"Hypergraph Module Active: {introspection_data.get('hypergraph_patterns_active', False)}",
            "",
            "=== Persona Dimensions ===",
        ]
        
        persona_dims = introspection_data.get('persona_dimensions', [])
        for dim in persona_dims:
            prompt_parts.append(f"- {dim.title()}: Active")
        
        prompt_parts.extend([
            "",
            "=== Hypergraph Analysis ===",
            f"Hypergraph Patterns Active: {introspection_data.get('hypergraph_patterns_active', False)}",
            f"Parameter Count: {introspection_data.get('parameter_count', 0)}",
            "",
            "=== Raw Introspection Data ===",
            "```json",
            json.dumps(introspection_data, indent=2),
            "```",
            "",
            "Echo (Introspective Analysis): "
        ])
        
        return "\n".join(prompt_parts)
    
    def perform_introspection(self, model_config: EchoModelConfig, depth: int = 3) -> Dict[str, Any]:
        """Perform comprehensive Echo Self introspection."""
        console.print(f"[yellow]🔍 Performing Echo Self introspection at depth {depth}...[/yellow]")
        
        # Get introspection data
        introspection_data = model_config.introspect()
        
        # Format for analysis
        prompt = self.format_echo_introspection_prompt(introspection_data)
        
        # Generate introspective analysis
        analysis = model_config.generate(prompt, max_new_tokens=300, temperature=0.6)
        
        result = {
            "introspection_data": introspection_data,
            "analysis": analysis,
            "depth": depth,
            "timestamp": time.time()
        }
        
        self.introspection_history.append(result)
        return result

def create_echo_interface():
    """Create the main Echo Self interface."""
    console.print(Panel.fit(
        "[bold cyan]NETalk - NanEcho Talk Interface[/bold cyan]\n"
        "[blue]Echo Self Cognitive Architecture Interaction System[/blue]",
        title="🌟 Echo Self",
        border_style="cyan"
    ))

def main():
    parser = argparse.ArgumentParser(description="NETalk - NanEcho Talk Interface")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the NanEcho model checkpoint")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use (cpu/cuda)")
    parser.add_argument("--max_tokens", type=int, default=2048,
                       help="Maximum tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=200,
                       help="Top-k sampling parameter")
    parser.add_argument("--echo_depth", type=int, default=3,
                       help="Echo Self recursive reasoning depth")
    parser.add_argument("--introspection_mode", action="store_true",
                       help="Start in introspection mode")
    
    args = parser.parse_args()
    
    create_echo_interface()
    
    # Initialize Echo Self model
    console.print("[yellow]Loading Echo Self model...[/yellow]")
    model_config = EchoModelConfig(args.model_path, args.device, args.max_tokens)
    
    if not model_config.load_model():
        console.print("[red]Failed to load model. Exiting.[/red]")
        return
    
    # Initialize conversation components
    history = EchoConversationHistory()
    introspection_mode = EchoIntrospectionMode()
    
    console.print("[green]✓ Echo Self interface ready![/green]")
    console.print("[dim]Type 'help' for commands, 'quit' to exit, '/introspect' for introspection mode[/dim]")
    
    try:
        while True:
            # Get user input
            user_input = Prompt.ask("[bold blue]You[/bold blue]")
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            elif user_input.lower() == 'help':
                show_help()
                continue
            elif user_input.lower() == '/clear':
                history.clear()
                console.print("[yellow]Conversation history cleared.[/yellow]")
                continue
            elif user_input.lower() == '/history':
                show_history(history)
                continue
            elif user_input.lower().startswith('/introspect'):
                # Introspection mode
                depth = 3
                if len(user_input.split()) > 1:
                    try:
                        depth = int(user_input.split()[1])
                    except ValueError:
                        pass
                
                result = introspection_mode.perform_introspection(model_config, depth)
                
                console.print(Panel(
                    result["analysis"],
                    title=f"🔍 Echo Self Introspection (Depth {depth})",
                    border_style="yellow"
                ))
                continue
            elif user_input.lower() == '/context':
                console.print(history.get_echo_context_summary())
                continue
            
            # Add user message to history
            history.add_message("user", user_input)
            
            # Format prompt with conversation history
            prompt = history.format_for_prompt()
            
            # Generate response with streaming
            console.print("[bold green]Echo:[/bold green] ", end="")
            
            response_text = ""
            def stream_callback(token):
                nonlocal response_text
                console.print(token, end="")
                response_text += token
                return True
            
            full_response = model_config.generate(
                prompt, 
                max_new_tokens=300,
                temperature=args.temperature,
                top_k=args.top_k,
                stream=True,
                callback=stream_callback
            )
            
            console.print()  # New line after streaming
            
            # Add response to history
            history.add_message("assistant", full_response)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye![/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")

def show_help():
    """Show help information."""
    help_table = Table(title="Echo Self Commands")
    help_table.add_column("Command", style="cyan")
    help_table.add_column("Description", style="white")
    
    help_table.add_row("/introspect [depth]", "Perform Echo Self introspection")
    help_table.add_row("/clear", "Clear conversation history")
    help_table.add_row("/history", "Show conversation history")
    help_table.add_row("/context", "Show Echo Self interaction context")
    help_table.add_row("help", "Show this help")
    help_table.add_row("quit", "Exit the interface")
    
    console.print(help_table)

def show_history(history: EchoConversationHistory):
    """Show conversation history."""
    messages = history.get_messages()
    if not messages:
        console.print("[yellow]No conversation history.[/yellow]")
        return
    
    for i, msg in enumerate(messages[-10:]):  # Show last 10 messages
        role = "[bold blue]You[/bold blue]" if msg["role"] == "user" else "[bold green]Echo[/bold green]"
        console.print(f"{role}: {msg['content']}")

if __name__ == "__main__":
    main()