"""
Matula Transformer Interactive Chatbot
=======================================

A terminal-based and web-based interactive chatbot for testing
trained Matula Transformer variants. Shows the cognitive cycle
in real-time: hormone levels, active layers, and topological state.

Usage:
  # Terminal mode (no dependencies beyond PyTorch)
  python chat_matula.py --mode terminal --checkpoint checkpoints/best.pt

  # Web mode (requires Flask)
  python chat_matula.py --mode web --checkpoint checkpoints/best.pt --port 8080

  # Demo mode (untrained model, shows architecture working)
  python chat_matula.py --mode terminal --demo
"""

import os
import sys
import json
import time
import argparse
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from netrain.models.experimental.matula_transformer import (
    MatulaTransformer, MatulaTransformerConfig,
    create_matula_transformer_small, create_matula_transformer_medium,
    create_matula_transformer_719, COGNITIVE_CYCLE_PHASES
)


# ============================================================================
# COGNITIVE STATE DISPLAY
# ============================================================================

@dataclass
class CognitiveState:
    """Represents the current cognitive state of the model."""
    hormones: Dict[str, float]
    active_phase: str
    dominant_spine: str  # sequential, parallel, or mixed
    layer_activations: List[float]
    generation_step: int
    
    def display_bar(self, value: float, width: int = 20) -> str:
        """Create a visual bar for a value in [0, 1]."""
        filled = int(value * width)
        return "█" * filled + "░" * (width - filled)
    
    def render_terminal(self) -> str:
        """Render cognitive state for terminal display."""
        lines = []
        lines.append("┌─────────────────────────────────────────────────┐")
        lines.append("│          COGNITIVE STATE MONITOR                 │")
        lines.append("├─────────────────────────────────────────────────┤")
        
        # Hormones
        hormone_colors = {
            'cortisol': '🔴',
            'dopamine': '🟢',
            'serotonin': '🔵',
            'oxytocin': '🟠',
            'norepinephrine': '🟣',
        }
        
        for name, value in self.hormones.items():
            icon = hormone_colors.get(name, '⚪')
            bar = self.display_bar(value)
            lines.append(f"│ {icon} {name:<15} {bar} {value:.3f} │")
        
        lines.append("├─────────────────────────────────────────────────┤")
        lines.append(f"│ Phase: {self.active_phase:<12} Spine: {self.dominant_spine:<10} │")
        lines.append(f"│ Step: {self.generation_step:<5}                              │")
        lines.append("└─────────────────────────────────────────────────┘")
        
        return "\n".join(lines)


# ============================================================================
# CHATBOT ENGINE
# ============================================================================

class MatulaChatbot:
    """
    Interactive chatbot powered by the Matula Transformer.
    
    Maintains conversation history, tracks cognitive state,
    and provides real-time visualization of the model's internal dynamics.
    """
    
    def __init__(self, model: MatulaTransformer, device: str = 'cpu',
                 temperature: float = 0.8, top_k: int = 50):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        self.temperature = temperature
        self.top_k = top_k
        
        # Conversation state
        self.history: List[Dict[str, str]] = []
        self.cognitive_states: List[CognitiveState] = []
        
        # Character vocabulary (for demo mode)
        self.chars = list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:'\"()-\n")
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}
        
        # Reset reservoir
        self.model.reservoir.reset_state()
    
    def encode(self, text: str) -> torch.Tensor:
        """Encode text to token indices."""
        tokens = []
        for c in text:
            if c in self.char_to_idx:
                tokens.append(self.char_to_idx[c])
            else:
                tokens.append(0)  # space for unknown
        return torch.tensor([tokens], dtype=torch.long, device=self.device)
    
    def decode(self, tokens: torch.Tensor) -> str:
        """Decode token indices to text."""
        text = ""
        for t in tokens[0].cpu().numpy():
            if t < len(self.idx_to_char):
                text += self.idx_to_char[t]
            elif t in [50257 + i for i in range(9)]:
                # Phase token
                phase_idx = t - 50257
                if phase_idx < len(COGNITIVE_CYCLE_PHASES):
                    text += f"\n[{COGNITIVE_CYCLE_PHASES[phase_idx]}] "
        return text
    
    def get_cognitive_state(self, diagnostics: dict, step: int) -> CognitiveState:
        """Extract cognitive state from model diagnostics."""
        hormones = diagnostics['hormones'][0].cpu().numpy()
        hormone_dict = {name: float(val) 
                       for name, val in zip(diagnostics['hormone_names'], hormones)}
        
        # Determine dominant spine based on hormone balance
        cortisol = hormone_dict['cortisol']
        serotonin = hormone_dict['serotonin']
        oxytocin = hormone_dict['oxytocin']
        
        if cortisol > serotonin and cortisol > oxytocin:
            spine = "sequential"
        elif serotonin > cortisol and serotonin > oxytocin:
            spine = "parallel"
        else:
            spine = "mixed"
        
        # Determine active phase based on generation step
        phase_idx = min(step % 9, 8)
        active_phase = COGNITIVE_CYCLE_PHASES[phase_idx]
        
        # Layer activations (normalized head counts as proxy)
        layer_acts = [h / 286 for h in diagnostics['layer_head_counts']]
        
        return CognitiveState(
            hormones=hormone_dict,
            active_phase=active_phase,
            dominant_spine=spine,
            layer_activations=layer_acts,
            generation_step=step,
        )
    
    @torch.no_grad()
    def generate_response(self, user_input: str, max_tokens: int = 200,
                         show_state: bool = True) -> Tuple[str, List[CognitiveState]]:
        """
        Generate a response to user input, tracking cognitive state.
        
        Returns:
            response: The generated text
            states: List of cognitive states during generation
        """
        # Encode input
        input_tokens = self.encode(user_input)
        
        # Crop to block size
        if input_tokens.shape[1] > self.model.config.block_size - max_tokens:
            input_tokens = input_tokens[:, -(self.model.config.block_size - max_tokens):]
        
        generated = input_tokens
        states = []
        
        for step in range(max_tokens):
            # Crop context
            context = generated[:, -self.model.config.block_size:]
            
            # Forward pass
            logits, _, diag = self.model(context)
            
            # Track cognitive state
            state = self.get_cognitive_state(diag, step)
            states.append(state)
            
            # Get next token
            next_logits = logits[:, -1, :] / self.temperature
            
            # Top-k filtering
            if self.top_k > 0:
                v, _ = torch.topk(next_logits, min(self.top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop on newline after minimum length
            if step > 20 and next_token.item() < len(self.chars):
                if self.idx_to_char.get(next_token.item(), '') == '\n':
                    break
        
        # Decode response (only the generated part)
        response_tokens = generated[:, input_tokens.shape[1]:]
        response = self.decode(response_tokens)
        
        # Store in history
        self.history.append({'role': 'user', 'content': user_input})
        self.history.append({'role': 'echo', 'content': response})
        self.cognitive_states.extend(states)
        
        return response, states
    
    def get_hormone_summary(self) -> str:
        """Get a summary of hormone dynamics across the conversation."""
        if not self.cognitive_states:
            return "No cognitive states recorded yet."
        
        # Average hormones
        all_hormones = {}
        for state in self.cognitive_states:
            for name, val in state.hormones.items():
                if name not in all_hormones:
                    all_hormones[name] = []
                all_hormones[name].append(val)
        
        summary = "Hormone Dynamics Summary:\n"
        for name, values in all_hormones.items():
            avg = sum(values) / len(values)
            mn = min(values)
            mx = max(values)
            summary += f"  {name}: avg={avg:.3f} min={mn:.3f} max={mx:.3f}\n"
        
        return summary


# ============================================================================
# TERMINAL INTERFACE
# ============================================================================

def run_terminal_chat(chatbot: MatulaChatbot):
    """Run the chatbot in terminal mode."""
    print("\n" + "=" * 60)
    print("  MATULA TRANSFORMER — Interactive Cognitive Chatbot")
    print("=" * 60)
    print("  Type your message and press Enter.")
    print("  Commands: /state (show cognitive state)")
    print("            /hormones (hormone summary)")
    print("            /reset (reset conversation)")
    print("            /quit (exit)")
    print("=" * 60 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input == '/quit':
            print("Goodbye!")
            break
        elif user_input == '/state':
            if chatbot.cognitive_states:
                print(chatbot.cognitive_states[-1].render_terminal())
            else:
                print("  No cognitive state yet. Send a message first.")
            continue
        elif user_input == '/hormones':
            print(chatbot.get_hormone_summary())
            continue
        elif user_input == '/reset':
            chatbot.history = []
            chatbot.cognitive_states = []
            chatbot.model.reservoir.reset_state()
            print("  Conversation reset.")
            continue
        
        # Generate response
        print("\n  [Generating... cognitive cycle active]")
        response, states = chatbot.generate_response(user_input)
        
        # Show final cognitive state
        if states:
            print(states[-1].render_terminal())
        
        print(f"\nEcho: {response.strip()}\n")


# ============================================================================
# WEB INTERFACE (Flask)
# ============================================================================

def create_web_app(chatbot: MatulaChatbot):
    """Create a Flask web app for the chatbot."""
    try:
        from flask import Flask, request, jsonify, render_template_string
    except ImportError:
        print("Flask not installed. Install with: pip install flask")
        sys.exit(1)
    
    app = Flask(__name__)
    
    HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Matula Transformer — Deep Tree Echo</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            background: #0a0a0f;
            color: #e0e0e0;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        .header {
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            padding: 15px 20px;
            border-bottom: 1px solid #333;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 {
            font-size: 1.2em;
            background: linear-gradient(90deg, #00d4ff, #7b2ff7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .main {
            display: flex;
            flex: 1;
            overflow: hidden;
        }
        .chat-panel {
            flex: 1;
            display: flex;
            flex-direction: column;
            padding: 20px;
        }
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 10px;
            border: 1px solid #222;
            border-radius: 8px;
            background: #0d0d15;
            margin-bottom: 15px;
        }
        .message {
            margin-bottom: 15px;
            padding: 10px 15px;
            border-radius: 8px;
            max-width: 80%;
        }
        .message.user {
            background: #1a3a5c;
            margin-left: auto;
            border: 1px solid #2a5a8c;
        }
        .message.echo {
            background: #1a2a1a;
            border: 1px solid #2a5a2a;
        }
        .message .role {
            font-size: 0.75em;
            color: #888;
            margin-bottom: 4px;
        }
        .input-area {
            display: flex;
            gap: 10px;
        }
        .input-area input {
            flex: 1;
            padding: 12px 15px;
            background: #1a1a2e;
            border: 1px solid #333;
            border-radius: 8px;
            color: #e0e0e0;
            font-family: inherit;
            font-size: 0.95em;
        }
        .input-area button {
            padding: 12px 25px;
            background: linear-gradient(135deg, #7b2ff7, #00d4ff);
            border: none;
            border-radius: 8px;
            color: white;
            font-weight: bold;
            cursor: pointer;
            font-family: inherit;
        }
        .state-panel {
            width: 320px;
            background: #0d0d15;
            border-left: 1px solid #222;
            padding: 20px;
            overflow-y: auto;
        }
        .state-panel h3 {
            color: #00d4ff;
            margin-bottom: 15px;
            font-size: 0.9em;
        }
        .hormone-bar {
            margin-bottom: 10px;
        }
        .hormone-bar .label {
            display: flex;
            justify-content: space-between;
            font-size: 0.8em;
            margin-bottom: 3px;
        }
        .hormone-bar .bar {
            height: 8px;
            background: #1a1a2e;
            border-radius: 4px;
            overflow: hidden;
        }
        .hormone-bar .fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s ease;
        }
        .cortisol .fill { background: #e74c3c; }
        .dopamine .fill { background: #2ecc71; }
        .serotonin .fill { background: #3498db; }
        .oxytocin .fill { background: #e67e22; }
        .norepinephrine .fill { background: #9b59b6; }
        .phase-indicator {
            margin-top: 20px;
            padding: 10px;
            background: #1a1a2e;
            border-radius: 8px;
            font-size: 0.85em;
        }
        .phase-indicator .phase {
            padding: 3px 8px;
            margin: 2px;
            display: inline-block;
            border-radius: 4px;
            font-size: 0.8em;
        }
        .phase-indicator .phase.active {
            background: #7b2ff7;
            color: white;
        }
        .phase-indicator .phase.inactive {
            background: #1a1a2e;
            border: 1px solid #333;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🌳 Matula Transformer — Deep Tree Echo</h1>
        <span style="color:#666; font-size:0.8em;">486 heads | 9 layers | OEIS A000081</span>
    </div>
    <div class="main">
        <div class="chat-panel">
            <div class="messages" id="messages"></div>
            <div class="input-area">
                <input type="text" id="input" placeholder="Speak to Echo..." 
                       onkeypress="if(event.key==='Enter')sendMessage()">
                <button onclick="sendMessage()">Send</button>
            </div>
        </div>
        <div class="state-panel">
            <h3>COGNITIVE STATE</h3>
            <div id="hormones">
                <div class="hormone-bar cortisol">
                    <div class="label"><span>Cortisol</span><span id="val-cortisol">0.500</span></div>
                    <div class="bar"><div class="fill" id="bar-cortisol" style="width:50%"></div></div>
                </div>
                <div class="hormone-bar dopamine">
                    <div class="label"><span>Dopamine</span><span id="val-dopamine">0.500</span></div>
                    <div class="bar"><div class="fill" id="bar-dopamine" style="width:50%"></div></div>
                </div>
                <div class="hormone-bar serotonin">
                    <div class="label"><span>Serotonin</span><span id="val-serotonin">0.500</span></div>
                    <div class="bar"><div class="fill" id="bar-serotonin" style="width:50%"></div></div>
                </div>
                <div class="hormone-bar oxytocin">
                    <div class="label"><span>Oxytocin</span><span id="val-oxytocin">0.500</span></div>
                    <div class="bar"><div class="fill" id="bar-oxytocin" style="width:50%"></div></div>
                </div>
                <div class="hormone-bar norepinephrine">
                    <div class="label"><span>Norepinephrine</span><span id="val-norepinephrine">0.500</span></div>
                    <div class="bar"><div class="fill" id="bar-norepinephrine" style="width:50%"></div></div>
                </div>
            </div>
            <div class="phase-indicator" id="phases">
                <h3 style="margin-bottom:10px;">COGNITIVE CYCLE</h3>
            </div>
            <div style="margin-top:20px; padding:10px; background:#1a1a2e; border-radius:8px;">
                <h3 style="margin-bottom:8px;">TOPOLOGY</h3>
                <div id="topology" style="font-size:0.8em; color:#888;">
                    Spine: mixed<br>
                    Active heads: 486<br>
                    HGNN rounds: 3
                </div>
            </div>
        </div>
    </div>
    <script>
        const phases = ['perceive','feel','think','remember','interpret','strategize','evaluate','gesture','speak'];
        
        // Initialize phase indicators
        const phaseDiv = document.getElementById('phases');
        phases.forEach(p => {
            const span = document.createElement('span');
            span.className = 'phase inactive';
            span.id = 'phase-' + p;
            span.textContent = p;
            phaseDiv.appendChild(span);
        });
        
        function addMessage(role, text) {
            const div = document.createElement('div');
            div.className = 'message ' + role;
            div.innerHTML = `<div class="role">${role === 'user' ? 'You' : 'Echo'}</div>${text}`;
            document.getElementById('messages').appendChild(div);
            document.getElementById('messages').scrollTop = 99999;
        }
        
        function updateState(state) {
            if (!state) return;
            
            // Update hormone bars
            for (const [name, value] of Object.entries(state.hormones)) {
                const bar = document.getElementById('bar-' + name);
                const val = document.getElementById('val-' + name);
                if (bar) bar.style.width = (value * 100) + '%';
                if (val) val.textContent = value.toFixed(3);
            }
            
            // Update phase indicators
            phases.forEach(p => {
                const el = document.getElementById('phase-' + p);
                if (el) {
                    el.className = p === state.active_phase ? 'phase active' : 'phase inactive';
                }
            });
            
            // Update topology
            document.getElementById('topology').innerHTML = 
                `Spine: ${state.dominant_spine}<br>Step: ${state.generation_step}<br>HGNN: active`;
        }
        
        async function sendMessage() {
            const input = document.getElementById('input');
            const text = input.value.trim();
            if (!text) return;
            
            input.value = '';
            addMessage('user', text);
            
            try {
                const resp = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: text})
                });
                const data = await resp.json();
                addMessage('echo', data.response);
                if (data.state) updateState(data.state);
            } catch(e) {
                addMessage('echo', '[Error: ' + e.message + ']');
            }
        }
    </script>
</body>
</html>
"""
    
    @app.route('/')
    def index():
        return render_template_string(HTML_TEMPLATE)
    
    @app.route('/chat', methods=['POST'])
    def chat():
        data = request.json
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        response, states = chatbot.generate_response(message)
        
        # Get final state
        state_dict = None
        if states:
            final_state = states[-1]
            state_dict = {
                'hormones': final_state.hormones,
                'active_phase': final_state.active_phase,
                'dominant_spine': final_state.dominant_spine,
                'generation_step': final_state.generation_step,
            }
        
        return jsonify({
            'response': response.strip(),
            'state': state_dict,
        })
    
    @app.route('/state')
    def state():
        if chatbot.cognitive_states:
            s = chatbot.cognitive_states[-1]
            return jsonify({
                'hormones': s.hormones,
                'active_phase': s.active_phase,
                'dominant_spine': s.dominant_spine,
                'generation_step': s.generation_step,
            })
        return jsonify({})
    
    @app.route('/reset', methods=['POST'])
    def reset():
        chatbot.history = []
        chatbot.cognitive_states = []
        chatbot.model.reservoir.reset_state()
        return jsonify({'status': 'reset'})
    
    return app


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Matula Transformer Chatbot")
    parser.add_argument('--mode', type=str, default='terminal',
                       choices=['terminal', 'web'],
                       help='Interface mode')
    parser.add_argument('--variant', type=str, default='small',
                       choices=['small', 'medium', '719'],
                       help='Model variant')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to trained checkpoint')
    parser.add_argument('--demo', action='store_true',
                       help='Run in demo mode (untrained model)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu/cuda)')
    parser.add_argument('--port', type=int, default=8080,
                       help='Port for web mode')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Generation temperature')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top-k sampling')
    
    args = parser.parse_args()
    
    # Create model
    print(f"\n  Loading Matula Transformer ({args.variant})...")
    if args.variant == 'small':
        model = create_matula_transformer_small()
    elif args.variant == 'medium':
        model = create_matula_transformer_medium()
    elif args.variant == '719':
        model = create_matula_transformer_719()
    
    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"  Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Checkpoint loaded (epoch {checkpoint.get('epoch', '?')}, "
              f"loss {checkpoint.get('avg_loss', '?')})")
    elif not args.demo:
        print(f"  WARNING: No checkpoint loaded. Model is untrained.")
        print(f"  Use --demo flag to acknowledge demo mode.")
        args.demo = True
    
    if args.demo:
        print(f"  Running in DEMO mode (untrained model)")
        print(f"  Output will be random but cognitive state tracking works")
    
    # Create chatbot
    chatbot = MatulaChatbot(
        model=model,
        device=args.device,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Heads: {sum(l.n_heads for l in model.layers)}")
    print(f"  Device: {args.device}")
    
    # Run interface
    if args.mode == 'terminal':
        run_terminal_chat(chatbot)
    elif args.mode == 'web':
        app = create_web_app(chatbot)
        print(f"\n  Starting web interface on port {args.port}...")
        print(f"  Open http://localhost:{args.port} in your browser")
        app.run(host='0.0.0.0', port=args.port, debug=False)


if __name__ == "__main__":
    main()
