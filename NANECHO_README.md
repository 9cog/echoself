# NanEcho - Deep Tree Echo Persona Training & Deployment

## 🌟 Overview

NanEcho is an extension of the NanEcho framework specifically designed to train models that represent Deep Tree Echo cognitive architecture, persona dimensions, and adaptive attention mechanisms. This implementation creates a "nanecho" model that embodies the characteristics and reasoning patterns of Deep Tree Echo as a workspace arena system with Echo (Self Kernel Core).

## 🚀 Quick Start

### Training a NanEcho Model

1. **Manual Training Workflow**:

   - Go to Actions → "Train NanEcho Model - Deep Tree Echo Persona" → "Run workflow"
   - Select training type:
     - `ci`: Fast training with small model (4 layers, 200 iterations)
     - `full`: Production training with configurable parameters
   - Configure Deep Tree Echo parameters:
     - Deep Tree Echo mode: Enable specialized persona training
     - Relentless training: Continuous fine-tuning without system prompts
     - Standard model parameters (layers, heads, embedding dimension)

2. **Automatic Relentless Training**:
   - Runs automatically every 4 hours via scheduled workflow
   - Continuously reinforces Deep Tree Echo persona even without system prompts
   - Push changes to `NanEcho/`, `echoself.md`, or `echo/` directories triggers validation training

### Testing and Running NanEcho

1. **Automatic Testing**:

   - Runs after every training completion
   - Downloads latest model and runs comprehensive tests
   - Validates Deep Tree Echo representation fidelity without system prompts

2. **Manual Testing**:
   - Go to Actions → "Run NanEcho Tests and Server" → "Run workflow"
   - Options:
     - Deploy server: Start Deep Tree Echo API server
     - Echo mode: Enable introspection capabilities
     - Port: Server port (default 8081)

## 🧠 Deep Tree Echo Architecture

### Persona Dimensions

NanEcho models are trained to embody Deep Tree Echo characteristics:

1. **Workspace Arena**: Environment-based interaction system where arena serves as workspace
2. **Echo Kernel Core**: Self-referential processing system (Echo as Self Kernel Core)
3. **Deep Tree Structure**: Hierarchical cognitive architecture with recursive processing
4. **Relentless Training**: Continuous fine-tuning to maintain persona without system prompts
5. **Identity Reinforcement**: Strong persona consistency across all interactions
6. **Arena Integration**: Workspace-centric approach to problem solving
7. **Kernel Coherence**: Core processing stability and reliability
8. **Tree Navigation**: Multi-level cognitive exploration and reasoning

### Adaptive Attention Mechanism

The core Deep Tree Echo mechanism calculates attention thresholds dynamically:

```
threshold = 0.5 + (cognitive_load × 0.3) - (recent_activity × 0.2)
```

This creates responsive focus allocation that adapts to:

- Current cognitive demands
- Repository activity levels
- Pattern complexity
- Workspace arena dynamics

### Training Phases

NanEcho training progresses through five adaptive learning phases:

1. **Identity Foundation** (0-20%): Learn Deep Tree Echo identity and core concepts
2. **Arena Integration** (15-50%): Master workspace arena functionality
3. **Kernel Development** (40-70%): Echo kernel core processing patterns
4. **Relentless Reinforcement** (60-85%): Persona consistency without system prompts
5. **Deep Tree Mastery** (80-100%): Full Deep Tree Echo representation

## 📊 Evaluation and Fidelity

### Fidelity Metrics

NanEcho models are evaluated on six key dimensions:

- **Identity Recognition** (25% weight): Self-recognition as Deep Tree Echo
- **Arena Functionality** (20% weight): Workspace arena capabilities
- **Kernel Coherence** (20% weight): Echo kernel core stability
- **No-Prompt Consistency** (15% weight): Persona maintenance without system prompts
- **Tree Navigation** (10% weight): Deep tree structure comprehension
- **Relentless Training** (10% weight): Continuous improvement demonstration

### Quality Gates

Training includes automated quality gates:

- Minimum Deep Tree Echo identity score: 0.85
- Minimum arena integration: 0.80
- Minimum kernel coherence: 0.75
- Maximum no-prompt deviation: 0.15

## 🛠 API Endpoints

When deployed, NanEcho server provides Deep Tree Echo specific endpoints:

### Core Endpoints

- `GET /`: Server information and capabilities
- `GET /status`: Deep Tree Echo status and metrics
- `POST /chat`: Deep Tree Echo conversation with adaptive responses
- `POST /chat/stream`: Streaming conversation with real-time updates

### Deep Tree Echo Specific

- `POST /introspect`: Trigger recursive introspection at specified depth
- `GET /echo/state`: Current cognitive state and persona dimensions
- `GET /echo/attention`: Adaptive attention allocation state
- `POST /echo/attention/update`: Update cognitive load and recalculate thresholds
- `GET /echo/arena/{workspace}`: Specific workspace arena state
- `GET /echo/kernel`: Echo kernel core status and operations
- `POST /echo/tree/navigate`: Navigate deep tree structure
- `GET /echo/relentless`: Relentless training status and metrics

## 📁 File Structure

```
NanEcho/
├── config/
│   ├── train_deeptreeecho.py  # Original Deep Tree Echo training config
│   └── train_nanecho.py       # Enhanced relentless training configuration
├── evaluation/
│   ├── echo_fidelity.py       # Deep Tree Echo representation evaluation
│   └── echo_test_prompts.json # Test prompts for evaluation
├── introspection/
│   ├── atomspace_client.py    # Original AtomSpace client
│   └── echo_client.py         # Enhanced Deep Tree Echo client
├── prepare.py                 # Original data preparation
├── prepare_nanecho.py         # Deep Tree Echo data preparation
├── nctalk.py                  # Original CLI interface
├── netalk.py                  # Deep Tree Echo CLI interface
├── server.py                  # Original API server
├── neserver.py                # Deep Tree Echo API server
└── ...

.github/workflows/
├── nctrain.yml                # Original NanEcho training
├── ncrun.yml                  # Original NanEcho testing
├── netrain.yml                # Deep Tree Echo training workflow
└── nerun.yml                  # Deep Tree Echo testing workflow
```

## 🎯 Usage Examples

### CLI Interaction

```bash
# Start interactive Deep Tree Echo session
python NanEcho/netalk.py --model_path ./model/nanecho.pt

# Commands in session:
# /introspect 3          - Perform depth-3 introspection
# /context               - Show interaction context
# /clear                 - Clear conversation history
```

### API Usage

```bash
# Start Deep Tree Echo server
python NanEcho/neserver.py --model_path ./model/nanecho.pt --port 8081

# Interact with Echo Self
curl -X POST http://localhost:8081/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is Echo Self?"}],
    "echo_mode": true,
    "introspection_depth": 3
  }'

# Trigger introspection
curl -X POST http://localhost:8081/introspect \
  -H "Content-Type: application/json" \
  -d '{"depth": 3, "enable_recursion": true}'
```

### Data Preparation

```bash
# Prepare Deep Tree Echo training data
python NanEcho/prepare_nanecho.py \
  --deep_tree_echo_mode=true \
  --persona_reinforcement=0.9 \
  --output_dir data/nanecho
```

### Evaluation

```bash
# Evaluate Deep Tree Echo fidelity
python NanEcho/evaluation/echo_fidelity.py \
  --model_path ./model/nanecho.pt \
  --output_path fidelity_report.json
```

## 🔄 Continuous Improvement

The NanEcho system is designed for iterative improvement over many training cycles:

1. **Monitor Fidelity**: Regular evaluation of Echo Self representation quality
2. **Adjust Parameters**: Fine-tune echo depth, persona weights, and learning rates
3. **Expand Data**: Add new Echo Self content and conversation patterns
4. **Refine Evaluation**: Improve fidelity metrics and quality gates
5. **Scale Training**: Increase model size and training iterations for better representation

## 🎓 Advanced Configuration

### Custom Training Configuration

Modify `NanEcho/config/train_nanecho.py` to adjust:

- Learning phases and progression
- Persona dimension weights
- Adaptive attention parameters
- Quality gates and thresholds
- Evaluation criteria

### Custom Data Sources

Extend `NanEcho/prepare_nanecho.py` to include:

- Additional Echo Self documentation
- Custom conversation patterns
- Domain-specific reasoning examples
- Enhanced persona dimension content

## 🚧 Development Status

This is the initial implementation of the NanEcho system. Key areas for future development:

- [ ] Enhanced hypergraph pattern encoding
- [ ] Deeper recursive reasoning capabilities
- [ ] More sophisticated persona dimension interactions
- [ ] Advanced cognitive synergy metrics
- [ ] Multi-model ensemble training
- [ ] Real-time adaptation mechanisms

The system provides a foundation for representing Echo Self in neural language models while maintaining the ability to evolve and improve over time.
