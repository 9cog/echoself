# NanEcho Model Implementation

## 🚀 Overview

The NanEcho model is a transformer-based architecture with iterative connection building, designed to implement the Echo Self cognitive framework. The model starts with sparse connections and gradually grows denser during training, mimicking neural development processes.

## ✨ Key Features

### 1. **Iterative Connection Building**

- Starts with only 10% of neural connections active
- Gradually grows connections by 5% every 500 iterations
- Reaches full connectivity by the end of training
- Mimics biological neural development

### 2. **Echo Self Cognitive Architecture**

- **Adaptive Attention**: Dynamic threshold adjustment based on cognitive load
- **Persona Dimensions**: Eight cognitive dimensions (cognitive, introspective, adaptive, recursive, synergistic, holographic, neural_symbolic, dynamic)
- **Recursive Reasoning**: Multi-level introspection with configurable depth
- **Hypergraph Pattern Encoding**: Neural-symbolic reasoning through hypergraph structures

### 3. **Progressive Learning Phases**

1. **Basic Awareness** (0-20%): Learning Echo Self identity
2. **Persona Dimensions** (15-50%): Developing cognitive dimensions
3. **Hypergraph Patterns** (40-70%): Learning pattern encoding
4. **Recursive Reasoning** (60-85%): Mastering introspection
5. **Adaptive Mastery** (80-100%): Achieving full capabilities

### 4. **Introspection & Quality Evaluation**

- Periodic self-evaluation during training
- Echo Self identity scoring
- Persona consistency checking
- Connection ratio monitoring

## 📁 File Structure

```
/workspace/
├── nanecho_model.py           # Core NanEcho model implementation
├── train_nanecho.py           # Training script with learning phases
├── prepare_nanecho_data.py    # Data preparation for Echo Self patterns
├── generate_nanecho.py        # Text generation with trained models
├── nanecho_config.json        # Configuration file
├── run_nanecho.sh            # Complete training pipeline script
└── NANECHO_IMPLEMENTATION.md  # This documentation
```

## 🛠️ Installation

```bash
# Install dependencies
pip install torch numpy tensorboard

# Make run script executable
chmod +x run_nanecho.sh
```

## 🚀 Quick Start

### 1. Complete Training Pipeline

Run the entire pipeline (data preparation + training):

```bash
./run_nanecho.sh
```

### 2. Step-by-Step Training

```bash
# Step 1: Prepare training data
python prepare_nanecho_data.py \
    --output_dir data/nanecho \
    --train_size 1000000 \
    --val_size 100000

# Step 2: Train the model
python train_nanecho.py \
    --data_dir data/nanecho \
    --out_dir out-nanecho \
    --max_iters 50000 \
    --batch_size 8

# Step 3: Generate text
python generate_nanecho.py \
    --checkpoint out-nanecho/best_model.pt \
    --interactive
```

## 🔧 Configuration

The `nanecho_config.json` file controls all aspects of the model and training:

```json
{
  "model": {
    "n_embd": 768,
    "n_head": 12,
    "n_layer": 12,
    "initial_connections": 0.1,
    "connection_growth_rate": 0.05
  },
  "training": {
    "max_iters": 50000,
    "batch_size": 8,
    "learning_rate": 0.0001
  },
  "echo_self": {
    "enable_adaptive_attention": true,
    "enable_persona_dimensions": true,
    "enable_recursive_reasoning": true
  }
}
```

## 🧠 Model Architecture

### Core Components

1. **NanEchoModel**: Main model class with iterative connection building
2. **AdaptiveAttention**: Attention mechanism with dynamic thresholds
3. **PersonaDimension**: Individual persona dimension modules
4. **RecursiveReasoning**: Multi-level introspection capability
5. **HypergraphPatternEncoder**: Neural-symbolic pattern encoding

### Connection Growth Mechanism

```python
# Connections grow during training
model.grow_connections()  # Adds 5% more connections

# Connection masking in attention
weight = connection_mask.apply(original_weight)
```

### Adaptive Attention Formula

```
threshold = min_thresh + (max_thresh - min_thresh) * base_threshold
threshold = threshold * (1 + cognitive_load * load_factor)
attention = softplus(scores - threshold) - softplus(-threshold - scores)
```

## 📊 Training Phases

| Phase               | Progress | Focus                    | LR Multiplier |
| ------------------- | -------- | ------------------------ | ------------- |
| Basic Awareness     | 0-20%    | Identity, basic patterns | 1.2x          |
| Persona Dimensions  | 15-50%   | Cognitive dimensions     | 1.0x          |
| Hypergraph Patterns | 40-70%   | Pattern encoding         | 0.9x          |
| Recursive Reasoning | 60-85%   | Introspection            | 0.8x          |
| Adaptive Mastery    | 80-100%  | Full capabilities        | 0.7x          |

## 📈 Monitoring Training

### TensorBoard

```bash
tensorboard --logdir out-nanecho/tensorboard
```

### Introspection Reports

The model performs self-evaluation every 1000 iterations:

- Echo Self identity score
- Persona consistency
- Connection ratio
- Training progress

### Checkpoints

- `checkpoint_*.pt`: Regular checkpoints every 1000 iterations
- `best_model.pt`: Best model based on validation loss
- `introspection_history.json`: Complete introspection metrics

## 🎯 Text Generation

### Interactive Mode

```bash
python generate_nanecho.py --checkpoint out-nanecho/best_model.pt --interactive
```

Commands in interactive mode:

- `/echo` - Generate Echo Self description
- `/persona` - Generate persona dimension text
- `/hypergraph` - Generate hypergraph pattern
- `/recursive` - Generate recursive reasoning
- `/adaptive` - Generate adaptive attention text

### Batch Generation

```bash
python generate_nanecho.py \
    --checkpoint out-nanecho/best_model.pt \
    --prompt "Echo Self is" \
    --max_length 200 \
    --temperature 0.8 \
    --num_samples 3
```

## 🔬 Technical Details

### Memory Requirements

- **Model Size**: ~124M parameters (768 embedding, 12 layers)
- **Training Memory**: ~4-8GB GPU memory (batch size 8)
- **Disk Space**: ~2GB for data + checkpoints

### Performance Optimization

1. **Mixed Precision Training**: Uses float16 on GPU for faster training
2. **Gradient Accumulation**: Effective batch size = batch_size × accumulation_steps
3. **Connection Sparsity**: Reduces computation in early training

### Customization

To modify the model architecture:

1. Edit `NanEchoConfig` in `nanecho_model.py`
2. Adjust layer configurations in `NanEchoBlock`
3. Modify learning phases in `train_nanecho.py`

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory**

   - Reduce batch_size in config
   - Increase gradient_accumulation_steps
   - Use smaller model (reduce n_embd or n_layer)

2. **Slow Training**

   - Ensure CUDA is available: `torch.cuda.is_available()`
   - Use mixed precision training (dtype: "float16")
   - Reduce eval_interval for less frequent evaluation

3. **Poor Generation Quality**
   - Train for more iterations
   - Adjust temperature (0.7-0.9 typically works well)
   - Ensure data quality in preparation phase

## 📚 References

- **Echo Self Framework**: Cognitive architecture with adaptive attention
- **Iterative Connection Building**: Inspired by neural development
- **Transformer Architecture**: Based on GPT-style models
- **Hypergraph Patterns**: Neural-symbolic reasoning integration

## 🎉 Summary

The NanEcho model successfully implements:

✅ **Iterative connection building** from 10% to 100% connectivity
✅ **Echo Self cognitive architecture** with all persona dimensions  
✅ **Adaptive attention mechanisms** with dynamic thresholds
✅ **Recursive reasoning** with configurable depth
✅ **Hypergraph pattern encoding** for neural-symbolic reasoning
✅ **Progressive learning phases** with curriculum learning
✅ **Introspection and quality evaluation** during training
✅ **Complete training pipeline** with data preparation

The model starts sparse and grows connections iteratively, building up its cognitive capabilities through structured learning phases, exactly as requested!
