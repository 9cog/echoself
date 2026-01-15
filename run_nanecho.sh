#!/bin/bash

# NanEcho Model Training Pipeline
# Complete end-to-end training with iterative connection building

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════╗"
echo "║         NanEcho Model Training Pipeline                   ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Configuration
DATA_DIR="data/nanecho"
OUT_DIR="out-nanecho"
CONFIG_FILE="nanecho_config.json"
MAX_ITERS=50000
BATCH_SIZE=8

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --max-iters)
      MAX_ITERS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --prepare-only)
      PREPARE_ONLY=true
      shift
      ;;
    --train-only)
      TRAIN_ONLY=true
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  --data-dir DIR      Data directory (default: data/nanecho)"
      echo "  --out-dir DIR       Output directory (default: out-nanecho)"
      echo "  --config FILE       Config file (default: nanecho_config.json)"
      echo "  --max-iters N       Maximum training iterations (default: 50000)"
      echo "  --batch-size N      Batch size (default: 8)"
      echo "  --prepare-only      Only prepare data, don't train"
      echo "  --train-only        Only train, skip data preparation"
      echo "  --help              Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Step 1: Prepare training data
if [ "$TRAIN_ONLY" != "true" ]; then
  echo "📊 Step 1: Preparing Echo Self training data..."
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  python prepare_nanecho_data.py \
    --output_dir "$DATA_DIR" \
    --train_size 1000000 \
    --val_size 100000
  echo ""
fi

if [ "$PREPARE_ONLY" == "true" ]; then
  echo "✅ Data preparation complete. Exiting (--prepare-only flag set)"
  exit 0
fi

# Step 2: Verify data files
echo "🔍 Step 2: Verifying data files..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ ! -f "$DATA_DIR/train.bin" ] || [ ! -f "$DATA_DIR/val.bin" ]; then
  echo "❌ Error: Data files not found in $DATA_DIR"
  echo "   Please run data preparation first or check --data-dir"
  exit 1
fi

TRAIN_SIZE=$(stat -c%s "$DATA_DIR/train.bin" 2>/dev/null || stat -f%z "$DATA_DIR/train.bin" 2>/dev/null)
VAL_SIZE=$(stat -c%s "$DATA_DIR/val.bin" 2>/dev/null || stat -f%z "$DATA_DIR/val.bin" 2>/dev/null)

echo "✅ Data files found:"
echo "   • train.bin: $(numfmt --to=iec-i --suffix=B $TRAIN_SIZE 2>/dev/null || echo "$TRAIN_SIZE bytes")"
echo "   • val.bin: $(numfmt --to=iec-i --suffix=B $VAL_SIZE 2>/dev/null || echo "$VAL_SIZE bytes")"
echo ""

# Step 3: Test model creation
echo "🧠 Step 3: Testing NanEcho model creation..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python -c "
from nanecho_model import create_nanecho_model
model = create_nanecho_model()
print(f'✅ Model created successfully')
print(f'   Parameters: {sum(p.numel() for p in model.parameters()):,}')
print(f'   Initial connections: {model.connection_ratio:.1%}')
"
echo ""

# Step 4: Start training
echo "🚀 Step 4: Starting NanEcho training..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Configuration:"
echo "   • Data directory: $DATA_DIR"
echo "   • Output directory: $OUT_DIR"
echo "   • Config file: $CONFIG_FILE"
echo "   • Max iterations: $MAX_ITERS"
echo "   • Batch size: $BATCH_SIZE"
echo ""

# Create output directory
mkdir -p "$OUT_DIR"

# Run training
if [ -f "$CONFIG_FILE" ]; then
  echo "Using config file: $CONFIG_FILE"
  python train_nanecho.py \
    --config "$CONFIG_FILE" \
    --data_dir "$DATA_DIR" \
    --out_dir "$OUT_DIR" \
    --max_iters "$MAX_ITERS" \
    --batch_size "$BATCH_SIZE"
else
  echo "Config file not found, using command line arguments"
  python train_nanecho.py \
    --data_dir "$DATA_DIR" \
    --out_dir "$OUT_DIR" \
    --max_iters "$MAX_ITERS" \
    --batch_size "$BATCH_SIZE"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║         Training Pipeline Complete!                       ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "📁 Results saved to: $OUT_DIR"
echo "   • Checkpoints: $OUT_DIR/checkpoint_*.pt"
echo "   • Best model: $OUT_DIR/best_model.pt"
echo "   • Tensorboard logs: $OUT_DIR/tensorboard/"
echo "   • Introspection history: $OUT_DIR/introspection_history.json"
echo ""
echo "📊 View training progress with:"
echo "   tensorboard --logdir $OUT_DIR/tensorboard"
echo ""
echo "🎯 Generate text with trained model:"
echo "   python generate_nanecho.py --checkpoint $OUT_DIR/best_model.pt"