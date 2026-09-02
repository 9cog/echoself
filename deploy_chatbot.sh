#!/bin/bash
# =============================================================================
# Matula Transformer: Train + Deploy as Interactive Chatbot
# =============================================================================
# 
# This script handles the full pipeline:
#   1. Train the model on Vast.ai GPU (or locally)
#   2. Deploy as an interactive chatbot (terminal or web)
#
# Usage:
#   # Train locally (CPU, slow but works)
#   ./deploy_chatbot.sh train-local small 5
#
#   # Train on Vast.ai (GPU, fast)
#   ./deploy_chatbot.sh train-vast medium 50
#
#   # Run chatbot (terminal)
#   ./deploy_chatbot.sh chat-terminal small checkpoints/best.pt
#
#   # Run chatbot (web, port 8080)
#   ./deploy_chatbot.sh chat-web small checkpoints/best.pt 8080
#
#   # Demo mode (no training needed)
#   ./deploy_chatbot.sh demo small
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${PURPLE}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║     MATULA TRANSFORMER — Training & Chatbot Deployment       ║"
echo "║     486 heads | 9 layers | OEIS A000081                      ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

CMD="${1:-help}"
VARIANT="${2:-small}"
ARG3="${3:-5}"
ARG4="${4:-8080}"

case "$CMD" in
    train-local)
        echo -e "${GREEN}Training locally (CPU)...${NC}"
        echo "  Variant: $VARIANT"
        echo "  Epochs: $ARG3"
        echo ""
        python3 train_matula.py \
            --variant "$VARIANT" \
            --device cpu \
            --epochs "$ARG3" \
            --batch_size 2 \
            --lr 3e-4 \
            --data data/deep_echo/cognitive_cycles.jsonl \
            --checkpoint_dir "checkpoints/${VARIANT}" \
            --block_size 256
        echo -e "${GREEN}Training complete! Checkpoint: checkpoints/${VARIANT}/best.pt${NC}"
        ;;
    
    train-vast)
        echo -e "${GREEN}Training on Vast.ai (GPU)...${NC}"
        echo "  Variant: $VARIANT"
        echo "  Epochs: $ARG3"
        echo ""
        
        # Check for VAST_API_KEY
        if [ -f ~/echoo/.env ]; then
            source ~/echoo/.env
        fi
        
        if [ -z "$VAST_API_KEY" ]; then
            echo -e "${RED}Error: VAST_API_KEY not set. Add it to ~/echoo/.env${NC}"
            exit 1
        fi
        
        echo "  VAST_API_KEY: ${VAST_API_KEY:0:8}..."
        echo ""
        echo "  This will:"
        echo "    1. Find cheapest GPU offer (>=16GB VRAM)"
        echo "    2. Upload training code"
        echo "    3. Run training"
        echo "    4. Download checkpoint"
        echo "    5. Destroy instance"
        echo ""
        echo "  Estimated cost: \$0.05-0.50 depending on variant"
        echo ""
        
        python3 launch_235_healing.py \
            --variant "$VARIANT" \
            --epochs "$ARG3"
        ;;
    
    chat-terminal)
        CHECKPOINT="$ARG3"
        echo -e "${BLUE}Starting terminal chatbot...${NC}"
        echo "  Variant: $VARIANT"
        echo "  Checkpoint: $CHECKPOINT"
        echo ""
        python3 chat_matula.py \
            --mode terminal \
            --variant "$VARIANT" \
            --checkpoint "$CHECKPOINT" \
            --device cpu
        ;;
    
    chat-web)
        CHECKPOINT="$ARG3"
        PORT="$ARG4"
        echo -e "${BLUE}Starting web chatbot on port $PORT...${NC}"
        echo "  Variant: $VARIANT"
        echo "  Checkpoint: $CHECKPOINT"
        echo "  URL: http://localhost:$PORT"
        echo ""
        python3 chat_matula.py \
            --mode web \
            --variant "$VARIANT" \
            --checkpoint "$CHECKPOINT" \
            --device cpu \
            --port "$PORT"
        ;;
    
    demo)
        echo -e "${BLUE}Starting demo chatbot (untrained model)...${NC}"
        echo "  Variant: $VARIANT"
        echo "  NOTE: Output will be random, but cognitive state tracking works"
        echo ""
        python3 chat_matula.py \
            --mode terminal \
            --variant "$VARIANT" \
            --demo \
            --device cpu
        ;;
    
    demo-web)
        PORT="$ARG3"
        echo -e "${BLUE}Starting web demo on port $PORT...${NC}"
        echo "  Variant: $VARIANT"
        echo "  URL: http://localhost:$PORT"
        echo ""
        python3 chat_matula.py \
            --mode web \
            --variant "$VARIANT" \
            --demo \
            --device cpu \
            --port "${PORT:-8080}"
        ;;
    
    help|*)
        echo "Usage: $0 <command> [variant] [arg3] [arg4]"
        echo ""
        echo "Commands:"
        echo "  train-local <variant> <epochs>     Train locally on CPU"
        echo "  train-vast  <variant> <epochs>     Train on Vast.ai GPU"
        echo "  chat-terminal <variant> <ckpt>     Terminal chatbot"
        echo "  chat-web <variant> <ckpt> <port>   Web chatbot"
        echo "  demo <variant>                     Demo (untrained, terminal)"
        echo "  demo-web <variant> <port>          Demo (untrained, web)"
        echo ""
        echo "Variants: small (33.8M), medium (134M), 719 (full 1205-head)"
        echo ""
        echo "Examples:"
        echo "  $0 train-local small 10"
        echo "  $0 demo small"
        echo "  $0 demo-web small 8080"
        echo "  $0 chat-web medium checkpoints/medium/best.pt 8080"
        ;;
esac
