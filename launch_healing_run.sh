#!/bin/bash
# Vast.ai Launcher for EchoSelf Healing Run
set -e

# Configuration
INSTANCE_TYPE="RTX 3090"
IMAGE="pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel"
DISK_GB=50

echo "🚀 Deep Tree Echo - Vast.ai Healing Run Launcher"

if [ -z "$VAST_API_KEY" ]; then
    echo "❌ Error: VAST_API_KEY environment variable not set."
    echo "Source the .env file first: source /mnt/cx5l1vpt0nguxqdwvte64scrh/ubuntu/echoo/.env"
    exit 1
fi

echo "🔍 Searching for suitable GPU instance..."
# Find cheapest available RTX 3090/4090
OFFER=$(curl -s -H "Authorization: Bearer $VAST_API_KEY" \
    "https://console.vast.ai/api/v0/bundles/?q={\"gpu_name\":{\"\$in\":[\"RTX 3090\",\"RTX 4090\"]},\"verified\":{\"\$eq\":true},\"rentable\":{\"\$eq\":true},\"disk_space\":{\"\$gte\":$DISK_GB}}&order=dph_base" | \
    python3 -c "
import sys, json
data = json.load(sys.stdin)
offers = data.get('offers', [])
if not offers:
    print('ERROR: No offers found')
    sys.exit(1)
best = offers[0]
print(f\"{best['id']}|{best['gpu_name']}|{best['dph_base']:.3f}\")
")

if [[ $OFFER == ERROR* ]]; then
    echo "❌ No suitable instances found."
    exit 1
fi

OFFER_ID=$(echo $OFFER | cut -d'|' -f1)
GPU_NAME=$(echo $OFFER | cut -d'|' -f2)
PRICE=$(echo $OFFER | cut -d'|' -f3)

echo "✅ Found $GPU_NAME at \$$PRICE/hr (Offer ID: $OFFER_ID)"

echo "🚀 Launching instance..."
LAUNCH_RES=$(curl -s -X PUT -H "Authorization: Bearer $VAST_API_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"client_id\": \"local\", \"image\": \"$IMAGE\", \"disk\": $DISK_GB, \"onstart\": \"apt-get update && apt-get install -y git && pip install -r /workspace/echoself/requirements.txt\"}" \
    "https://console.vast.ai/api/v0/asks/$OFFER_ID/")

NEW_ID=$(echo $LAUNCH_RES | python3 -c "import sys, json; print(json.load(sys.stdin).get('new_contract', ''))")

if [ -z "$NEW_ID" ]; then
    echo "❌ Failed to launch instance: $LAUNCH_RES"
    exit 1
fi

echo "✅ Instance launched! Contract ID: $NEW_ID"
echo "⏳ Waiting for instance to boot and SSH to become available (this takes a few minutes)..."

# In a real run, we would loop and check status here, then rsync the repo and run:
# ssh -p $PORT root@$IP "cd /workspace/echoself && python3 prepare_healing_data.py && python3 netrain/cli/train.py build netrain_gpu.yml"

echo "
To connect and start training once booted:
1. Check status: vastai show instances
2. Sync repo: rsync -avz -e 'ssh -p PORT' /home/ubuntu/echoself/ root@IP:/workspace/echoself/
3. Run: ssh -p PORT root@IP 'cd /workspace/echoself && python3 netrain/cli/__main__.py build netrain_gpu.yml'
"
