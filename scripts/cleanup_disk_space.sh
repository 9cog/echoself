#!/bin/bash

# Disk Space Cleanup Script for GitHub Actions
# This script removes common space-consuming tools that are pre-installed
# on GitHub Actions runners but may not be needed for all workflows.

set -e

# Define paths for easier maintenance
ANDROID_SDK_PATH="/usr/local/lib/android"
GHCUP_PATH="/usr/local/.ghcup"
DOTNET_PATH="/usr/share/dotnet"
SWIFT_PATH="/usr/share/swift"
TOOLCACHE_PATH="/opt/hostedtoolcache"

# Create temporary directory with restrictive permissions
TMPDIR=$(mktemp -d -m 700)

# Ensure cleanup on exit or error
trap 'cleanup_temp_dir' EXIT ERR

cleanup_temp_dir() {
    # Safety checks before cleanup:
    # 1. Variable is not empty
    # 2. Not root directory
    # 3. Directory exists
    # 4. Directory is actually a directory (not a symlink)
    if [ -n "$TMPDIR" ] && [ "$TMPDIR" != "/" ] && [ -d "$TMPDIR" ] && [ ! -L "$TMPDIR" ]; then
        rm -rf "$TMPDIR"
    fi
}

echo "=== Disk Space Cleanup Script ==="
echo "This script will remove common pre-installed tools to free up disk space."
echo ""

# Show current disk usage
echo "=== Disk Usage BEFORE Cleanup ==="
df -h /
echo ""

# Calculate space to be freed (parallel for performance)
echo "Calculating sizes..."
(du -sh "$ANDROID_SDK_PATH" 2>/dev/null | cut -f1 > "$TMPDIR/android_size.txt" || echo "0" > "$TMPDIR/android_size.txt") &
(du -sh "$GHCUP_PATH" 2>/dev/null | cut -f1 > "$TMPDIR/ghcup_size.txt" || echo "0" > "$TMPDIR/ghcup_size.txt") &
(du -sh "$DOTNET_PATH" 2>/dev/null | cut -f1 > "$TMPDIR/dotnet_size.txt" || echo "0" > "$TMPDIR/dotnet_size.txt") &
(du -sh "$SWIFT_PATH" 2>/dev/null | cut -f1 > "$TMPDIR/swift_size.txt" || echo "0" > "$TMPDIR/swift_size.txt") &
(du -sh "$TOOLCACHE_PATH" 2>/dev/null | cut -f1 > "$TMPDIR/toolcache_size.txt" || echo "0" > "$TMPDIR/toolcache_size.txt") &
wait

ANDROID_SIZE=$(cat "$TMPDIR/android_size.txt" 2>/dev/null || echo "0")
GHCUP_SIZE=$(cat "$TMPDIR/ghcup_size.txt" 2>/dev/null || echo "0")
DOTNET_SIZE=$(cat "$TMPDIR/dotnet_size.txt" 2>/dev/null || echo "0")
SWIFT_SIZE=$(cat "$TMPDIR/swift_size.txt" 2>/dev/null || echo "0")
TOOLCACHE_SIZE=$(cat "$TMPDIR/toolcache_size.txt" 2>/dev/null || echo "0")

# Note: Temp directory cleanup is handled by trap on EXIT

echo "Estimated space to be freed:"
echo "  - Android SDK: $ANDROID_SIZE"
echo "  - Haskell (.ghcup): $GHCUP_SIZE"
echo "  - .NET SDK: $DOTNET_SIZE"
echo "  - Swift: $SWIFT_SIZE"
echo "  - Hosted Toolcache: $TOOLCACHE_SIZE"
echo ""

# Confirm cleanup (skip in non-interactive mode)
if [ -t 0 ]; then
    read -p "Do you want to proceed with cleanup? (yes/no): " CONFIRM
    if [ "$CONFIRM" != "yes" ]; then
        echo "Cleanup cancelled."
        exit 0
    fi
fi

echo "Starting cleanup..."
echo ""

# Remove Android SDK
if [ -d "$ANDROID_SDK_PATH" ]; then
    echo "Removing Android SDK..."
    sudo rm -rf "$ANDROID_SDK_PATH"
    echo "✓ Android SDK removed"
fi

# Remove Haskell toolchain
if [ -d "$GHCUP_PATH" ]; then
    echo "Removing Haskell toolchain (.ghcup)..."
    sudo rm -rf "$GHCUP_PATH"
    echo "✓ Haskell toolchain removed"
fi

# Remove .NET SDK
if [ -d "$DOTNET_PATH" ]; then
    echo "Removing .NET SDK..."
    sudo rm -rf "$DOTNET_PATH"
    echo "✓ .NET SDK removed"
fi

# Remove Swift
if [ -d "$SWIFT_PATH" ]; then
    echo "Removing Swift..."
    sudo rm -rf "$SWIFT_PATH"
    echo "✓ Swift removed"
fi

# Remove hosted toolcache
# WARNING: This may affect GitHub Actions setup-* actions (e.g., actions/setup-python,
# actions/setup-node, actions/setup-java) if they expect to find cached tool versions.
# These actions will still work but may need to download tools fresh, increasing build time.
# Only remove if you're installing tools fresh in each workflow or using specific versions
# via package managers (apt, pip, npm, etc.) instead of relying on setup-* actions.
if [ -d "$TOOLCACHE_PATH" ]; then
    echo "Removing hosted toolcache..."
    sudo rm -rf "$TOOLCACHE_PATH"
    echo "✓ Hosted toolcache removed"
fi

# Clean apt cache
echo "Cleaning apt cache..."
sudo apt-get clean
echo "✓ apt cache cleaned"

# Clean old logs
echo "Cleaning old logs..."
sudo journalctl --vacuum-time=1d 2>/dev/null || echo "  (journalctl not available)"
echo "✓ Logs cleaned"

echo ""
echo "=== Disk Usage AFTER Cleanup ==="
df -h /
echo ""

echo "✓ Cleanup complete!"
echo ""
echo "Note: If you need any of these tools in your workflow, do not run this script"
echo "or modify it to keep the tools you need."
