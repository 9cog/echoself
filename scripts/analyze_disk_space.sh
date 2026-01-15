#!/bin/bash

# Disk Space Analysis Script
# Analyzes disk usage on the system and identifies major space consumers

set -e

# Configuration
FOCUS_DIRS="/usr /opt /var /home"  # Primary directories to analyze

echo "=== Disk Space Analysis ==="
echo ""
echo "Timestamp: $(date)"
echo ""

# Overall disk usage
echo "=== Overall Disk Usage ==="
df -h
echo ""

# Top-level directories (focused on major directories)
echo "=== Top-level Directory Usage ==="
for dir in $FOCUS_DIRS; do
    if [ -d "$dir" ]; then
        du -sh "$dir" 2>/dev/null
    fi
done | sort -hr
# Also show root level summary
du -sh / 2>/dev/null | head -1
echo ""

# /usr breakdown (typically the largest)
echo "=== /usr Directory Breakdown ==="
du -h -d 1 /usr 2>/dev/null | sort -hr | head -15
echo ""

# /usr/local breakdown
echo "=== /usr/local Directory Breakdown ==="
du -h -d 1 /usr/local 2>/dev/null | sort -hr | head -15
echo ""

# /usr/local/lib breakdown (often contains large SDKs)
echo "=== /usr/local/lib Directory Breakdown ==="
du -h -d 1 /usr/local/lib 2>/dev/null | sort -hr | head -15
echo ""

# /usr/share breakdown
echo "=== /usr/share Directory Breakdown ==="
du -h -d 1 /usr/share 2>/dev/null | sort -hr | head -15
echo ""

# /opt breakdown
echo "=== /opt Directory Breakdown ==="
du -h -d 1 /opt 2>/dev/null | sort -hr | head -15
echo ""

# /var breakdown
echo "=== /var Directory Breakdown ==="
du -h -d 1 /var 2>/dev/null | sort -hr | head -15
echo ""

# Docker usage (if available)
echo "=== Docker Disk Usage ==="
if command -v docker &> /dev/null; then
    docker system df 2>/dev/null || echo "Docker not accessible"
else
    echo "Docker not installed"
fi
echo ""

# Package cache sizes
echo "=== Package Cache Sizes ==="
echo -n "APT cache: "
du -sh /var/cache/apt 2>/dev/null || echo "N/A"
echo -n "pip cache: "
du -sh ~/.cache/pip 2>/dev/null || echo "N/A"
echo -n "npm cache: "
du -sh ~/.npm 2>/dev/null || echo "N/A"
echo ""

# Large files in the system (limited to focus directories for performance)
echo "=== Top 20 Largest Files (> 100MB in key directories) ==="
for dir in $FOCUS_DIRS; do
    find "$dir" -type f -size +100M -exec du -h {} + 2>/dev/null
done | sort -hr | head -20 || echo "Unable to scan all directories"
echo ""

# GitHub Actions runner cache
if [ -d "/opt/hostedtoolcache" ]; then
    echo "=== GitHub Actions Hosted Toolcache ==="
    du -h -d 1 /opt/hostedtoolcache 2>/dev/null | sort -hr | head -10
    echo ""
fi

# Summary
echo "=== SUMMARY ==="
echo "Analysis complete. Review the output above to identify space-consuming directories."
echo "Common culprits in CI/CD environments:"
echo "  - Android SDK (/usr/local/lib/android)"
echo "  - .NET SDK (/usr/share/dotnet)"
echo "  - Swift toolchain (/usr/share/swift)"
echo "  - Haskell toolchain (/usr/local/.ghcup)"
echo "  - Julia installation (/usr/local/julia*)"
echo "  - Hosted toolcache (/opt/hostedtoolcache)"
echo ""
