# Disk Space Management Scripts - Quick Reference

## Overview

This directory contains scripts for analyzing and managing disk space, particularly useful in CI/CD environments.

## Scripts

### 1. analyze_disk_space.sh

**Purpose**: Comprehensive disk space analysis

**Usage**:

```bash
bash scripts/analyze_disk_space.sh
```

**Output**: Detailed report showing:

- Overall disk usage
- Top-level directory breakdown
- Analysis of /usr, /opt, /var directories
- Identification of large files (>100MB)
- GitHub Actions toolcache analysis

**When to use**:

- Before running disk-intensive operations
- To identify what's consuming space
- In CI/CD pipelines for monitoring

### 2. cleanup_disk_space.sh

**Purpose**: Remove common pre-installed tools to free up disk space

**Usage**:

```bash
bash scripts/cleanup_disk_space.sh
```

**Interactive mode**: Script will ask for confirmation before cleanup

**Non-interactive mode**: Set environment variable to skip confirmation

```bash
echo "yes" | bash scripts/cleanup_disk_space.sh
```

**What it removes**:

- Android SDK (~12G) - `/usr/local/lib/android`
- Haskell toolchain (~6.4G) - `/usr/local/.ghcup`
- .NET SDK (~4G) - `/usr/share/dotnet`
- Swift toolchain (~3.2G) - `/usr/share/swift`
- Hosted toolcache (~5.8G) - `/opt/hostedtoolcache`
- APT cache
- Old logs

**⚠️ Warning**: Only run this if you're sure you don't need these tools!

**When to use**:

- In GitHub Actions workflows that need more disk space
- When disk usage is >80%
- Before large builds or tests

## GitHub Actions Integration

### Manual Workflow

Use the "Disk Space Management" workflow in GitHub Actions:

```yaml
# .github/workflows/disk-space-management.yml
# Trigger: workflow_dispatch (manual)
```

Options:

- **Action**: analyze, cleanup, or analyze-and-cleanup
- **Selective removal**: Choose which SDKs to remove (checkboxes)

### In Your Workflows

Add cleanup step to free space:

```yaml
steps:
  - name: Free Disk Space
    run: |
      echo "=== Disk Space Before ===" 
      df -h
      sudo rm -rf /usr/local/lib/android
      sudo rm -rf /usr/local/.ghcup
      sudo rm -rf /usr/share/dotnet
      sudo rm -rf /usr/share/swift
      echo "=== Disk Space After ===" 
      df -h
```

Or use the cleanup script:

```yaml
steps:
  - name: Checkout
    uses: actions/checkout@v4

  - name: Free Disk Space
    run: echo "yes" | bash scripts/cleanup_disk_space.sh
```

### Using Third-Party Actions

Alternative: Use existing cleanup actions:

```yaml
steps:
  - name: Free Disk Space (Ubuntu)
    # Using specific version for reproducible builds
    uses: jlumbroso/free-disk-space@v1.3.1
    with:
      android: true
      dotnet: true
      haskell: true
      large-packages: true
      docker-images: true
```

## Common Disk Space Issues

### Issue: "No space left on device"

**Solution**:

```bash
# Quick cleanup
sudo rm -rf /usr/local/lib/android
sudo apt-get clean
df -h  # Check if space freed
```

### Issue: Build fails due to disk space

**Solution**: Add cleanup at start of workflow

```yaml
- name: Free space before build
  run: bash scripts/cleanup_disk_space.sh
```

### Issue: Want to keep some tools but free space

**Solution**: Edit `cleanup_disk_space.sh` and comment out sections you want to keep

## Monitoring Disk Space

### Quick check:

```bash
df -h /
```

### Detailed analysis:

```bash
bash scripts/analyze_disk_space.sh > disk-report.txt
cat disk-report.txt
```

### Find largest directories:

```bash
du -h -d 1 / 2>/dev/null | sort -hr | head -20
```

### Find largest files:

```bash
find / -type f -size +100M -exec du -h {} + 2>/dev/null | sort -hr | head -20
```

## Best Practices

1. **Run analysis first**: Always analyze before cleanup to know what you're removing
2. **Selective cleanup**: Only remove what you don't need for your project
3. **Document decisions**: Add comments in workflows explaining why you removed specific tools
4. **Monitor regularly**: Set up periodic checks in CI/CD to catch space issues early
5. **Use artifacts wisely**: Clean up old artifacts to save space

## Troubleshooting

### Script permission denied

```bash
chmod +x scripts/analyze_disk_space.sh
chmod +x scripts/cleanup_disk_space.sh
```

### Cleanup doesn't free enough space

```bash
# Check Docker images
docker system df
docker system prune -a

# Check package caches
sudo apt-get clean
pip cache purge
npm cache clean --force
```

### Need to restore removed tools

Unfortunately, removed tools cannot be easily restored. You'll need to:

- Re-run the workflow in a fresh environment
- Or manually reinstall the tools:

**Android SDK**:

```bash
# Via Android Studio
# https://developer.android.com/studio

# Or via command line tools
wget https://dl.google.com/android/repository/commandlinetools-linux-latest.zip
unzip commandlinetools-linux-latest.zip -d /usr/local/lib/android
```

**.NET SDK**:

```bash
# Via Microsoft installer
wget https://dot.net/v1/dotnet-install.sh
sudo bash dotnet-install.sh --install-dir /usr/share/dotnet
```

**Haskell (GHCup)**:

```bash
# Via GHCup installer
curl --proto '=https' --tlsv1.2 -sSf https://get-ghcup.haskell.org | sh
```

**Swift**:

```bash
# Download from Swift.org
# https://swift.org/download/
wget https://download.swift.org/swift-5.x.x-release/ubuntu2204/swift-5.x.x-RELEASE/swift-5.x.x-RELEASE-ubuntu22.04.tar.gz
sudo tar xzf swift-*.tar.gz -C /usr/share/
```

**Hosted Toolcache**:

- This will be automatically repopulated by GitHub Actions setup actions (setup-python, setup-node, etc.)
- No manual restoration needed

## For More Information

See [DISK_SPACE_ANALYSIS.md](../DISK_SPACE_ANALYSIS.md) for detailed analysis and recommendations.
