# Disk Space Analysis Report

## Problem Statement

The `/dev/root` filesystem was using 84% (61G/72G) of available space, causing concerns about running out of disk space during builds and operations.

## Analysis Performed

Date: 2025-12-03

### Overall Disk Usage

```
Filesystem      Size  Used Avail Use% Mounted on
/dev/root        72G   58G   15G  80% /
tmpfs           3.9G   84K  3.9G   1% /dev/shm
tmpfs           1.6G  1.1M  1.6G   1% /run
tmpfs           5.0M     0  5.0M   0% /run/lock
/dev/sda16      881M   62M  758M   8% /boot
/dev/sda15      105M  6.2M   99M   6% /boot/efi
```

### Major Space Consumers

#### 1. /usr Directory (41G - 71% of total usage)

**Breakdown:**

- `/usr/local`: 24G

  - `/usr/local/lib/android`: 12G - Android SDK
  - `/usr/local/.ghcup`: 6.4G - Haskell toolchain
  - `/usr/local/share`: 2.1G
  - `/usr/local/julia1.12.2`: 1015M - Julia installation
  - `/usr/local/bin`: 933M
  - `/usr/local/aws-cli`: 238M
  - `/usr/local/aws-sam-cli`: 206M
  - `/usr/local/n`: 167M

- `/usr/share`: 9.3G

  - `/usr/share/dotnet`: 4G - .NET SDK
  - `/usr/share/swift`: 3.2G - Swift toolchain
  - `/usr/share/miniconda`: 802M
  - `/usr/share/az_12.5.0`: 496M - Azure CLI
  - `/usr/share/gradle-9.2.1`: 144M
  - `/usr/share/kotlinc`: 83M

- `/usr/lib`: 6.6G

#### 2. /opt Directory (8.5G - 15% of total usage)

**Breakdown:**

- `/opt/hostedtoolcache`: 5.8G - GitHub Actions hosted toolcache
- `/opt/microsoft`: 783M
- `/opt/az`: 666M
- `/opt/pipx`: 514M
- `/opt/google`: 374M
- `/opt/actionarchivecache`: 243M
- `/opt/runner-cache`: 217M

#### 3. Other Directories

- `/home`: 2.6G
- `/var`: 496M
- `/etc`: 690M

## Root Causes

The disk space usage is primarily due to:

1. **Pre-installed Development Tools**: GitHub Actions runners come with numerous pre-installed SDKs and toolchains
   - Android SDK (12G)
   - Haskell toolchain (6.4G)
   - .NET SDK (4G)
   - Swift toolchain (3.2G)
2. **Hosted Toolcache**: GitHub Actions caching system (5.8G)

3. **Multiple Language Runtimes**: Julia, AWS tools, Azure tools, etc.

## Recommendations

### For GitHub Actions Workflows

If this analysis is being run in a GitHub Actions context, consider:

1. **Use Minimal Runners**: If available, use minimal/slim runner images that don't include all pre-installed tools

2. **Remove Unused Tools**: At the start of workflows, remove unused SDKs:

   ```yaml
   - name: Free Disk Space
     run: |
       sudo rm -rf /usr/local/lib/android
       sudo rm -rf /usr/local/.ghcup
       sudo rm -rf /usr/share/dotnet
       sudo rm -rf /usr/share/swift
       sudo rm -rf /opt/hostedtoolcache
   ```

3. **Use cleanup-disk-space Action**: Consider using pre-built actions like:
   ```yaml
   - name: Free Disk Space (Ubuntu)
     uses: jlumbroso/free-disk-space@v1.3.1
     with:
       android: true
       dotnet: true
       haskell: true
       large-packages: true
       docker-images: true
       swap-storage: true
   ```

### For Local Development

1. **Clean Package Caches**: Regularly clean package manager caches

   ```bash
   sudo apt-get clean
   pip cache purge
   npm cache clean --force
   ```

2. **Remove Unused Docker Images**: If Docker is in use

   ```bash
   docker system prune -a
   ```

3. **Uninstall Unused SDKs**: Remove development tools that aren't needed

## Monitoring

Use the provided script to monitor disk usage:

```bash
bash scripts/analyze_disk_space.sh
```

This will generate a comprehensive report of disk usage across the system.

## Conclusion

The high disk usage (80-84%) is primarily caused by pre-installed development tools in the GitHub Actions runner environment. These can be safely removed at the start of workflows if they're not needed for the specific build process. For this repository, which appears to focus on Python-based cognitive architecture and machine learning, the Android SDK, Haskell toolchain, .NET SDK, and Swift toolchain are likely unnecessary and can be removed to free up approximately 25G of space.
