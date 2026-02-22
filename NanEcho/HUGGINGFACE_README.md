# HuggingFace Deployment for EchoSelf NanEcho

This directory contains scripts and workflows for deploying trained EchoSelf NanEcho models to HuggingFace Hub and downloading them for incremental training.

## Overview

The HuggingFace integration enables:
- **Model Sharing**: Deploy trained models to HuggingFace Hub for easy sharing
- **Incremental Training**: Download pre-trained models to continue training
- **Version Control**: Track model versions through HuggingFace releases
- **Dataset Distribution**: Share training datasets alongside models

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    HuggingFace Integration                       │
└─────────────────────────────────────────────────────────────────┘
                                  │
                ┌─────────────────┴─────────────────┐
                │                                   │
         ┌──────▼──────┐                    ┌──────▼──────┐
         │  Upload to  │                    │ Download    │
         │ HuggingFace │                    │  from HF    │
         └──────┬──────┘                    └──────┬──────┘
                │                                   │
    ┌───────────┴────────────┐           ┌─────────┴──────────┐
    │                        │           │                    │
┌───▼───┐              ┌────▼────┐  ┌───▼────┐         ┌────▼────┐
│Convert│              │ Dataset │  │Convert │         │Training │
│  to   │              │ Upload  │  │  from  │         │  with   │
│  HF   │              │         │  │   HF   │         │   HF    │
│Format │              │         │  │ Format │         │ Model   │
└───────┘              └─────────┘  └────────┘         └─────────┘
```

## Setup

### 1. Configure HuggingFace Token

Create a HuggingFace access token with **write** permissions:

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Give it a name (e.g., "EchoSelf Deployment")
4. Select **"Write"** permission
5. Click "Generate token"
6. Copy the token

### 2. Add GitHub Secret

Add the token as a GitHub secret named `HFESELF`:

1. Go to your repository settings
2. Navigate to **Secrets and variables** → **Actions**
3. Click **"New repository secret"**
4. Name: `HFESELF`
5. Value: Paste your HuggingFace token
6. Click **"Add secret"**

### 3. Create HuggingFace Repository

Create a model repository on HuggingFace:

1. Go to https://huggingface.co/new
2. Repository type: **Model**
3. Owner: `9cog` (or your organization)
4. Name: `echoself-nanecho`
5. License: MIT (or your choice)
6. Click **"Create model"**

## Usage

### Deploying to HuggingFace

#### Method 1: Manual Workflow Dispatch

1. Go to **Actions** → **Deploy to HuggingFace Hub**
2. Click **"Run workflow"**
3. Configure options:
   - **Source workflow**: `netrain-cached` or `netrain`
   - **Training type**: `ci`, `scheduled`, or `full`
   - **Repo ID**: `9cog/echoself-nanecho`
   - **Upload datasets**: `true` or `false`
   - **Create release**: `true` or `false`
4. Click **"Run workflow"**

#### Method 2: Automatic Post-Training Deployment

The workflow automatically triggers after successful training runs of `netrain-cached.yml`.

#### Method 3: Command Line

```bash
# Convert checkpoint to HuggingFace format
python NanEcho/convert_to_huggingface.py \
  --checkpoint .training-progress/checkpoints/latest_checkpoint.pt \
  --output-dir hf-model

# Upload to HuggingFace Hub
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload 9cog/echoself-nanecho hf-model/ .
```

### Downloading from HuggingFace for Training

#### Method 1: Workflow Dispatch with Download

1. Go to **Actions** → **Train NanEcho Model with Caching**
2. Click **"Run workflow"**
3. Set **"Download from HuggingFace Hub"**: `true`
4. Set **"HuggingFace repository ID"**: `9cog/echoself-nanecho`
5. Click **"Run workflow"**

The training will:
1. Download the model from HuggingFace
2. Convert it to NanEcho checkpoint format
3. Use it as the starting point for incremental training
4. Upload the improved model back after training

#### Method 2: Command Line

```bash
# Download and convert model
python NanEcho/download_from_huggingface.py \
  --repo-id 9cog/echoself-nanecho \
  --output-checkpoint .training-progress/checkpoints/hf_init.pt

# Start training from downloaded checkpoint
python train_cached.py --config training_config.json
```

## Scripts

### `convert_to_huggingface.py`

Converts NanEcho PyTorch checkpoint to HuggingFace format.

**Features:**
- Converts model weights to HuggingFace GPT-2 format
- Creates `config.json` with model architecture
- Generates `tokenizer_config.json` for tiktoken compatibility
- Creates comprehensive model card (`README.md`)
- Preserves training metadata and Echo Self features

**Usage:**
```bash
python NanEcho/convert_to_huggingface.py \
  --checkpoint path/to/checkpoint.pt \
  --output-dir hf-model \
  --config optional_config.json
```

**Output:**
```
hf-model/
├── pytorch_model.bin       # Model weights
├── config.json             # Model configuration
├── tokenizer_config.json   # Tokenizer settings
├── training_metadata.json  # Training info
└── README.md               # Model card
```

### `download_from_huggingface.py`

Downloads model from HuggingFace Hub and converts to NanEcho format.

**Features:**
- Downloads model from HuggingFace Hub
- Converts HuggingFace format back to NanEcho checkpoint
- Preserves Echo Self cognitive architecture features
- Validates model compatibility

**Usage:**
```bash
python NanEcho/download_from_huggingface.py \
  --repo-id 9cog/echoself-nanecho \
  --output-checkpoint out-nanecho/hf_init.pt \
  --token $HF_TOKEN
```

**Environment Variables:**
- `HF_TOKEN`: HuggingFace access token (optional for public repos)

## Workflows

### `deploy-huggingface.yml`

Deploys trained models to HuggingFace Hub.

**Triggers:**
- Manual workflow dispatch
- Automatic after successful `netrain-cached` runs
- Can be scheduled

**Steps:**
1. Locate best checkpoint (git, artifacts, cache)
2. Convert to HuggingFace format
3. Prepare datasets for upload
4. Upload to HuggingFace Hub
5. Optionally create release tag
6. Generate deployment summary

### Training Workflow Integration

Both `netrain.yml` and `netrain-cached.yml` support HuggingFace model downloads:

**New Workflow Inputs:**
- `download_from_hf`: Boolean, download model before training
- `hf_repo_id`: Repository ID to download from

**Integration Flow:**
```
[HuggingFace Hub] ──download──> [Local Checkpoint] ──train──> [Improved Model] ──upload──> [HuggingFace Hub]
```

## Model Card

Each deployed model includes a comprehensive model card with:

- **Model Description**: Overview of Deep Tree Echo architecture
- **Architecture Details**: Layers, dimensions, parameters
- **Training Information**: Iteration, loss, quality score
- **Echo Self Features**: Adaptive attention, persona dimensions
- **Usage Examples**: Code snippets for inference
- **Training Data**: Dataset description
- **Limitations**: Research model disclaimers
- **Citation**: BibTeX citation

## Best Practices

### Deployment

1. **Always test locally first**: Use `convert_to_huggingface.py` to verify conversion
2. **Review model card**: Ensure metadata is accurate before public deployment
3. **Use release tags**: Enable version tracking with `create_release: true`
4. **Monitor disk space**: Large models require significant storage

### Training with HuggingFace Models

1. **Verify compatibility**: Check model architecture matches training config
2. **Test download**: Manually test download before automated runs
3. **Monitor checkpoints**: Ensure HF checkpoint integrates with cache system
4. **Backup checkpoints**: Always maintain git-committed checkpoints as fallback

### Security

1. **Protect tokens**: Never commit `HFESELF` token to repository
2. **Use scoped tokens**: Create tokens with minimal required permissions
3. **Rotate regularly**: Update tokens periodically
4. **Monitor usage**: Check HuggingFace account for unauthorized access

## Continuous Improvement Workflow

The recommended workflow for continuous model improvement:

```
1. Train with netrain-cached.yml (every 6 hours)
   ↓
2. Automatically deploy best checkpoint to HuggingFace
   ↓
3. Next training run downloads latest HuggingFace model
   ↓
4. Continue training for further improvement
   ↓
5. Deploy improved model to HuggingFace
   ↓
6. Repeat cycle
```

This creates a continuous improvement loop where:
- Each training session builds on the best previous model
- Models are versioned and shared via HuggingFace
- Training history is preserved through checkpoints
- Community can access latest versions

## Troubleshooting

### "HFESELF secret not configured"

**Solution:** Add HuggingFace token as GitHub secret (see Setup section)

### "No checkpoint found"

**Causes:**
- Training hasn't completed successfully
- Checkpoint artifacts expired
- Cache was cleared

**Solution:**
- Verify training workflow completed
- Check artifact retention settings (30-90 days)
- Run fresh training session

### "Model architecture mismatch"

**Causes:**
- Training config changed between upload and download
- Incompatible model versions

**Solution:**
- Ensure consistent `n_layer`, `n_embd`, `n_head` settings
- Check config.json in HuggingFace repository
- Use matching training configuration

### "Upload failed"

**Causes:**
- Invalid token
- Insufficient permissions
- Repository doesn't exist

**Solution:**
- Verify token has write permissions
- Check repository exists at specified `repo_id`
- Test token with `huggingface-cli whoami`

## Examples

### Example 1: Deploy After Training

```yaml
# Trigger training
workflow_dispatch:
  - netrain-cached.yml
    training_type: full
    max_iters: 5000

# Wait for completion
# Checkpoint automatically backed up

# Deploy to HuggingFace
workflow_dispatch:
  - deploy-huggingface.yml
    source_workflow: netrain-cached
    training_type: full
    upload_datasets: true
    create_release: true
```

### Example 2: Train from HuggingFace Model

```yaml
# Download and train
workflow_dispatch:
  - netrain-cached.yml
    download_from_hf: true
    hf_repo_id: 9cog/echoself-nanecho
    training_type: full
    max_iters: 2000
```

### Example 3: Full Cycle

```bash
# 1. Train model
gh workflow run netrain-cached.yml \
  -f training_type=full \
  -f max_iters=5000

# 2. Wait for completion (~30 minutes)

# 3. Deploy to HuggingFace
gh workflow run deploy-huggingface.yml \
  -f source_workflow=netrain-cached \
  -f training_type=full \
  -f create_release=true

# 4. Continue training from HuggingFace model
gh workflow run netrain-cached.yml \
  -f download_from_hf=true \
  -f training_type=full \
  -f max_iters=2000
```

## Advanced Configuration

### Custom Repository

Deploy to a different HuggingFace repository:

```bash
gh workflow run deploy-huggingface.yml \
  -f repo_id=myorg/my-custom-model
```

### Private Repositories

For private HuggingFace repositories:
1. Ensure token has access to private repos
2. Set repository visibility on HuggingFace
3. Use same deployment workflow

### Dataset-Only Upload

To update datasets without changing the model:
1. Modify datasets in `data/nanecho/`
2. Run deployment with current checkpoint
3. Datasets will be updated in HuggingFace repository

## Contributing

When contributing to HuggingFace integration:

1. Test conversions locally before committing
2. Verify workflows in fork before creating PR
3. Update documentation for new features
4. Maintain backward compatibility with existing checkpoints

## Support

For issues or questions:
- **GitHub Issues**: https://github.com/9cog/echoself/issues
- **Documentation**: See repository README
- **HuggingFace**: https://huggingface.co/docs

## License

This HuggingFace integration follows the same license as the main EchoSelf repository (MIT).
