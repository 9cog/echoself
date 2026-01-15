# Training Data Size Validation Fix

## Problem

The training process was failing with the following error:

```
RuntimeError: random_ expects 'from' to be less than 'to', but got from=0 >= to=-390
```

This occurred in the `get_batch()` function at line:

```python
ix = torch.randint(len(data) - block_size, (batch_size,))
```

**Root Cause**: The dataset size (`len(data)`) was smaller than the configured `block_size`, making `len(data) - block_size` negative, which creates an invalid range for `torch.randint()`.

## Solution

### 1. Enhanced Training Script (`train.py`)

Created a local training script with comprehensive data validation:

- **Guard Clause**: Validates dataset size before attempting to create batches
- **Clear Error Messages**: Provides actionable error messages with suggestions
- **Compatibility**: Works with existing configuration files
- **Enhanced `get_batch()` Function**: Includes proper validation

**Key Features:**

```python
def get_batch(split, data, block_size, batch_size, device, device_type):
    # Validate data size before attempting to create batches
    if len(data) <= block_size:
        raise ValueError(
            f"Insufficient data for batch generation: "
            f"len(data)={len(data)}, block_size={block_size}. "
            f"Dataset must be larger than block_size."
        )

    ix = torch.randint(len(data) - block_size, (batch_size,))
    # ... rest of function
```

### 2. Data Preparation Validation

Enhanced both data preparation scripts:

#### `NanEcho/prepare.py`

- Added dataset size validation against typical block sizes (256-1024)
- Provides early warnings when datasets might be too small
- Prevents creation of unusable datasets

#### `NanEcho/prepare_nanecho.py`

- Validates Echo Self dataset size during preparation
- Returns appropriate error codes when data is insufficient
- Suggests solutions for small datasets

### 3. Validation Thresholds

```python
MIN_BLOCK_SIZE = 256   # Minimum reasonable block size
MAX_BLOCK_SIZE = 1024  # From train configs (train_nanecho.py, train_cogprime.py)
```

## Usage

### Using the Enhanced Training Script

```bash
# Use the local training script instead of external nanoGPT
python train.py config/train_nanecho.py

# Or with custom configuration
python train.py config/train_cogprime.py
```

### Data Preparation with Validation

```bash
# Prepare data with automatic validation
cd NanEcho
python prepare_nanecho.py --echo_depth 3 --persona_weight 0.7

# Check output for validation warnings:
# ✅ Dataset validation passed: 15,432 tokens
# ⚠️  Warning: Dataset smaller than block_size - consider adding more data
```

### Configuration Adjustment

If you encounter validation errors, you have these options:

1. **Increase Data Size** (Recommended):

   ```bash
   # Generate more training data
   python prepare_nanecho.py --echo_depth 5 --persona_weight 0.9
   ```

2. **Reduce Block Size**:

   ```python
   # In your train config file
   block_size = 512  # Reduce from 1024 if dataset is smaller
   ```

3. **Add More Source Content**:
   - Ensure `echoself.md` and related files exist and have sufficient content
   - Add more Echo Self patterns to source files

## Error Messages and Solutions

### "Insufficient data for batch generation"

```
ValueError: Insufficient data for batch generation: len(data)=300, block_size=1024.
Dataset must be larger than block_size.
```

**Solutions:**

1. Run data preparation with higher parameters: `--echo_depth 5 --persona_weight 1.0`
2. Reduce `block_size` in training config to `< 300`
3. Add more source content to Echo Self files

### "Dataset too small for training"

```
❌ Error: Dataset too small (150 tokens < 256)
Cannot proceed with training.
```

**Solutions:**

1. Check that source files (`echoself.md`, etc.) exist and contain content
2. Verify Echo Self pattern matching is working correctly
3. Increase synthetic sample generation parameters

### "Training split smaller than block_size"

```
⚠️  Warning: Training split (800 tokens) smaller than block_size (1024)
```

**Solutions:**

1. Generate more training data
2. Adjust train/validation split ratio
3. Reduce `block_size` in configuration

## Validation Workflow

The enhanced system performs validation at multiple stages:

1. **Data Preparation**: Validates dataset size during creation
2. **Training Start**: Validates datasets before loading model
3. **Batch Creation**: Validates data size for each batch request

## Testing

Test the validation logic:

```bash
# Simple logic test
python3 -c "
data_len, block_size = 300, 390
if data_len <= block_size:
    print('✅ Would catch the error')
else:
    print('❌ Error condition not detected')
"
```

## Integration with CI/CD

The GitHub workflows will automatically use the enhanced validation:

1. **Data Preparation** (`prepare_nanecho.py`) validates during dataset creation
2. **Training** uses the local `train.py` with validation
3. **Error Reporting** provides clear messages for debugging

## Backward Compatibility

- Existing configuration files continue to work unchanged
- External nanoGPT repository can still be used (but without validation)
- All validation is additive - no breaking changes to existing functionality

## Summary

This fix prevents the `torch.randint` error by:

- ✅ Validating dataset size before training
- ✅ Providing clear, actionable error messages
- ✅ Suggesting specific solutions for each error case
- ✅ Maintaining compatibility with existing configurations
- ✅ Adding validation at multiple pipeline stages

The solution is **minimal, surgical, and focused** on preventing the specific error while providing guidance for resolution.
