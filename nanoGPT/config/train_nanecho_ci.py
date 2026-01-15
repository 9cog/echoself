# train_nanecho_ci.py
# Minimal configuration for CI testing

# Output directory for checkpoints
out_dir = 'out-nanecho-ci'

# Data
dataset = 'nanecho'
gradient_accumulation_steps = 1  # Reduced for CI
batch_size = 4  # Reduced for CI
block_size = 1024

# Model - smaller for CI
n_layer = 4  # Reduced for CI
n_head = 4  # Reduced for CI
n_embd = 128  # Reduced for CI
dropout = 0.0
bias = False

# AdamW optimizer
learning_rate = 6e-4
max_iters = 10  # Very few iterations for CI
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate decay
decay_lr = True
warmup_iters = 2
lr_decay_iters = 10
min_lr = 6e-5

# Logging
eval_interval = 5
log_interval = 1
eval_iters = 2  # Reduced for CI
eval_only = False
always_save_checkpoint = False  # Don't save in CI

# DDP settings
backend = 'nccl'

# System
device = 'cpu'  # Use CPU for CI
dtype = 'float32'  # Use float32 for CI compatibility
compile = False  # Disable compilation for CI