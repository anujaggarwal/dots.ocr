# DOTS OCR Complete Installation Guide

This guide documents the complete process to install DOTS OCR with all dependencies, including solutions to common issues we encountered.

## Overview of Installation Challenges

During our installation, we faced several critical issues:
1. **CUDA Memory Issues** - Model running out of GPU memory without flash-attn
2. **Flash-attn Compilation Failures** - Missing CUDA toolkit and nvcc compiler
3. **Dependency Conflicts** - PyTorch versions, transformers compatibility
4. **Model Loading Issues** - Import errors with model names containing dots
5. **Memory Optimization** - Need for gradient checkpointing and proper settings

## Prerequisites

- **GPU**: NVIDIA GPU with CUDA support (tested on A10G with 22GB VRAM)
- **OS**: Ubuntu 20.04+ (tested on 24.04)
- **Python**: 3.10+ (tested on 3.12)
- **Memory**: 16GB+ RAM recommended
- **Disk**: 50GB+ free space for models and dependencies

## Step-by-Step Installation

### 1. Check GPU and Driver Status

```bash
# Check GPU availability
nvidia-smi

# Should show:
# - GPU name and memory
# - CUDA Version (we had 12.9)
# - Driver version
```

**Expected Output:**
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 575.57.08              Driver Version: 575.57.08      CUDA Version: 12.9     |
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv dots_env
source dots_env/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 3. Install PyTorch with CUDA Support

**Critical**: Install the correct PyTorch version that matches your CUDA driver.

```bash
# We used PyTorch 2.7.0 with CUDA 12.8 (compatible with driver 12.9)
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

**Issue Encountered**: Initially tried different versions, this combination worked.

### 4. Install CUDA Toolkit (Critical for Flash-Attn)

This was our biggest challenge. Flash-attn requires `nvcc` compiler which comes with CUDA toolkit.

```bash
# Download CUDA keyring
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb

# Install keyring
sudo dpkg -i cuda-keyring_1.0-1_all.deb

# Update package lists
sudo apt update

# Install CUDA compiler and development tools
sudo apt install -y cuda-nvcc-12-1 cuda-cudart-dev-12-1
```

**Why This Failed Initially**: We tried installing the full `cuda-toolkit-12-1` package which had dependency conflicts with `libtinfo5`. The minimal installation above works.

### 5. Set CUDA Environment Variables

```bash
# Set environment variables (add to ~/.bashrc for persistence)
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$PATH:/usr/local/cuda-12.1/bin

# Verify nvcc is available
nvcc --version
```

**Expected Output:**
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Mon_Apr__3_17:16:06_PDT_2023
Cuda compilation tools, release 12.1, V12.1.105
```

### 6. Install Flash-Attention

```bash
# Install with no-build-isolation to access PyTorch in build environment
CUDA_HOME=/usr/local/cuda-12.1 PATH=$PATH:/usr/local/cuda-12.1/bin \
pip install flash-attn==2.8.0.post2 --no-build-isolation
```

**Critical Issues We Solved**:
- **Without CUDA toolkit**: `nvcc was not found` error
- **Without no-build-isolation**: `No module named 'torch'` during build
- **Wrong CUDA version**: Compilation errors

### 7. Install Transformers and Dependencies

```bash
# Install specific transformers version for compatibility
pip install transformers==4.51.3

# Install other required packages
pip install qwen_vl_utils gradio PyMuPDF openai huggingface_hub modelscope accelerate
```

**Issue**: We initially had `transformers==4.56.2` which caused conflicts. Version 4.51.3 is required.

### 8. Clone and Install DOTS OCR

```bash
# Clone repository
git clone https://github.com/rednote-hilab/dots.ocr.git
cd dots.ocr

# Edit requirements.txt to comment out flash-attn temporarily
# (We'll install it separately as done above)

# Install in editable mode
pip install -e .
```

**Issue Encountered**: The `pip install -e .` initially failed because it tried to compile flash-attn without proper CUDA setup.

### 9. Download Model Weights

```bash
# Download model from HuggingFace
python -c "from huggingface_hub import snapshot_download; snapshot_download('rednote-hilab/dots.ocr', local_dir='./weights/DotsOCR')"
```

**Issue**: Model names with dots cause import issues, so local download is recommended.

### 10. Test Installation

```bash
cd demo

# Test with flash attention
CUDA_HOME=/usr/local/cuda-12.1 PATH=$PATH:/usr/local/cuda-12.1/bin python demo_hf.py
```

## Common Issues and Solutions

### Issue 1: CUDA Out of Memory

**Error**: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 8.52 GiB`

**Root Cause**: Without flash-attn, eager attention has O(n²) memory complexity.

**Solutions**:
1. ✅ Install flash-attn (primary solution)
2. ✅ Reduce `max_new_tokens` from 24000 to 2048
3. ✅ Enable gradient checkpointing
4. ✅ Use `low_cpu_mem_usage=True`

### Issue 2: Flash-Attn Import Errors

**Error**: `ModuleNotFoundError: No module named 'transformers_modules.rednote-hilab.dots'`

**Solutions**:
1. ✅ Download model locally instead of using HuggingFace name directly
2. ✅ Use absolute paths in scripts

### Issue 3: NVCC Not Found

**Error**: `nvcc was not found. Are you sure your environment has nvcc available?`

**Solution**: Install CUDA toolkit development packages (not just runtime).

### Issue 4: Torch Import During Flash-Attn Build

**Error**: `No module named 'torch'` during flash-attn installation

**Solution**: Use `--no-build-isolation` flag to allow access to installed PyTorch.

### Issue 5: Data Type Mismatch

**Error**: `Input type (c10::BFloat16) and bias type (c10::Half) should be the same`

**Solution**: Stick with `torch.bfloat16` (don't switch to float16).

## Memory Usage Comparison

| Configuration | Memory Usage | Status |
|---------------|-------------|---------|
| Eager attention | ~22GB+ | ❌ CUDA OOM |
| Flash-attn | ~15GB | ✅ Works |
| + Gradient checkpointing | ~12GB | ✅ Optimal |

## Verification Commands

```bash
# Check PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check flash-attn installation
python -c "import flash_attn; print('Flash-attn installed successfully')"

# Check model loading
python -c "from transformers import AutoModelForCausalLM; print('Transformers working')"

# Check NVCC
nvcc --version

# Check GPU memory
nvidia-smi
```

## Production Configuration

For production use, ensure these settings in your scripts:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    attn_implementation="flash_attention_2",  # Critical for memory efficiency
    torch_dtype=torch.bfloat16,  # Required data type
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True  # Reduces CPU memory during loading
)

# Enable gradient checkpointing for inference
model.gradient_checkpointing_enable()

# Use reasonable max_new_tokens
generated_ids = model.generate(**inputs, max_new_tokens=2048)  # Not 24000
```

## Environment Variables for Persistence

Add to `~/.bashrc`:

```bash
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$PATH:/usr/local/cuda-12.1/bin
```

## System Requirements Met

- ✅ **GPU Memory**: 22GB A10G (minimum 16GB recommended)
- ✅ **CUDA Driver**: 12.9 (supports toolkit 12.1)
- ✅ **Python**: 3.12 (3.10+ required)
- ✅ **PyTorch**: 2.7.0+cu128
- ✅ **Flash-attn**: 2.8.0.post2
- ✅ **Transformers**: 4.51.3

## Performance Results

After proper installation:
- ✅ **High-resolution images**: 1700x2250 pixels processed successfully
- ✅ **Complex documents**: Multi-table research papers, legal documents
- ✅ **Multiple formats**: JSON, markdown, pure text extraction
- ✅ **Memory efficient**: ~15GB usage vs 22GB+ without flash-attn
- ✅ **Stable inference**: No CUDA OOM errors

## Installation Time

- **Total time**: ~2-3 hours (including troubleshooting)
- **Model download**: ~10 minutes (depends on connection)
- **CUDA toolkit**: ~15 minutes
- **Flash-attn compilation**: ~10 minutes

## Key Learnings

1. **Flash-attn is essential** for high-resolution document processing
2. **CUDA toolkit** (not just runtime) is required for flash-attn compilation
3. **Model names with dots** cause import issues - use local downloads
4. **Memory optimization** requires multiple techniques, not just flash-attn
5. **Environment variables** must be set for both installation and runtime
6. **PyTorch CUDA version** must be compatible with driver version

## Next Steps

With this installation complete, you can:
1. **Deploy with vLLM** for production use
2. **Process batch documents** with high throughput
3. **Extract to multiple formats** (JSON, markdown, text)
4. **Handle complex layouts** including tables, formulas, headers

This guide should enable a smooth installation without the trial-and-error we experienced!