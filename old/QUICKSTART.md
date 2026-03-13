# Quick Start Guide

## Current Status

✅ **Completed:**
- Python virtual environment created (`venv/`)
- All required packages installed
- Dataset download script ready (`fetch_dataset.py`)
- Setup verification script ready (`verify_setup.py`)

⏳ **Pending (requires your action):**
- Download HAM10000 dataset (requires Kaggle API setup)

## Next Steps

### 1. Set up Kaggle API (one-time)

```bash
# Get your API token from https://www.kaggle.com/settings
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 2. Download Dataset

```bash
source venv/bin/activate
python fetch_dataset.py
```

This will download ~3GB of data and extract it to `data/`.

### 3. Verify Setup

```bash
python verify_setup.py
```

### 4. Test Run (Quick Test)

For a quick test with just 1 epoch, temporarily modify `Normal_ResNet_HAM10000.py`:

```python
# Change line 303 from:
epochs = 200
# To:
epochs = 1
```

Then run:
```bash
python Normal_ResNet_HAM10000.py
```

### 5. Full Training

Once verified, restore `epochs = 200` and run any of the scripts:

- `Normal_ResNet_HAM10000.py` - Centralized baseline
- `FL_ResNet_HAM10000.py` - Federated Learning
- `SL_ResNet_HAM10000.py` - Split Learning
- `SFLV1_ResNet_HAM10000.py` - SplitFed V1
- `SFLV2_ResNet_HAM10000.py` - SplitFed V2

## Files Created

- `requirements.txt` - Python dependencies
- `fetch_dataset.py` - Dataset download script
- `verify_setup.py` - Setup verification script
- `SETUP.md` - Detailed setup instructions
- `QUICKSTART.md` - This file
