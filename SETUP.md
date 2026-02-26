# Setup Instructions

## 1. Python Virtual Environment

Already created! Activate it with:
```bash
source venv/bin/activate
```

## 2. Install Dependencies

Dependencies are already installed. If you need to reinstall:
```bash
pip install -r requirements.txt
```

## 3. Download HAM10000 Dataset

### Option A: Using Kaggle API (Recommended)

1. **Get Kaggle API credentials:**
   - Go to https://www.kaggle.com/settings
   - Scroll to "API" section
   - Click "Create New Token" - this downloads `kaggle.json`

2. **Set up credentials:**
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. **Run the download script:**
   ```bash
   source venv/bin/activate
   python fetch_dataset.py
   ```

### Option B: Manual Download from Harvard Dataverse

1. Visit: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
2. Download:
   - `HAM10000_metadata.csv`
   - `HAM10000_images_part1.zip`
   - `HAM10000_images_part2.zip`
3. Extract and organize:
   ```bash
   mkdir -p data
   # Extract metadata.csv to data/
   # Extract images from part1.zip to data/HAM10000_images_part1/
   # Extract images from part2.zip to data/HAM10000_images_part2/
   ```

### Option C: Direct Download (if available)

If you have the dataset URL, you can download directly:
```bash
mkdir -p data
cd data
# Download and extract files here
```

## 4. Verify Dataset Structure

After downloading, verify your `data/` directory contains:
```
data/
├── HAM10000_metadata.csv
└── <subdirectories with *.jpg files>
```

The code will automatically find all `.jpg` files in subdirectories under `data/`.

## 5. Test Run

Once the dataset is ready, test with a short run:

```bash
source venv/bin/activate
python Normal_ResNet_HAM10000.py
```

**Note:** Full training runs 200 epochs. For quick testing, you can temporarily modify `epochs = 200` to `epochs = 1` in the script.
