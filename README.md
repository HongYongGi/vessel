# Vessel

Vessel segmentation dataset download & preprocessing CLI tool for medical imaging research.

Automates the full pipeline for managing CT/CTA vessel segmentation datasets: **download** from public repositories, **preprocess** with standardized medical imaging operations, and **export** to nnUNet v2 format.

## Features

- **Dataset Registry** -- 7 curated vessel segmentation datasets with standardized metadata
- **Automated Download** -- Supports Zenodo, Kaggle, Google Drive, HTTP, and Grand Challenge sources with resume capability
- **Medical Image Preprocessing** -- Resampling, CT windowing, intensity normalization, and label harmonization
- **Unified Label Taxonomy** -- Maps heterogeneous dataset labels to a consistent anatomical naming scheme
- **nnUNet Export** -- Direct export to nnUNet v2 raw format with single or merged dataset support
- **Quality Validation** -- Automated checks for shape consistency, spacing, NaN/Inf values, and label integrity
- **Reproducible Splits** -- Deterministic train/val/test splitting with configurable ratios

## Supported Datasets

| ID | Name | Modality | Region | Cases | Size (GB) |
|----|------|----------|--------|------:|----------:|
| `aortaseg24` | AortaSeg24 | CTA | Thoracoabdominal | 100 | 10.0 |
| `imagecas` | ImageCAS | CTA | Cardiac | 1,000 | 55.0 |
| `mmwhs` | MM-WHS | CT | Cardiac | 120 | 4.0 |
| `msd_hepatic_vessel` | MSD Task08 Hepatic Vessel | CT | Abdomen | 443 | 15.0 |
| `parse2022` | PARSE 2022 | CT | Thorax | 200 | 12.0 |
| `topcow` | TopCoW 2023 | CTA | Head & Neck | 140 | 8.0 |
| `totalsegmentator_v2` | TotalSegmentator v2 | CT | Whole Body | 1,204 | 115.0 |

## Installation

```bash
# Install from source
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"
```

### Requirements

- Python >= 3.10
- Core: `torch`, `SimpleITK`, `nibabel`, `numpy`, `scipy`, `scikit-image`
- CLI: `typer`, `rich`, `tqdm`
- Config: `pydantic`, `pyyaml`

## Quick Start

### 1. Set data root directory

```bash
export VESSEL_DATA_ROOT=/path/to/your/data
```

### 2. Browse available datasets

```bash
# List all registered datasets
vessel list

# Show detailed info for a specific dataset
vessel info aortaseg24
```

### 3. Download a dataset

```bash
# Download a single dataset
vessel download one aortaseg24

# Download with resume support
vessel download one aortaseg24 --resume

# Dry run (show what would be downloaded)
vessel download one aortaseg24 --dry-run

# Download all tier-1 datasets
vessel download all --tier 1
```

### 4. Preprocess

```bash
# Preprocess a single dataset
vessel preprocess run aortaseg24

# Preprocess with multiple workers
vessel preprocess run aortaseg24 --workers 4

# Validate preprocessed data
vessel preprocess validate aortaseg24
```

### 5. Export to nnUNet

```bash
# Export single dataset
vessel export nnunet aortaseg24 --task-id 100

# Merge multiple datasets into one nnUNet task
vessel export nnunet --merge aortaseg24 imagecas --task-id 200 --task-name VesselMerged
```

### 6. Check status

```bash
vessel show-status
```

## CLI Reference

```
vessel                          # Main entry point
  list                          # Show all registered datasets (shortcut)
  info <dataset_id>             # Show dataset details (shortcut)
  show-status                   # Show download/preprocess status (shortcut)
  registry                      # Dataset registry commands
    list [--tier N] [--region R]  # List datasets with optional filters
    info <dataset_id>             # Detailed dataset information
  download                      # Download commands
    one <dataset_id> [--resume] [--dry-run]  # Download single dataset
    all [--tier N] [--resume] [--dry-run]    # Download multiple datasets
  preprocess                    # Preprocessing commands
    run <dataset_id> [--workers N]   # Run preprocessing pipeline
    all [--tier N] [--workers N]     # Preprocess multiple datasets
    validate <dataset_id>            # Validate processed data
  export                        # Export commands
    nnunet <dataset_id> --task-id N  # Export to nnUNet v2 format
    nnunet --merge ID1 ID2 ... --task-id N --task-name NAME  # Merge export
  status                        # Status commands
    show                             # Display status table
```

## Configuration

Each dataset is defined by a YAML configuration file in `configs/`:

```yaml
version: "1.0"

dataset:
  id: aortaseg24
  name: AortaSeg24
  tier: 1
  body_region: thoracoabdominal
  modality: CTA
  num_cases: 100
  estimated_size_gb: 10.0

source:
  type: zenodo
  record_id: "10991212"
  filename: aortaseg24.zip

format:
  image_pattern: "images/*.nii.gz"
  label_pattern: "labels/*.nii.gz"

labels:
  type: multiclass
  num_classes: 24
  mapping:
    0: background
    1: aorta
    # ...

preprocess:
  target_spacing: [1.0, 1.0, 1.0]
  intensity_window:
    center: 300
    width: 700
  normalize: minmax
```

### Label Taxonomy

The unified label taxonomy (`configs/_label_taxonomy.yaml`) maps anatomical structures to consistent numeric IDs across all datasets. This enables:

- Merging datasets with different labeling schemes
- Consistent evaluation across heterogeneous data
- Selective label export (e.g., only aorta-related classes)

## Data Directory Structure

```
$VESSEL_DATA_ROOT/
├── raw/                    # Downloaded raw data
│   ├── aortaseg24/
│   ├── imagecas/
│   └── ...
├── processed/              # Preprocessed NIfTI files
│   ├── aortaseg24/
│   │   ├── images/
│   │   ├── labels/
│   │   └── splits.json
│   └── ...
├── exports/                # nnUNet formatted exports
│   └── nnUNet_raw/
│       └── Dataset100_AortaSeg24/
└── .vessel/                # Internal status tracking
    └── download_status.json
```

## Project Structure

```
vessel/
├── configs/                    # Dataset YAML configurations
│   ├── _label_taxonomy.yaml    # Unified label mapping
│   └── *.yaml                  # Per-dataset configs
├── src/vessel/
│   ├── cli/                    # Typer CLI commands
│   ├── core/                   # Data models, registry, paths
│   ├── download/               # Source-specific downloaders
│   ├── preprocess/             # Preprocessing pipeline
│   │   ├── pipeline.py         # Orchestration
│   │   ├── resample.py         # Isotropic resampling
│   │   ├── intensity.py        # CT windowing & normalization
│   │   ├── label_harmonize.py  # Label taxonomy mapping
│   │   ├── validate.py         # Quality checks
│   │   ├── split.py            # Train/val/test splitting
│   │   └── crop_pad.py         # Foreground cropping
│   ├── export/                 # nnUNet v2 exporter
│   └── utils/                  # I/O, hashing, progress
├── tests/
└── pyproject.toml
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=vessel --cov-report=term-missing

# Lint
ruff check src/
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `VESSEL_DATA_ROOT` | Root directory for all data operations | Yes |
| `KAGGLE_USERNAME` | Kaggle API username (for Kaggle datasets) | Optional |
| `KAGGLE_KEY` | Kaggle API key | Optional |

## License

See [LICENSE](LICENSE) for details.
