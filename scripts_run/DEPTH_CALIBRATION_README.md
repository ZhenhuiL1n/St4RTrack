# Depth Calibration Pipeline

Pipeline for creating calibrated MoGe depth maps for DNA-Rendering sequences.

---

## Quick Start

```bash
conda activate st4rtrack

# All 48 views, parallel (8 workers), FAST mode
python scripts_run/process_sequence.py --seq 0012_09 --views all --fast --workers 8

# Single view with full debugging output
python scripts_run/process_sequence.py --seq 0012_09 --view 22 --step all
```

---

## Usage

### Command Line Options

| Flag | Description | Example |
|------|-------------|---------|
| `--seq` | Sequence name (required) | `--seq 0012_09` |
| `--view` | Single view index | `--view 22` |
| `--views` | Multiple views | `--views 0,10,22` or `--views all` |
| `--fast` | Skip intermediate files, only output final npy | `--fast` |
| `--load_moge` | Load existing MoGe data (view 22 only!) | `--load_moge` |
| `--workers` | Parallel workers (max = num GPUs) | `--workers 7` |
| `--gpus` | GPU IDs to use (default: 0,1,2,3,4,5,6) | `--gpus 0,1,2,3` |
| `--step` | Specific steps to run | `--step 1,2,3,4,5,6,7,8,9` |

---

## Examples

### Single View (Full Pipeline)
```bash
# Process view 22 with all intermediate files
python scripts_run/process_sequence.py --seq 0012_09 --view 22 --step all
```

### Single View (Fast Mode)
```bash
# Only produce moge_calibrated.npy
python scripts_run/process_sequence.py --seq 0012_09 --view 22 --fast
```

### All 48 Views (Sequential)
```bash
# Run MoGe for each view sequentially
python scripts_run/process_sequence.py --seq 0012_09 --views all --fast
```

### All 48 Views (Parallel - Recommended)
```bash
# Process 8 views in parallel
python scripts_run/process_sequence.py --seq 0012_09 --views all --fast --workers 8
```

### Using Existing MoGe Data (View 22 Only!)
```bash
# Only works for the original view the MoGe data was created for
python scripts_run/process_sequence.py --seq 0012_09 --view 22 --fast --load_moge
```

---

## Important Notes

### MoGe Inference

| Situation | What Happens |
|-----------|--------------|
| `--load_moge` | Loads pre-computed MoGe (⚠️ **only works for original view 22!**) |
| No `--load_moge` | Runs MoGe inference for each view (GPU required) |

> **⚠️ Warning**: For views other than 22, you MUST run MoGe inference.
> The existing MoGe data at `data/DNA_Seq/depth_maps/DNA_02/DNA_Seqmoge_results.npy` is **only for view 22**.

### Parallel Processing

| Workers | Best For |
|---------|----------|
| 1 | Single GPU, debugging |
| 2-4 | MoGe inference (GPU memory limited) |
| 8-16 | Fast mode without MoGe inference |

> **⚠️ GPU OOM**: Parallel MoGe inference may cause GPU out-of-memory. Use fewer workers or `--fast` mode.

---

## Output Structure

### Fast Mode (`--fast`)
```
reproduction/
├── 0012_09_view0/moge_calibrated.npy
├── 0012_09_view1/moge_calibrated.npy
...
└── 0012_09_view47/moge_calibrated.npy
```

### Full Mode (default)
```
reproduction/0012_09_view22/
├── 1_frames_raw/           # Raw RGB (150 frames)
├── 2_frames_512/           # 512x512 RGB
├── 3_moge_depth/           # MoGe depth per frame
├── 4_rendered_depth_raw/   # PCL depth at 1024×1224
├── 5_rendered_depth_512/   # PCL depth at 512×512
├── 6_calibrated_depth/     # Calibrated MoGe per frame
├── 7_visualizations/       # Comparison grids
├── 8_unprojected_ply/      # 3D point clouds
└── moge_calibrated.npy     # Final combined output
```

---

## Pipeline Steps

| Step | Description | Output |
|------|-------------|--------|
| 1 | Extract RGB frames | `1_frames_raw/` |
| 2 | Center crop & resize | `2_frames_512/` |
| 3 | MoGe depth estimation | `3_moge_depth/` |
| 4 | Render PCL depth | `4_rendered_depth_raw/` |
| 5 | Crop & resize depth | `5_rendered_depth_512/` |
| 6 | Calibrate MoGe | `6_calibrated_depth/` |
| 7 | Create visualizations | `7_visualizations/` |
| 8 | Create PLY comparisons | `8_unprojected_ply/` |
| 9 | Create combined npy | `moge_calibrated.npy` |

---

## Calibration Results

| Metric | Typical Value |
|--------|---------------|
| Scale Factor Mean | ~2.67 |
| Scale Factor Std | ~0.09 |

*Scale factor indicates MoGe depth is ~2.67× smaller than metric GT depth.*
