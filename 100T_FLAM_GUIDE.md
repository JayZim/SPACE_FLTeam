# 100 Timesteps FLAM Generation Guide

## Overview

This guide explains how to generate 100-timestep FLAM files for the SPACE FLOMPS project, supporting federated learning simulation with 4 satellites.

## Quick Start

### Method 1: Using Dedicated Script

```bash
# Generate 100-timestep FLAM file (4 satellites)
python generate_100t_flam.py
```

### Method 2: Using Custom Generator

```bash
# Generate FLAM file with custom parameters
python generate_custom_flam.py --satellites 4 --timesteps 100

# Other examples
python generate_custom_flam.py -s 8 -t 200    # 8 satellites, 200 timesteps
python generate_custom_flam.py -s 40 -t 500   # 40 satellites, 500 timesteps
```

### Method 3: Using main.py with Custom Parameters

```bash
# Generate custom FLAM through main.py workflow
python main.py flomps TLEs/SatCount4.tle --timesteps 100
python main.py flomps TLEs/SatCount8.tle --timesteps 200
```

## Generated File Format

The generated 100-timestep FLAM file will contain:
- **Filename format**: `flam_4n_100t_flomps_YYYY-MM-DD_HH-MM-SS.csv`
- **File size**: ~8.9KB
- **Lines**: 601 lines (100 timesteps × 6 lines/timestep + 1 end line)
- **Timestep range**: Timestep: 1 to Timestep: 100

## File Content Example

```
Timestep: 1, Round: 1, Target Node: 3, Phase: TRAINING
0,0,0,0
0,0,0,0
0,0,0,0
0,0,0,0

Timestep: 2, Round: 1, Target Node: 3, Phase: TRAINING
0,0,0,0
0,0,0,0
0,0,0,0
0,0,0,0
...
Timestep: 100, Round: 15, Target Node: 1, Phase: TRAINING
0,0,0,0
0,0,0,0
0,0,0,0
0,0,0,0
```

## Verification Methods

### 1. Verify Timestep Count

```bash
grep -c "Timestep:" synth_FLAMs/flam_4n_100t_*.csv
# Should output: 100
```

### 2. View File Header and Tail

```bash
head -10 synth_FLAMs/flam_4n_100t_*.csv
tail -10 synth_FLAMs/flam_4n_100t_*.csv
```

### 3. Test with FL Core

```python
from federated_learning.fl_core import FederatedLearning

fl_core = FederatedLearning()
fl_core.set_num_clients(4)
fl_core.set_num_rounds(3)
fl_core.run(flam_path=None)  # Auto-detect latest file
```

## Usage in System

### 1. FL Core Auto-Detection

FL Core automatically detects the latest FLAM file in the `synth_FLAMs/` directory:

```python
from federated_learning.fl_core import FederatedLearning
fl_core = FederatedLearning()
fl_core.run(flam_path=None)  # Automatically use latest 100-timestep file
```

### 2. Using via main.py

```bash
# Basic usage - uses default parameters
python main.py flomps TLEs/SatCount4.tle

# Custom timesteps
python main.py flomps TLEs/SatCount4.tle --timesteps 100
python main.py flomps TLEs/SatCount8.tle --timesteps 200
```

The workflow will generate new FLAM files, then the FL module will automatically use the latest generated file.

### 3. Using with FL Handler

```python
from federated_learning.fl_handler import FLHandler
from federated_learning.fl_core import FederatedLearning

fl_core = FederatedLearning()
handler = FLHandler(fl_core)
handler.run_module()  # Auto-detect latest FLAM file
```

## Technical Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Satellites | 4 | Using TLEs/SatCount4.tle |
| Timesteps | 100 | 1 timestep per minute |
| Time Span | 100 minutes | 1 hour 40 minutes |
| Start Time | 2025-06-04 00:00:00 | Configurable |
| End Time | 2025-06-04 01:40:00 | Auto-calculated |
| Phases | TRAINING/TRANSMITTING | Controlled by Sam algorithm |

## Algorithm Features

The generated FLAM file includes Sam's algorithm features:
- **Training Phase**: 3-timestep training periods
- **Transmitting Phase**: Dynamic connection matrices for data transmission
- **Round Cycling**: Automatic round completion detection and new round start
- **Target Node**: Randomly selected federated learning aggregation nodes

## Troubleshooting

### Issue 1: Generation Failed
- Check if TLE file exists: `ls TLEs/SatCount4.tle`
- Check Python environment and dependencies

### Issue 2: Timestep Count Mismatch
- Verify SatSim time configuration
- Check algorithm parameter settings

### Issue 3: FL Core Cannot Load
- Confirm file format is correct
- Check file permissions

## Related Files

- `generate_100t_flam.py` - Dedicated 100-timestep generator
- `generate_custom_flam.py` - Custom parameter generator  
- `generate_flam_csv.py` - Base FLAM generation functions
- `federated_learning/fl_core.py` - FL core engine
- `TLEs/SatCount4.tle` - TLE data for 4 satellites
- `main.py` - Main workflow entry point

## Command Line Options

### generate_custom_flam.py Options

```bash
python generate_custom_flam.py --help

Options:
  -s, --satellites SATELLITES  Number of satellites (default: 4)
  -t, --timesteps TIMESTEPS    Number of timesteps (default: 100)
  --start-time START_TIME      Start time in YYYY-MM-DD HH:MM:SS format
  --test                       Test the generated FLAM file with FL Core
```

### main.py Integration

```bash
python main.py flomps [TLE_FILE] [OPTIONS]

Options:
  --timesteps TIMESTEPS        Number of timesteps to generate
  --start-time START_TIME      Simulation start time
  --rounds ROUNDS              Number of federated learning rounds
```

## Changelog

- **2025-06-06**: Initial version, supports 100-timestep generation
- **2025-06-06**: Added custom parameter generator
- **2025-06-06**: Enhanced verification and testing functionality
- **2025-06-06**: Added main.py integration for custom parameters

---

**Note**: This functionality is part of the SPACE FLOMPS project 2025 upgrade, ensuring FL Core and Handler can automatically detect and use the latest generated FLAM files. 