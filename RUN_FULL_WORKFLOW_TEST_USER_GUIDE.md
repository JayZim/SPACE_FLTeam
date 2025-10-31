# User Guide: run_full_workflow_test.py

## 📋 Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Modes](#usage-modes)
- [Detailed Examples](#detailed-examples)
- [Output Files](#output-files)
- [Troubleshooting](#troubleshooting)
- [Configuration Options](#configuration-options)
- [Advanced Usage](#advanced-usage)

---

## 🎯 Overview

`run_full_workflow_test.py` is a comprehensive testing script that automates the complete FLOMPS (Federated Learning on Mobile Proximity-based Systems) workflow. It runs the entire pipeline from satellite orbital data (TLE files) to federated learning results, including visualizations and metrics.

### What It Does

The script orchestrates three main stages:

1. **Satellite Simulation (SatSim)**: Generates or reads satellite connectivity matrices
2. **FLOMPS Algorithm**: Creates federated learning activity matrices (FLAM)
3. **Federated Learning**: Trains ML models using FL across simulated satellites

### Key Features

- ✅ Automatic TLE file processing and SatSim generation
- ✅ Support for multiple satellite constellation sizes
- ✅ Interactive and batch execution modes
- ✅ Customizable timesteps and duration
- ✅ Automatic result file detection and reporting
- ✅ Quiet mode for minimal output

---

## 📦 Prerequisites

### Required Software

- **Python**: Version 3.10
- **Dependencies**: See `P45_Requirements.txt` for complete list

### Required Modules

Install dependencies using:

```bash
pip install -r P45_Requirements.txt
```

Key dependencies include:
- `skyfield` (≥1.46) - Satellite orbital calculations
- `torch` (≥2.2.0) - Deep learning framework
- `numpy`, `pandas`, `matplotlib` - Data processing and visualization

### Project Structure

Ensure your project has these directories:
```
SPACE_FLTeam/
├── TLEs/                           # Satellite orbital data
│   ├── SatCount4.tle
│   ├── SatCount8.tle
│   └── ...
├── sat_sim/output/                 # SatSim output files
├── synth_FLAMs/                    # FLAM CSV files
└── federated_learning/
    └── results_from_output/        # Training results
```

---

## 🚀 Installation

### Step 1: Navigate to Project Root

```bash
cd /path/to/SPACE_FLTeam
```

### Step 2: Verify Python Version

```bash
python -V  # Should show 3.10 or higher
```

### Step 3: Install Dependencies

```bash
pip install -r P45_Requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import skyfield, torch, numpy; print('All modules installed successfully')"
```

---

## ⚡ Quick Start

### Simplest Usage (Auto Mode)

```bash
python run_full_workflow_test.py --auto
```

This will:
- Use the latest SatSim file from `sat_sim/output/`
- Or generate a new one from 8-satellite TLE with 10 timesteps
- Run the complete FLOMPS workflow
- Display output file locations

### Interactive Mode

```bash
python run_full_workflow_test.py
```

Follow the prompts to:
1. Choose input source (existing file or generate from TLE)
2. Select satellite constellation size
3. Configure optional parameters

---

## 🎮 Usage Modes

### Mode 1: Auto Mode (Recommended for Quick Testing)

**Purpose**: Run with sensible defaults without user interaction

```bash
python run_full_workflow_test.py --auto
```

**Behavior**:
- Uses latest existing SatSim file, or
- Generates from `TLEs/SatCount8.tle` with 10 timesteps
- No prompts for user input
- Shows final output paths

**Best for**: Quick tests, automated runs, CI/CD pipelines

---

### Mode 2: Interactive Mode

**Purpose**: Full control over configuration

```bash
python run_full_workflow_test.py
```

**Interactive Flow**:

1. **Choose input source**:
   ```
   Choose input source:
     1) Use existing sat_sim/output/*.txt
     2) Generate from TLEs (interactive)
   Enter 1 or 2:
   ```

2. **If selecting option 1**: Choose from existing SatSim files

3. **If selecting option 2**: Choose TLE configuration:
   ```
   Select satellite set:
     1) 1 -> SatCount1.tle
     2) 3 -> SatCount3.tle
     3) 4 -> SatCount4.tle
     4) 8 -> SatCount8.tle
     5) 40 -> SatCount40.tle
     6) NovaSar -> NovaSar.tle
     7) Walker -> Walker.tle
   ```

4. **Configure timing** (if using existing SatSim):
   ```
   Timesteps (integer, each = 1 minute) [default empty]: 
   Custom duration (HH:MM:SS) [default 00:10:00]: 
   ```

**Best for**: First-time users, custom configurations, learning the system

---

### Mode 3: File-Specific Mode

**Purpose**: Use a specific SatSim file with optional parameters

```bash
python run_full_workflow_test.py path/to/sat_sim_output.txt
```

**Examples**:

```bash
# Use specific file with defaults
python run_full_workflow_test.py sat_sim/output/sat_sim_20250605_001448.txt

# Use specific file with custom timesteps
python run_full_workflow_test.py sat_sim/output/sat_sim_20250605_001448.txt --timesteps 50

# Use specific file with custom duration
python run_full_workflow_test.py sat_sim/output/sat_sim_20250605_001448.txt --custom-duration "01:00:00"
```

**Best for**: Testing specific scenarios, reproducing results

---

### Mode 4: Quiet Mode

**Purpose**: Minimal console output for batch processing

```bash
python run_full_workflow_test.py --auto --quiet
```

**Behavior**:
- Suppresses detailed output
- Shows only last 50 lines of execution
- Still displays output file locations

**Best for**: Batch processing, automated testing

---

## 💡 Detailed Examples

### Example 1: First-Time User Testing with 4 Satellites

```bash
# Start interactive mode
python run_full_workflow_test.py

# Follow prompts:
# 1) Enter "2" to generate from TLEs
# 2) Enter "3" to select SatCount4.tle (4 satellites)
# 3) Enter "10" for timesteps (or press Enter for default)
# 4) Press Enter to accept default duration
# 5) Select FLAM file when prompted
```

**Expected Output**:
```
🚀 Starting FLOMPS Complete Workflow...

📡 Step 1: Running SatSim (Satellite Simulation)...
Generated SatSim file: sat_sim/output/sat_sim_20250605_001448.txt

🧮 Step 2: Running Algorithm (FLOMPS Algorithm)...
✅ Algorithm completed: Generated FLAM data

🤖 Step 3: Running Federated Learning...

=== Outputs ===
Latest FLAM: synth_FLAMs/flam_4n_100t_flomps_3phase_2025-06-05_00-14-48.csv
Results dir: federated_learning/results_from_output/output_test_20250605_001448
Dashboard: federated_learning/results_from_output/output_test_20250605_001448/dashboard.html
Metrics JSON: federated_learning/results_from_output/output_test_20250605_001448/metrics_20250605_001448.json
Model Weights: federated_learning/results_from_output/output_test_20250605_001448/model_20250605_001448.pt
```

---

### Example 2: Quick Automated Test

```bash
python run_full_workflow_test.py --auto
```

**Workflow**:
- Automatically uses latest SatSim or generates from 8-sat TLE
- No user interaction required
- Complete execution in 15-30 seconds

---

### Example 3: Custom Duration for Extended Simulation

```bash
python run_full_workflow_test.py --auto --custom-duration "00:30:00"
```

**Result**: 30-minute simulation with approximately 30 timesteps

---

### Example 4: Batch Testing Multiple Configurations

```bash
# Test with 4 satellites
python run_full_workflow_test.py --auto --quiet

# Test with 8 satellites
python run_full_workflow_test.py --auto --quiet

# Test with 40 satellites
python run_full_workflow_test.py --auto --quiet
```

---

## 📁 Output Files

### FLAM CSV File

**Location**: `synth_FLAMs/flam_*n_*t_flomps_3phase_YYYY-MM-DD_HH-MM-SS.csv`

**Naming Convention**:
- `*n` = Number of satellites
- `*t` = Number of timesteps
- `flomps` = Algorithm used
- `3phase` = Three-phase round structure
- Timestamp = Generation date and time

**Example**: `flam_8n_100t_flomps_3phase_2025-10-31_04-33-44.csv`

**Content Format**:
```csv
Timestep: 1, Round: 1, Target Node: 0, Phase: TRAINING
0,0,0,0    # No communication during training
0,0,0,0
0,0,0,0
0,0,0,0

Timestep: 4, Round: 1, Target Node: 0, Phase: TRANSMITTING  
0,1,1,0    # Client-to-server transmission
1,0,0,1
1,0,0,1
0,1,1,0
```

---

### Federated Learning Results

**Base Directory**: `federated_learning/results_from_output/output_*`

**Files Generated**:

1. **`dashboard.html`**
   - Interactive visualization dashboard
   - View accuracy, loss, client participation
   - Launch with: `open dashboard.html` (Mac) or double-click

2. **`metrics_*.json`**
   - Training metrics (accuracy, loss per round)
   - Client participation statistics
   - Performance measurements

3. **`model_*.pt`**
   - PyTorch model weights
   - Trained global model parameters
   - Load with: `torch.load('model_*.pt')`

4. **`results_*.log`**
   - Detailed execution log
   - Error messages and warnings
   - Debugging information

---

## 🔧 Configuration Options

### Command-Line Arguments

| Argument | Type | Description | Example |
|----------|------|-------------|---------|
| `sat_sim_file` | Path (optional) | Path to SatSim output file | `sat_sim/output/sat_sim_*.txt` |
| `--timesteps` | Integer | Number of timesteps (1 minute each) | `--timesteps 100` |
| `--custom-duration` | String | Duration in HH:MM:SS format | `--custom-duration "01:30:00"` |
| `--auto` | Flag | Use defaults without prompts | `--auto` |
| `--quiet` | Flag | Reduce console output | `--quiet` |

### Usage Patterns

```bash
# Pattern 1: Auto mode only
python run_full_workflow_test.py --auto

# Pattern 2: Auto mode with quiet output
python run_full_workflow_test.py --auto --quiet

# Pattern 3: Specific file with custom timesteps
python run_full_workflow_test.py file.txt --timesteps 50

# Pattern 4: Specific file with custom duration
python run_full_workflow_test.py file.txt --custom-duration "00:45:00"

# Pattern 5: Full customization
python run_full_workflow_test.py file.txt --timesteps 100 --custom-duration "01:30:00"
```

---

## 🐛 Troubleshooting

### Problem 1: "No SatSim .txt files found"

**Cause**: No existing SatSim files in `sat_sim/output/`

**Solution**:
```bash
# Generate new SatSim file
python run_full_workflow_test.py

# Select option 2 (Generate from TLEs)
# Choose any satellite configuration
```

**Prevention**: Run SatSim generation at least once before using auto mode

---

### Problem 2: "Failed to import SatSim"

**Cause**: Missing or incorrectly installed SatSim module

**Solution**:
```bash
# Verify installation
python -c "from sat_sim.sat_sim import SatSim; print('OK')"

# Reinstall if needed
pip install -r P45_Requirements.txt --force-reinstall
```

---

### Problem 3: "skyfield not available"

**Cause**: Missing Skyfield library

**Solution**:
```bash
pip install skyfield>=1.46
```

---

### Problem 4: "Error: SatSim file not found"

**Cause**: Invalid file path provided

**Solution**:
```bash
# Check if file exists
ls -la path/to/your/file.txt

# Use relative path from project root
python run_full_workflow_test.py sat_sim/output/sat_sim_*.txt
```

---

### Problem 5: "Workflow exited with code 1"

**Cause**: Error during workflow execution

**Solution**:
1. Check console output for specific error message
2. Verify dependencies are installed correctly
3. Check file permissions in output directories
4. Review logs in `federated_learning/results_from_output/`

---

### Problem 6: "No valid TLE records parsed"

**Cause**: Corrupted or improperly formatted TLE file

**Solution**:
```bash
# Verify TLE file format
head -n 20 TLEs/SatCount4.tle

# Should show:
# SATNAME
# TLE LINE 1
# TLE LINE 2
# (repeated for each satellite)
```

---

### Problem 7: FLAM selection prompt too many times

**Cause**: Script waiting for FLAM file selection

**Solution**:
- The prompt appears during FL stage
- Use existing FLAM file from `synth_FLAMs/`
- Select option 1 (existing file)

---

## 🎓 Advanced Usage

### Custom TLE Configuration

To add your own TLE file:

1. Place `.tle` file in `TLEs/` directory
2. Run interactive mode:
   ```bash
   python run_full_workflow_test.py
   ```
3. Select option 2 (Generate from TLEs)
4. Your TLE file should appear in the list

**TLE Format**:
```
SATELLITE NAME
1 25544U 98067A   12345.67890123  .00000000  00000-0  00000+0 0  9999
2 25544  51.6448  123.4567 0001234   0.0000 359.9999 15.12345678 12345
```

---

### Automation with Scripts

Create a batch test script:

```bash
#!/bin/bash
# batch_test.sh

echo "Testing with 4 satellites..."
python run_full_workflow_test.py --auto --quiet

echo "Testing with 8 satellites..."
python run_full_workflow_test.py --auto --quiet

echo "Testing with 40 satellites..."
python run_full_workflow_test.py --auto --quiet

echo "All tests completed!"
```

---

### Extracting Results Programmatically

```python
from pathlib import Path

# Find latest results
results_dir = Path("federated_learning/results_from_output")
latest_run = sorted(results_dir.glob("output_*"), 
                   key=lambda p: p.stat().st_ctime)[-1]

print(f"Results: {latest_run}")
print(f"Dashboard: {latest_run / 'dashboard.html'}")
```

---

## 📊 Performance Expectations

| Satellites | Timesteps | Runtime | Output Size |
|------------|-----------|---------|-------------|
| 4          | 100       | ~10 sec | ~15KB       |
| 8          | 100       | ~15 sec | ~30KB       |
| 40         | 100       | ~45 sec | ~150KB      |

**Note**: Actual times depend on hardware, dataset size, and model complexity.

---

## 🔗 Related Documentation

- **[README.md](README.md)** - Project overview
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - 30-second start guide
- **[FL_SYSTEM_ARCHITECTURE.md](FL_SYSTEM_ARCHITECTURE.md)** - System architecture
- **[100T_FLAM_GUIDE.md](100T_FLAM_GUIDE.md)** - FLAM generation details
- **[P45_Requirements.txt](P45_Requirements.txt)** - Dependencies

---

## 💬 Support

### Getting Help

1. **Check error messages**: Often provide specific guidance
2. **Review logs**: Located in `federated_learning/results_from_output/output_*/results_*.log`
3. **Run verification**: `python -c "import torch, skyfield, numpy; print('All OK')"`
4. **Consult documentation**: See related files above

### Common Issues Summary

| Error | Quick Fix |
|-------|-----------|
| Module not found | `pip install -r P45_Requirements.txt` |
| File not found | Check path and use `--auto` |
| Permission error | Check directory write permissions |
| Out of memory | Reduce timesteps or satellites |

---

## 📝 License & Attribution

**Project**: Distributed Model Training for Simulated Satellite Swarms  
**Institution**: Swinburne University of Technology  
**Script**: run_full_workflow_test.py  
**Last Updated**: 2025-01-07

---

## 🎯 Summary

**Quickest Way to Run**:
```bash
python run_full_workflow_test.py --auto
```

**Most Control**:
```bash
python run_full_workflow_test.py
```

**Batch Processing**:
```bash
python run_full_workflow_test.py --auto --quiet
```

**With Custom Timing**:
```bash
python run_full_workflow_test.py --auto --timesteps 100
```

---

*This guide covers all essential usage scenarios for run_full_workflow_test.py. For advanced configurations or debugging, refer to the source code comments and related documentation.*

