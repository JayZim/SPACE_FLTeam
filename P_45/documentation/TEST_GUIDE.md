# SPACE Project Testing Guide

## Overview

This guide explains how to run various tests for the SPACE project, including complete workflow tests, individual module tests, and full TLE configuration tests.

## Test Environment Setup

### 1. Environment Requirements

- Python 3.8+
- All dependencies installed (see `requirements.txt`)

### 2. Directory Structure

```
P_45/
├── test_scripts/          # Test scripts
│   ├── test_complete_workflow.py
│   ├── test_all_tle_configs.py
│   ├── test_sat_sim_only.py
│   ├── test_algorithm_only.py
│   ├── test_fl_only.py
│   └── test_gif_animation.py  # GIF animation tests
├── test_data/             # Test data (TLE file references)
├── test_results/          # Test result outputs
│   ├── sat_sim_output/
│   ├── algorithm_output/  # FLAM files
│   └── fl_output/
├── test_logs/             # Test logs
└── documentation/         # Test documentation
    ├── TEST_GUIDE.md
    ├── USER_MANUAL.md
    └── TEST_REPORT_TEMPLATE.md
```

## Test Scripts Description

### 1. Complete Workflow Tests

#### `test_complete_workflow.py`
- **Function**: Tests complete workflow TLE → SatSim → Algorithm → FL
- **Use case**: End-to-end functionality verification
- **Test files**: SatCount1.tle, SatCount4.tle, SatCount8.tle

```bash
python3 P_45/test_scripts/test_complete_workflow.py
```

#### `test_gif_animation.py` 🎬
- **Function**: Tests GIF animation generation functionality
- **Use case**: Animation functionality verification
- **Test content**: Accuracy animation, client participation animation

```bash
cd /Users/stephentsang/Documents/GitHub/SPACE_FLTeam
python3 P_45/test_scripts/test_gif_animation.py
```

#### `test_all_tle_configs.py`
- **Function**: Tests complete workflow for all 7 TLE files
- **Use case**: Comprehensive functionality verification
- **Test files**: All files in TLEs/ directory

```bash
cd /Users/stephentsang/Documents/GitHub/SPACE_FLTeam
python3 P_45/test_scripts/test_all_tle_configs.py
```

### 2. Individual Module Tests

#### `test_sat_sim_only.py`
- **Function**: Tests satellite simulation module standalone functionality
- **Use case**: SatSim module functionality verification
- **Output**: Adjacency matrices and satellite orbital data

```bash
cd /Users/stephentsang/Documents/GitHub/SPACE_FLTeam
python3 P_45/test_scripts/test_sat_sim_only.py
```

#### `test_algorithm_only.py`
- **Function**: Tests FLOMPS algorithm module standalone functionality
- **Use case**: Algorithm module functionality verification
- **Features**: Automatically runs SatSim from TLE to generate adjacency matrices
- **Output**: FLAM files to `flomps_algorithm/output/`

```bash
cd /Users/stephentsang/Documents/GitHub/SPACE_FLTeam
python3 P_45/test_scripts/test_algorithm_only.py
```

#### `test_fl_only.py`
- **Function**: Tests federated learning module standalone functionality
- **Use case**: FL module functionality verification
- **Includes**: Basic FL tests and FLAM file tests

```bash
cd /Users/stephentsang/Documents/GitHub/SPACE_FLTeam
python3 P_45/test_scripts/test_fl_only.py
```

## Test Parameters Description

### 1. Time Configuration
- **Simulation duration**: 1 hour (current date 00:00:00 to 01:00:00)
- **Timestep length**: 1 minute
- **Timestamp**: Uses current date (2025-10-24)

### 2. Satellite Configuration
- **TLE files**: Located in `TLEs/` directory
- **Number of satellites**: 1-40 satellites (based on TLE file)
- **Orbit types**: Supports various orbital configurations

### 3. Algorithm Configuration
- **Mode**: FLOMPS three-phase algorithm
- **Server selection**: Dynamic selection of aggregation and redistribution servers
- **Output format**: CSV format FLAM files

### 4. Federated Learning Configuration
- **Model type**: EfficientNetB0
- **Dataset**: EuroSAT
- **Rounds**: 3 rounds
- **Number of clients**: 4 clients

## Expected Results Description

### 1. Success Criteria
- ✅ All test steps complete without errors
- ✅ Generate expected output files
- ✅ Correct timestamps (current date)
- ✅ Correct file paths

### 2. Output Files
- **SatSim Output**: Adjacency matrix files
- **Algorithm Output**: FLAM CSV files (located in `flomps_algorithm/output/`)
- **FL Output**: Model training results and evaluation metrics
- **Animation Files**: GIF animation files (located in `federated_learning/results_from_output/`)
  - `accuracy_progress.gif`: Accuracy progress animation
  - `client_participation.gif`: Client participation animation

### 3. Performance Metrics
- **Execution time**: Varies based on satellite count and configuration
- **Memory usage**: Monitor peak memory usage
- **File size**: Reasonable output file sizes

## Troubleshooting

### 1. Common Issues

#### Module Import Error
```
ModuleNotFoundError: No module named 'xxx'
```
**Solution**: Ensure tests are run from project root directory, or check Python path settings

#### TLE File Not Found
```
❌ Error: TLE file does not exist
```
**Solution**: Check TLE file path, ensure file exists in `TLEs/` directory

#### Permission Error
```
PermissionError: [Errno 13] Permission denied
```
**Solution**: Check output directory permissions, ensure write permissions

### 2. Debug Mode

#### Enable Detailed Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Check Intermediate Results
- View JSON result files in `P_45/test_results/` directory
- Check output directories of each module

### 3. Performance Optimization

#### Reduce Test Time
- Modify simulation duration (reduce end_time)
- Reduce FL rounds
- Use fewer satellite configurations

#### Memory Optimization
- Monitor memory usage
- Clean up intermediate variables promptly
- Use smaller datasets

## Test Reports

### 1. Automatic Report Generation
Each test script automatically generates JSON format test reports containing:
- Test configuration
- Execution time
- Success/failure status
- Detailed result data

### 2. Report Location
- Test results: `P_45/test_results/`
- Report format: `{test_type}_test_results_{timestamp}.json`

### 3. Report Analysis
Using report data you can:
- Analyze performance trends
- Identify problem patterns
- Optimize test configurations
- Generate test summaries

## Best Practices

### 1. Test Order
1. Run individual module tests first
2. Then run complete workflow tests
3. Finally run full TLE configuration tests

### 2. Environment Isolation
- Use virtual environments
- Avoid conflicts with other projects
- Regularly clean test outputs

### 3. Version Control
- Record test configuration changes
- Save important test results
- Tag test versions

## Support

If you encounter problems or need help, please:
1. Check the troubleshooting section of this guide
2. Review error messages in test result files
3. Check project documentation and code comments
4. Contact project maintainers
