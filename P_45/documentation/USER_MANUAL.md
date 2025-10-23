# SPACE Project User Manual

## Project Overview

SPACE (Satellite-based Parameter Aggregation and Communication Enhancement) is a satellite constellation-based federated learning system that implements the FLOMPS (Federated Learning Over Moving Parameter Server) algorithm. The system enables distributed machine learning in dynamic satellite networks.

### System Architecture

```
TLE Files → Satellite Simulation → Adjacency Matrix → FLOMPS Algorithm → FLAM Files → Federated Learning → Result Output + GIF Animations
```

### Core Components

1. **Satellite Simulation (SatSim)**: Simulates satellite orbits and communication links based on TLE data
2. **FLOMPS Algorithm**: Three-phase federated learning algorithm (Transmission, Check, Redistribution)
3. **Federated Learning (FL)**: PyTorch-based distributed machine learning framework
4. **Model Evaluation and Selection**: Intelligent model selection and performance evaluation system
5. **Visualization and Animation**: Generates GIF animations and interactive dashboards

## Quick Start

### 1. Environment Setup

```bash
# Clone the project
cd SPACE_FLTeam

# Install dependencies
pip install -r requirements.txt

# Verify installation
python3 -c "import torch, skyfield, numpy; print('Environment setup complete')"
```

### 2. Basic Usage

#### Running Complete Workflow
```bash
# Run complete workflow with default configuration (including GIF animations)
python3 main.py flomps TLEs/SatCount4.tle

# Generate animations with custom parameters
python3 main.py flomps TLEs/SatCount8.tle --timesteps 15 --model-type SimpleCNN --data-set MNIST --num-rounds 3 --num-clients 4

# Generate longer animations (more timesteps)
python3 main.py flomps TLEs/SatCount8.tle --timesteps 30 --num-rounds 5
```

#### Running Individual Modules
```bash
# Run satellite simulation only
python3 main.py flomps TLEs/SatCount4.tle --sat-sim-only

# Run algorithm only
python3 main.py flomps TLEs/SatCount4.tle --algorithm-only

# Run federated learning only
python3 main.py flomps TLEs/SatCount4.tle --fl-only
```

## Complete Workflow Description

### Phase 1: Satellite Simulation (SatSim)

**Input**: TLE files
**Output**: Adjacency matrix files

```python
# Satellite simulation configuration
{
    "start_time": "2025-10-24 00:00:00",
    "end_time": "2025-10-24 01:40:00", 
    "timestep": 1,
    "output_file_type": "txt"
}
```

**Features**:
- Reads satellite orbital data from TLE files
- Calculates visibility and communication links between satellites
- Generates time-series adjacency matrices
- Outputs satellite positions and communication status

### Phase 2: FLOMPS Algorithm

**Input**: Adjacency matrices
**Output**: FLAM files (located in `flomps_algorithm/output/`)

```python
# Algorithm configuration
{
    "fedavg_mode": false,
    "static_server_id": 0,
    "server_selection": {
        "connect_to_all_satellites": false,
        "max_lookahead": 20,
        "minimum_connected_satellites": 5
    }
}
```

**Three-phase execution**:
1. **TRANSMITTING**: Inter-satellite model parameter transmission
2. **CHECK**: Check connection status and data integrity
3. **REDISTRIBUTION**: Redistribute aggregated models

**FLAM File Format**:
```csv
Time: 2025-10-24 00:00:00, Timestep: 1, Round: 1, Phase: TRANSMITTING, 
Aggregation Server: 0, Redistribution Server: 1, Target Node: 0, 
Phase Length: 1, Timestep in Phase: 1, Current Connections: 3, 
Cumulative Connections: 3/4, Connected Sats: [0,1,2], 
Missing Sats: [3], Target Sats: [0,1,2,3], Phase Complete: false
1,0,1,0
0,1,0,1
1,0,1,0
0,1,0,1
```

### Phase 3: Federated Learning (FL)

**Input**: FLAM files
**Output**: Training results and model files

```python
# FL configuration
{
    "num_rounds": 3,
    "num_clients": 4,
    "model_type": "EfficientNetB0",
    "data_set": "EuroSAT"
}
```

**Features**:
- Schedules federated learning based on FLAM files
- Distributes and aggregates model parameters between satellites
- Executes distributed training
- Evaluates model performance

## Command Line Parameters

### Main Commands

```bash
python3 main.py <workflow> <tle_file> [options]
```

### Workflow Options

- `flomps`: Run complete FLOMPS workflow
- `sat-sim-only`: Run satellite simulation only
- `algorithm-only`: Run algorithm only
- `fl-only`: Run federated learning only

### Common Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--timesteps` | Simulation timesteps | 100 | `--timesteps 50` |
| `--rounds` | FL training rounds | 3 | `--rounds 5` |
| `--clients` | Number of clients | 4 | `--clients 8` |
| `--model` | Model type | EfficientNetB0 | `--model ResNet50` |
| `--dataset` | Dataset | EuroSAT | `--dataset MNIST` |
| `--gui` | Enable GUI | false | `--gui true` |

### Advanced Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--start-time` | Simulation start time | Current date 00:00:00 |
| `--end-time` | Simulation end time | Current date 01:40:00 |
| `--timestep` | Timestep length (minutes) | 1 |
| `--fedavg-mode` | Use FedAvg mode | false |
| `--static-server` | Static server ID | 0 |

## Configuration Files

### options.json

Main configuration file containing parameters for all modules:

```json
{
    "sat_sim": {
        "start_time": "2025-10-24 00:00:00",
        "end_time": "2025-10-24 01:40:00",
        "timestep": 1,
        "output_file_type": "txt",
        "ground_station": {
            "location": {
                "lat": 35.6895,
                "long": 139.6917
            }
        },
        "gui": false,
        "module_settings": {
            "output_to_file": true
        }
    },
    "algorithm": {
        "fedavg_mode": false,
        "static_server_id": 0,
        "server_selection": {
            "connect_to_all_satellites": false,
            "max_lookahead": 20,
            "minimum_connected_satellites": 5
        },
        "module_settings": {
            "output_to_file": true
        }
    },
    "federated_learning": {
        "num_rounds": 3,
        "num_clients": 4,
        "model_type": "EfficientNetB0",
        "data_set": "EuroSAT",
        "available_models": [
            "SimpleCNN", "ResNet50", "CustomCNN", 
            "EfficientNetB0", "VisionTransformer"
        ],
        "available_datasets": [
            "MNIST", "CIFAR10", "EuroSAT"
        ],
        "module_settings": {
            "output_to_file": true
        }
    }
}
```

### settings.json

Global settings file:

```json
{
    "options_file": "options.json"
}
```

## 🎬 GIF Animation Features

### Animation Types

The system automatically generates two types of GIF animations:

#### 1. Accuracy Progress Animation (accuracy_progress.gif)
- **Content**: Shows gradual improvement of accuracy during federated learning
- **Format**: Bar chart animation
- **Data**: Accuracy changes based on training rounds and timesteps
- **Size**: Typically 150-200KB

#### 2. Client Participation Animation (client_participation.gif)
- **Content**: Visualizes communication patterns and client participation in satellite constellation
- **Format**: Circular network graph animation
- **Data**: Satellite communication data based on FLAM files
- **Size**: Typically 400-600KB

### Animation Generation Conditions

Animations are automatically generated under the following conditions:
- Running complete workflow (`python3 main.py flomps`)
- Sufficient training data (at least 2 data points)
- Successful execution of federated learning module

### Animation Location

```
federated_learning/results_from_output/YYYYMMDD_HHMMSS/
├── accuracy_progress.gif      # Accuracy animation
├── client_participation.gif   # Client participation animation
├── dashboard.html             # Interactive dashboard
├── fl_metrics_*.json          # Detailed metrics
├── fl_model_*.pt              # Trained model
└── fl_results_*.log           # Execution logs
```

### Animation Testing

```bash
# Run GIF animation tests
cd P_45/test_scripts
python3 test_gif_animation.py

# View latest generated animations
ls -la federated_learning/results_from_output/*/accuracy_progress.gif
ls -la federated_learning/results_from_output/*/client_participation.gif
```

### Animation Optimization Tips

1. **Longer animations**: Increase `--timesteps` and `--num-rounds` parameters
2. **More data**: Use more clients (`--num-clients`)
3. **Different models**: Try different model and dataset combinations

## Output Files Description

### 1. Satellite Simulation Output

**Location**: `sat_sim/output/`
**Format**: Text files
**Content**: Adjacency matrices and satellite position data

### 2. Algorithm Output

**Location**: `flomps_algorithm/output/`
**Format**: CSV files
**Naming**: `flam_{satellites}n_{timesteps}t_flomps_3phase_{timestamp}.csv`
**Content**: FLAM scheduling files and communication topology

### 3. Federated Learning Output

**Location**: `federated_learning/results_from_output/`
**Format**: JSON, PyTorch model files
**Content**: Training results, model weights, evaluation metrics

### 4. Test Results

**Location**: `P_45/test_results/`
**Format**: JSON files
**Content**: Test configuration, execution time, success/failure status

## Troubleshooting Guide

### 1. Common Errors

#### Module Import Error
```
ModuleNotFoundError: No module named 'torch'
```
**Solution**:
```bash
pip install torch torchvision
```

#### TLE File Format Error
```
ValueError: Invalid TLE format
```
**Solution**: Check TLE file format, ensure it contains satellite names and two lines of orbital data

#### Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution**: 
- Reduce number of satellites
- Reduce number of timesteps
- Use smaller models

### 2. Performance Optimization

#### Accelerate Satellite Simulation
- Reduce simulation time range
- Increase timestep length
- Use fewer satellites

#### Optimize Algorithm Performance
- Enable FedAvg mode (simplified algorithm)
- Reduce maximum lookahead time
- Adjust minimum connected satellites

#### Improve FL Training
- Reduce training rounds
- Use smaller models
- Reduce number of clients

### 3. Debugging Tips

#### Enable Detailed Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Check Intermediate Results
- View output directories of each module
- Check JSON configuration files
- Verify data formats

#### Use Test Scripts
```bash
# Run test suite
python3 P_45/test_scripts/test_complete_workflow.py
```

## Advanced Usage

### 1. Custom Models

```python
# Add new models in federated_learning/model_selection.py
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Define model architecture
        pass
```

### 2. Custom Datasets

```python
# Add new datasets in federated_learning/fl_core.py
def load_custom_dataset():
    # Implement data loading logic
    pass
```

### 3. Modify Algorithm Parameters

```python
# Adjust algorithm logic in flomps_algorithm/algorithm_core.py
class AlgorithmCore:
    def __init__(self):
        self.phase_length = 5  # Custom phase length
        self.max_rounds = 10   # Custom maximum rounds
```

## Best Practices

### 1. Project Organization
- Use version control
- Regularly backup configurations
- Record experimental results

### 2. Performance Monitoring
- Monitor memory usage
- Record execution time
- Analyze output quality

### 3. Testing Strategy
- Test small-scale configurations first
- Gradually increase complexity
- Run complete tests regularly

### 4. Documentation Maintenance
- Update configuration descriptions
- Record problem solutions
- Share usage experiences

## Support

For help or to report issues, please:

1. Check the troubleshooting section of this manual
2. Review the project README and documentation
3. Run test scripts to verify functionality
4. Contact project maintainers

---

**Version**: 1.0  
**Last Updated**: 2025-10-24  
**Maintainers**: SPACE Project Team
