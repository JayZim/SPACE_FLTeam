# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SPACE (Satellite Federated Learning Project) is a simulation system that validates the FLOMPS (Federated Learning Over Moving Parameter Server) concept for satellite swarms. The system simulates satellite orbital motion, calculates communication links, and runs federated learning algorithms based on dynamic topology changes.

**Core Requirements:**
- Python 3.12
- TensorFlow 2.16.1
- TensorFlow Federated 0.75.0
- Key libraries: NumPy 1.26.4, Pandas 2.2.2, Matplotlib 3.8.4, PyTorch 2.7.0
- Skyfield for satellite position calculation (SGP4 propagator)

## Development Commands

### Testing
```bash
# Test complete system integration
python test_complete_integration.py

# Test FL compatibility
python test_fl_compatibility.py

# Test algorithm component
python unit_test_algorithm_component.py

# Test complete workflow
python test_complete_workflow.py

# Test FL 2025 optimizations
python test_fl_2025_optimizations.py
```

### Running the System
```bash
# Generate FLAM CSV files (quickest start)
python generate_flam_csv.py

# Generate with specific TLE file
python generate_flam_csv.py TLEs/SatCount4.tle

# Run complete FLOMPS workflow
python main.py flomps --input-file TLEs/SatCount4.tle --start-time "2024-09-12 12:00:00" --end-time "2024-09-12 13:40:00"

# Run with custom timesteps
python main.py flomps --input-file TLEs/SatCount8.tle --timesteps 100

# Run with custom duration
python main.py flomps --input-file TLEs/SatCount8.tle --custom-duration "01:40:00"

# Launch GUI interface
python SPACEGUI.py
```

### Configuration
```bash
# View current configuration
python main.py settings --show-options

# Change options file
python main.py settings --options-file custom_options.json
```

## Architecture

### Three-Stage Pipeline
The system follows a clear data flow:
```
TLE Orbital Data → SatSim → Algorithm → Federated Learning → Visualization
```

1. **SatSim** (`sat_sim/`): Reads TLE files, calculates satellite positions/communications, generates time-series adjacency matrices
2. **Algorithm** (`flomps_algorithm/`): Processes adjacency matrices through FLOMPS algorithm, generates FLAM files
3. **Federated Learning** (`federated_learning/`): Executes actual FL training using FLAM schedules

### Module Factory Pattern
The system uses a factory pattern (`module_factory.py`) to instantiate modules with their Config, Handler, and Output components:
- `create_sat_sim_module()` → SatSimModule(config, handler, output)
- `create_algorithm_module()` → AlgorithmModule(config, handler, output)
- `create_fl_module()` → FLModule(config, handler, output)
- `create_fl_module_with_model_evaluation()` → FLModule with evaluation enabled

### Configuration System
Configuration follows a three-tier hierarchy:
1. `settings.json` - Global system settings (which config file to use)
2. `options.json` - Module-specific configuration (sat_sim, algorithm, federated_learning)
3. Command-line arguments - Runtime overrides

**Priority:** CLI args > options.json > defaults

### Interface Abstraction Layer
The `interfaces/` directory defines standard interfaces implemented by all modules:
- `config.py` - Configuration interface
- `handler.py` - Handler interface for data processing
- `output.py` - Output interface for results
- `federated_learning.py` - FL-specific interface

This ensures modules are interchangeable and compatible.

### Path Management
Always use `utilities/path_manager.py` for cross-platform path handling:
```python
from utilities.path_manager import ProjectPathManager
pm = ProjectPathManager()
csv_path = pm.get_latest_csv_file()
synth_dir = pm.synth_flams_dir
```

## Key Components

### FLOMPS Algorithm (`flomps_algorithm/algorithm_core.py`)
The core algorithm uses multi-criteria server selection:
1. **Primary:** Maximum connectivity (satellite that can reach most others)
2. **Secondary:** Speed to maximum (fastest to achieve max connectivity)
3. **Tertiary:** Load balancing (least frequently selected)

**Cumulative Connectivity Model:** Connections persist over time within a round (realistic satellite behavior).

**Key Functions:**
- `analyze_all_satellites()` - Comprehensive constellation analysis
- `analyze_single_satellite()` - Deep dive into individual satellite connectivity
- `find_best_server_for_round()` - Multi-criteria server selection
- `start_algorithm_steps()` - Main algorithm execution with round-based processing

### FLAM File Format
Output CSV format (in `synth_FLAMs/`):
```csv
Timestep: 1, Round: 1, Target Node: 0, Phase: TRAINING
0,0,0,0
0,0,0,0
...

Timestep: 4, Round: 1, Target Node: 0, Phase: TRANSMITTING
0,1,1,0
1,0,0,1
...
```

**Filename convention:** `flam_<n>n_<t>t_flomps_<timestamp>.csv`
- `n` = number of satellites
- `t` = number of timesteps
- `flomps` = algorithm type
- `timestamp` = generation time

### TLE Files
Available satellite configurations in `TLEs/`:
- `SatCount4.tle` - 4 satellites (quick test)
- `SatCount8.tle` - 8 satellites (medium scale)
- `SatCount40.tle` - 40 satellites (large scale)
- `Walker.tle` - Walker constellation
- `NovaSar.tle` - Single real satellite

### Workflow System
Workflows are in `workflows/` directory. The `flomps.py` workflow:
1. Creates modules via factory
2. Runs SatSim → generates adjacency matrices
3. Runs Algorithm → generates FLAM files
4. Runs FL → trains models using FLAM schedules

Custom workflows can be added by:
1. Creating file in `workflows/`
2. Adding CLI parser in `main.py`
3. Implementing run function

### Model Evaluation Module
The FL module includes automatic model selection (`federated_learning/model_evaluation.py`):
- Evaluates multiple ML models (SimpleCNN, ResNet variants, etc.)
- Selects best model based on accuracy, training time, memory
- Configurable via `options.json` → `federated_learning.model_evaluation`
- Enable with `create_fl_module_with_model_evaluation()`

## Important Implementation Notes

### When Modifying Modules
- Each module must implement the interfaces defined in `interfaces/`
- Update `module_factory.py` if adding new modules
- Maintain Config/Handler/Output pattern
- Add CLI arguments in `cli_args.py` for new options

### Output Files
- SatSim outputs are intermediate (not typically saved to disk)
- Algorithm outputs to `synth_FLAMs/` as CSV
- FL outputs to `federated_learning/results_from_output/`
- All modules respect `output_to_file` setting in options.json

### Time and Timesteps
- Default timestep = 1 minute
- Simulation time = (end_time - start_time)
- Number of timesteps = simulation time in minutes
- Adjacency matrices generated per timestep

### Known Issues
- TensorFlow 2.16.1 not compatible with Python 3.13 (use Python 3.12)
- Core algorithm functionality doesn't depend on TensorFlow
- Large satellite counts (>40) may cause memory issues

## Testing Strategy

Tests follow integration pattern:
1. Unit tests for individual components (`unit_test_*.py`)
2. Integration tests for module chains (`test_integration.py`)
3. Complete workflow tests (`test_complete_integration.py`, `test_complete_workflow.py`)
4. Compatibility tests (`test_fl_compatibility.py`)

Always run `test_complete_integration.py` before commits to verify system functionality.

## Documentation Files
- `README.md` - Basic project intro
- `PROJECT_ARCHITECTURE_GUIDE.md` - Detailed architecture
- `QUICK_REFERENCE.md` - 30-second overview
- `TEAM_FLAM_GENERATOR_GUIDE.md` - FLAM generator usage
- `flomps_algorithm/ALGORITHM_CORE_README_UPDATED.md` - Algorithm details
- `federated_learning/read_fl_output.md` - FL output documentation
