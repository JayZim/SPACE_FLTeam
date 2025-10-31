<div align="center">
<img src="https://www.swinburne.edu.au/content/dam/media/brand/logo-long-full.svg" alt="Alt text" width="500"/>
</div>

# Project 45 – Distributed model training for simulated swarms

This project requires the following Core Software Libraries:
* Python 3.10
* TensorFlow (Version: 2.16.1)
* TensorFlow Federated (Version 0.75.0)
* NumPy (Version 1.26.4)
* Pandas (Version 2.2.2)
* Matplotlib (Version 3.8.4)

## Project 45 Overview
**What does it do?**

The Distributed Model Training for Simulated Satellite Swarms project is focused on improving the way Machine Learning (ML) can be implemented and trained in space, specifically across multiple satellites that operate as a coordinated group. As interest increases in enabling ML systems to function independently in orbit, a key challenge is identifying the best method for training models on-site, where data stays on the satellite instead of being transmitted back to Earth.

Building on previous student research, this study seeks to test a new approach using Federated Learning (FL) to train ML models across a network of simulated satellites.

Federated Learning enables each satellite to train its model locally and share only the changes made to the model, which reduces the need for constant communication and improves data privacy—factors that are especially important for systems operating in space.

The project will involve reviewing existing FL frameworks, choosing or adapting an appropriate approach, or creating a custom FL architecture using TensorFlow. A simulated environment and testing setup are already in place, which will allow for comparisons with existing benchmark systems. Through thorough testing and evaluation, the project will assess how effective this FL model is for satellite constellations. The final goal is to produce research findings that add value to the emerging field of space-based ML and distributed AI systems.

### 🎬 New Features

- **GIF Animation Generation**: Automatic generation of animated visualizations
  - Accuracy progress animations showing federated learning convergence
  - Client participation animations visualizing satellite communication patterns
- **Interactive Dashboards**: Real-time monitoring and visualization tools
- **Complete Workflow**: End-to-end simulation from TLE files to animated results 

## 📚 Documentation

### Quick Start
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - 30 seconds to get started
- **[RUN_FULL_WORKFLOW_TEST_USER_GUIDE.md](RUN_FULL_WORKFLOW_TEST_USER_GUIDE.md)** - Complete user guide for run_full_workflow_test.py
- **[PROJECT_ARCHITECTURE_GUIDE.md](PROJECT_ARCHITECTURE_GUIDE.md)** - full project architecture guide
- **[TEAM_FLAM_GENERATOR_GUIDE.md](TEAM_FLAM_GENERATOR_GUIDE.md)** - FLAM generator guide



### Installation & Testing

#### 1. Environment Setup

To install the required dependencies, run the following command:
```bash
# Install core dependencies from P45_Requirements.txt
python -V
pip install -r P45_Requirements.txt
```

#### 2. Running the Workflow

You have two main options to run the federated learning simulation:

##### Option A: Full Workflow (Recommended)
Run the complete end-to-end workflow from TLE files to results:
```bash
# Run with auto mode (uses latest SatSim or generates from TLE)
python run_full_workflow_test.py --auto

# Run with interactive FL output (prompts for GIF generation and dashboard)
python run_full_workflow_test.py --interactive-fl-output

# Run interactively with custom options
python run_full_workflow_test.py

# Run with specific SatSim file
python run_full_workflow_test.py path/to/sat_sim_output.txt
```

##### Option B: Direct Federated Learning Core
Run the federated learning module directly:
```bash
cd federated_learning
python fl_core.py
# Follow the prompts to provide FLAM file path and configuration
```

#### 3. Testing & Additional Tools
```bash
# Test system requirements
python test_complete_integration.py

# Generate FLAM CSV file
python generate_flam_csv.py

# Start GUI
python SPACEGUI.py
```

## Full Documentation
For full documentation, please refer to the [Wiki](https://github.com/samhallSwin/SPACE/wiki) for system and operational details.
