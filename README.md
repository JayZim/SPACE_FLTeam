<div align="center">
<img src="https://www.swinburne.edu.au/content/dam/media/brand/logo-long-full.svg" alt="Alt text" width="500"/>
</div>

# Project I54 - Validating a new concept in Federated Learning for Satellite Swarms

This project requires the following Core Software Libraries:
* Python 3.12
* TensorFlow (Version: 2.16.1)
* TensorFlow Federated (Version 0.75.0)
* NumPy (Version 1.26.4)
* Pandas (Version 2.2.2)
* Matplotlib (Version 3.8.4)

## Project I54 Overview
**What does it do?**

The S.P.A.C.E project is a space simulation suite intending to validate a concept designed by the project's client – Swinburne's Space Instrumentation Engineering Group (SIEG). This concept aims to address current Earth Observation limitations, through the implementation of decentralised learning of satellite swarms, with the S.P.A.C.E project providing the framework for a virtual suite to which the client can use to further progress the initial FLOMPS (Federated Learning Over Moving Parameter Server) concept.

### 🎬 New Features
- **GIF Animation Generation**: Automatic generation of animated visualizations
  - Accuracy progress animations showing federated learning convergence
  - Client participation animations visualizing satellite communication patterns
- **Interactive Dashboards**: Real-time monitoring and visualization tools
- **Complete Workflow**: End-to-end simulation from TLE files to animated results 

## 📚 Documentation

### Quick Start
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - 30 seconds to get started
- **[PROJECT_ARCHITECTURE_GUIDE.md](PROJECT_ARCHITECTURE_GUIDE.md)** - full project architecture guide
- **[TEAM_FLAM_GENERATOR_GUIDE.md](TEAM_FLAM_GENERATOR_GUIDE.md)** - FLAM generator guide

### Installation & Testing
To install the required dependencies, run the following command:
```bash
# Install dependencies
pip install -r requirements.txt

# Test complete workflow with GIF animations
python3 main.py flomps TLEs/SatCount8.tle --timesteps 15 --model-type SimpleCNN --data-set MNIST --num-rounds 3 --num-clients 4

# Test GIF animation generation
cd P_45/test_scripts
python3 test_gif_animation.py

# Generate FLAM CSV file
python generate_flam_csv.py

# Start GUI
python SPACEGUI.py
```

## Full Documentation
For full documentation, please refer to the [Wiki](https://github.com/samhallSwin/SPACE/wiki) for system and operational details.
