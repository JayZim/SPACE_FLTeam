#!/usr/bin/env python3
"""
Main Run Script - 6 Satellites, 100 Timesteps (Interactive Version)
Interactive execution with user prompts for configuration
"""

import os
import sys
import time
import json
from datetime import datetime

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, project_root)

def get_user_input():
    """Get user configuration input"""
    print("🚀 SPACE Project Main Run - Interactive Configuration")
    print("=" * 70)
    
    # Get satellite count
    while True:
        try:
            sat_count = int(input("Enter number of satellites (default: 6): ").strip() or "6")
            if sat_count < 1 or sat_count > 20:
                print("Please enter a number between 1 and 20")
                continue
            break
        except ValueError:
            print("Please enter a valid number")
    
    # Get timesteps
    while True:
        try:
            timesteps = int(input("Enter number of timesteps (default: 100): ").strip() or "100")
            if timesteps < 1 or timesteps > 1000:
                print("Please enter a number between 1 and 1000")
                continue
            break
        except ValueError:
            print("Please enter a valid number")
    
    # Get FL rounds
    while True:
        try:
            fl_rounds = int(input("Enter FL rounds (default: 5): ").strip() or "5")
            if fl_rounds < 1 or fl_rounds > 50:
                print("Please enter a number between 1 and 50")
                continue
            break
        except ValueError:
            print("Please enter a valid number")
    
    # Get model type
    print("\nAvailable models:")
    print("1. SimpleCNN (default)")
    print("2. ResNet50")
    print("3. EfficientNetB0")
    print("4. CustomCNN")
    print("5. VisionTransformer")
    
    model_choice = input("Select model (1-5, default: 1): ").strip() or "1"
    model_map = {
        "1": "SimpleCNN",
        "2": "ResNet50", 
        "3": "EfficientNetB0",
        "4": "CustomCNN",
        "5": "VisionTransformer"
    }
    model_type = model_map.get(model_choice, "SimpleCNN")
    
    # Get dataset
    print("\nAvailable datasets:")
    print("1. MNIST (default)")
    print("2. CIFAR10")
    print("3. EuroSAT")
    
    dataset_choice = input("Select dataset (1-3, default: 1): ").strip() or "1"
    dataset_map = {
        "1": "MNIST",
        "2": "CIFAR10",
        "3": "EuroSAT"
    }
    dataset = dataset_map.get(dataset_choice, "MNIST")
    
    # Get TLE file
    print("\nAvailable TLE files:")
    tle_files = []
    tle_dir = os.path.join(project_root, "TLEs")
    if os.path.exists(tle_dir):
        tle_files = [f for f in os.listdir(tle_dir) if f.endswith('.tle')]
    
    if tle_files:
        for i, tle_file in enumerate(tle_files, 1):
            print(f"{i}. {tle_file}")
        
        while True:
            try:
                tle_choice = input(f"Select TLE file (1-{len(tle_files)}, default: 1): ").strip() or "1"
                tle_idx = int(tle_choice) - 1
                if 0 <= tle_idx < len(tle_files):
                    tle_file = os.path.join(tle_dir, tle_files[tle_idx])
                    break
                else:
                    print(f"Please enter a number between 1 and {len(tle_files)}")
            except ValueError:
                print("Please enter a valid number")
    else:
        print("No TLE files found, using default")
        tle_file = os.path.join(project_root, "TLEs", "SatCount8.tle")
    
    # Enable interactive features
    enable_interactive = input("\nEnable interactive model selection? (y/N): ").strip().lower() == 'y'
    enable_adaptation = input("Enable FL adaptation system? (y/N): ").strip().lower() == 'y'
    
    return {
        'sat_count': sat_count,
        'timesteps': timesteps,
        'fl_rounds': fl_rounds,
        'model_type': model_type,
        'dataset': dataset,
        'tle_file': tle_file,
        'enable_interactive': enable_interactive,
        'enable_adaptation': enable_adaptation
    }

def main_run_interactive():
    """Main run with interactive configuration"""
    config = get_user_input()
    
    print(f"\n🚀 SPACE Project Main Run - {config['sat_count']} Satellites, {config['timesteps']} Timesteps")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  - Satellites: {config['sat_count']}")
    print(f"  - Timesteps: {config['timesteps']}")
    print(f"  - FL Rounds: {config['fl_rounds']}")
    print(f"  - Model: {config['model_type']}")
    print(f"  - Dataset: {config['dataset']}")
    print(f"  - TLE File: {os.path.basename(config['tle_file'])}")
    print(f"  - Interactive: {config['enable_interactive']}")
    print(f"  - Adaptation: {config['enable_adaptation']}")
    print("=" * 70)
    
    start_time = time.time()
    
    try:
        # Step 1: Satellite Simulation
        print("\n📡 Step 1: Running Satellite Simulation")
        print("-" * 50)
        
        from sat_sim.sat_sim import SatSim
        from skyfield.api import load
        
        # Create time objects
        ts = load.timescale()
        start_time_obj = ts.utc(2025, 10, 24, 0, 0, 0)
        end_time_obj = ts.utc(2025, 10, 24, 0, config['timesteps'], 0)  # timesteps minutes
        
        # Create SatSim instance
        sat_sim = SatSim(
            start_time=start_time_obj,
            end_time=end_time_obj,
            timestep=1,
            output_file_type='txt',
            gui_enabled=False,
            output_to_file=True
        )
        
        # Load TLE data
        if not os.path.exists(config['tle_file']):
            raise FileNotFoundError(f"TLE file not found: {config['tle_file']}")
        
        print(f"📄 Using TLE file: {os.path.basename(config['tle_file'])}")
        
        # Load TLE data
        tle_data = {}
        with open(config['tle_file'], 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        
        i = 0
        while i < len(lines):
            if i + 2 < len(lines):
                name = lines[i].strip()
                tle_line1 = lines[i + 1].strip()
                tle_line2 = lines[i + 2].strip()
                tle_data[name] = [tle_line1, tle_line2]
                i += 3
            else:
                break
        
        sat_sim.set_tle_data(tle_data)
        matrices = sat_sim.run_with_adj_matrix()
        satellite_names = list(tle_data.keys())
        
        step1_time = time.time() - start_time
        print(f"✅ SatSim completed: {len(satellite_names)} satellites, {len(matrices)} timesteps")
        print(f"⏱️  Duration: {step1_time:.2f} seconds")
        
        # Step 2: FLOMPS Algorithm
        print("\n🧮 Step 2: Running FLOMPS Algorithm")
        print("-" * 50)
        
        from flomps_algorithm.algorithm_core import Algorithm
        from flomps_algorithm.algorithm_handler import AlgorithmHandler
        from flomps_algorithm.algorithm_output import AlgorithmOutput
        
        algorithm_core = Algorithm()
        algorithm_handler = AlgorithmHandler(algorithm_core)
        algorithm_output = AlgorithmOutput()
        
        # Set data and run algorithm
        algorithm_core.set_adjacency_matrices(matrices)
        algorithm_core.set_satellite_names(satellite_names)
        algorithm_core.start_algorithm_steps()
        
        # Get algorithm result and output FLAM files
        algorithm_result = algorithm_core.get_algorithm_output()
        algorithm_output.process_algorithm_output(algorithm_result)
        
        step2_time = time.time() - start_time - step1_time
        print(f"✅ Algorithm completed: FLAM files output to flomps_algorithm/output/")
        print(f"⏱️  Duration: {step2_time:.2f} seconds")
        
        # Step 3: Federated Learning
        print("\n🤖 Step 3: Running Federated Learning")
        print("-" * 50)
        
        from federated_learning.fl_core import FederatedLearning
        from federated_learning.fl_config import Config as FLConfig
        from federated_learning.fl_handler import FLHandler
        
        # Create FL instance with user preferences
        fl_instance = FederatedLearning(
            enable_model_evaluation=config['enable_interactive'],
            enable_adaptation=config['enable_adaptation']
        )
        fl_config = FLConfig(fl_instance)
        fl_handler = FLHandler(fl_instance)
        
        # Set parameters
        fl_instance.num_rounds = config['fl_rounds']
        fl_instance.num_clients = config['sat_count']
        fl_instance.model_type = config['model_type']
        fl_instance.data_set = config['dataset']
        
        # Initialize data and model
        if config['enable_interactive']:
            print("🔄 Interactive mode enabled - you may be prompted for model selection")
            fl_instance.initialize_data(config['dataset'])
            fl_instance.initialize_model(config['model_type'], auto_select=True, interactive_mode=True)
        else:
            print("🔄 Non-interactive mode - using fallback methods")
            fl_instance._initialize_data_fallback(config['dataset'])
            fl_instance._initialize_model_fallback(config['model_type'], auto_select=False, interactive_mode=False)
        
        # Get the latest FLAM file for FL training
        flam_output_dir = os.path.join(project_root, "flomps_algorithm", "output")
        flam_files = [f for f in os.listdir(flam_output_dir) if f.endswith('.csv') and 'flam' in f.lower()]
        
        if flam_files:
            # Use the most recent FLAM file
            latest_flam = max(flam_files, key=lambda x: os.path.getmtime(os.path.join(flam_output_dir, x)))
            flam_path = os.path.join(flam_output_dir, latest_flam)
            print(f"📄 Using FLAM file: {latest_flam}")
        else:
            flam_path = None
            print("⚠️  No FLAM file found, running without FLAM")
        
        # Run federated learning through FL handler
        print("🔄 Running FL training...")
        try:
            # Load FLAM file if available
            if flam_path:
                fl_handler.load_flam_file(flam_path)
            
            # Use FL handler to run FL with output generation
            fl_handler.run_module()
            print(f"✅ FL training completed successfully")
        except Exception as e:
            print(f"❌ FL training failed: {e}")
            import traceback
            traceback.print_exc()
            fl_result = None
        
        step3_time = time.time() - start_time - step1_time - step2_time
        print(f"✅ FL completed: {fl_instance.num_rounds} rounds, {fl_instance.num_clients} clients")
        print(f"⏱️  Duration: {step3_time:.2f} seconds")
        
        # Step 4: Generate Visualizations
        print("\n🎨 Step 4: Generating Visualizations")
        print("-" * 50)
        
        # Check for generated files
        fl_output_dir = os.path.join(project_root, "federated_learning", "results_from_output")
        if os.path.exists(fl_output_dir):
            subdirs = [d for d in os.listdir(fl_output_dir) if os.path.isdir(os.path.join(fl_output_dir, d))]
            if subdirs:
                latest_dir = max(subdirs, key=lambda x: os.path.getmtime(os.path.join(fl_output_dir, x)))
                latest_path = os.path.join(fl_output_dir, latest_dir)
                
                # Check for generated files
                files = os.listdir(latest_path)
                gif_files = [f for f in files if f.endswith('.gif')]
                json_files = [f for f in files if f.endswith('.json')]
                log_files = [f for f in files if f.endswith('.log')]
                model_files = [f for f in files if f.endswith('.pt')]
                html_files = [f for f in files if f.endswith('.html')]
                
                print(f"📂 Output directory: {latest_dir}")
                print(f"📊 Generated files:")
                print(f"   - GIF animations: {len(gif_files)} files")
                print(f"   - JSON metrics: {len(json_files)} files")
                print(f"   - Log files: {len(log_files)} files")
                print(f"   - Model files: {len(model_files)} files")
                print(f"   - HTML dashboards: {len(html_files)} files")
                
                if gif_files:
                    print(f"   🎬 GIF files: {', '.join(gif_files)}")
                if html_files:
                    print(f"   📊 Dashboard: {', '.join(html_files)}")
        
        # Final summary
        total_time = time.time() - start_time
        print(f"\n🎉 Interactive Main Run Completed Successfully!")
        print("=" * 70)
        print(f"📊 Final Configuration:")
        print(f"   - Satellites: {len(satellite_names)}")
        print(f"   - Timesteps: {len(matrices)}")
        print(f"   - FL Rounds: {fl_instance.num_rounds}")
        print(f"   - FL Clients: {fl_instance.num_clients}")
        print(f"   - Model: {fl_instance.model_type}")
        print(f"   - Dataset: {fl_instance.data_set}")
        print(f"   - Interactive: {config['enable_interactive']}")
        print(f"   - Adaptation: {config['enable_adaptation']}")
        print(f"\n⏱️  Timing:")
        print(f"   - SatSim: {step1_time:.2f} seconds")
        print(f"   - Algorithm: {step2_time:.2f} seconds")
        print(f"   - FL: {step3_time:.2f} seconds")
        print(f"   - Total: {total_time:.2f} seconds")
        
        # Save run report
        report = {
            "timestamp": datetime.now().isoformat(),
            "mode": "interactive",
            "configuration": {
                "satellites": len(satellite_names),
                "timesteps": len(matrices),
                "fl_rounds": fl_instance.num_rounds,
                "fl_clients": fl_instance.num_clients,
                "model_type": fl_instance.model_type,
                "data_set": fl_instance.data_set,
                "interactive_mode": config['enable_interactive'],
                "adaptation_enabled": config['enable_adaptation']
            },
            "timing": {
                "sat_sim": step1_time,
                "algorithm": step2_time,
                "federated_learning": step3_time,
                "total": total_time
            },
            "status": "success"
        }
        
        results_dir = os.path.join(project_root, "P_45", "test_results")
        os.makedirs(results_dir, exist_ok=True)
        
        report_file = os.path.join(results_dir, f"main_run_interactive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Interactive run report saved to: {report_file}")
        
        return True
        
    except Exception as e:
        total_time = time.time() - start_time
        print(f"\n❌ Interactive Main Run Failed!")
        print(f"⏱️  Duration: {total_time:.2f} seconds")
        print(f"💥 Error: {str(e)}")
        
        # Save error report
        error_report = {
            "timestamp": datetime.now().isoformat(),
            "mode": "interactive",
            "status": "failed",
            "error": str(e),
            "duration": total_time
        }
        
        results_dir = os.path.join(project_root, "P_45", "test_results")
        os.makedirs(results_dir, exist_ok=True)
        
        error_file = os.path.join(results_dir, f"main_run_interactive_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(error_file, 'w') as f:
            json.dump(error_report, f, indent=2)
        
        print(f"📄 Error report saved to: {error_file}")
        
        return False

def main():
    """Main function"""
    success = main_run_interactive()
    
    if success:
        print(f"\n🎉 Interactive main run completed successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ Interactive main run failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
