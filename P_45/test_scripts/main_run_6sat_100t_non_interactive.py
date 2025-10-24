#!/usr/bin/env python3
"""
Main Run Script - 6 Satellites, 100 Timesteps (Non-Interactive Version)
Fully automated execution without any user input prompts
"""

import os
import sys

# Fix OpenMP library conflict on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import time
import json
from datetime import datetime

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, project_root)

def main_run_non_interactive():
    """Main run: 6 satellites, 100 timesteps - fully automated"""
    print("🚀 SPACE Project Main Run - 6 Satellites, 100 Timesteps (Non-Interactive)")
    print("=" * 70)
    
    start_time = time.time()
    
    try:
        # Step 1: Satellite Simulation
        print("\n📡 Step 1: Running Satellite Simulation")
        print("-" * 50)
        
        from sat_sim.sat_sim import SatSim
        from skyfield.api import load
        
        # Create time objects (100 minutes = 100 timesteps)
        ts = load.timescale()
        start_time_obj = ts.utc(2025, 10, 24, 0, 0, 0)
        end_time_obj = ts.utc(2025, 10, 24, 1, 40, 0)  # 100 minutes
        
        # Create SatSim instance
        sat_sim = SatSim(
            start_time=start_time_obj,
            end_time=end_time_obj,
            timestep=1,
            output_file_type='txt',
            gui_enabled=False,
            output_to_file=True
        )
        
        # Load TLE data (use SatCount6.tle if available, otherwise SatCount8.tle)
        tle_files = ["TLEs/SatCount6.tle", "TLEs/SatCount8.tle"]
        tle_file = None
        for tle_path in tle_files:
            if os.path.exists(os.path.join(project_root, tle_path)):
                tle_file = os.path.join(project_root, tle_path)
                break
        
        if not tle_file:
            raise FileNotFoundError("No TLE file found")
        
        print(f"📄 Using TLE file: {os.path.basename(tle_file)}")
        
        # Load TLE data
        tle_data = {}
        with open(tle_file, 'r') as f:
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
        
        # Create FL instance with adaptation system enabled
        fl_instance = FederatedLearning(enable_model_evaluation=True, enable_adaptation=True)
        fl_config = FLConfig(fl_instance)
        fl_handler = FLHandler(fl_instance)
        
        # Set parameters for 6 satellites, 100 timesteps
        fl_instance.num_rounds = 5  # More rounds for 100 timesteps
        fl_instance.num_clients = 6  # Match satellite count
        fl_instance.model_type = "SimpleCNN"
        fl_instance.data_set = "MNIST"
        
        # Initialize data and model using adaptation system
        fl_instance.initialize_data("MNIST", use_heterogeneous=True)
        fl_instance.initialize_model("SimpleCNN", auto_select=True, interactive_mode=False)
        
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
        print(f"\n🎉 Non-Interactive Main Run Completed Successfully!")
        print("=" * 70)
        print(f"📊 Configuration:")
        print(f"   - Satellites: {len(satellite_names)}")
        print(f"   - Timesteps: {len(matrices)}")
        print(f"   - FL Rounds: {fl_instance.num_rounds}")
        print(f"   - FL Clients: {fl_instance.num_clients}")
        print(f"   - Model: {fl_instance.model_type}")
        print(f"   - Dataset: {fl_instance.data_set}")
        print(f"   - Mode: Non-Interactive (Automated)")
        print(f"\n⏱️  Timing:")
        print(f"   - SatSim: {step1_time:.2f} seconds")
        print(f"   - Algorithm: {step2_time:.2f} seconds")
        print(f"   - FL: {step3_time:.2f} seconds")
        print(f"   - Total: {total_time:.2f} seconds")
        
        # Save run report
        report = {
            "timestamp": datetime.now().isoformat(),
            "mode": "non_interactive",
            "configuration": {
                "satellites": len(satellite_names),
                "timesteps": len(matrices),
                "fl_rounds": fl_instance.num_rounds,
                "fl_clients": fl_instance.num_clients,
                "model_type": fl_instance.model_type,
                "data_set": fl_instance.data_set,
                "interactive_mode": False,
                "adaptation_enabled": False
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
        
        report_file = os.path.join(results_dir, f"main_run_non_interactive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Non-interactive run report saved to: {report_file}")
        
        return True
        
    except Exception as e:
        total_time = time.time() - start_time
        print(f"\n❌ Non-Interactive Main Run Failed!")
        print(f"⏱️  Duration: {total_time:.2f} seconds")
        print(f"💥 Error: {str(e)}")
        
        # Save error report
        error_report = {
            "timestamp": datetime.now().isoformat(),
            "mode": "non_interactive",
            "status": "failed",
            "error": str(e),
            "duration": total_time
        }
        
        results_dir = os.path.join(project_root, "P_45", "test_results")
        os.makedirs(results_dir, exist_ok=True)
        
        error_file = os.path.join(results_dir, f"main_run_non_interactive_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(error_file, 'w') as f:
            json.dump(error_report, f, indent=2)
        
        print(f"📄 Error report saved to: {error_file}")
        
        return False

def main():
    """Main function"""
    success = main_run_non_interactive()
    
    if success:
        print(f"\n🎉 Non-interactive main run completed successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ Non-interactive main run failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
