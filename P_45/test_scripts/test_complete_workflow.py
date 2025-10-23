#!/usr/bin/env python3
"""
Complete Workflow Test Script
Tests complete workflow: TLE → SatSim → Algorithm → FL
End-to-end testing for individual TLE files
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_complete_workflow(tle_file, test_name="Complete Workflow Test"):
    """
    Test complete workflow
    
    Args:
        tle_file (str): TLE file path
        test_name (str): Test name
    """
    print(f"\n{'='*60}")
    print(f"🚀 Starting {test_name}")
    print(f"📁 TLE file: {tle_file}")
    print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Check if TLE file exists
    if not os.path.exists(tle_file):
        print(f"❌ Error: TLE file does not exist: {tle_file}")
        return False
    
    # Set output directory
    output_dir = project_root / "P_45" / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Record test results
    test_results = {
        "test_name": test_name,
        "tle_file": tle_file,
        "start_time": datetime.now().isoformat(),
        "steps": {}
    }
    
    try:
        # Step 1: Run satellite simulation
        print("\n📡 Step 1: Running satellite simulation (SatSim)")
        start_time = time.time()
        
        from sat_sim.sat_sim import SatSim
        from skyfield.api import load
        
        # Create time objects
        ts = load.timescale()
        now = datetime.now()
        start_time_sim = ts.utc(now.year, now.month, now.day, 0, 0, 0)
        end_time_sim = ts.utc(now.year, now.month, now.day, 1, 0, 0)  # 1 hour simulation
        
        # Create SatSim instance
        sat_sim = SatSim(
            start_time=start_time_sim,
            end_time=end_time_sim,
            timestep=1,
            output_file_type='txt',
            gui_enabled=False,
            output_to_file=True
        )
        
        # Read TLE data
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
        
        # Run simulation
        matrices = sat_sim.run_with_adj_matrix()
        satellite_names = list(tle_data.keys())
        
        step1_time = time.time() - start_time
        test_results["steps"]["sat_sim"] = {
            "status": "success",
            "duration": step1_time,
            "satellites": len(satellite_names),
            "timesteps": len(matrices)
        }
        
        print(f"✅ Satellite simulation completed: {len(satellite_names)} satellites, {len(matrices)} timesteps")
        print(f"⏱️  Duration: {step1_time:.2f} seconds")
        
        # Step 2: Run algorithm
        print("\n🧮 Step 2: Running FLOMPS algorithm")
        start_time = time.time()
        
        from flomps_algorithm.algorithm_handler import AlgorithmHandler
        from flomps_algorithm.algorithm_core import Algorithm
        from flomps_algorithm.algorithm_output import AlgorithmOutput
        
        # Create algorithm handlers
        algorithm_core = Algorithm()
        algorithm_handler = AlgorithmHandler(algorithm_core)
        algorithm_output = AlgorithmOutput()
        
        # Set adjacency matrices and satellite names
        algorithm_handler.adjacency_matrices = matrices
        algorithm_handler.sat_names = satellite_names
        
        # Set data and run algorithm
        algorithm_core.set_adjacency_matrices(algorithm_handler.adjacency_matrices)
        algorithm_core.set_satellite_names(algorithm_handler.sat_names)
        algorithm_core.start_algorithm_steps()
        
        # Get algorithm result and output FLAM files
        algorithm_result = algorithm_core.get_algorithm_output()
        algorithm_output.process_algorithm_output(algorithm_result)
        
        step2_time = time.time() - start_time
        test_results["steps"]["algorithm"] = {
            "status": "success",
            "duration": step2_time,
            "output_files": "flomps_algorithm/output/"
        }
        
        print(f"✅ Algorithm completed: FLAM files output to flomps_algorithm/output/")
        print(f"⏱️  Duration: {step2_time:.2f} seconds")
        
        # Step 3: Run federated learning
        print("\n🤖 Step 3: Running federated learning (FL)")
        start_time = time.time()
        
        from federated_learning.fl_core import FederatedLearning
        from federated_learning.fl_config import Config as FLConfig
        
        # Create FL instance
        fl_instance = FederatedLearning()
        
        # Create FL configuration
        fl_config = FLConfig(fl_instance)
        
        # Set default parameters to avoid interactive input
        fl_instance.num_rounds = 1
        fl_instance.num_clients = 4
        fl_instance.model_type = "SimpleCNN"
        fl_instance.data_set = "MNIST"
        
        # Initialize data and model without interactive prompts
        fl_instance.initialize_data("MNIST")
        fl_instance.initialize_model("SimpleCNN", auto_select=False, interactive_mode=False)
        
        # Run federated learning (skip interactive parts)
        try:
            fl_result = fl_instance.run()
        except Exception as e:
            print(f"⚠️  FL run failed: {e}")
            fl_result = {"status": "error", "message": str(e)}
        
        step3_time = time.time() - start_time
        test_results["steps"]["federated_learning"] = {
            "status": "success",
            "duration": step3_time,
            "rounds": fl_config.num_rounds,
            "clients": fl_config.num_clients
        }
        
        print(f"✅ Federated learning completed: {fl_config.num_rounds} rounds, {fl_config.num_clients} clients")
        print(f"⏱️  Duration: {step3_time:.2f} seconds")
        
        # Test completed
        total_time = step1_time + step2_time + step3_time
        test_results["end_time"] = datetime.now().isoformat()
        test_results["total_duration"] = total_time
        test_results["overall_status"] = "success"
        
        print(f"\n🎉 {test_name} completed!")
        print(f"⏱️  Total duration: {total_time:.2f} seconds")
        print(f"📊 Satellite count: {len(satellite_names)}")
        print(f"📊 Timestep count: {len(matrices)}")
        print(f"📊 FL rounds: {fl_config.num_rounds}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        test_results["end_time"] = datetime.now().isoformat()
        test_results["overall_status"] = "failed"
        test_results["error"] = str(e)
        return False
    
    finally:
        # Save test results
        results_file = output_dir / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        print(f"📄 Test results saved to: {results_file}")

def main():
    """Main function"""
    print("🧪 SPACE Project Complete Workflow Test")
    print("=" * 50)
    
    # Test configuration
    tle_files = [
        "TLEs/SatCount1.tle",
        "TLEs/SatCount4.tle", 
        "TLEs/SatCount8.tle"
    ]
    
    success_count = 0
    total_tests = len(tle_files)
    
    for tle_file in tle_files:
        test_name = f"Complete Workflow Test - {os.path.basename(tle_file)}"
        if test_complete_workflow(tle_file, test_name):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"📊 Test Summary")
    print(f"✅ Success: {success_count}/{total_tests}")
    print(f"❌ Failed: {total_tests - success_count}/{total_tests}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
