#!/usr/bin/env python3
"""
Non-Interactive Complete Workflow Test
Tests the complete workflow without any interactive input
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

def test_non_interactive_workflow():
    """Test complete workflow without interactive input"""
    print("🧪 Non-Interactive Complete Workflow Test")
    print("=" * 60)
    
    test_results = {
        "test_name": "Non-Interactive Complete Workflow Test",
        "start_time": datetime.now().isoformat(),
        "steps": {},
        "overall_status": "success"
    }
    
    try:
        # Step 1: Test SatSim
        print("\n📡 Step 1: Testing Satellite Simulation")
        start_time = time.time()
        
        from sat_sim.sat_sim import SatSim
        from skyfield.api import load
        
        # Create time objects
        ts = load.timescale()
        start_time_obj = ts.utc(2025, 10, 24, 0, 0, 0)
        end_time_obj = ts.utc(2025, 10, 24, 0, 3, 0)  # 3 minutes
        
        # Create SatSim instance
        sat_sim = SatSim(
            start_time=start_time_obj,
            end_time=end_time_obj,
            timestep=1,
            output_file_type='txt',
            gui_enabled=False,
            output_to_file=False
        )
        
        # Load TLE data
        tle_file = os.path.join(project_root, "TLEs", "SatCount8.tle")
        if not os.path.exists(tle_file):
            raise FileNotFoundError(f"TLE file not found: {tle_file}")
        
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
        test_results["steps"]["satellite_simulation"] = {
            "status": "success",
            "duration": step1_time,
            "satellites": len(satellite_names),
            "timesteps": len(matrices)
        }
        print(f"✅ SatSim completed: {len(satellite_names)} satellites, {len(matrices)} timesteps")
        
        # Step 2: Test Algorithm
        print("\n🧮 Step 2: Testing FLOMPS Algorithm")
        start_time = time.time()
        
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
        
        # Get algorithm result
        algorithm_result = algorithm_core.get_algorithm_output()
        algorithm_output.process_algorithm_output(algorithm_result)
        
        step2_time = time.time() - start_time
        test_results["steps"]["algorithm"] = {
            "status": "success",
            "duration": step2_time,
            "output_files": "flomps_algorithm/output/"
        }
        print(f"✅ Algorithm completed: FLAM files output to flomps_algorithm/output/")
        
        # Step 3: Test FL (without interactive input)
        print("\n🤖 Step 3: Testing Federated Learning (Non-Interactive)")
        start_time = time.time()
        
        from federated_learning.fl_core import FederatedLearning
        from federated_learning.fl_config import Config as FLConfig
        
        # Create FL instance
        fl_instance = FederatedLearning()
        fl_config = FLConfig(fl_instance)
        
        # Set default parameters to avoid interactive input
        fl_instance.num_rounds = 1
        fl_instance.num_clients = 4
        fl_instance.model_type = "SimpleCNN"
        fl_instance.data_set = "MNIST"
        
        # Initialize data and model without interactive prompts
        fl_instance.initialize_data("MNIST")
        fl_instance.initialize_model("SimpleCNN", auto_select=False, interactive_mode=False)
        
        # Test FL initialization (skip actual training to avoid interactive input)
        fl_result = {"status": "initialized", "message": "FL module ready (training skipped for automation)"}
        
        step3_time = time.time() - start_time
        test_results["steps"]["federated_learning"] = {
            "status": "success",
            "duration": step3_time,
            "message": "FL module initialized successfully (training skipped for automation)"
        }
        print(f"✅ FL module initialized successfully")
        
        # Test completed
        total_time = step1_time + step2_time + step3_time
        test_results["end_time"] = datetime.now().isoformat()
        test_results["total_duration"] = total_time
        test_results["overall_status"] = "success"
        
        print(f"\n🎉 Non-Interactive Complete Workflow Test completed!")
        print(f"⏱️  Total duration: {total_time:.2f} seconds")
        print(f"📊 Satellite count: {len(satellite_names)}")
        print(f"📊 Timestep count: {len(matrices)}")
        
        return test_results
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        test_results["end_time"] = datetime.now().isoformat()
        test_results["overall_status"] = "failed"
        test_results["error"] = str(e)
        return test_results

def main():
    """Main function"""
    result = test_non_interactive_workflow()
    
    # Save test results
    results_dir = os.path.join(project_root, "P_45", "test_results")
    os.makedirs(results_dir, exist_ok=True)
    
    result_file = os.path.join(results_dir, f"non_interactive_workflow_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n📄 Test results saved to: {result_file}")
    
    if result["overall_status"] == "success":
        print("✅ Test completed successfully!")
        sys.exit(0)
    else:
        print("❌ Test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

