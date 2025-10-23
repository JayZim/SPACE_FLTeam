#!/usr/bin/env python3
"""
Algorithm Module Standalone Test Script
Test standalone functionality of algorithm module
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

def test_algorithm_module(test_name="Algorithm Standalone Test"):
    """
    Test algorithm module
    
    Args:
        test_name (str): Test name
    
    Returns:
        dict: Test results
    """
    print(f"\n{'='*60}")
    print(f"🚀 Starting {test_name}")
    print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Record test results
    test_result = {
        "test_name": test_name,
        "start_time": datetime.now().isoformat(),
        "steps": {}
    }
    
    try:
        # Import algorithm modules
        from flomps_algorithm.algorithm_handler import AlgorithmHandler
        from flomps_algorithm.algorithm_core import Algorithm
        from flomps_algorithm.algorithm_output import AlgorithmOutput
        
        print("\n🧮 Running Algorithm Module")
        
        # Record start time
        start_time = time.time()
        
        # Create algorithm handlers
        algorithm_core = Algorithm()
        algorithm_handler = AlgorithmHandler(algorithm_core)
        algorithm_output = AlgorithmOutput()
        
        # Create dummy adjacency matrices for testing
        print("🏗️  Creating test data...")
        import numpy as np
        
        # Create test adjacency matrices
        num_satellites = 8
        num_timesteps = 10
        
        test_matrices = []
        test_sat_names = [f"SAT_{i+1}" for i in range(num_satellites)]
        
        for t in range(num_timesteps):
            # Create random adjacency matrix
            adj_matrix = np.random.rand(num_satellites, num_satellites)
            # Make it symmetric
            adj_matrix = (adj_matrix + adj_matrix.T) / 2
            # Set diagonal to 0
            np.fill_diagonal(adj_matrix, 0)
            # Threshold to create sparse matrix
            adj_matrix = (adj_matrix > 0.3).astype(float)
            
            test_matrices.append((t, adj_matrix))
        
        print(f"✅ Created test data: {num_satellites} satellites, {num_timesteps} timesteps")
        
        # Set test data
        algorithm_handler.adjacency_matrices = test_matrices
        algorithm_handler.sat_names = test_sat_names
        
        # Set data and run algorithm
        print("🔄 Running FLOMPS algorithm...")
        algorithm_core.set_adjacency_matrices(algorithm_handler.adjacency_matrices)
        algorithm_core.set_satellite_names(algorithm_handler.sat_names)
        algorithm_core.start_algorithm_steps()
        
        # Get algorithm result and output FLAM files
        print("📤 Outputting algorithm results...")
        algorithm_result = algorithm_core.get_algorithm_output()
        algorithm_output.process_algorithm_output(algorithm_result)
        
        # Calculate test results
        duration = time.time() - start_time
        test_result["steps"]["algorithm"] = {
            "status": "success",
            "duration": duration,
            "satellites": num_satellites,
            "timesteps": num_timesteps,
            "satellite_names": test_sat_names
        }
        
        test_result["end_time"] = datetime.now().isoformat()
        test_result["total_duration"] = duration
        test_result["overall_status"] = "success"
        
        print(f"✅ Algorithm module test completed!")
        print(f"⏱️  Duration: {duration:.2f} seconds")
        print(f"📊 Satellite count: {num_satellites}")
        print(f"📊 Timestep count: {num_timesteps}")
        print(f"📊 Satellite names: {', '.join(test_sat_names)}")
        
        # Check output files
        output_dir = project_root / "flomps_algorithm" / "output"
        if output_dir.exists():
            output_files = list(output_dir.glob("*.csv"))
            if output_files:
                print(f"📄 Output files: {len(output_files)} files")
                for file in output_files[:3]:  # Show first 3 files
                    file_size = file.stat().st_size
                    print(f"   - {file.name} ({file_size} bytes)")
                if len(output_files) > 3:
                    print(f"   ... and {len(output_files) - 3} more files")
        
        return test_result
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        test_result["end_time"] = datetime.now().isoformat()
        test_result["overall_status"] = "failed"
        test_result["error"] = str(e)
        return test_result

def main():
    """Main function"""
    print("🧪 SPACE Project Algorithm Module Standalone Test")
    print("=" * 60)
    
    # Run algorithm test
    result = test_algorithm_module("Algorithm Standalone Test")
    
    # Generate test report
    print(f"\n{'='*60}")
    print(f"📊 Test Summary")
    
    if result["overall_status"] == "success":
        print(f"✅ Algorithm module test completed successfully!")
        print(f"⏱️  Total duration: {result.get('total_duration', 0):.2f} seconds")
        
        # Show detailed results
        algorithm_step = result.get("steps", {}).get("algorithm", {})
        if algorithm_step:
            print(f"📊 Satellite count: {algorithm_step.get('satellites', 'N/A')}")
            print(f"📊 Timestep count: {algorithm_step.get('timesteps', 'N/A')}")
    else:
        print(f"❌ Algorithm module test failed!")
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    # Save test results
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "test_suite": "Algorithm Module Standalone Test",
        "result": result
    }
    
    results_file = project_root / "P_45" / "test_results" / f"algorithm_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Test results saved to: {results_file}")

if __name__ == "__main__":
    main()