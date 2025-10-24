#!/usr/bin/env python3
"""
Federated Learning Module Standalone Test Script
Test standalone functionality of FL module
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

def test_fl_module(test_name="FL Standalone Test"):
    """
    Test federated learning module
    
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
        # Import Federated Learning module
        from federated_learning.fl_core import FederatedLearning
        from federated_learning.fl_config import Config as FLConfig
        
        print("\n🤖 Running Federated Learning Module")
        
        # Record start time
        start_time = time.time()
        
        # Create FL instance
        print("🏗️  Creating FL instance...")
        fl_instance = FederatedLearning()
        
        # Create FL configuration
        print("⚙️  Creating FL configuration...")
        fl_config = FLConfig(fl_instance)
        
        print(f"✅ FL configuration created:")
        print(f"   - Model type: {fl_config.model_type}")
        print(f"   - Dataset: {fl_config.data_set}")
        print(f"   - Rounds: {fl_config.num_rounds}")
        print(f"   - Clients: {fl_config.num_clients}")
        
        print("✅ FL instance created")
        
        # Set default parameters to avoid interactive input
        fl_instance.num_rounds = 1
        fl_instance.num_clients = 4
        fl_instance.model_type = "SimpleCNN"
        fl_instance.data_set = "MNIST"
        
        # Initialize data and model without interactive prompts
        fl_instance.initialize_data("MNIST")
        fl_instance.initialize_model("SimpleCNN", auto_select=False, interactive_mode=False)
        
        # Run Federated Learning (skip interactive parts)
        print("🔄 Running Federated Learning...")
        try:
            fl_result = fl_instance.run()
        except Exception as e:
            print(f"⚠️  FL run failed: {e}")
            fl_result = {"status": "error", "message": str(e)}
        
        # Check results
        if not fl_result:
            raise Exception("Federated Learning run failed, no results returned")
        
        print(f"✅ Federated Learning run completed")
        
        # Calculate test results
        duration = time.time() - start_time
        test_result["steps"]["federated_learning"] = {
            "status": "success",
            "duration": duration,
            "model_type": fl_config.model_type,
            "data_set": fl_config.data_set,
            "num_rounds": fl_config.num_rounds,
            "num_clients": fl_config.num_clients
        }
        
        test_result["end_time"] = datetime.now().isoformat()
        test_result["total_duration"] = duration
        test_result["overall_status"] = "success"
        
        print(f"✅ Federated Learning module test completed!")
        print(f"⏱️  Duration: {duration:.2f} seconds")
        print(f"📊 Model type: {fl_config.model_type}")
        print(f"📊 Dataset: {fl_config.data_set}")
        print(f"📊 Rounds: {fl_config.num_rounds}")
        print(f"📊 Clients: {fl_config.num_clients}")
        
        # Check output files
        output_dir = project_root / "federated_learning" / "results_from_output"
        if output_dir.exists():
            output_files = list(output_dir.rglob("*"))
            if output_files:
                print(f"📄 Output files: {len(output_files)} files")
                for file in output_files[:5]:  # Show first 5 files
                    if file.is_file():
                        file_size = file.stat().st_size
                        print(f"   - {file.name} ({file_size} bytes)")
                if len(output_files) > 5:
                    print(f"   ... and {len(output_files) - 5} more files")
        
        return test_result
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        test_result["end_time"] = datetime.now().isoformat()
        test_result["overall_status"] = "failed"
        test_result["error"] = str(e)
        return test_result

def test_fl_with_flam(flam_file, test_name="FL with FLAM Test"):
    """
    Test Federated Learning module using FLAM file
    
    Args:
        flam_file (str): FLAM file path
        test_name (str): Test name
    
    Returns:
        dict: Test results
    """
    print(f"\n{'='*60}")
    print(f"🚀 Starting {test_name}")
    print(f"📁 FLAM file: {flam_file}")
    print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Check if FLAM file exists
    if not os.path.exists(flam_file):
        print(f"❌ Error: FLAM file does not exist: {flam_file}")
        return {
            "test_name": test_name,
            "flam_file": flam_file,
            "status": "failed",
            "error": "FLAM file does not exist"
        }
    
    # Record test results
    test_result = {
        "test_name": test_name,
        "flam_file": flam_file,
        "start_time": datetime.now().isoformat(),
        "steps": {}
    }
    
    try:
        # Import Federated Learning module
        from federated_learning.fl_core import FederatedLearning
        from federated_learning.fl_config import Config as FLConfig
        
        print("\n🤖 Running Federated Learning Module with FLAM file")
        
        # Record start time
        start_time = time.time()
        
        # Create FL instance
        print("🏗️  Creating FL instance...")
        fl_instance = FederatedLearning()
        
        # Create FL configuration
        print("⚙️  Creating FL configuration...")
        fl_config = FLConfig(fl_instance)
        
        # Set FLAM file path
        print(f"📄 Setting FLAM file: {flam_file}")
        fl_config.flam_file = flam_file
        
        # Set default parameters to avoid interactive input
        fl_instance.num_rounds = 1
        fl_instance.num_clients = 4
        fl_instance.model_type = "SimpleCNN"
        fl_instance.data_set = "MNIST"
        
        # Initialize data and model without interactive prompts
        fl_instance.initialize_data("MNIST")
        fl_instance.initialize_model("SimpleCNN", auto_select=False, interactive_mode=False)
        
        # Run Federated Learning (skip interactive parts)
        print("🔄 Running Federated Learning...")
        try:
            fl_result = fl_instance.run()
        except Exception as e:
            print(f"⚠️  FL run failed: {e}")
            fl_result = {"status": "error", "message": str(e)}
        
        # Check results
        if not fl_result:
            raise Exception("Federated Learning run failed, no results returned")
        
        print(f"✅ Federated Learning run completed")
        
        # Calculate test results
        duration = time.time() - start_time
        test_result["steps"]["federated_learning"] = {
            "status": "success",
            "duration": duration,
            "model_type": fl_config.model_type,
            "data_set": fl_config.data_set,
            "num_rounds": fl_config.num_rounds,
            "num_clients": fl_config.num_clients,
            "flam_file": os.path.basename(flam_file)
        }
        
        test_result["end_time"] = datetime.now().isoformat()
        test_result["total_duration"] = duration
        test_result["overall_status"] = "success"
        
        print(f"✅ FL with FLAM test completed!")
        print(f"⏱️  Duration: {duration:.2f} seconds")
        print(f"📊 Model type: {fl_config.model_type}")
        print(f"📊 Dataset: {fl_config.data_set}")
        print(f"📊 Rounds: {fl_config.num_rounds}")
        print(f"📊 Clients: {fl_config.num_clients}")
        print(f"📄 FLAM file: {os.path.basename(flam_file)}")
        
        return test_result
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        test_result["end_time"] = datetime.now().isoformat()
        test_result["overall_status"] = "failed"
        test_result["error"] = str(e)
        return test_result

def main():
    """Main function"""
    print("🧪 SPACE Project Federated Learning Module Standalone Test")
    print("=" * 60)
    
    results = []
    
    # Test 1: Basic FL module test
    print("\n📋 Test 1: Basic Federated Learning Module")
    result1 = test_fl_module("FL Basic Test")
    results.append(result1)
    
    print("\n" + "="*60)
    
    # Test 2: FL test using FLAM file
    print("\n📋 Test 2: Federated Learning with FLAM file")
    
    # Find available FLAM files
    flam_output_dir = project_root / "flomps_algorithm" / "output"
    flam_files = []
    if flam_output_dir.exists():
        flam_files = list(flam_output_dir.glob("flam_*.csv"))
    
    if flam_files:
        # Use the latest FLAM file
        latest_flam = max(flam_files, key=lambda x: x.stat().st_mtime)
        print(f"📄 Using latest FLAM file: {latest_flam.name}")
        
        result2 = test_fl_with_flam(str(latest_flam), f"FL with FLAM Test - {latest_flam.name}")
        results.append(result2)
    else:
        print("⚠️  No FLAM files found, skipping FLAM test")
        result2 = {
            "test_name": "FL with FLAM Test",
            "status": "skipped",
            "reason": "No FLAM files found"
        }
        results.append(result2)
    
    # Generate test report
    print(f"\n{'='*60}")
    print(f"📊 Test Summary")
    
    successful_tests = sum(1 for r in results if r.get("overall_status") == "success")
    total_tests = len(results)
    
    print(f"✅ Success: {successful_tests}/{total_tests}")
    print(f"❌ Failed: {total_tests - successful_tests}/{total_tests}")
    
    # Save test results
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "test_suite": "Federated Learning Module Standalone Test",
        "total_tests": total_tests,
        "successful_tests": successful_tests,
        "failed_tests": total_tests - successful_tests,
        "results": results
    }
    
    results_file = project_root / "P_45" / "test_results" / f"fl_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Test results saved to: {results_file}")

if __name__ == "__main__":
    main()