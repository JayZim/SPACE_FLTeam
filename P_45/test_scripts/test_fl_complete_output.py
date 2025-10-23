#!/usr/bin/env python3
"""
Complete FL Output Test Script
Test complete output functionality of FL module, including GIF animation generation
"""

import os
import sys
import time
import json
from datetime import datetime

# Add project root directory to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

def test_fl_complete_output():
    """Test complete output functionality of FL module"""
    
    print("🧪 Testing FL Module Complete Output Functionality")
    print("=" * 60)
    
    # Test configuration
    test_config = {
        "tle_file": "TLEs/SatCount8.tle",
        "timesteps": 60,
        "model": "SimpleCNN",
        "dataset": "MNIST",
        "rounds": 3,
        "clients": 4
    }
    
    print(f"📋 Test Configuration:")
    for key, value in test_config.items():
        print(f"   {key}: {value}")
    
    # Step 1: Run complete FLOMPS workflow
    print(f"\n🚀 Step 1: Running Complete FLOMPS Workflow")
    start_time = time.time()
    
    try:
        import subprocess
        
        # Build command line
        cmd = [
            "python3", "main.py", "flomps", test_config["tle_file"],
            "--timesteps", str(test_config["timesteps"]),
            "--model-type", test_config["model"],
            "--data-set", test_config["dataset"],
            "--num-rounds", str(test_config["rounds"]),
            "--num-clients", str(test_config["clients"])
        ]
        
        print(f"🔧 Executing command: {' '.join(cmd)}")
        
        # Run complete workflow
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
        
        if result.returncode == 0:
            print("✅ Workflow executed successfully")
        else:
            print(f"❌ Workflow execution failed (return code: {result.returncode})")
            print(f"Error output: {result.stderr}")
            return False
        
        workflow_time = time.time() - start_time
        print(f"✅ Complete workflow finished, time taken: {workflow_time:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        return False
    
    # Step 2: Check FL output files
    print(f"\n📁 Step 2: Checking FL Output Files")
    
    # Find the latest FL output directory
    fl_output_root = os.path.join(project_root, "federated_learning", "results_from_output")
    if not os.path.exists(fl_output_root):
        print(f"❌ FL output directory does not exist: {fl_output_root}")
        return False
    
    # Get the latest output directory
    output_dirs = [d for d in os.listdir(fl_output_root) if os.path.isdir(os.path.join(fl_output_root, d))]
    if not output_dirs:
        print(f"❌ No FL output directories found")
        return False
    
    latest_dir = max(output_dirs, key=lambda x: os.path.getctime(os.path.join(fl_output_root, x)))
    latest_output_dir = os.path.join(fl_output_root, latest_dir)
    
    print(f"📂 Latest output directory: {latest_output_dir}")
    
    # Check output files
    expected_files = [
        "fl_results_*.log",
        "fl_metrics_*.json", 
        "fl_model_*.pt",
        "dashboard.html",
        "accuracy_progress.gif",
        "client_participation.gif"
    ]
    
    found_files = []
    missing_files = []
    
    for file_pattern in expected_files:
        if "*" in file_pattern:
            # Use glob to find matching files
            import glob
            matching_files = glob.glob(os.path.join(latest_output_dir, file_pattern))
            if matching_files:
                found_files.extend(matching_files)
            else:
                missing_files.append(file_pattern)
        else:
            file_path = os.path.join(latest_output_dir, file_pattern)
            if os.path.exists(file_path):
                found_files.append(file_path)
            else:
                missing_files.append(file_pattern)
    
    print(f"\n📊 File Check Results:")
    print(f"   ✅ Found files: {len(found_files)}")
    for file_path in found_files:
        file_size = os.path.getsize(file_path)
        print(f"      - {os.path.basename(file_path)} ({file_size} bytes)")
    
    if missing_files:
        print(f"   ❌ Missing files: {len(missing_files)}")
        for file_pattern in missing_files:
            print(f"      - {file_pattern}")
    
    # Step 3: Check FLAM files
    print(f"\n📁 Step 3: Checking FLAM Files")
    
    flam_output_dir = os.path.join(project_root, "flomps_algorithm", "output")
    if os.path.exists(flam_output_dir):
        flam_files = [f for f in os.listdir(flam_output_dir) if f.startswith("flam_") and f.endswith(".csv")]
        if flam_files:
            latest_flam = max(flam_files, key=lambda x: os.path.getctime(os.path.join(flam_output_dir, x)))
            flam_path = os.path.join(flam_output_dir, latest_flam)
            flam_size = os.path.getsize(flam_path)
            print(f"✅ Latest FLAM file: {latest_flam} ({flam_size} bytes)")
        else:
            print(f"❌ No FLAM files found")
    else:
        print(f"❌ FLAM output directory does not exist: {flam_output_dir}")
    
    # Step 4: Generate test report
    print(f"\n📋 Step 4: Generating Test Report")
    
    test_result = {
        "timestamp": datetime.now().isoformat(),
        "test_config": test_config,
        "workflow_time": workflow_time,
        "fl_output_dir": latest_output_dir,
        "found_files": [os.path.basename(f) for f in found_files],
        "missing_files": missing_files,
        "success": len(missing_files) == 0
    }
    
    # Save test report
    report_file = os.path.join(project_root, "P_45", "test_results", f"fl_complete_output_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    with open(report_file, 'w') as f:
        json.dump(test_result, f, indent=2)
    
    print(f"📄 Test report saved: {report_file}")
    
    # Summary
    print(f"\n🎯 Test Summary:")
    if test_result["success"]:
        print(f"   ✅ All FL output files generated successfully")
        print(f"   📊 Output directory: {latest_output_dir}")
        print(f"   ⏱️  Total time: {workflow_time:.2f} seconds")
    else:
        print(f"   ❌ Some files missing, please check output logic")
        print(f"   📊 Output directory: {latest_output_dir}")
    
    return test_result["success"]

if __name__ == "__main__":
    success = test_fl_complete_output()
    sys.exit(0 if success else 1)
