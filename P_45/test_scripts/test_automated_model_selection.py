#!/usr/bin/env python3
"""
Automated Model and Dataset Selection Test Script
Test different model and dataset combinations
"""

import os
import sys
import subprocess
import json
from datetime import datetime

# Add project root directory to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

def test_model_dataset_combinations():
    """Test different model and dataset combinations"""
    
    # Test configurations
    test_configs = [
        {"model": "SimpleCNN", "dataset": "MNIST", "name": "SimpleCNN + MNIST"},
        {"model": "SimpleCNN", "dataset": "CIFAR10", "name": "SimpleCNN + CIFAR10"},
        {"model": "ResNet50", "dataset": "CIFAR10", "name": "ResNet50 + CIFAR10"},
        {"model": "ResNet50", "dataset": "EuroSAT", "name": "ResNet50 + EuroSAT"},
        {"model": "EfficientNetB0", "dataset": "EuroSAT", "name": "EfficientNetB0 + EuroSAT"}
    ]
    
    results = []
    
    for i, config in enumerate(test_configs, 1):
        print(f"🧪 Test {i}/{len(test_configs)}: {config['name']}")
        print(f"   Model: {config['model']}")
        print(f"   Dataset: {config['dataset']}")
        
        # Build command
        cmd = [
            "python3", "main.py", "flomps", "TLEs/SatCount8.tle",
            "--timesteps", "30",
            "--model-type", config["model"],
            "--data-set", config["dataset"],
            "--num-rounds", "3",
            "--num-clients", "4"
        ]
        
        print(f"🔧 Executing command: {' '.join(cmd)}")
        print("\n⏳ Running test...")
        
        # Record start time
        start_time = datetime.now()
        
        try:
            # Execute command
            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            # Record end time
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Record results
            test_result = {
                "config": config,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration": duration,
                "return_code": result.returncode,
                "stdout": result.stdout[-1000:] if result.stdout else "",  # Keep only last 1000 chars
                "stderr": result.stderr[-1000:] if result.stderr else "",  # Keep only last 1000 chars
                "success": result.returncode == 0
            }
            
            if result.returncode == 0:
                print(f"✅ Test successful! Duration: {duration}")
            else:
                print(f"❌ Test failed! Return code: {result.returncode}")
                print(f"Error output: {result.stderr[-500:] if result.stderr else 'None'}")
            
        except subprocess.TimeoutExpired:
            duration = (datetime.now() - start_time).total_seconds()
            test_result = {
                "config": config,
                "start_time": start_time.isoformat(),
                "duration": duration,
                "success": False,
                "error": "Test timeout (5 minutes)"
            }
            print("⏰ Test timeout (5 minutes)")
        except Exception as e:
            test_result = {
                "config": config,
                "start_time": start_time.isoformat(),
                "success": False,
                "error": str(e)
            }
            print(f"❌ Execution error: {e}")
        
        results.append(test_result)
        print("-" * 60)
    
    # Save test results
    save_test_results(results)
    
    # Display summary
    display_summary(results)

def save_test_results(results):
    """Save test results to file"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(project_root, "P_45", "test_results", f"automated_model_selection_test_{timestamp}.json")
    
    # Save detailed results
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Test results saved to: {results_file}")

def display_summary(results):
    """Display test summary"""
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r.get('success', False))
    failed_tests = total_tests - successful_tests
    
    print(f"Total tests: {total_tests}")
    print(f"Successful: {successful_tests} ✅")
    print(f"Failed: {failed_tests} ❌")
    print(f"Success rate: {successful_tests/total_tests*100:.1f}%")
    
    print(f"\n📋 Detailed Results:")
    for result in results:
        config = result['config']
        status = "✅" if result.get('success', False) else "❌"
        duration = f"{result.get('duration', 0):.1f}s" if 'duration' in result else "N/A"
        print(f"  {status} {config['name']} - {duration}")

def main():
    """Main function"""
    print("🚀 Starting Automated Model and Dataset Combination Test")
    print("="*60)
    test_model_dataset_combinations()

if __name__ == "__main__":
    main()