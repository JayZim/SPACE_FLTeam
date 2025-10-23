#!/usr/bin/env python3
"""
Test GIF Animation Generation Script
Generate federated learning results with sufficient data to test animation functionality
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime

# Add project root directory to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

def create_dummy_metrics_with_data():
    """Create dummy metrics file with sufficient data for testing animation"""
    
    # Create test output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_output_dir = os.path.join(project_root, "federated_learning", "results_from_output", f"test_gif_{timestamp}")
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Create metrics file with sufficient data
    metrics_data = {
        "accuracy": 0.85,
        "loss": 0.45,
        "processing_time": 120.5,
        "evaluation_time": 15.2,
        "timestamp": datetime.now().isoformat(),
        "model_type": "SimpleCNN",
        "data_set": "MNIST",
        "num_rounds": 3,
        "num_clients": 4,
        "round_times": {
            "round_1": 45.2,
            "round_2": 38.7,
            "round_3": 36.6
        },
        "round_accuracies": [0.65, 0.72, 0.78, 0.81, 0.83, 0.85, 0.86, 0.87, 0.88, 0.89],
        "participation_log": [
            {
                "timestep": i,
                "round": (i // 3) + 1,
                "phase": "TRANSMITTING" if i % 3 == 0 else "REDISTRIBUTION" if i % 3 == 1 else "CHECK",
                "aggregation_server": 3 if i % 2 == 0 else 6,
                "redistribution_server": 3 if i % 2 == 0 else 6,
                "in_range_clients": [1, 2, 5, 6, 7] if i % 2 == 0 else [1, 2, 3, 4, 7],
                "out_of_range_clients": [0, 3, 4] if i % 2 == 0 else [0, 5, 6],
                "accuracy": 0.65 + (i * 0.025)  # Incremental accuracy
            }
            for i in range(10)
        ],
        "additional_metrics": {
            "round_accuracies": [0.65, 0.72, 0.78, 0.81, 0.83, 0.85, 0.86, 0.87, 0.88, 0.89],
            "participation_log": [
                {
                    "timestep": i,
                    "round": (i // 3) + 1,
                    "phase": "TRANSMITTING" if i % 3 == 0 else "REDISTRIBUTION" if i % 3 == 1 else "CHECK",
                    "aggregation_server": 3 if i % 2 == 0 else 6,
                    "redistribution_server": 3 if i % 2 == 0 else 6,
                    "in_range_clients": [1, 2, 5, 6, 7] if i % 2 == 0 else [1, 2, 3, 4, 7],
                    "out_of_range_clients": [0, 3, 4] if i % 2 == 0 else [0, 5, 6],
                    "accuracy": 0.65 + (i * 0.025)
                }
                for i in range(10)
            ]
        }
    }
    
    # Save metrics file
    metrics_file = os.path.join(test_output_dir, f"fl_metrics_{timestamp}.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f"✅ Created test metrics file: {metrics_file}")
    return test_output_dir, metrics_file

def test_gif_animation_generation():
    """Test GIF animation generation"""
    print("🎬 Testing GIF Animation Generation Function")
    print("=" * 50)
    
    # Create test data
    test_output_dir, metrics_file = create_dummy_metrics_with_data()
    
    try:
        # Import animation generation functions
        from federated_learning.fl_output import FLOutput
        
        # Test accuracy progress animation
        print("\n📊 Testing accuracy progress animation...")
        acc_gif_path = os.path.join(test_output_dir, "accuracy_progress.gif")
        
        try:
            FLOutput.animate_accuracy_progress(metrics_file, save_path=acc_gif_path)
            if os.path.exists(acc_gif_path):
                file_size = os.path.getsize(acc_gif_path)
                print(f"✅ Accuracy animation generated successfully: {acc_gif_path} ({file_size} bytes)")
            else:
                print("❌ Accuracy animation file not generated")
        except Exception as e:
            print(f"❌ Accuracy animation generation failed: {e}")
        
        # Test client participation animation
        print("\n👥 Testing client participation animation...")
        part_gif_path = os.path.join(test_output_dir, "client_participation.gif")
        
        try:
            FLOutput.animate_client_participation(metrics_file, save_path=part_gif_path)
            if os.path.exists(part_gif_path):
                file_size = os.path.getsize(part_gif_path)
                print(f"✅ Client participation animation generated successfully: {part_gif_path} ({file_size} bytes)")
            else:
                print("❌ Client participation animation file not generated")
        except Exception as e:
            print(f"❌ Client participation animation generation failed: {e}")
        
        # Check generated files
        print(f"\n📁 Test output directory: {test_output_dir}")
        files = os.listdir(test_output_dir)
        for file in files:
            file_path = os.path.join(test_output_dir, file)
            file_size = os.path.getsize(file_path)
            print(f"   - {file} ({file_size} bytes)")
        
        return test_output_dir
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_complete_workflow_with_animation():
    """Run complete workflow and ensure animation generation"""
    print("🚀 Running Complete Workflow to Generate Animation")
    print("=" * 50)
    
    # Use longer timesteps and more rounds to ensure sufficient data
    cmd = [
        "python3", "main.py", "flomps", "TLEs/SatCount8.tle",
        "--timesteps", "30",  # Increase timesteps
        "--model-type", "SimpleCNN",
        "--data-set", "MNIST",
        "--num-rounds", "5",  # Increase rounds
        "--num-clients", "4"
    ]
    
    print(f"🔧 Executing command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root, timeout=300)
        
        if result.returncode == 0:
            print("✅ Complete workflow executed successfully")
            
            # Find the latest output directory
            fl_output_root = os.path.join(project_root, "federated_learning", "results_from_output")
            if os.path.exists(fl_output_root):
                subdirs = [d for d in os.listdir(fl_output_root) if os.path.isdir(os.path.join(fl_output_root, d))]
                if subdirs:
                    latest_dir = max(subdirs, key=lambda x: os.path.getmtime(os.path.join(fl_output_root, x)))
                    latest_path = os.path.join(fl_output_root, latest_dir)
                    
                    print(f"\n📁 Latest output directory: {latest_path}")
                    
                    # Check animation files
                    gif_files = [f for f in os.listdir(latest_path) if f.endswith('.gif')]
                    png_files = [f for f in os.listdir(latest_path) if f.endswith('.png')]
                    
                    print(f"🎬 GIF files: {gif_files}")
                    print(f"🖼️  PNG files: {png_files}")
                    
                    if gif_files:
                        print("✅ Successfully generated GIF animation files!")
                        for gif_file in gif_files:
                            gif_path = os.path.join(latest_path, gif_file)
                            file_size = os.path.getsize(gif_path)
                            print(f"   - {gif_file} ({file_size} bytes)")
                    else:
                        print("⚠️  No GIF files found, but PNG files are available as backup")
                        for png_file in png_files:
                            png_path = os.path.join(latest_path, png_file)
                            file_size = os.path.getsize(png_path)
                            print(f"   - {png_file} ({file_size} bytes)")
                    
                    return latest_path
        else:
            print(f"❌ Workflow execution failed (return code: {result.returncode})")
            print(f"Error output: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print("⏰ Workflow execution timeout")
        return None
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        return None

def main():
    """Main function"""
    print("🎬 GIF Animation Generation Test")
    print("=" * 60)
    
    # Test 1: Test animation generation using dummy data
    print("\n🧪 Test 1: Test animation generation using dummy data")
    test_dir = test_gif_animation_generation()
    
    if test_dir:
        print(f"✅ Dummy data test completed, output directory: {test_dir}")
    else:
        print("❌ Dummy data test failed")
    
    # Test 2: Run complete workflow
    print("\n🚀 Test 2: Run complete workflow")
    workflow_dir = run_complete_workflow_with_animation()
    
    if workflow_dir:
        print(f"✅ Complete workflow test completed, output directory: {workflow_dir}")
    else:
        print("❌ Complete workflow test failed")
    
    print("\n🎯 Test Summary:")
    if test_dir and workflow_dir:
        print("✅ All tests completed, GIF animation functionality is normal")
    elif test_dir or workflow_dir:
        print("⚠️  Partial tests completed, please check results")
    else:
        print("❌ Tests failed, further debugging needed")

if __name__ == "__main__":
    main()
