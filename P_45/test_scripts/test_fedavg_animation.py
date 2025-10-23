#!/usr/bin/env python3
"""
FedAvg model
Test the animation generated function and availability of fedavg mode
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime

# 添加項目根目錄到路徑
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

def test_fedavg_animation():
    """test the animation generated function and availability of fedavg mode""" 
    print("🎬 test the animation generated function and availability of fedavg mode")
    print("=" * 50)
    
    # 1. modify options.json to enable fedavg mode
    print("📝 step 1: enable fedavg mode")
    try:
        with open('options.json', 'r') as f:
            options = json.load(f)
        
        # enable fedavg mode
        options['algorithm']['fedavg_mode'] = True
        
        with open('options.json', 'w') as f:
            json.dump(options, f, indent=4)
        
        print("✅ fedavg mode enabled")
    except Exception as e:
        print(f"❌ enable fedavg mode failed: {e}")
        return False
    
    # 2. run the full workflow in fedavg mode
    print("\n🚀 step 2: run the full workflow in fedavg mode")
    try:
        cmd = [
            "python3", "main.py", "flomps", "TLEs/SatCount8.tle",
            "--timesteps", "15",
            "--model-type", "SimpleCNN",
            "--data-set", "MNIST",
            "--num-rounds", "3",
            "--num-clients", "4"
        ]
        
        print(f"🔧 executed command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root, timeout=300)
        
        if result.returncode == 0:
            print("✅ FedAvg wrokflow executed successfully")
        else:
            print(f"❌ FedAvg wrokflow executed failed (return code: {result.returncode})")
            print(f"error output: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ wrokflow execution timeout")
        return False
    except Exception as e:
        print(f"❌ execution failed: {e}")
        return False
    
    # 3. check the generated animation files
    print("\n📁 step 3: check the generated animation files")
    
    # find the latest output directory
    fl_output_root = os.path.join(project_root, "federated_learning", "results_from_output")
    if not os.path.exists(fl_output_root):
        print("❌ FL output directory does not exist")
        return False
    
    subdirs = [d for d in os.listdir(fl_output_root) if os.path.isdir(os.path.join(fl_output_root, d))]
    if not subdirs:
        print("❌ no output directory found")
        return False
    
    # sort by directory name（timestamp），then check which has files
    subdirs.sort(reverse=True)  # latest timestamp comes first
    latest_dir = None
    for subdir in subdirs:
        subdir_path = os.path.join(fl_output_root, subdir)
        if os.listdir(subdir_path):  # if the directory is not empty
            latest_dir = subdir
            break
    
    if not latest_dir:
        print("❌ no output directory with files found")
        return False
    
    latest_path = os.path.join(fl_output_root, latest_dir)
    
    print(f"📂 latest output directory: {latest_path}")
    
    # check the FedAvg animation files
    expected_files = [
        "accuracy_fedavg.gif",
        "participation_fedavg.gif",
        "dashboard.html",
        "fl_metrics_*.json",
        "fl_model_*.pt",
        "fl_results_*.log"
    ]
    
    found_files = []
    missing_files = []
    
    for filename in expected_files:
        if '*' in filename:
            # handle wildcard matching
            import glob
            matches = glob.glob(os.path.join(latest_path, filename))
            if matches:
                for match in matches:
                    file_size = os.path.getsize(match)
                    found_files.append((os.path.basename(match), file_size))
                    print(f"✅ {os.path.basename(match)} ({file_size} bytes)")
            else:
                missing_files.append(filename)
                print(f"❌ {filename} not found")
        else:
            # handle exact match
            file_path = os.path.join(latest_path, filename)
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                found_files.append((filename, file_size))
                print(f"✅ {filename} ({file_size} bytes)")
            else:
                missing_files.append(filename)
                print(f"❌ {filename} not found")
    
    # 4. restore FLOMPS mode
    print("\n🔄 step 4: restore FLOMPS mode")
    try:
        with open('options.json', 'r') as f:
            options = json.load(f)
        
        # restore FLOMPS mode
        options['algorithm']['fedavg_mode'] = False
        
        with open('options.json', 'w') as f:
            json.dump(options, f, indent=4)
        
        print("✅ FLOMPS mode restored")
    except Exception as e:
        print(f"⚠️ restore FLOMPS mode failed: {e}")
    
    # 5. generate test report
    print("\n📋 step 5: generate test report")
    test_result = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "test_type": "fedavg_animation",
        "status": "success" if not missing_files else "failure",
        "found_files": found_files,
        "missing_files": missing_files,
        "output_directory": latest_path
    }
    
    report_dir = os.path.join(project_root, "P_45", "test_results")
    os.makedirs(report_dir, exist_ok=True)
    report_file = os.path.join(report_dir, f"fedavg_animation_test_{test_result['timestamp']}.json")
    
    with open(report_file, "w") as f:
        json.dump(test_result, f, indent=4)
    
    print(f"📄 test report saved: {report_file}")
    
    # 6. summarize: summary the test
    print("\n🎯 test summary:")
    if not missing_files:
        print("✅ FedAvg animation test passed")
        print(f"   - found {len(found_files)} files")
        for filename, size in found_files:
            print(f"     - {filename}: {size} bytes")
        return True
    else:
        print("❌ FedAvg animation test failed")
        print(f"   - missing {len(missing_files)} files")
        for filename in missing_files:
            print(f"     - {filename}")
        return False

def main():
    """main function"""
    print("🎬 test FedAvg animation")     
    print("=" * 60)
    
    success = test_fedavg_animation()
    
    if success:
        print("only some file part test passed!")
        sys.exit(0)
    else:
        print("\n💥 test failed! Please check the error")    
        sys.exit(1)

if __name__ == "__main__":
    main()
