#!/usr/bin/env python3
"""
Test script to verify the interactive version works correctly
by simulating user input through stdin redirection
"""

import subprocess
import sys
import os

def test_interactive_version():
    """Test the interactive version with simulated input"""
    
    # Test input that simulates user responses
    test_input = """6
100
1
5
6
SimpleCNN
MNIST
y
"""
    
    script_path = "/Users/stephentsang/Documents/GitHub/SPACE_FLTeam/P_45/test_scripts/main_run_6sat_100t_interactive.py"
    
    print("🧪 Testing Interactive Version with Simulated Input")
    print("=" * 60)
    print("Simulated input:")
    print(test_input)
    print("=" * 60)
    
    try:
        # Run the interactive script with simulated input
        result = subprocess.run(
            [sys.executable, script_path],
            input=test_input,
            text=True,
            capture_output=True,
            timeout=300  # 5 minute timeout
        )
        
        print("📊 Test Results:")
        print(f"Exit code: {result.returncode}")
        print(f"STDOUT length: {len(result.stdout)} characters")
        print(f"STDERR length: {len(result.stderr)} characters")
        
        if result.returncode == 0:
            print("✅ Interactive version test PASSED")
            
            # Check if output files were generated
            output_dir = "/Users/stephentsang/Documents/GitHub/SPACE_FLTeam/federated_learning/results_from_output"
            if os.path.exists(output_dir):
                latest_dir = max([d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))], 
                               key=lambda x: os.path.getctime(os.path.join(output_dir, x)))
                latest_path = os.path.join(output_dir, latest_dir)
                
                gif_files = [f for f in os.listdir(latest_path) if f.endswith('.gif')]
                print(f"📁 Generated {len(gif_files)} GIF files in {latest_path}")
                for gif in gif_files:
                    print(f"   - {gif}")
            else:
                print("⚠️  No output directory found")
        else:
            print("❌ Interactive version test FAILED")
            print("STDERR:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out after 5 minutes")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")

if __name__ == "__main__":
    test_interactive_version()
