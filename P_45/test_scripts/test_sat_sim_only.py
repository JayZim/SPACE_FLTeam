#!/usr/bin/env python3
"""
Satellite Simulation Module Standalone Test Script
Test standalone functionality of SatSim module
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

def test_sat_sim_module(tle_file, test_name="SatSim Standalone Test"):
    """
    Test Satellite Simulation module
    
    Args:
        tle_file (str): TLE file path
        test_name (str): Test name
    
    Returns:
        dict: Test results
    """
    print(f"\n{'='*60}")
    print(f"🚀 Starting {test_name}")
    print(f"📁 TLE file: {tle_file}")
    print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Check if TLE file exists
    if not os.path.exists(tle_file):
        print(f"❌ Error: TLE file does not exist: {tle_file}")
        return {
            "test_name": test_name,
            "tle_file": tle_file,
            "status": "failed",
            "error": "TLE file does not exist"
        }
    
    # Record test results
    test_result = {
        "test_name": test_name,
        "tle_file": tle_file,
        "start_time": datetime.now().isoformat(),
        "steps": {}
    }
    
    try:
        # Import Satellite Simulation module
        from sat_sim.sat_sim import SatSim
        from skyfield.api import load
        
        print("\n📡 Running Satellite Simulation (SatSim)")
        
        # Record start time
        start_time = time.time()
        
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
        print("📖 Reading TLE data...")
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
        
        print(f"✅ Read TLE data for {len(tle_data)} satellites")
        
        # Set TLE data
        sat_sim.set_tle_data(tle_data)
        
        # Run simulation
        print("🔄 Running satellite orbit simulation...")
        matrices = sat_sim.run_with_adj_matrix()
        satellite_names = list(tle_data.keys())
        
        # Calculate test results
        duration = time.time() - start_time
        test_result["steps"]["sat_sim"] = {
            "status": "success",
            "duration": duration,
            "satellites": len(satellite_names),
            "timesteps": len(matrices),
            "satellite_names": satellite_names
        }
        
        test_result["end_time"] = datetime.now().isoformat()
        test_result["total_duration"] = duration
        test_result["overall_status"] = "success"
        
        print(f"✅ Satellite Simulation completed!")
        print(f"⏱️  Duration: {duration:.2f} seconds")
        print(f"📊 Satellite count: {len(satellite_names)}")
        print(f"📊 Timestep count: {len(matrices)}")
        print(f"📊 Satellite names: {', '.join(satellite_names)}")
        
        # Check adjacency matrices
        if matrices and len(matrices) > 0:
            print(f"📊 Adjacency matrix dimensions: {matrices[0][1].shape}")
            print(f"📊 Matrix data type: {type(matrices[0][1])}")
            
            # Check matrix content
            first_matrix = matrices[0][1]
            print(f"📊 First matrix statistics:")
            print(f"   - Min value: {first_matrix.min():.4f}")
            print(f"   - Max value: {first_matrix.max():.4f}")
            print(f"   - Mean value: {first_matrix.mean():.4f}")
            print(f"   - Non-zero elements: {(first_matrix != 0).sum()}")
        
        return test_result
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        test_result["end_time"] = datetime.now().isoformat()
        test_result["overall_status"] = "failed"
        test_result["error"] = str(e)
        return test_result

def main():
    """Main function"""
    print("🧪 SPACE Project Satellite Simulation Module Standalone Test")
    print("=" * 60)
    
    # Test configuration
    tle_files = [
        "TLEs/SatCount1.tle",
        "TLEs/SatCount3.tle",
        "TLEs/SatCount4.tle",
        "TLEs/SatCount8.tle"
    ]
    
    results = []
    success_count = 0
    total_tests = len(tle_files)
    
    for i, tle_file in enumerate(tle_files, 1):
        print(f"\n📋 Test {i}/{total_tests}")
        test_name = f"SatSim Test - {os.path.basename(tle_file)}"
        result = test_sat_sim_module(tle_file, test_name)
        results.append(result)
        
        if result["overall_status"] == "success":
            success_count += 1
    
    # Generate test report
    print(f"\n{'='*60}")
    print(f"📊 Test Summary")
    print(f"✅ Success: {success_count}/{total_tests}")
    print(f"❌ Failed: {total_tests - success_count}/{total_tests}")
    print(f"📈 Success Rate: {(success_count/total_tests)*100:.1f}%")
    
    # Detailed results table
    print(f"\n📋 Detailed Results:")
    print(f"{'Test File':<20} {'Satellites':<8} {'Timesteps':<8} {'Status':<8} {'Duration(s)':<10}")
    print("-" * 60)
    
    for result in results:
        satellites = result.get("steps", {}).get("sat_sim", {}).get("satellites", "N/A")
        timesteps = result.get("steps", {}).get("sat_sim", {}).get("timesteps", "N/A")
        status = "✅" if result["overall_status"] == "success" else "❌"
        duration = f"{result.get('total_duration', 0):.1f}"
        test_file = os.path.basename(result["tle_file"])
        print(f"{test_file:<20} {satellites:<8} {timesteps:<8} {status:<8} {duration:<10}")
    
    # Save test results
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "test_suite": "Satellite Simulation Module Standalone Test",
        "total_tests": total_tests,
        "successful_tests": success_count,
        "failed_tests": total_tests - success_count,
        "success_rate": (success_count/total_tests)*100,
        "results": results
    }
    
    results_file = project_root / "P_45" / "test_results" / f"sat_sim_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Test results saved to: {results_file}")

if __name__ == "__main__":
    main()