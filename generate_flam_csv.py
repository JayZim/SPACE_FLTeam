#!/usr/bin/env python3
"""
FLAM CSV Generator
Main utility to generate FLAM CSV files using SatSim → Algorithm workflow.
Uses direct core classes (bypasses handlers) for better reliability.
Author: stephen zeng
Date: 2025-06-04
Version: 2.0
Python Version: 3.12

Changelog:
- 2025-06-04: Initial creation. 
- 2025-06-04: Added command line argument support for time parameters
"""

import sys
import os
import argparse # for command line arguments
from datetime import datetime, timedelta # for time handling
sys.path.append(os.path.abspath('.'))

from sat_sim.sat_sim import SatSim
from flomps_algorithm.algorithm_core import Algorithm
import json
from skyfield.api import load
import glob

# add path manager
try:
    from utilities.path_manager import get_synth_flams_dir
    use_path_manager = True
except ImportError:
    use_path_manager = False

# main function
def generate_flam_csv(tle_file="TLEs/SatCount4.tle", start_time=None, end_time=None, timestep=1):
    """
    Generate FLAM CSV using SatSim + Algorithm direct approach.

    This uses core classes directly instead of handlers for better reliability.
    The handler-based approach had compatibility issues and was removed.
    """

    print("🚀 Starting FLAM CSV Generation...")
    print(f"📡 Using TLE file: {tle_file}")

    # If no time specified, use current time as start and add 10 minutes as end
    if start_time is None:
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"No start time specified, using current time: {start_time}")
    
    if end_time is None:
        start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        end_dt = start_dt + timedelta(minutes=10)
        end_time = end_dt.strftime("%Y-%m-%d %H:%M:%S")
        print(f"No end time specified, using 10 minutes later: {end_time}")

    # Calculate expected timesteps
    start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    timestep_minutes = timestep
    total_minutes = int((end_dt - start_dt).total_seconds() / 60)
    expected_timesteps = total_minutes // timestep_minutes

    print(f"Expected timesteps: {expected_timesteps}")
    print(f"    Start: {start_dt}")
    print(f"    End: {end_dt}")
    print(f"    Interval: {timestep_minutes} minutes")

    try:
        # Step 1: Run SatSim directly
        print("\n📡 Setting up SatSim...")

        # Convert to Skyfield time objects
        ts = load.timescale()
        start_skyfield = ts.utc(start_dt.year, start_dt.month, start_dt.day,
                               start_dt.hour, start_dt.minute, start_dt.second)
        end_skyfield = ts.utc(end_dt.year, end_dt.month, end_dt.day,
                             end_dt.hour, end_dt.minute, end_dt.second)

        # Create SatSim instance
        sat_sim = SatSim(
            start_time=start_skyfield,
            end_time=end_skyfield,
            timestep=timestep_minutes,
            output_file_type='txt',  # We'll handle CSV ourselves
            gui_enabled=False,
            output_to_file=False  # We don't need SatSim's file output
        )

        # Read TLE data
        print(f"📡 Loading TLE file: {tle_file}")
        tle_data = {}
        with open(tle_file, 'r') as f:
            lines = [line.strip() for line in f.readlines()]

        # Parse TLE data (3LE format)
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
        print(f"✅ Loaded TLE data for {len(tle_data)} satellites")

        # Run SatSim to get adjacency matrices
        print("📡 Running SatSim simulation...")
        adjacency_matrices = sat_sim.run_with_adj_matrix()
        print(f"✅ SatSim completed: {len(adjacency_matrices)} timesteps generated")

        # Step 2: Run Algorithm with SatSim data
        print("\n🧮 Setting up Algorithm...")
        algorithm = Algorithm()

        # Set satellite names from TLE data
        satellite_names = list(tle_data.keys())
        algorithm.set_satellite_names(satellite_names)

        # Set up algorithm with SatSim output
        algorithm.set_adjacency_matrices(adjacency_matrices)

        print(f"🧮 Running algorithm with {len(satellite_names)} satellites...")

        # Run the algorithm
        algorithm.start_algorithm_steps()
        print("✅ Algorithm completed!")

        # Step 3: Check output files
        print("\n📄 Checking output files...")

        # Find the generated CSV file
        if use_path_manager:
            csv_dir = get_synth_flams_dir()
            csv_files = list(csv_dir.glob("*.csv"))
            print(f"CSV directory: {csv_dir}")
        else:
            # use backup path
            csv_dir_str = "synth_FLAMs/"
            if os.path.exists(csv_dir_str):
                csv_files = [os.path.join(csv_dir_str, f) for f in os.listdir(csv_dir_str) if f.endswith('.csv')]
            else:
                csv_files = []
            print(f"CSV directory: {csv_dir_str}")
        
        if csv_files:
            print(f"Found {len(csv_files)} CSV files")
            
            # Check the latest file
            if use_path_manager:
                latest_file = max(csv_files, key=lambda x: x.stat().st_ctime)
                print(f"Latest file: {latest_file.name}")
                
                # Count timesteps in the latest file
                with open(latest_file, 'r') as f:
                    timestep_count = sum(1 for line in f if line.startswith("Timestep:")) # count the number of lines that start with "Timestep:"
                print(f"CSV contains {timestep_count} timesteps")
            else:
                latest_file = max(csv_files, key=lambda x: os.path.getctime(x))
                print(f"Latest file: {os.path.basename(latest_file)}")
                
                # Count timesteps in the latest file
                with open(latest_file, 'r') as f:
                    timestep_count = sum(1 for line in f if line.startswith("Timestep:"))
                print(f"📊 CSV contains {timestep_count} timesteps")
            
            return latest_file
        else:
            print("❌ No CSV file found in synth_FLAMs/")
            return None

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function to generate FLAM CSV with command line argument support"""
    parser = argparse.ArgumentParser(description='Generate FLAM CSV files from TLE data')
    parser.add_argument('tle_file', nargs='?', default='TLEs/SatCount4.tle', 
                       help='TLE file path (default: TLEs/SatCount4.tle)')
    parser.add_argument('--start-time', type=str, 
                       help='Start time in YYYY-MM-DD HH:MM:SS format (default: current time)')
    parser.add_argument('--end-time', type=str, 
                       help='End time in YYYY-MM-DD HH:MM:SS format (default: start + 10 minutes)')
    parser.add_argument('--timestep', type=int, default=1, 
                       help='Timestep in minutes (default: 1)')
    
    args = parser.parse_args()
    
    print("Starting FLAM CSV Generation...")
    print(f"Parameters:")
    print(f"   TLE file: {args.tle_file}")
    if args.start_time:
        print(f"   Start time: {args.start_time}")
    else:
        print(f"   Start time: [Current time]")
    if args.end_time:
        print(f"   End time: {args.end_time}")
    else:
        print(f"   End time: [Start + 10 minutes]")
    print(f"   Timestep: {args.timestep} minutes")

    result = generate_flam_csv(args.tle_file, args.start_time, args.end_time, args.timestep)

    if result:
        filename = result.name if use_path_manager else os.path.basename(result)
        print(f"\n🎉 SUCCESS! FLAM CSV generated: {filename}")
        print(f"📁 Full path: {result}")
        print("\n🔗 This CSV can now be used by FL core!")
    else:
        print("\n💥 FAILED to generate FLAM CSV")

if __name__ == "__main__":
    main()
