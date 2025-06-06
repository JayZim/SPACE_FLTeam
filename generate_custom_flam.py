#!/usr/bin/env python3
"""
Custom FLAM Generator - Generate FLAM files with custom parameters
Author: stephen zeng
Date: 2025-06-06
Version: 1.0

Usage:
    python generate_custom_flam.py --satellites 4 --timesteps 100
    python generate_custom_flam.py --satellites 5 --timesteps 50
    python generate_custom_flam.py --satellites 8 --timesteps 200
"""

import argparse
import os
import sys
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generate_flam_csv import generate_flam_csv

def get_tle_file(num_satellites):
    """Get the appropriate TLE file for the number of satellites"""
    tle_files = {
        1: "TLEs/SatCount1.tle",
        3: "TLEs/SatCount3.tle", 
        4: "TLEs/SatCount4.tle",
        8: "TLEs/SatCount8.tle",
        40: "TLEs/SatCount40.tle"
    }
    
    if num_satellites in tle_files:
        return tle_files[num_satellites]
    else:
        # Find the closest available TLE file
        available = sorted(tle_files.keys())
        closest = min(available, key=lambda x: abs(x - num_satellites))
        print(f"⚠️ No TLE file for {num_satellites} satellites, using {closest} satellites instead")
        return tle_files[closest]

def generate_custom_flam(satellites, timesteps, start_time=None):
    """Generate custom FLAM file with specified parameters"""
    print("🚀 Starting Custom FLAM Generation...")
    print(f"📊 Parameters:")
    print(f"   Satellites: {satellites}")
    print(f"   Timesteps: {timesteps}")
    
    # Get appropriate TLE file
    tle_file = get_tle_file(satellites)
    print(f"   TLE file: {tle_file}")
    
    # Check if TLE file exists
    if not os.path.exists(tle_file):
        print(f"❌ TLE file not found: {tle_file}")
        return None
    
    # Calculate time span
    if start_time is None:
        start_time = "2025-06-04 00:00:00"
    
    start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    # timesteps * 1 minute per timestep
    end_dt = start_dt + timedelta(minutes=timesteps)
    end_time = end_dt.strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"⏱️ Time span:")
    print(f"   Start: {start_time}")
    print(f"   End: {end_time}")
    print(f"   Duration: {timesteps} minutes (1 minute per timestep)")
    
    # Generate the FLAM file
    result = generate_flam_csv(
        tle_file=tle_file,
        start_time=start_time,
        end_time=end_time,
        timestep=1
    )
    
    if result:
        print(f"\n🎉 SUCCESS! Generated custom FLAM file")
        print(f"📁 File: {os.path.basename(result)}")
        
        # Verify timestep count
        with open(result, 'r') as f:
            actual_timesteps = sum(1 for line in f if line.startswith("Timestep:"))
        
        print(f"✅ Verified: {actual_timesteps} timesteps generated")
        
        if actual_timesteps == timesteps:
            print("🎯 Perfect! Exact timestep count achieved")
        else:
            print(f"ℹ️ Note: {actual_timesteps} timesteps generated (requested: {timesteps})")
            
        return result
    else:
        print("❌ Failed to generate FLAM file")
        return None

def test_custom_flam(result_file, satellites):
    """Test the generated FLAM file with FL Core"""
    print(f"\n🧪 Testing generated FLAM file...")
    
    from federated_learning.fl_core import FederatedLearning
    
    fl_core = FederatedLearning()
    fl_core.set_num_clients(satellites)
    fl_core.set_num_rounds(2)  # Just 2 rounds for testing
    
    try:
        # Load the specific file
        schedule = fl_core.load_flam_schedule(result_file)
        print(f"✅ FL Core can load the file: {len(schedule)} timesteps")
        
        # Show some sample phases
        sample_phases = [s['phase'] for s in schedule[:10]]
        print(f"📋 Sample phases: {sample_phases}")
        
        print("🎉 FLAM file is ready for use!")
        return True
    except Exception as e:
        print(f"❌ Error testing FLAM file: {e}")
        return False

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(
        description='Generate custom FLAM files for SPACE FLOMPS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --satellites 4 --timesteps 100
  %(prog)s --satellites 5 --timesteps 50 
  %(prog)s --satellites 8 --timesteps 200
  %(prog)s -s 40 -t 500
        """
    )
    
    parser.add_argument('-s', '--satellites', type=int, default=4,
                       help='Number of satellites (default: 4)')
    parser.add_argument('-t', '--timesteps', type=int, default=100, 
                       help='Number of timesteps (default: 100)')
    parser.add_argument('--start-time', type=str,
                       help='Start time in YYYY-MM-DD HH:MM:SS format (default: 2025-06-04 00:00:00)')
    parser.add_argument('--test', action='store_true',
                       help='Test the generated FLAM file with FL Core')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 SPACE FLOMPS - Custom FLAM Generator")
    print("="*60)
    
    # Generate custom FLAM file
    result = generate_custom_flam(args.satellites, args.timesteps, args.start_time)
    
    if result:
        if args.test:
            test_success = test_custom_flam(result, args.satellites)
        else:
            test_success = True
            
        print("\n" + "="*60)
        print("📋 Summary")
        print("="*60)
        print("✅ FLAM Generation: Success")
        if args.test:
            print(f"✅ FL Core Test: {'Success' if test_success else 'Failed'}")
        print(f"📁 Generated file: {os.path.basename(result)}")
        print(f"🎯 You now have a {args.timesteps}-timestep FLAM file ready for use!")
        print("\n🔧 Usage options:")
        print("   - FL Core auto-detection: FL Core will automatically use this file")
        print("   - main.py workflow: python main.py flomps [tle_file]")
        print("   - Direct file path: specify the file path manually")
    else:
        print("\n❌ FLAM generation failed")

if __name__ == "__main__":
    main() 