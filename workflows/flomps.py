import sys
import os.path
import time
from datetime import datetime, timedelta
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import module_factory

def build_modules(options):
    sat_sim_module = module_factory.create_sat_sim_module()
    sat_sim_module.config.read_options(options["sat_sim"])
        
    algorithm_module = module_factory.create_algorithm_module()
    algorithm_module.config.read_options(options["algorithm"])

    fl_module = module_factory.create_fl_module()
    fl_module.config.read_options(options["federated_learning"])

    return sat_sim_module, algorithm_module, fl_module

def calculate_time_duration(timesteps, custom_duration=None):
    """Calculate end time based on timesteps or custom duration"""
    if custom_duration:
        # Parse custom duration HH:MM:SS
        try:
            time_parts = custom_duration.split(':')
            hours = int(time_parts[0])
            minutes = int(time_parts[1])
            seconds = int(time_parts[2]) if len(time_parts) > 2 else 0
            total_minutes = hours * 60 + minutes + seconds / 60
            return total_minutes
        except (ValueError, IndexError):
            print(f"⚠️ Invalid custom duration format: {custom_duration}, using timesteps instead")
            return timesteps
    else:
        # 1 minute per timestep
        return timesteps

def apply_custom_timesteps(sat_sim_module, options, timesteps=None, custom_duration=None):
    """Apply custom timesteps to satellite simulation module"""
    if timesteps is None and custom_duration is None:
        print("ℹ️ Using default timesteps from options.json")
        return
    
    print(f"🔧 Applying custom parameters:")
    if timesteps:
        print(f"   Timesteps: {timesteps}")
    if custom_duration:
        print(f"   Custom Duration: {custom_duration}")
    
    # Calculate simulation duration in minutes
    duration_minutes = calculate_time_duration(timesteps or 100, custom_duration)
    print(f"   Calculated Duration: {duration_minutes} minutes")
    
    # Get current sat_sim options
    sat_sim_options = options["sat_sim"]
    start_time = sat_sim_options.get('start_time', '2025-01-07 00:00:00')
    
    # Calculate end time
    start_dt = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    end_dt = start_dt + timedelta(minutes=duration_minutes)
    end_time = end_dt.strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"   Start Time: {start_time}")
    print(f"   End Time: {end_time}")
    
    # Update sat_sim options in the options dictionary
    options["sat_sim"]['start_time'] = start_time
    options["sat_sim"]['end_time'] = end_time
    options["sat_sim"]['timestep'] = 1  # Always 1 minute per timestep
    
    # Reload the configuration with updated options
    sat_sim_module.config.config_loaded = False  # Reset the config loaded flag
    sat_sim_module.config.read_options(options["sat_sim"])
    
    print("✅ Custom timesteps applied to satellite simulation")

def run(input_file, options, timesteps=None, custom_duration=None):
    """Run the complete FLOMPS workflow: SatSim → Algorithm → FL"""
    print("🚀 Starting FLOMPS Complete Workflow...")
    
    # Apply custom timesteps to options before creating modules
    if timesteps is not None or custom_duration is not None:
        # Make a copy of options to avoid modifying the original
        import copy
        options = copy.deepcopy(options)
    
    # Create Modules
    sat_sim_module, algorithm_module, fl_module = build_modules(options)
    
    # Apply custom timesteps if provided (after modules are created but before running)
    if timesteps is not None or custom_duration is not None:
        apply_custom_timesteps(sat_sim_module, options, timesteps, custom_duration)
    
    # Step 1: SatSim - Satellite Simulation
    print("\n📡 Step 1: Running SatSim (Satellite Simulation)...")
    sat_sim_module.handler.parse_file(input_file)
    sat_sim_module.handler.run_module()
    sat_sim_result = sat_sim_module.output.get_result()
    
    # Handle SatSimResult properly
    if hasattr(sat_sim_result, 'matrices'):
        matrices = sat_sim_result.matrices
        print(f"✅ SatSim completed: Generated {len(matrices) if matrices else 0} adjacency matrices")
    else:
        matrices = sat_sim_result
        print(f"✅ SatSim completed: Generated {len(matrices) if matrices else 0} adjacency matrices")
    
    # Step 2: Algorithm - Algorithm Processing
    print("\n🧮 Step 2: Running Algorithm (FLOMPS Algorithm)...")
    algorithm_module.handler.parse_data(matrices)
    algorithm_module.handler.run_module()
    flam = algorithm_module.output.get_result()
    print(f"✅ Algorithm completed: Generated FLAM data")
    
    # Ensure algorithm output is written to file
    print("\n📁 Step 2.5: Writing Algorithm Output to Files...")
    algorithm_output_data = algorithm_module.handler.algorithm.get_algorithm_output()
    if algorithm_output_data:
        algorithm_module.output.write_to_file(algorithm_output_data)
        print("✅ FLAM files written to disk")
    else:
        print("⚠️ No algorithm output data to write")
    
    # Give file system a moment to ensure file writing is complete
    time.sleep(1)
    
    # Step 3: FL - Federated Learning
    print("\n🤖 Step 3: Running Federated Learning...")
    
    # Approach 1: Use data passing approach (handler data passing)
    if flam is not None:
        print("[INFO] Using data pipeline approach (handler data passing)...")
        fl_module.handler.parse_data(flam)
        fl_module.handler.run_module()
    else:
        # Approach 2: Let FL handler auto-detect the latest FLAM file
        print("[INFO] Using auto-detection approach (latest FLAM file)...")
        fl_module.handler.run_module()  # This will auto-detect the latest FLAM file
    
    print("\n🎉 FLOMPS Complete Workflow Finished!")
    print("✅ SatSim → Algorithm → FL pipeline completed successfully")
    
    # Display final summary
    if timesteps or custom_duration:
        print(f"\n📊 Simulation Summary:")
        if timesteps:
            print(f"   Custom Timesteps: {timesteps}")
        if custom_duration:
            print(f"   Custom Duration: {custom_duration}")
        print(f"   FLAM Files: Available in synth_FLAMs/ directory")
        print(f"   FL Core: Can auto-detect and use latest FLAM files")
