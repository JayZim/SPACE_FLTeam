import sys
import os.path
from pathlib import Path
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

def run(input_file, options, timesteps=None, custom_duration=None, flam_file: str | None = None, interactive_select_flam: bool = False, interactive_fl_output: bool = False):
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
    
    # Step 1: SatSim - Satellite Simulation (or reuse existing SatSim output)
    matrices = None
    try:
        # Detect if input_file is a SatSim adjacency output (starts with 'Number of satellites:')
        if isinstance(input_file, str) and input_file.endswith('.txt') and os.path.isfile(input_file):
            with open(input_file, 'r') as f:
                first_line = f.readline().strip()
            if first_line.startswith('Number of satellites:'):
                print("\n📡 Step 1: Reusing existing SatSim output (adjacency txt)...")
                # Let algorithm handler read the adjacency file directly later
                pass
            else:
                print("\n📡 Step 1: Running SatSim (Satellite Simulation)...")
                sat_sim_module.handler.parse_file(input_file)
                sat_sim_module.handler.run_module()
                sat_sim_result = sat_sim_module.output.get_result()
                if hasattr(sat_sim_result, 'matrices'):
                    matrices = sat_sim_result.matrices
                else:
                    matrices = sat_sim_result
                print(f"✅ SatSim completed: Generated {len(matrices) if matrices else 0} adjacency matrices")
        else:
            print("\n📡 Step 1: Running SatSim (Satellite Simulation)...")
            sat_sim_module.handler.parse_file(input_file)
            sat_sim_module.handler.run_module()
            sat_sim_result = sat_sim_module.output.get_result()
            if hasattr(sat_sim_result, 'matrices'):
                matrices = sat_sim_result.matrices
            else:
                matrices = sat_sim_result
            print(f"✅ SatSim completed: Generated {len(matrices) if matrices else 0} adjacency matrices")
    except Exception as e:
        print(f"⚠️ SatSim step encountered an error: {e}. Proceeding if existing adjacency provided.")
    
    # Step 2: Algorithm - Algorithm Processing
    print("\n🧮 Step 2: Running Algorithm (FLOMPS Algorithm)...")
    if matrices is None and isinstance(input_file, str) and input_file.endswith('.txt') and os.path.isfile(input_file):
        # Parse adjacency txt directly
        algorithm_module.handler.parse_file(input_file)
    else:
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
    
    # Optionally allow explicit FLAM selection or path override
    chosen_flam_path = None
    try:
        if interactive_select_flam:
            # Build FLAM list from synth_FLAMs
            try:
                # Prefer utilities.path_manager if available via FL handler
                from utilities.path_manager import get_synth_flams_dir  # type: ignore
                flam_dir = get_synth_flams_dir()
                flam_candidates = sorted(list(flam_dir.glob("flam_*.csv")), key=lambda p: p.stat().st_ctime)
            except Exception:
                flam_dir = (Path(__file__).parent.parent / "synth_FLAMs").resolve()
                flam_candidates = sorted(list(flam_dir.glob("flam_*.csv")), key=lambda p: p.stat().st_ctime)

            if not flam_candidates:
                print(f"[INFO] No FLAM files found in {flam_dir}, falling back to auto-detect.")
            else:
                print("\nSelect a FLAM file to use for FL training:")
                for i, p in enumerate(flam_candidates, 1):
                    print(f"  {i}) {p.name}")
                idx = None
                while idx is None:
                    s = input(f"Enter number (1-{len(flam_candidates)}): ").strip()
                    if s.isdigit() and 1 <= int(s) <= len(flam_candidates):
                        idx = int(s) - 1
                chosen_flam_path = str(flam_candidates[idx])
        elif flam_file:
            chosen_flam_path = flam_file
    except KeyboardInterrupt:
        print("\n[INFO] Selection cancelled. Falling back to auto-detect.")
        chosen_flam_path = None

    if chosen_flam_path:
        # Use the full FL core run path to enable interactive model/dataset selection like fl_core.py
        try:
            print(f"[INFO] Using explicitly selected FLAM: {os.path.basename(chosen_flam_path)}")
            # Call the FederatedLearning.run to leverage interactive ModelSelection path
            fl_module.handler.federated_learning.run(
                flam_path=chosen_flam_path,
                interactive_mode=True,
            )
        except Exception as e:
            print(f"[WARN] FL core run failed with selected FLAM ({e}), falling back to handler auto-detect")
            fl_module.handler.flam = None
            fl_module.handler.run_module()
    else:
        # No explicit FLAM selected; let the handler auto-detect latest FLAM
        fl_module.handler.flam = None
        fl_module.handler.run_module()
    
    # Write FL results to disk similar to fl_core standalone
    try:
        from federated_learning.fl_output import FLOutput
        from federated_learning.fl_visualization import FLVisualization
        from torchvision import datasets, transforms

        fl_core = fl_module.handler.federated_learning

        # Create timestamped run directory under federated_learning/results_from_output
        results_root = (Path(__file__).parent.parent / "federated_learning" / "results_from_output").resolve()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = results_root / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        # Build test dataset based on current dataset (align with training transforms)
        data_set = getattr(fl_core, "current_dataset", "MNIST")
        if data_set == "MNIST":
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                # MNIST 是灰度，重复到 3 通道以匹配 RGB 模型
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            test_dataset = datasets.MNIST(
                root=str((Path(__file__).parent.parent / "federated_learning" / "data" / "MNIST").resolve()),
                train=False,
                download=True,
                transform=transform
            )
        elif data_set == "CIFAR10":
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            test_dataset = datasets.CIFAR10(
                root=str((Path(__file__).parent.parent / "federated_learning" / "data" / "CIFAR10").resolve()),
                train=False,
                download=True,
                transform=transform
            )
        elif data_set == "EuroSAT":
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.3443, 0.3804, 0.4086], std=[0.1814, 0.1535, 0.1311])
            ])
            test_dataset = datasets.EuroSAT(
                root=str((Path(__file__).parent.parent / "federated_learning" / "data" / "EuroSAT").resolve()),
                download=True,
                transform=transform
            )
        else:
            # Fallback to MNIST (aligned to RGB 64x64)
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            test_dataset = datasets.MNIST(
                root=str((Path(__file__).parent.parent / "federated_learning" / "data" / "MNIST").resolve()),
                train=False,
                download=True,
                transform=transform
            )

        output = FLOutput(test_dataset=test_dataset)
        try:
            output.evaluate_model(fl_core.global_model, fl_core.total_training_time)
        except Exception as eval_err:
            print(f"[WARN] Evaluation failed: {eval_err}. Proceeding to save artifacts anyway.")

        # Add metrics captured during run
        for metric_name, value in fl_core.get_round_metrics().items():
            output.add_metric(metric_name, value)
        output.add_metric("round_accuracies", getattr(fl_core, "round_accuracies", []))
        output.add_metric("participation_log", getattr(fl_core, "participation_log", []))

        # Save artifacts
        log_file = run_dir / f"results_{timestamp}.log"
        metrics_file = run_dir / f"metrics_{timestamp}.json"
        model_file = run_dir / f"model_{timestamp}.pt"

        try:
            output.log_result(str(log_file))
        except Exception as e_log:
            print(f"[WARN] Logging failed: {e_log}")
        try:
            output.write_to_file(str(metrics_file), format="json")
        except Exception as e_json:
            print(f"[WARN] Writing metrics failed: {e_json}")
        try:
            output.save_model(str(model_file))
        except Exception as e_model:
            print(f"[WARN] Saving model failed: {e_model}")

        # Generate charts
        try:
            viz = FLVisualization(results_dir=str(run_dir))
            viz.visualize_from_json(str(metrics_file))
        except Exception as viz_err:
            print(f"[WARN] Visualization failed: {viz_err}")

        # Generate GIFs either interactively or automatically based on flag
        if interactive_fl_output:
            try:
                choice = input("\nGenerate animations for this run? (1 = Yes, 2 = No) : ").strip()
            except KeyboardInterrupt:
                choice = "2"
            
            if choice == "1":
                print("Generating animations...")
                try:
                    acc_gif = run_dir / "accuracy_progress.gif"
                    part_gif = run_dir / "client_participation.gif"
                    FLOutput.animate_accuracy_progress(str(metrics_file), save_path=str(acc_gif))
                    FLOutput.animate_client_participation(str(metrics_file), save_path=str(part_gif))
                    print(f"Animations saved to {run_dir}")
                except Exception as gif_err:
                    print(f"[WARN] GIF generation failed: {gif_err}")
        else:
            # Auto-generate GIFs
            try:
                acc_gif = run_dir / "accuracy_progress.gif"
                part_gif = run_dir / "client_participation.gif"
                FLOutput.animate_accuracy_progress(str(metrics_file), save_path=str(acc_gif))
                FLOutput.animate_client_participation(str(metrics_file), save_path=str(part_gif))
                print(f"[INFO] Animations auto-generated and saved to {run_dir}")
            except Exception as gif_err:
                print(f"[WARN] GIF generation failed: {gif_err}")

        print("\n=== Outputs ===")
        print(f"Results dir: {run_dir}")
        
        # Interactive dashboard creation option if enabled
        if interactive_fl_output:
            try:
                dash_choice = input("\nCreate comparison dashboard now? (1 = Yes, 2 = No) : ").strip()
            except KeyboardInterrupt:
                dash_choice = "2"
            
            if dash_choice == "1":
                try:
                    from federated_learning.dashboard_compare import run_dashboard_creator
                    run_dashboard_creator()
                except Exception as dash_err:
                    print(f"[WARN] Dashboard creation failed: {dash_err}")
    except Exception as e:
        print(f"[WARN] Failed to write FL results: {e}")

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
