"""
Filename: algorithm_output.py
Description: Prepares and sends data to Federated Learning module
Initial Creator: Elysia Guglielmo (System Architect),
Contributor: Gagandeep Singh
Date: 2025-06-04
Version: 1.
Python Version:

Changelog:
- 2024-08-07: Initial creation.
- 2024-08-24: Added method to write Federated Learning Adjacency Matrix (FLAM) to a file by Yuganya Perumal
- 2024-08-31: Added method to set result as data frame object to be passed as input FL component by Yuganya Perumal.
- 2024-09-09: Algorithm output returns the FLAM.
- 2024-09-21: Algorithm output re organised to remove redundancy and added timestamp and satellite name to data structure passed to FL.
- 2024-10-14: Output file name changed to FLAM.txt as per client feedback.
- 2025-06-04: Removed hardcoded paths, added universal path management.
- 2025-09-05: Enhanced output generation to support new time-optimized algorithm with improved CSV export format.
- 2025-09-05: Added support for round-based output structure with predictive timestamp information.
- 2025-09-05: Improved file naming conventions to include satellite count and timestep information in CSV exports.

Usage:
Instantiate AlgorithmOutput and assign FLInput.
"""

"""
Description: Prepares and sends data to Federated Learning module - CSV format only
"""
from interfaces.output import Output
import os
import sys
from datetime import datetime
import numpy as npy
import pandas as pds

# add path manager
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from utilities.path_manager import path_manager
except ImportError:
    # if path manager is not found, use backup path
    print("Warning: Path manager not found, using backup path")
    path_manager = None

class AlgorithmOutput(Output):

    def __init__(self):
        self.flam_output = None

        if path_manager:
            # use path manager
            self.output_path = str(path_manager.algorithm_output_dir)
            self.csv_output_path = str(path_manager.synth_flams_dir)
        else:
            # use backup path
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.output_path = os.path.join(script_dir, 'output')
            project_root = os.path.dirname(script_dir)
            self.csv_output_path = os.path.join(project_root, "synth_FLAMs")

            # ensure directories exist
            os.makedirs(self.output_path, exist_ok=True)
            os.makedirs(self.csv_output_path, exist_ok=True)

    def get_result(self):
        return self.flam_output

    def set_flam(self, algo_output):
        self.flam_output = algo_output

    def get_flam(self):
        return self.flam_output

    def process_algorithm_output(self, algorithm_output):
        algo_op_rows = []

        for timestamp, algo_op in algorithm_output.items():
            # Get all the standard fields
            aggregator_flag = algo_op['aggregator_flag'] if algo_op['aggregator_flag'] is not None else 'None'
            fl_am = algo_op['federatedlearning_adjacencymatrix']
            satellite_count = algo_op['satellite_count']
            selected_satellite = algo_op['selected_satellite']
            round_number = algo_op.get('round_number', 1)
            phase = algo_op.get('phase', 'TRANSMITTING')
            round_length = algo_op.get('round_length', 1)
            timestep_in_round = algo_op.get('timestep_in_round', 1)
            aggregator_id = algo_op.get('aggregator_id', 0)

            # Get enhanced server analysis fields
            server_connections_current = algo_op.get('server_connections_current', 0)
            server_connections_cumulative = algo_op.get('server_connections_cumulative', 0)
            target_connections = algo_op.get('target_connections', 0)
            connected_satellites = algo_op.get('connected_satellites', [])
            missing_satellites = algo_op.get('missing_satellites', [])
            target_satellites = algo_op.get('target_satellites', [])
            round_complete = algo_op.get('round_complete', False)

            # Append the row with all the information
            algo_op_rows.append({
                'time_stamp': timestamp,
                'satellite_count': satellite_count,
                'satellite_names': algo_op.get('satellite_names', []),
                'selected_satellite': selected_satellite,
                'aggregator_flag': aggregator_flag,
                'aggregator_id': aggregator_id,
                'federatedlearning_adjacencymatrix': fl_am,
                'round_number': round_number,
                'phase': phase,
                'round_length': round_length,
                'timestep_in_round': timestep_in_round,
                'server_connections_current': server_connections_current,
                'server_connections_cumulative': server_connections_cumulative,
                'target_connections': target_connections,
                'connected_satellites': connected_satellites,
                'missing_satellites': missing_satellites,
                'target_satellites': target_satellites,
                'round_complete': round_complete
            })

        # Convert to DataFrame
        self.set_flam(pds.DataFrame(algo_op_rows))

    def set_result(self, algorithm_output):
        self.process_algorithm_output(algorithm_output)

    def log_result(self):
        print(self.flam_output)

    def write_to_file(self, algorithm_output):
        # Ensure output directory exists
        os.makedirs(self.csv_output_path, exist_ok=True)

        # Write original format - COMMENTED OUT as requested
        # self._write_original_format(algorithm_output)

        # Write CSV format only
        self._write_csv_format(algorithm_output)

    def _write_original_format(self, algorithm_output):
        """Write the original FLAM format to txt file - COMMENTED OUT"""
        # Generate timestamp string
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Generate unique output file name with 'sat_sim_output' and date/time
        output_file_name = f"flam_{timestamp_str}.txt"
        # Construct full output file path
        output_file = os.path.join(self.output_path, output_file_name)
        print(f"Writing output to {output_file}")

        self.process_algorithm_output(algorithm_output)
        with open(output_file, 'w') as file:
            file.write(self.get_flam().to_string(index=False))

    def _write_csv_format(self, algorithm_output):
        """Write CSV format with comprehensive server analysis for FL core compatibility"""
        # Generate timestamp string
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Extract parameters from first entry for filename
        first_entry = next(iter(algorithm_output.values()))
        num_devices = first_entry['satellite_count']
        timesteps = len(algorithm_output)

        # Create filename
        filename = f"flam_{num_devices}n_{timesteps}t_flomps_{timestamp_str}.csv"
        filepath = os.path.join(self.csv_output_path, filename)

        print(f"Writing FLOMPS CSV format to {filepath}")

        with open(filepath, mode='w', newline='') as f:
            timestep = 1
            for time_stamp, data in algorithm_output.items():
                # Get data for this timestep
                matrix = data['federatedlearning_adjacencymatrix']
                round_num = data.get('round_number', 1)
                target_node = data.get('aggregator_id', 0)
                phase = data.get('phase', 'TRANSMITTING')
                round_length = data.get('round_length', 1)
                timestep_in_round = data.get('timestep_in_round', 1)

                # Enhanced server analysis fields
                server_connections_current = data.get('server_connections_current', 0)
                server_connections_cumulative = data.get('server_connections_cumulative', 0)
                target_connections = data.get('target_connections', 0)
                connected_satellites = data.get('connected_satellites', [])
                missing_satellites = data.get('missing_satellites', [])
                target_satellites = data.get('target_satellites', [])
                round_complete = data.get('round_complete', False)

                # Write header line with comprehensive server analysis
                f.write(f"Time: {time_stamp}, Timestep: {timestep}, Round: {round_num}, "
                    f"Target Node: {target_node}, Phase: {phase}, "
                    f"Round Length: {round_length}, Timestep in Round: {timestep_in_round}, "
                    f"Current Connections: {server_connections_current}, "
                    f"Cumulative Connections: {server_connections_cumulative}/{target_connections}, "
                    f"Connected Sats: {connected_satellites}, "
                    f"Missing Sats: {missing_satellites}, "
                    f"Target Sats: {target_satellites}, "
                    f"Round Complete: {round_complete}\n")

                # Write matrix rows
                for i in range(len(matrix)):
                    line = ",".join(str(matrix[i][j]) for j in range(len(matrix[i])))
                    f.write(line + "\n")

                timestep += 1

        print(f"FLOMPS CSV format exported to {filename}")

        # Calculate and display comprehensive round statistics
        rounds = {}
        for data in algorithm_output.values():
            round_num = data.get('round_number', 1)
            if round_num not in rounds:
                target_connections = data.get('target_connections', 0)
                final_connections = data.get('server_connections_cumulative', 0)
                success_rate = (final_connections / target_connections * 100) if target_connections > 0 else 0

                rounds[round_num] = {
                    'length': data.get('round_length', 1),
                    'server': data.get('aggregator_id', 0),
                    'server_name': data.get('selected_satellite', 'Unknown'),
                    'completed': data.get('round_complete', False),
                    'final_connections': final_connections,
                    'target_connections': target_connections,
                    'success_rate': success_rate,
                    'target_satellites': data.get('target_satellites', [])
                }

        print(f"\nGenerated {timesteps} timesteps across {len(rounds)} rounds:")
        print("=" * 80)

        for round_num, info in rounds.items():
            status = "✓ SUCCESS" if info['completed'] else "✗ TIMEOUT"
            conn_status = f"{info['final_connections']}/{info['target_connections']}"

            print(f"Round {round_num}: Server {info['server']} ({info['server_name']})")
            print(f"  Length: {info['length']} timestamps")
            print(f"  Connections: {conn_status} ({info['success_rate']:.1f}%)")
            print(f"  Target Satellites: {info['target_satellites']}")
            print(f"  Status: {status}")
            print("-" * 40)
