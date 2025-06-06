"""
Filename: algorithm_output.py
Description: Prepares and sends data to Federated Learning module
Initial Creator: Elysia Guglielmo (System Architect), stephen zeng
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

Usage:
Instantiate AlgorithmOutput and assign FLInput.
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

    # Constructor
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

    # Pass Federated Learning Adjacency Matrix (FLAM) to Federated Learning Component
    # Aggregator Flag has three values true, false, None
        # Aggregator Flag set to true when a client is selected with most number of connectivity with other clients.
        # Aggregator Flag set to false when a client is not selected due to less number of connectivity with other clients.
        # Aggregator Flag set to None when No connectivity exists between clients.
    def process_algorithm_output(self, algorithm_output):
        algo_op_rows = []

        for timestamp, algo_op in algorithm_output.items():
            # Convert the aggregator_flag to 'None' if there is no connectivity among satellites
            aggregator_flag = algo_op['aggregator_flag'] if algo_op['aggregator_flag'] is not None else 'None'
            # Get FLAM
            fl_am = algo_op['federatedlearning_adjacencymatrix']
            # Get satellite count
            satellite_count = algo_op['satellite_count']
            # Get chosen satellite name
            selected_satellite = algo_op['selected_satellite']
            # Append the row with the aggregator flag and matrix
            algo_op_rows.append({
                'time_stamp': timestamp,
                'satellite_count': satellite_count,
                'satellite_name':selected_satellite,
                'aggregator_flag': aggregator_flag,
                'federatedlearning_adjacencymatrix': fl_am
            })

        # Convert the list of algorithm output rows to a DataFrame for easy processing and manipulation
        self.set_flam(pds.DataFrame(algo_op_rows))

    # Data structure that can be passed to Federated Learning (FL) Component
    def set_result(self, algorithm_output):
        self.process_algorithm_output(algorithm_output)

    def log_result(self):
        print(self.flam_output)

    # Write the Federated Learning Adjacency Matrix (FLAM) to the file.
    def write_to_file(self, algorithm_output):
        # Ensure output directory exists
        os.makedirs(self.output_path, exist_ok=True)

        # Write original format
        self._write_original_format(algorithm_output)

        # Write Sam's CSV format
        self._write_sam_csv_format(algorithm_output)

    def _write_original_format(self, algorithm_output):
        """Write the original FLAM format to txt file"""
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

    def _write_sam_csv_format(self, algorithm_output):
        """Write Sam's CSV format for FL core compatibility"""
        # Generate timestamp string
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Extract parameters from first entry for filename
        first_entry = next(iter(algorithm_output.values()))
        num_devices = first_entry['satellite_count']
        timesteps = len(algorithm_output)

        # Create filename similar to Sam's format
        filename = f"flam_{num_devices}n_{timesteps}t_flomps_{timestamp_str}.csv"
        filepath = os.path.join(self.csv_output_path, filename)

        print(f"Writing Sam's CSV format to {filepath}")

        with open(filepath, mode='w', newline='') as f:
            timestep = 1
            for time_stamp, data in algorithm_output.items():
                # Get data for this timestep
                matrix = data['federatedlearning_adjacencymatrix']
                round_num = data.get('round_number', 1)
                target_node = data.get('target_node', 0)
                phase = data.get('phase', 'TRAINING')

                # Write header line with Time first, then Timestep number
                f.write(f"Time: {time_stamp}, Timestep: {timestep}, Round: {round_num}, "
                       f"Target Node: {target_node}, Phase: {phase}\n")

                # Write matrix rows
                for i in range(len(matrix)):
                    line = ",".join(str(matrix[i][j]) for j in range(len(matrix[i])))
                    f.write(line + "\n")

                # Add blank line between timesteps
                f.write("\n")
                timestep += 1

        print(f"Sam's CSV format exported to {filename}")
