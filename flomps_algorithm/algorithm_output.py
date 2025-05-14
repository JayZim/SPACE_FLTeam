"""
Filename: algorithm_output.py
Description: Prepares and sends data to Federated Learning module
Initial Creator: Elysia Guglielmo (System Architect)
Author: Yuganya Perumal,stephen zeng
Date: 2025-05-14
Version: 1.0
Python Version: 

Changelog:
- 2024-08-07: Initial creation.
- 2024-08-24: Added method to write Federated Learning Adjacency Matrix (FLAM) to a file by Yuganya Perumal
- 2024-08-31: Added method to set result as data frame object to be passed as input FL component by Yuganya Perumal.
- 2024-09-09: Algorithm output returns the FLAM.
- 2024-09-21: Algorithm output re organised to remove redundancy and added timestamp and satellite name to data structure passed to FL.
- 2024-10-14: Output file name changed to FLAM.txt as per client feedback.
- 2025-05-14: Added Algorithm Tuner by Stephen ZENG
- 2025-05-14: Added method to log result to console by Stephen ZENG
Usage: 
Instantiate AlgorithmOutput and assign FLInput. 
"""
from interfaces.output import Output
import os
from datetime import datetime
import numpy as npy
import pandas as pds
class AlgorithmOutput(Output):

    # Constructor
    def __init__(self):
        self.flam_output = None

        # Get the directory of the current script (sat_sim_output.py)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Set output_path to 'sat_sim_output' folder within the 'sat_sim' directory
        self.output_path = os.path.join(script_dir, 'output')

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
                'satellite_server':selected_satellite,
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
    def write_to_file(self, algorithm_output, output_file=None):
        if not output_file:
            output_file = self._generate_output_filename()
        
        # add timestamp to output file name
        if self.flam_output is None:
            self.process_algorithm_output(algorithm_output)  # output is a dictionary
        
        import os
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w') as file:
            file.write(self.get_flam().to_string(index=False))

    def _generate_output_filename(self):
        """Generate output filename with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.output_path, f"flam_{timestamp}.txt")