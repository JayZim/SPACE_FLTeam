"""
Filename: algorithm_core.py
Description: Manage FLOMPS algorithm processes. 
Initial Creator: Elysia Guglielmo (System Architect)
Author: Yuganya Perumal,Stephen ZENG
Date: 2025-05-14
Version: 1.0
Python Version: 3.10

Changelog:
- 2024-07-31: Initial creation.
- 2024-08-24: getters and setters for satellite names by Yuganya Perumal
- 2024-09-09: Move Algorithm Steps from Algorithm Handler to Algorithm Core by Yuganya Perrumal
- 2024-09-21: Implemented Load Balancing based on past selection of the satellite when there more than one satellite with max no. of connectivity.
- 2024-10-03: Removed validation for satellite names. auto generation of satellite names implemented if none found.
- 2025-05-14: Added Algorithm Tuner by Stephen ZENG

Usage: 
Instantiate to setup Algorithmhandler and AlgorithmConfig.
"""
from flomps_algorithm.algorithm_output import AlgorithmOutput
import numpy as npy
class Algorithm():

    # Constructor
    def __init__(self):
        self.satellite_names = []
        self.adjacency_matrices = None
        self.output = AlgorithmOutput()
        self.selection_counts = None
        self.output_to_file = True 
        self.tuner = None  # add tuner attribute
        self.connectivity_threshold = None  # add connectivity threshold attribute
        
    def set_satellite_names(self, satellite_names):
        self.satellite_names = satellite_names
        self.selection_counts = npy.zeros(len(satellite_names))  

    def get_satellite_names(self):
        return self.satellite_names
    
    def set_adjacency_matrices(self, adjacency_matrices):       
        self.adjacency_matrices = adjacency_matrices
    
    def get_adjacency_matrices(self):
        return self.adjacency_matrices
    
    def set_output_to_file(self, output_to_file):
        self.output_to_file = output_to_file
    
    def set_tuner(self, tuner):  
        self.tuner = tuner  
  
    def set_connectivity_threshold(self, threshold):  
        # Set the connectivity threshold for the algorithm.
        self.connectivity_threshold = threshold
    
    def select_satellite_with_max_connections(self, each_matrix):
        satellite_connections = npy.sum(each_matrix, axis=1)
        max_connections = npy.max(satellite_connections)
        max_connected_satellites = [i for i, conn in enumerate(satellite_connections) if conn == max_connections]
        
        if len(max_connected_satellites) > 1:
            # Select satellite with the fewest past selections in case there is more than one satellite with max no.of connectivity.
            selected_satellite_index = min(max_connected_satellites, key=lambda index: self.selection_counts[index])
        else:
            selected_satellite_index = max_connected_satellites[0]

        # Update the selection count for the chosen satellite.
        self.selection_counts[selected_satellite_index] += 1
        return selected_satellite_index, max_connections
    
    def get_selected_satellite_name(self, satellite_index):
        satellite_names = self.get_satellite_names()
        if 0 <= satellite_index < len(satellite_names):
            return satellite_names[satellite_index]
        else:
            raise IndexError("Satellite does not exist for selection.")
    
    def start_algorithm_steps(self):
        adjacency_matrices = self.get_adjacency_matrices()
        algorithm_output = {}
        # Start Algorithm Component Steps
        # Step 1: Initialize aggregator flags for all satellite to False
        aggregator_flags = {sat_name: False for sat_name in self.get_satellite_names()}

        # Step 2: Loop through each incoming Adjacency Matrix associated with timestamp. 
        #         If No connectivitiy found at a particular timestamp set the matrix it to algorithm output 
        for time_stamp, each_matrix in adjacency_matrices:
            nrows, mcolumns = each_matrix.shape
            # get matrix size to know number of satellites involved.
            if(nrows == mcolumns):
                satellite_count = nrows 
            # Check if there are no connections (all zeros)
            if npy.all(each_matrix == 0):
                # If no connectivity between them, set algorithm output (Federated Learning Adjacency Matrix (FLAM)) without changes.
                algorithm_output[time_stamp] = {
                    'satellite_count': satellite_count,
                    'selected_satellite': None,
                    'federatedlearning_adjacencymatrix': each_matrix.copy(),
                    'aggregator_flag': None
                }
                continue  # Skip to the next iteration
        # Step 3: Incomming Adjacency Matrix has connections.
        # Step 4: Perform sum of rows
            satellite_connections = npy.sum(each_matrix, axis=1)
        # Step 5: Find the maximum number of connections in that particular timestamp.
            max_connections = npy.max(satellite_connections)
  
        # Step 6: Choose the satellite with fewest selection in case there are more than one satellites with maximum number of connectivity.
            selected_satellite_index, max_connections = self.select_satellite_with_max_connections(each_matrix)
            selected_satellite = self.get_selected_satellite_name(selected_satellite_index)
            
        # Step 7: Set the aggregator flag to true for the selected satellite for that particular timestamp.
            aggregator_flags[selected_satellite] = True

        # Step 8: Create a copy of the Adjacency Matrix to trim down the connectivity for non-selected satellites.
            fl_am = each_matrix.copy()

        # Step 9: Retain connections (1s) for the selected node
            for i, satellite_name in enumerate(self.get_satellite_names()):
                if i != selected_satellite_index:
                    # Remove connections to/from non-selected nodes that are not connected to the selected node
                    if each_matrix[selected_satellite_index, i] == 0 and each_matrix[i, selected_satellite_index] == 0:
                        fl_am[i, :] = 0  # Remove all connections from this node
                        fl_am[:, i] = 0  # Remove all connections to this node

        # Step 10: Store algorithm output (Federated Learning Adjacency Matrix (FLAM)) as a dictionary data structure.
            algorithm_output[time_stamp] = {
                'satellite_count': satellite_count,
                'selected_satellite': selected_satellite,
                'federatedlearning_adjacencymatrix': fl_am,
                'aggregator_flag': aggregator_flags[selected_satellite]
            }
        
        if self.output_to_file:
            self.output.write_to_file(algorithm_output) # write to file.
            
        self.output.set_result(algorithm_output) # set result to AlgorithmOutput
