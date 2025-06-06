"""
Filename: algorithm_core.py
Description: Manage FLOMPS algorithm processes.
Initial Creator: Elysia Guglielmo (System Architect)
Author: Yuganya Perumal, Gagandeep Singh
Date: 2024-07-31
Version: 1.0
Python Version:

Changelog:
- 2024-07-31: Initial creation.
- 2024-08-24: getters and setters for satellite names by Yuganya Perumal
- 2024-09-09: Move Algorithm Steps from Algorithm Handler to Algorithm Core by Yuganya Perrumal
- 2024-09-21: Implemented Load Balancing based on past selection of the satellite when there more than one satellite with max no. of connectivity.
- 2024-10-03: Removed validation for satellite names. auto generation of satellite names implemented if none found.


Usage:
Instantiate to setup Algorithmhandler and AlgorithmConfig.
"""
from flomps_algorithm.algorithm_output import AlgorithmOutput
import numpy as npy
import random
from collections import deque

class Algorithm():

    # Constructor
    def __init__(self):
        self.satellite_names = []
        self.adjacency_matrices = None
        self.output = AlgorithmOutput()
        self.selection_counts = None
        self.output_to_file = True
        self.algorithm_output_data = None  # Store algorithm output data

        # Sam's algorithm parameters
        self.toggle_chance = 0.1  # Default toggle chance for connection evolution
        self.training_time = 3    # Default training time in timesteps
        self.down_bias = 2.0      # Bias for breaking connections vs forming them

        # Round tracking variables
        self.round_number = 1
        self.target_node = None
        self.training_counter = 0
        self.in_training = True
        self.connections = None  # Will store the evolving connection matrix

    def set_satellite_names(self, satellite_names):
        self.satellite_names = satellite_names
        self.selection_counts = npy.zeros(len(satellite_names))
        # Initialize the connection matrix based on number of satellites
        num_satellites = len(satellite_names)
        self.connections = npy.zeros((num_satellites, num_satellites), dtype=int)
        # Select initial target node
        if num_satellites > 0:
            self.target_node = random.randint(0, num_satellites - 1)

    def get_satellite_names(self):
        return self.satellite_names

    def set_adjacency_matrices(self, adjacency_matrices):
        self.adjacency_matrices = adjacency_matrices

    def get_adjacency_matrices(self):
        return self.adjacency_matrices

    def set_output_to_file(self, output_to_file):
        self.output_to_file = output_to_file

    def get_algorithm_output(self):
        """Get algorithm output data, for workflow use"""
        return self.algorithm_output_data

    def set_algorithm_parameters(self, toggle_chance=0.1, training_time=3, down_bias=2.0):
        """Set parameters for Sam's algorithm"""
        self.toggle_chance = toggle_chance
        self.training_time = training_time
        self.down_bias = down_bias

    def is_reachable(self, matrix, target):
        """Check if all nodes can reach the target via BFS - Sam's reachability logic"""
        num_nodes = len(matrix)
        visited = [False] * num_nodes
        queue = deque([target])
        visited[target] = True

        while queue:
            node = queue.popleft()
            for neighbor, connected in enumerate(matrix[node]):
                if connected and not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

        return all(visited)

    def evolve_connections(self):
        """Evolve connections using Sam's toggle logic with bias"""
        num_devices = len(self.satellite_names)

        # Toggle connections with directional bias
        for i in range(num_devices):
            for j in range(i + 1, num_devices):
                if self.connections[i][j] == 1:
                    # Existing connection - bias towards breaking it
                    if random.random() < self.toggle_chance * self.down_bias:
                        self.connections[i][j] = 0
                        self.connections[j][i] = 0
                else:
                    # No connection - normal chance to form one
                    if random.random() < self.toggle_chance:
                        self.connections[i][j] = 1
                        self.connections[j][i] = 1

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
        """Sam's round-based algorithm with TRAINING/TRANSMITTING phases"""
        adjacency_matrices = self.get_adjacency_matrices()
        algorithm_output = {}

        # Initialize if this is the first run
        if self.target_node is None and len(self.satellite_names) > 0:
            self.target_node = random.randint(0, len(self.satellite_names) - 1)

        # Process each timestep
        for time_stamp, incoming_matrix in adjacency_matrices:
            nrows, mcolumns = incoming_matrix.shape
            satellite_count = nrows if nrows == mcolumns else 0

            # If using SatSim data, update our evolving connections based on actual connectivity
            # For real integration, we evolve from the incoming SatSim data
            if npy.any(incoming_matrix):
                # Update our connections based on SatSim input and apply evolution
                self.connections = incoming_matrix.copy()
                self.evolve_connections()
            else:
                # No connectivity from SatSim, just evolve existing connections
                self.evolve_connections()

            # Update training counter and phase status
            if self.in_training:
                self.training_counter += 1
                if self.training_counter > self.training_time:
                    self.in_training = False

            # Create effective matrix based on current training phase
            effective_matrix = npy.copy(self.connections)

            if self.in_training:
                # During training, zero out all connections
                effective_matrix[:, :] = 0
                phase = "TRAINING"
            else:
                phase = "TRANSMITTING"

            # Determine selected satellite and aggregator flag
            selected_satellite = None
            aggregator_flag = None

            if not self.in_training and self.target_node is not None:
                selected_satellite = self.get_selected_satellite_name(self.target_node)
                aggregator_flag = True

            # Store algorithm output in original format for compatibility
            algorithm_output[time_stamp] = {
                'satellite_count': satellite_count,
                'selected_satellite': selected_satellite,
                'federatedlearning_adjacencymatrix': effective_matrix,
                'aggregator_flag': aggregator_flag,
                'round_number': self.round_number,  # Additional info for Sam's format
                'phase': phase,  # Additional info for Sam's format
                'target_node': self.target_node  # Additional info for Sam's format
            }

            # Check for round completion during transmitting phase
            if not self.in_training and self.target_node is not None:
                if self.is_reachable(effective_matrix, self.target_node):
                    print(f"Round {self.round_number} complete — all nodes can reach target {self.target_node}.")
                    # Start new round
                    self.round_number += 1
                    self.target_node = random.randint(0, len(self.satellite_names) - 1)
                    self.training_counter = 0
                    self.in_training = True
                    print(f" Starting Round {self.round_number} | New Target Node: {self.target_node}")

        # Store the algorithm output data for external access
        self.algorithm_output_data = algorithm_output

        if self.output_to_file:
            self.output.write_to_file(algorithm_output)  # write to file.

        self.output.set_result(algorithm_output)  # set result to AlgorithmOutput

