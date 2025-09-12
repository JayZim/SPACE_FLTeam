"""
Filename: algorithm_core.py
Description: Manage FLOMPS algorithm processes.
Initial Creator: Elysia Guglielmo (System Architect)
Contributors: Yuganya Perumal, Gagandeep Singh
Date: 2024-07-31
Version: 1.0
Python Version:

Changelog:
- 2024-07-31: Initial creation.
- 2024-08-24: getters and setters for satellite names by Yuganya Perumal
- 2024-09-09: Move Algorithm Steps from Algorithm Handler to Algorithm Core by Yuganya Perrumal
- 2024-09-21: Implemented Load Balancing based on past selection of the satellite when there more than one satellite with max no. of connectivity.
- 2024-10-03: Removed validation for satellite names. auto generation of satellite names implemented if none found.
- 2025-08-29: Fixed import path issues for interfaces module by moving sys.path.append before imports.
- 2025-09-05: Major algorithm optimization - replaced connection-count based server selection with time-based optimization.
- 2025-09-05: Added calculate_cumulative_connection_time() function for predictive server selection.
- 2025-09-05: Implemented cumulative connectivity model for more realistic satellite communication behavior.
- 2025-09-05: Enhanced load balancing with increased penalty factor (0.1 to 0.5) for better satellite rotation.
- 2025-09-12: Code cleanup - removed redundant find_best_server_for_round() and unused calculate_cumulative_connection_time() functions.
- 2025-09-12: Optimized function hierarchy for better maintainability and eliminated dead code.


Usage:
Instantiate to setup Algorithmhandler and AlgorithmConfig.

quick run: cd (path)/flomps_algorithm && python algorithm_handler.py (path)/sat_sim/output/sat_sim_xxxx.txt
"""


from flomps_algorithm.algorithm_output import AlgorithmOutput
import numpy as npy
import random

class Algorithm():

    def __init__(self):
        self.satellite_names = []
        self.adjacency_matrices = None
        self.output = AlgorithmOutput()
        self.selection_counts = None
        self.output_to_file = True
        self.algorithm_output_data = None

        # Round tracking variables
        self.round_number = 1
        self.current_server = None

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

    def get_algorithm_output(self):
        return self.algorithm_output_data

    def analyze_all_satellites(self, start_matrix_index, max_lookahead=20):
        """
        Pre-analyze ALL satellites to find their maximum connectivity potential.
        Returns detailed analysis for each satellite.
        """
        num_satellites = len(self.satellite_names)
        satellite_analysis = {}

        print(f"\n=== Analyzing all {num_satellites} satellites for Round {self.round_number} ===")

        for sat_idx in range(num_satellites):
            analysis = self.analyze_single_satellite(sat_idx, start_matrix_index, max_lookahead)
            satellite_analysis[sat_idx] = analysis

            print(f"Satellite {sat_idx} ({self.satellite_names[sat_idx]}): "
                f"Max {analysis['max_connections']}/{num_satellites-1} connections "
                f"in {analysis['timestamps_to_max']} timestamps")

        return satellite_analysis

    def analyze_single_satellite(self, sat_idx, start_matrix_index, max_lookahead=20):
        """
        Analyze a single satellite's connectivity potential over the lookahead window.
        Returns: {
            'max_connections': int,
            'timestamps_to_max': int,
            'connected_satellites': set,
            'connection_timeline': list
        }
        """
        num_satellites = len(self.satellite_names)
        target_satellites = set(range(num_satellites))
        target_satellites.remove(sat_idx)  # Remove self

        connected_satellites = set()
        max_connections = 0
        timestamps_to_max = max_lookahead
        connection_timeline = []

        for timesteps_ahead in range(max_lookahead):
            matrix_idx = start_matrix_index + timesteps_ahead
            if matrix_idx >= len(self.adjacency_matrices):
                break

            _, matrix = self.adjacency_matrices[matrix_idx]

            # Check which satellites this server connects to in this timestamp
            current_connections = set()
            for target_sat in range(num_satellites):
                if target_sat != sat_idx and matrix[sat_idx][target_sat] == 1:
                    current_connections.add(target_sat)
                    connected_satellites.add(target_sat)  # Add to cumulative

            # Track timeline
            connection_timeline.append({
                'timestep': timesteps_ahead + 1,
                'current_connections': list(current_connections),
                'cumulative_connections': list(connected_satellites),
                'cumulative_count': len(connected_satellites)
            })

            # Update max if we found more connections
            if len(connected_satellites) > max_connections:
                max_connections = len(connected_satellites)
                timestamps_to_max = timesteps_ahead + 1

        return {
            'max_connections': max_connections,
            'timestamps_to_max': timestamps_to_max,
            'connected_satellites': connected_satellites,
            'connection_timeline': connection_timeline,
            'satellite_idx': sat_idx,
            'satellite_name': self.satellite_names[sat_idx]
        }

    def find_best_server_for_round(self, start_matrix_index=0):
        """
        Find the satellite with the best connectivity performance.
        1. Analyze all satellites first
        2. Pick the one with highest max connections
        3. If tied, pick the fastest
        4. If still tied, use load balancing
        """
        if start_matrix_index >= len(self.adjacency_matrices):
            return 0, 5  # Default fallback

        # Step 1: Analyze all satellites
        satellite_analysis = self.analyze_all_satellites(start_matrix_index)

        # Step 2: Find the best performance
        best_server = 0
        best_analysis = satellite_analysis[0]

        print(f"\n=== Server Selection for Round {self.round_number} ===")

        for sat_idx, analysis in satellite_analysis.items():
            is_better = False
            reason = ""

            # Primary criteria: Higher max connections
            if analysis['max_connections'] > best_analysis['max_connections']:
                is_better = True
                reason = f"Higher max connections ({analysis['max_connections']} vs {best_analysis['max_connections']})"

            # Secondary criteria: Same max connections but faster
            elif (analysis['max_connections'] == best_analysis['max_connections'] and
                analysis['timestamps_to_max'] < best_analysis['timestamps_to_max']):
                is_better = True
                reason = f"Same max connections ({analysis['max_connections']}) but faster ({analysis['timestamps_to_max']} vs {best_analysis['timestamps_to_max']} timestamps)"

            # Tertiary criteria: Load balancing (less frequently selected)
            elif (analysis['max_connections'] == best_analysis['max_connections'] and
                analysis['timestamps_to_max'] == best_analysis['timestamps_to_max'] and
                self.selection_counts[sat_idx] < self.selection_counts[best_server]):
                is_better = True
                reason = f"Load balancing (selected {self.selection_counts[sat_idx]} vs {self.selection_counts[best_server]} times)"

            if is_better:
                best_server = sat_idx
                best_analysis = analysis
                print(f"  New best: Satellite {sat_idx} ({self.satellite_names[sat_idx]}) - {reason}")

        # Update selection count
        self.selection_counts[best_server] += 1

        print(f"\n✓ Selected Server {best_server} ({self.satellite_names[best_server]})")
        print(f"  Will achieve {best_analysis['max_connections']}/{len(self.satellite_names)-1} connections in {best_analysis['timestamps_to_max']} timestamps")
        print(f"  Target satellites: {sorted(list(best_analysis['connected_satellites']))}")

        return best_server, best_analysis['timestamps_to_max'], best_analysis

    def start_algorithm_steps(self):
        """
        Proper FLOMPS algorithm with comprehensive server analysis.
        """
        adjacency_matrices = self.get_adjacency_matrices()
        algorithm_output = {}

        current_matrix_index = 0

        while current_matrix_index < len(adjacency_matrices):
            # Step 1: Analyze and select best server for this round
            selected_server, estimated_timestamps, server_analysis = self.find_best_server_for_round(current_matrix_index)
            self.current_server = selected_server

            # Step 2: Run the round with the selected server
            round_start_index = current_matrix_index
            timestep_in_round = 0
            num_satellites = len(self.satellite_names)
            target_satellites = server_analysis['connected_satellites']
            target_connections_count = len(target_satellites)
            connected_satellites = set()
            round_complete = False

            print(f"\n=== Running Round {self.round_number} ===")
            print(f"Server {selected_server} targeting {target_connections_count} satellites: {sorted(list(target_satellites))}")

            while current_matrix_index < len(adjacency_matrices) and not round_complete:
                timestep_in_round += 1
                time_stamp, matrix = adjacency_matrices[current_matrix_index]

                # Check which satellites this server connects to in this timestamp
                current_timestamp_connections = set()
                for target_sat in range(num_satellites):
                    if target_sat != selected_server and matrix[selected_server][target_sat] == 1:
                        current_timestamp_connections.add(target_sat)
                        connected_satellites.add(target_sat)  # Add to cumulative

                # Check if server has achieved its maximum connectivity
                if connected_satellites >= target_satellites:
                    round_complete = True
                    print(f"  ✓ Server {selected_server} achieved maximum connectivity at timestep {timestep_in_round}")
                    print(f"    Connected to all target satellites: {sorted(list(connected_satellites))}")
                else:
                    missing_satellites = target_satellites - connected_satellites
                    print(f"  → Timestep {timestep_in_round}: Connected {len(connected_satellites)}/{target_connections_count}")
                    print(f"    Current: {sorted(list(current_timestamp_connections))}")
                    print(f"    Cumulative: {sorted(list(connected_satellites))}")
                    print(f"    Still need: {sorted(list(missing_satellites))}")

                # Store algorithm output
                algorithm_output[time_stamp] = {
                    'satellite_count': len(matrix),
                    'satellite_names': self.satellite_names,
                    'selected_satellite': self.satellite_names[selected_server],
                    'aggregator_id': selected_server,
                    'federatedlearning_adjacencymatrix': matrix,
                    'aggregator_flag': True,
                    'round_number': self.round_number,
                    'phase': "TRANSMITTING",
                    'target_node': selected_server,
                    'round_length': timestep_in_round,  # Will be updated when round completes
                    'timestep_in_round': timestep_in_round,
                    'server_connections_current': len(current_timestamp_connections),
                    'server_connections_cumulative': len(connected_satellites),
                    'target_connections': target_connections_count,
                    'connected_satellites': sorted(list(connected_satellites)),
                    'missing_satellites': sorted(list(target_satellites - connected_satellites)),
                    'target_satellites': sorted(list(target_satellites)),
                    'round_complete': round_complete
                }

                current_matrix_index += 1

                # Safety timeout
                if timestep_in_round >= 20:
                    print(f"  ⚠ Round {self.round_number} timeout after {timestep_in_round} timestamps")
                    print(f"    Achieved {len(connected_satellites)}/{target_connections_count} target connections")
                    round_complete = True

            # Update round_length for all timestamps in this completed round
            actual_round_length = timestep_in_round
            for i in range(round_start_index, current_matrix_index):
                if i < len(adjacency_matrices):
                    timestamp_key = adjacency_matrices[i][0]
                    if timestamp_key in algorithm_output:
                        algorithm_output[timestamp_key]['round_length'] = actual_round_length

            success_rate = len(connected_satellites) / target_connections_count * 100 if target_connections_count > 0 else 0
            print(f"\nRound {self.round_number} completed in {actual_round_length} timestamps")
            print(f"Success rate: {success_rate:.1f}% ({len(connected_satellites)}/{target_connections_count} targets reached)")

            # Move to next round
            self.round_number += 1

        # Store the algorithm output data for external access
        self.algorithm_output_data = algorithm_output

        if self.output_to_file:
            self.output.write_to_file(algorithm_output)

        self.output.set_result(algorithm_output)
