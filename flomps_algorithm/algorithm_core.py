"""
Filename: algorithm_core.py
Description: Manage FLOMPS algorithm processes with three-phase round execution.
Initial Creator: Elysia Guglielmo (System Architect)
Contributors: Yuganya Perumal, Gagandeep Singh
Date: 2024-07-31
Version: 2.0
Python Version: 3.12

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
- 2025-10-03: Implemented three-phase round structure (TRANSMITTING → CHECK → REDISTRIBUTION)
- 2025-10-03: Added find_best_redistribution_server() with two-hop optimization (A+B)
- 2025-10-03: Fixed phase_length to update retroactively after phase completion

Three-Phase Algorithm:
1. TRANSMITTING: Clients send local models to aggregation server (uplink)
2. CHECK: Select and transfer global model to redistribution server (relay)
3. REDISTRIBUTION: Redistribution server sends global model to all clients (downlink)

Usage:
cd /path/to/flomps_algorithm && python algorithm_handler.py ../sat_sim/output/sat_sim_xxxx.txt

Sample Output:
================================================================================
ROUND 2 - THREE PHASE EXECUTION
================================================================================
### PHASE 1: TRANSMITTING (Model Aggregation) ###
  Selected Server 7 (Satellite 8)
  Will achieve 6/7 connections in 7 timestamps
  ✓ TRANSMITTING complete at timestep 7

### PHASE 2: CHECK (Select Redistribution Server) ###
  Selected Redistribution Server 5 (Satellite 6)
  Time A (aggregation→redistribution): 1 timestamps
  Time B (redistribution→clients): 1 timestamps
  → CHECK phase duration: 1 timesteps

### PHASE 3: REDISTRIBUTION (Distribute Global Model) ###
  Redistribution Server 5 targeting 5 satellites: [0, 2, 3, 4, 7]
  ✓ REDISTRIBUTION complete at timestep 1

Round 2 Summary:
  TRANSMITTING: 7 timesteps (Server 7)
  CHECK: 1 timesteps (Server 7 → 5)
  REDISTRIBUTION: 1 timesteps (Server 5)
  Total: 9 timesteps

CSV Output: synth_FLAMs/flam_8n_120t_flomps_3phase_YYYY-MM-DD_HH-MM-SS.csv
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
        self.aggregation_server = None
        self.redistribution_server = None

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

    def find_best_redistribution_server(self, aggregation_server, start_matrix_index, max_lookahead=20):
        """
        Find the best server for redistribution using two-hop optimization.
        Calculates: (aggregation_server -> candidate) + (candidate -> all clients)
        Returns: (redistribution_server, time_to_reach_redistribution_server, redistribution_analysis)
        """
        num_satellites = len(self.satellite_names)

        print(f"\n=== Finding Redistribution Server (from Aggregation Server {aggregation_server}) ===")

        # Step 1: Analyze path from aggregation_server to each candidate
        path_to_candidates = {}
        for candidate_idx in range(num_satellites):
            if candidate_idx == aggregation_server:
                # Same server, no time needed
                path_to_candidates[candidate_idx] = {
                    'reachable': True,
                    'timesteps': 0
                }
                continue

            # Analyze how long it takes aggregation_server to reach this candidate
            connected_to_candidate = False
            timesteps_to_candidate = max_lookahead

            for timesteps_ahead in range(max_lookahead):
                matrix_idx = start_matrix_index + timesteps_ahead
                if matrix_idx >= len(self.adjacency_matrices):
                    break

                _, matrix = self.adjacency_matrices[matrix_idx]

                # Check if aggregation_server can reach candidate
                if matrix[aggregation_server][candidate_idx] == 1:
                    connected_to_candidate = True
                    timesteps_to_candidate = timesteps_ahead + 1  # +1 because we need to wait for this timestep
                    break

            path_to_candidates[candidate_idx] = {
                'reachable': connected_to_candidate,
                'timesteps': timesteps_to_candidate
            }

        # Step 2: For each reachable candidate, analyze its redistribution capability
        redistribution_analysis = {}

        for candidate_idx in range(num_satellites):
            if not path_to_candidates[candidate_idx]['reachable']:
                continue

            time_a = path_to_candidates[candidate_idx]['timesteps']

            # Analyze from the timestep when candidate receives the model
            candidate_start_index = start_matrix_index + time_a

            # Analyze candidate's ability to redistribute to all clients
            candidate_analysis = self.analyze_single_satellite(candidate_idx, candidate_start_index, max_lookahead)

            time_b = candidate_analysis['timestamps_to_max']
            total_time = time_a + time_b

            redistribution_analysis[candidate_idx] = {
                'time_a': time_a,  # aggregation -> candidate
                'time_b': time_b,  # candidate -> all clients
                'total_time': total_time,
                'max_connections': candidate_analysis['max_connections'],
                'connected_satellites': candidate_analysis['connected_satellites']
            }

            print(f"Candidate {candidate_idx} ({self.satellite_names[candidate_idx]}): "
                  f"A={time_a} + B={time_b} = {total_time} timesteps, "
                  f"reaches {candidate_analysis['max_connections']}/{num_satellites-1} clients")

        # Step 3: Select best redistribution server
        # Primary: minimum total_time
        # Secondary: max_connections (in case of tie)
        # Tertiary: random selection (for ties)

        if not redistribution_analysis:
            # Fallback to aggregation_server if no candidates found
            print(f"⚠ No reachable candidates, using aggregation_server {aggregation_server}")
            return aggregation_server, 0, {'max_connections': 0, 'connected_satellites': set()}

        candidates_by_time = {}
        for candidate_idx, analysis in redistribution_analysis.items():
            total_time = analysis['total_time']
            if total_time not in candidates_by_time:
                candidates_by_time[total_time] = []
            candidates_by_time[total_time].append((candidate_idx, analysis))

        # Get minimum time
        min_time = min(candidates_by_time.keys())
        best_candidates = candidates_by_time[min_time]

        # If multiple candidates with same time, pick by max_connections
        max_connections = max(analysis['max_connections'] for _, analysis in best_candidates)
        best_candidates = [(idx, analysis) for idx, analysis in best_candidates
                          if analysis['max_connections'] == max_connections]

        # Random selection from remaining candidates
        best_server_idx, best_analysis = random.choice(best_candidates)

        print(f"\n✓ Selected Redistribution Server {best_server_idx} ({self.satellite_names[best_server_idx]})")
        print(f"  Time A (aggregation→redistribution): {best_analysis['time_a']} timestamps")
        print(f"  Time B (redistribution→clients): {best_analysis['time_b']} timestamps")
        print(f"  Total time: {best_analysis['total_time']} timestamps")
        print(f"  Will reach {best_analysis['max_connections']}/{num_satellites-1} clients")

        return best_server_idx, best_analysis['time_a'], best_analysis

    def start_algorithm_steps(self):
        """
        Three-phase FLOMPS algorithm: TRANSMITTING → CHECK → REDISTRIBUTION
        """
        adjacency_matrices = self.get_adjacency_matrices()
        algorithm_output = {}
        current_matrix_index = 0
        num_satellites = len(self.satellite_names)

        while current_matrix_index < len(adjacency_matrices):
            print(f"\n{'='*80}")
            print(f"ROUND {self.round_number} - THREE PHASE EXECUTION")
            print(f"{'='*80}")

            # ============================================================
            # PHASE 1: TRANSMITTING (Aggregation)
            # ============================================================
            print(f"\n### PHASE 1: TRANSMITTING (Model Aggregation) ###")

            # Select aggregation server
            aggregation_server, _, aggregation_analysis = self.find_best_server_for_round(current_matrix_index)
            self.aggregation_server = aggregation_server

            # Run TRANSMITTING phase
            phase_start_index = current_matrix_index
            timestep_in_phase = 0
            target_satellites = aggregation_analysis['connected_satellites']
            target_connections_count = len(target_satellites)
            connected_satellites = set()
            phase_complete = False

            print(f"Aggregation Server {aggregation_server} targeting {target_connections_count} satellites: {sorted(list(target_satellites))}")

            while current_matrix_index < len(adjacency_matrices) and not phase_complete:
                timestep_in_phase += 1
                time_stamp, matrix = adjacency_matrices[current_matrix_index]

                # Check connections
                current_timestamp_connections = set()
                for target_sat in range(num_satellites):
                    if target_sat != aggregation_server and matrix[aggregation_server][target_sat] == 1:
                        current_timestamp_connections.add(target_sat)
                        connected_satellites.add(target_sat)

                # Check completion
                if connected_satellites >= target_satellites:
                    phase_complete = True
                    print(f"  ✓ TRANSMITTING complete at timestep {timestep_in_phase}")
                    print(f"    Aggregated models from: {sorted(list(connected_satellites))}")
                else:
                    missing_satellites = target_satellites - connected_satellites
                    print(f"  → Timestep {timestep_in_phase}: {len(connected_satellites)}/{target_connections_count} models aggregated")

                # Store output
                algorithm_output[time_stamp] = {
                    'satellite_count': num_satellites,
                    'satellite_names': self.satellite_names,
                    'selected_satellite': self.satellite_names[aggregation_server],
                    'aggregator_id': aggregation_server,
                    'redistribution_id': None,  # Not determined yet
                    'federatedlearning_adjacencymatrix': matrix,
                    'aggregator_flag': True,
                    'round_number': self.round_number,
                    'phase': "TRANSMITTING",
                    'target_node': aggregation_server,
                    'phase_length': timestep_in_phase,
                    'timestep_in_phase': timestep_in_phase,
                    'server_connections_current': len(current_timestamp_connections),
                    'server_connections_cumulative': len(connected_satellites),
                    'target_connections': target_connections_count,
                    'connected_satellites': sorted(list(connected_satellites)),
                    'missing_satellites': sorted(list(target_satellites - connected_satellites)),
                    'target_satellites': sorted(list(target_satellites)),
                    'phase_complete': phase_complete
                }

                current_matrix_index += 1

                # Safety timeout
                if timestep_in_phase >= 20:
                    print(f"  ⚠ TRANSMITTING timeout after {timestep_in_phase} timestamps")
                    phase_complete = True

            transmitting_length = timestep_in_phase

            # Update phase_length for all TRANSMITTING timesteps
            for i in range(phase_start_index, current_matrix_index):
                if i < len(adjacency_matrices):
                    timestamp_key = adjacency_matrices[i][0]
                    if timestamp_key in algorithm_output:
                        algorithm_output[timestamp_key]['phase_length'] = transmitting_length

            # ============================================================
            # PHASE 2: CHECK (Select Redistribution Server)
            # ============================================================
            print(f"\n### PHASE 2: CHECK (Select Redistribution Server) ###")

            if current_matrix_index >= len(adjacency_matrices):
                print("⚠ No more timesteps available for CHECK phase")
                break

            # Find best redistribution server
            redistribution_server, check_duration, redistribution_analysis = \
                self.find_best_redistribution_server(aggregation_server, current_matrix_index)
            self.redistribution_server = redistribution_server

            # Run CHECK phase (waiting for connection to redistribution server)
            check_start_index = current_matrix_index

            if check_duration == 0:
                print(f"  ✓ Redistribution server is same as aggregation server (Server {redistribution_server})")
                print(f"  ✓ CHECK phase duration: 0 timesteps (no transfer needed)")
            else:
                print(f"  → Transferring model from Server {aggregation_server} to Server {redistribution_server}")
                print(f"  → CHECK phase duration: {check_duration} timesteps")

                for i in range(check_duration):
                    if current_matrix_index >= len(adjacency_matrices):
                        break

                    timestep_in_check = i + 1
                    time_stamp, matrix = adjacency_matrices[current_matrix_index]

                    check_complete = (timestep_in_check == check_duration)

                    print(f"  → CHECK timestep {timestep_in_check}/{check_duration}: Waiting for connection to Server {redistribution_server}")

                    # Store output
                    algorithm_output[time_stamp] = {
                        'satellite_count': num_satellites,
                        'satellite_names': self.satellite_names,
                        'selected_satellite': self.satellite_names[redistribution_server],
                        'aggregator_id': aggregation_server,
                        'redistribution_id': redistribution_server,
                        'federatedlearning_adjacencymatrix': matrix,
                        'aggregator_flag': False,
                        'round_number': self.round_number,
                        'phase': "CHECK",
                        'target_node': redistribution_server,
                        'phase_length': check_duration,
                        'timestep_in_phase': timestep_in_check,
                        'server_connections_current': 0,
                        'server_connections_cumulative': 0,
                        'target_connections': 0,
                        'connected_satellites': [],
                        'missing_satellites': [],
                        'target_satellites': [],
                        'phase_complete': check_complete
                    }

                    current_matrix_index += 1

                # Note: CHECK phase_length is already known upfront, so no need to update retroactively

            # ============================================================
            # PHASE 3: REDISTRIBUTION (Distribute Global Model)
            # ============================================================
            print(f"\n### PHASE 3: REDISTRIBUTION (Distribute Global Model) ###")

            if current_matrix_index >= len(adjacency_matrices):
                print("⚠ No more timesteps available for REDISTRIBUTION phase")
                break

            # Run REDISTRIBUTION phase
            redistribution_start_index = current_matrix_index
            timestep_in_phase = 0
            target_satellites_redist = redistribution_analysis['connected_satellites']
            target_connections_count_redist = len(target_satellites_redist)
            connected_satellites_redist = set()
            phase_complete = False

            print(f"Redistribution Server {redistribution_server} targeting {target_connections_count_redist} satellites: {sorted(list(target_satellites_redist))}")

            while current_matrix_index < len(adjacency_matrices) and not phase_complete:
                timestep_in_phase += 1
                time_stamp, matrix = adjacency_matrices[current_matrix_index]

                # Check connections
                current_timestamp_connections = set()
                for target_sat in range(num_satellites):
                    if target_sat != redistribution_server and matrix[redistribution_server][target_sat] == 1:
                        current_timestamp_connections.add(target_sat)
                        connected_satellites_redist.add(target_sat)

                # Check completion
                if connected_satellites_redist >= target_satellites_redist:
                    phase_complete = True
                    print(f"  ✓ REDISTRIBUTION complete at timestep {timestep_in_phase}")
                    print(f"    Global model distributed to: {sorted(list(connected_satellites_redist))}")
                else:
                    missing_satellites = target_satellites_redist - connected_satellites_redist
                    print(f"  → Timestep {timestep_in_phase}: {len(connected_satellites_redist)}/{target_connections_count_redist} models distributed")

                # Store output
                algorithm_output[time_stamp] = {
                    'satellite_count': num_satellites,
                    'satellite_names': self.satellite_names,
                    'selected_satellite': self.satellite_names[redistribution_server],
                    'aggregator_id': aggregation_server,
                    'redistribution_id': redistribution_server,
                    'federatedlearning_adjacencymatrix': matrix,
                    'aggregator_flag': False,
                    'round_number': self.round_number,
                    'phase': "REDISTRIBUTION",
                    'target_node': redistribution_server,
                    'phase_length': timestep_in_phase,
                    'timestep_in_phase': timestep_in_phase,
                    'server_connections_current': len(current_timestamp_connections),
                    'server_connections_cumulative': len(connected_satellites_redist),
                    'target_connections': target_connections_count_redist,
                    'connected_satellites': sorted(list(connected_satellites_redist)),
                    'missing_satellites': sorted(list(target_satellites_redist - connected_satellites_redist)),
                    'target_satellites': sorted(list(target_satellites_redist)),
                    'phase_complete': phase_complete
                }

                current_matrix_index += 1

                # Safety timeout
                if timestep_in_phase >= 20:
                    print(f"  ⚠ REDISTRIBUTION timeout after {timestep_in_phase} timestamps")
                    phase_complete = True

            redistribution_length = timestep_in_phase

            # Update phase_length for all REDISTRIBUTION timesteps
            for i in range(redistribution_start_index, current_matrix_index):
                if i < len(adjacency_matrices):
                    timestamp_key = adjacency_matrices[i][0]
                    if timestamp_key in algorithm_output:
                        algorithm_output[timestamp_key]['phase_length'] = redistribution_length

            # Round summary
            total_round_length = transmitting_length + check_duration + redistribution_length
            print(f"\n{'='*80}")
            print(f"Round {self.round_number} Summary:")
            print(f"  TRANSMITTING: {transmitting_length} timesteps (Server {aggregation_server})")
            print(f"  CHECK: {check_duration} timesteps (Server {aggregation_server} → {redistribution_server})")
            print(f"  REDISTRIBUTION: {redistribution_length} timesteps (Server {redistribution_server})")
            print(f"  Total: {total_round_length} timesteps")
            print(f"{'='*80}")

            # Move to next round
            self.round_number += 1

        # Store the algorithm output data for external access
        self.algorithm_output_data = algorithm_output

        if self.output_to_file:
            self.output.write_to_file(algorithm_output)

        self.output.set_result(algorithm_output)
