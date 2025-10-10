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

        # Server selection parameters
        self.connect_to_all_satellites = False
        self.max_lookahead = 20
        self.minimum_connected_satellites = 5

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

    def set_connect_to_all_satellites(self, connect_to_all_satellites):
        self.connect_to_all_satellites = connect_to_all_satellites

    def set_max_lookahead(self, max_lookahead):
        self.max_lookahead = max_lookahead

    def set_minimum_connected_satellites(self, minimum_connected_satellites):
        self.minimum_connected_satellites = minimum_connected_satellites

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
        Uses configurable criteria:
        - connect_to_all_satellites: If True, server must connect to all satellites
        - minimum_connected_satellites: Minimum number of satellites to connect to
        - max_lookahead: Maximum timesteps to look ahead

        Selection criteria:
        1. Must meet connectivity requirements (all satellites OR minimum satellites)
        2. Must meet connectivity within max_lookahead timesteps
        3. Among valid candidates: highest max connections
        4. If tied: fastest to achieve max connections
        5. If still tied: load balancing
        """
        if start_matrix_index >= len(self.adjacency_matrices):
            return 0, 5, {'max_connections': 0, 'connected_satellites': set()}  # Default fallback

        num_satellites = len(self.satellite_names)

        # Determine required connections based on settings
        if self.connect_to_all_satellites:
            required_connections = num_satellites - 1  # All satellites except self
        else:
            required_connections = self.minimum_connected_satellites

        print(f"\n=== Server Selection for Round {self.round_number} ===")
        print(f"Configuration: connect_to_all={self.connect_to_all_satellites}, "
              f"min_satellites={self.minimum_connected_satellites}, max_lookahead={self.max_lookahead}")
        print(f"Required connections: {required_connections}/{num_satellites-1}")

        # Step 1: Analyze all satellites
        satellite_analysis = self.analyze_all_satellites(start_matrix_index, self.max_lookahead)

        # Step 2: Filter satellites that meet requirements
        valid_candidates = {}
        for sat_idx, analysis in satellite_analysis.items():
            # Check if satellite meets minimum connectivity requirement
            if analysis['max_connections'] < required_connections:
                print(f"  ✗ Satellite {sat_idx}: Only {analysis['max_connections']}/{required_connections} connections (insufficient)")
                continue

            # Check if satellite achieves required connections within max_lookahead
            # Need to find when it reaches required_connections
            timestep_to_required = None
            for timeline_entry in analysis['connection_timeline']:
                if timeline_entry['cumulative_count'] >= required_connections:
                    timestep_to_required = timeline_entry['timestep']
                    break

            if timestep_to_required is None or timestep_to_required > self.max_lookahead:
                print(f"  ✗ Satellite {sat_idx}: Doesn't reach {required_connections} connections within {self.max_lookahead} timesteps")
                continue

            valid_candidates[sat_idx] = {
                'analysis': analysis,
                'timestep_to_required': timestep_to_required
            }
            print(f"  ✓ Satellite {sat_idx}: Valid candidate - {analysis['max_connections']}/{num_satellites-1} connections in {timestep_to_required} timesteps")

        # Step 3: If no valid candidates, search through ALL remaining timesteps
        if not valid_candidates:
            # Calculate total remaining timesteps
            total_remaining = len(self.adjacency_matrices) - start_matrix_index
            print(f"\n⚠ No satellites meet criteria within {self.max_lookahead} timesteps")
            print(f"   Searching through all remaining {total_remaining} timesteps...")

            satellite_analysis = self.analyze_all_satellites(start_matrix_index, total_remaining)

            for sat_idx, analysis in satellite_analysis.items():
                if analysis['max_connections'] >= required_connections:
                    timestep_to_required = None
                    for timeline_entry in analysis['connection_timeline']:
                        if timeline_entry['cumulative_count'] >= required_connections:
                            timestep_to_required = timeline_entry['timestep']
                            break

                    if timestep_to_required is not None:
                        valid_candidates[sat_idx] = {
                            'analysis': analysis,
                            'timestep_to_required': timestep_to_required
                        }
                        print(f"  ✓ Satellite {sat_idx}: Found - {analysis['max_connections']}/{num_satellites-1} connections in {timestep_to_required} timesteps")

        # Step 4: If still no valid candidates, declare it impossible
        if not valid_candidates:
            if self.connect_to_all_satellites:
                print(f"\n⚠ IMPOSSIBLE: No satellite can connect to all {num_satellites-1} satellites within the given {len(self.adjacency_matrices) - start_matrix_index} timesteps")
            else:
                print(f"\n⚠ IMPOSSIBLE: No satellite can connect to {required_connections} satellites within the given {len(self.adjacency_matrices) - start_matrix_index} timesteps")

            print(f"   Falling back to best available satellite...")

            best_server = 0
            best_analysis = satellite_analysis[0]
            for sat_idx, analysis in satellite_analysis.items():
                if analysis['max_connections'] > best_analysis['max_connections']:
                    best_server = sat_idx
                    best_analysis = analysis

            self.selection_counts[best_server] += 1
            print(f"\n✓ Selected Server {best_server} ({self.satellite_names[best_server]}) (fallback)")
            print(f"  Will achieve {best_analysis['max_connections']}/{num_satellites-1} connections in {best_analysis['timestamps_to_max']} timestamps")
            return best_server, best_analysis['timestamps_to_max'], best_analysis

        # Step 5: Select best from valid candidates
        best_server = None
        best_candidate = None

        for sat_idx, candidate in valid_candidates.items():
            analysis = candidate['analysis']
            timestep_to_required = candidate['timestep_to_required']

            if best_server is None:
                best_server = sat_idx
                best_candidate = candidate
                continue

            is_better = False
            reason = ""

            # Primary criteria: Higher max connections
            if analysis['max_connections'] > best_candidate['analysis']['max_connections']:
                is_better = True
                reason = f"Higher max connections ({analysis['max_connections']} vs {best_candidate['analysis']['max_connections']})"

            # Secondary criteria: Same max connections but faster to reach required
            elif (analysis['max_connections'] == best_candidate['analysis']['max_connections'] and
                  timestep_to_required < best_candidate['timestep_to_required']):
                is_better = True
                reason = f"Same max connections but faster to required ({timestep_to_required} vs {best_candidate['timestep_to_required']} timestamps)"

            # Tertiary criteria: Load balancing
            elif (analysis['max_connections'] == best_candidate['analysis']['max_connections'] and
                  timestep_to_required == best_candidate['timestep_to_required'] and
                  self.selection_counts[sat_idx] < self.selection_counts[best_server]):
                is_better = True
                reason = f"Load balancing (selected {self.selection_counts[sat_idx]} vs {self.selection_counts[best_server]} times)"

            if is_better:
                best_server = sat_idx
                best_candidate = candidate
                print(f"  New best: Satellite {sat_idx} ({self.satellite_names[sat_idx]}) - {reason}")

        # Update selection count
        self.selection_counts[best_server] += 1
        best_analysis = best_candidate['analysis']

        print(f"\n✓ Selected Server {best_server} ({self.satellite_names[best_server]})")
        print(f"  Will achieve {best_analysis['max_connections']}/{num_satellites-1} connections")
        print(f"  Reaches {required_connections} required connections in {best_candidate['timestep_to_required']} timestamps")
        print(f"  Target satellites: {sorted(list(best_analysis['connected_satellites']))}")

        return best_server, best_analysis['timestamps_to_max'], best_analysis

    def find_best_redistribution_server(self, aggregation_server, start_matrix_index, max_lookahead=None):
        """
        Find the best server for redistribution using two-hop optimization.
        Uses configurable criteria:
        - connect_to_all_satellites: If True, server must connect to all satellites
        - minimum_connected_satellites: Minimum number of satellites to connect to
        - max_lookahead: Maximum timesteps to look ahead

        Calculates: (aggregation_server -> candidate) + (candidate -> all clients)
        Returns: (redistribution_server, time_to_reach_redistribution_server, redistribution_analysis)
        """
        # Use instance max_lookahead if not provided
        if max_lookahead is None:
            max_lookahead = self.max_lookahead

        num_satellites = len(self.satellite_names)

        # Determine required connections based on settings
        if self.connect_to_all_satellites:
            required_connections = num_satellites - 1  # All satellites except self
        else:
            required_connections = self.minimum_connected_satellites

        print(f"\n=== Finding Redistribution Server (from Aggregation Server {aggregation_server}) ===")
        print(f"Configuration: connect_to_all={self.connect_to_all_satellites}, "
              f"min_satellites={self.minimum_connected_satellites}, max_lookahead={max_lookahead}")
        print(f"Required connections: {required_connections}/{num_satellites-1}")

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

            # Analyze candidate's ability to redistribute to clients
            candidate_analysis = self.analyze_single_satellite(candidate_idx, candidate_start_index, max_lookahead)

            # Check if candidate meets minimum connectivity requirement
            if candidate_analysis['max_connections'] < required_connections:
                print(f"  ✗ Candidate {candidate_idx}: Only {candidate_analysis['max_connections']}/{required_connections} connections (insufficient)")
                continue

            # Find when it reaches required_connections
            timestep_to_required = None
            for timeline_entry in candidate_analysis['connection_timeline']:
                if timeline_entry['cumulative_count'] >= required_connections:
                    timestep_to_required = timeline_entry['timestep']
                    break

            if timestep_to_required is None or timestep_to_required > max_lookahead:
                print(f"  ✗ Candidate {candidate_idx}: Doesn't reach {required_connections} connections within {max_lookahead} timesteps")
                continue

            time_b = timestep_to_required
            total_time = time_a + time_b

            redistribution_analysis[candidate_idx] = {
                'time_a': time_a,  # aggregation -> candidate
                'time_b': time_b,  # candidate -> required connections
                'total_time': total_time,
                'max_connections': candidate_analysis['max_connections'],
                'connected_satellites': candidate_analysis['connected_satellites']
            }

            print(f"  ✓ Candidate {candidate_idx} ({self.satellite_names[candidate_idx]}): "
                  f"A={time_a} + B={time_b} = {total_time} timesteps, "
                  f"reaches {candidate_analysis['max_connections']}/{num_satellites-1} clients")

        # Step 3: If no valid candidates, search through ALL remaining timesteps
        if not redistribution_analysis:
            total_remaining = len(self.adjacency_matrices) - start_matrix_index
            print(f"\n⚠ No candidates meet criteria within {max_lookahead} timesteps")
            print(f"   Searching through all remaining {total_remaining} timesteps...")

            # Re-analyze paths with all remaining timesteps
            for candidate_idx in range(num_satellites):
                if candidate_idx == aggregation_server:
                    path_to_candidates[candidate_idx] = {
                        'reachable': True,
                        'timesteps': 0
                    }
                    continue

                connected_to_candidate = False
                timesteps_to_candidate = total_remaining

                for timesteps_ahead in range(total_remaining):
                    matrix_idx = start_matrix_index + timesteps_ahead
                    if matrix_idx >= len(self.adjacency_matrices):
                        break

                    _, matrix = self.adjacency_matrices[matrix_idx]

                    if matrix[aggregation_server][candidate_idx] == 1:
                        connected_to_candidate = True
                        timesteps_to_candidate = timesteps_ahead + 1
                        break

                path_to_candidates[candidate_idx] = {
                    'reachable': connected_to_candidate,
                    'timesteps': timesteps_to_candidate
                }

            # Re-analyze redistribution capability
            for candidate_idx in range(num_satellites):
                if not path_to_candidates[candidate_idx]['reachable']:
                    continue

                time_a = path_to_candidates[candidate_idx]['timesteps']
                candidate_start_index = start_matrix_index + time_a
                candidate_analysis = self.analyze_single_satellite(candidate_idx, candidate_start_index, total_remaining)

                if candidate_analysis['max_connections'] >= required_connections:
                    timestep_to_required = None
                    for timeline_entry in candidate_analysis['connection_timeline']:
                        if timeline_entry['cumulative_count'] >= required_connections:
                            timestep_to_required = timeline_entry['timestep']
                            break

                    if timestep_to_required is not None:
                        time_b = timestep_to_required
                        total_time = time_a + time_b

                        redistribution_analysis[candidate_idx] = {
                            'time_a': time_a,
                            'time_b': time_b,
                            'total_time': total_time,
                            'max_connections': candidate_analysis['max_connections'],
                            'connected_satellites': candidate_analysis['connected_satellites']
                        }
                        print(f"  ✓ Candidate {candidate_idx}: Found - A={time_a} + B={time_b} = {total_time} timesteps")

        # Step 4: If still no valid candidates, select the one with most connections
        if not redistribution_analysis:
            if self.connect_to_all_satellites:
                print(f"\n⚠ IMPOSSIBLE: No redistribution server can reach all {num_satellites-1} satellites within the given {len(self.adjacency_matrices) - start_matrix_index} timesteps")
            else:
                print(f"\n⚠ IMPOSSIBLE: No redistribution server can reach {required_connections} satellites within the given {len(self.adjacency_matrices) - start_matrix_index} timesteps")

            print(f"   Selecting server with most connections...")

            # Analyze all reachable candidates without requirement filter
            best_redist_server = aggregation_server
            best_max_connections = 0
            best_time_a = 0
            best_time_b = 0
            best_connected_sats = set()

            for candidate_idx in range(num_satellites):
                if not path_to_candidates[candidate_idx]['reachable']:
                    continue

                time_a = path_to_candidates[candidate_idx]['timesteps']
                candidate_start_index = start_matrix_index + time_a
                candidate_analysis = self.analyze_single_satellite(candidate_idx, candidate_start_index, total_remaining)

                # Find fastest time to reach maximum connections
                timestep_to_max = candidate_analysis['timestamps_to_max']
                max_connections = candidate_analysis['max_connections']

                # Select if this candidate has more connections
                if max_connections > best_max_connections:
                    best_redist_server = candidate_idx
                    best_max_connections = max_connections
                    best_time_a = time_a
                    best_time_b = timestep_to_max
                    best_connected_sats = candidate_analysis['connected_satellites']
                    print(f"  ✓ Candidate {candidate_idx}: Best so far - {max_connections}/{num_satellites-1} connections")

            print(f"\n✓ Selected Redistribution Server {best_redist_server} ({self.satellite_names[best_redist_server]}) (best available)")
            print(f"  Time A (aggregation→redistribution): {best_time_a} timestamps")
            print(f"  Time B (redistribution→max clients): {best_time_b} timestamps")
            print(f"  Total time: {best_time_a + best_time_b} timestamps")
            print(f"  Will reach {best_max_connections}/{num_satellites-1} clients")

            return best_redist_server, best_time_a, {
                'time_a': best_time_a,
                'time_b': best_time_b,
                'total_time': best_time_a + best_time_b,
                'max_connections': best_max_connections,
                'connected_satellites': best_connected_sats
            }

        # Step 5: Select best redistribution server
        # Primary: minimum total_time
        # Secondary: max_connections (in case of tie)
        # Tertiary: random selection (for ties)

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
        print(f"  Time B (redistribution→required clients): {best_analysis['time_b']} timestamps")
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

                # Safety timeout - when connect_to_all=true, continue with most connected
                if timestep_in_phase >= 20:
                    if self.connect_to_all_satellites:
                        print(f"  ⚠ TRANSMITTING timeout after {timestep_in_phase} timestamps")
                        print(f"    Cannot connect to all satellites - proceeding with {len(connected_satellites)}/{target_connections_count} connected")
                    else:
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
            if check_duration == 0:
                print(f"  ✓ Redistribution server is same as aggregation server (Server {redistribution_server})")
                print(f"  ✓ CHECK phase duration: 0 timesteps (no transfer needed)")
                # Note: No output written for 0-duration CHECK phase, goes directly to REDISTRIBUTION
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
