"""
Filename: fl_handler.py
Description: Reads algorithm data to perform preprocessing before running federated learning round.
Author: Nicholas Paul Candra & Stephen Zeng
Date: 2025-10-10
Version: 2.7
Python Version: 3.10+

Changelog:
- 2025-10-10: Added path manager for universal path handling.
- 2025-10-10: Added auto-detection of latest FLAM file.
- 2025-10-10: Added dual-format support for FLAM file parsing.
- 2025-10-10: Added support for old and new FLAM file formats.
- 2025-10-10: Added support for custom timesteps.
- 2025-10-10: Added support for custom duration.
"""

import sys
import os
import ast
import pandas as pd
import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from interfaces.handler import Handler
from federated_learning.fl_core import FederatedLearning

# Add path manager for universal path handling
try:
    from utilities.path_manager import get_synth_flams_dir
    use_path_manager = True
except ImportError:
    use_path_manager = False

class FLHandler(Handler):
    def __init__(self, fl_core: FederatedLearning):
        super().__init__()
        self.federated_learning = fl_core
        self.current_round = 1

    def parse_input(self, file):
        return self.parse_file(file)

    def get_latest_flam_file(self):
        """获取最新生成的FLAM文件路径"""
        if use_path_manager:
            csv_dir = get_synth_flams_dir()
            csv_files = list(csv_dir.glob("flam_*.csv"))
        else:
            # Use backup path
            script_dir = os.path.dirname(os.path.abspath(__file__))
            csv_dir_str = os.path.join(script_dir, "..", "synth_FLAMs")
            if os.path.exists(csv_dir_str):
                csv_files = [os.path.join(csv_dir_str, f) for f in os.listdir(csv_dir_str) 
                           if f.startswith('flam_') and f.endswith('.csv')]
            else:
                csv_files = []
        
        if not csv_files:
            raise FileNotFoundError("No FLAM files found in synth_FLAMs directory")
        
        # Return the most recently created file
        if use_path_manager:
            latest_file = max(csv_files, key=lambda x: x.stat().st_ctime)
            return str(latest_file)
        else:
            latest_file = max(csv_files, key=lambda x: os.path.getctime(x))
            return latest_file

    def run_module(self):
        """Run FL module, automatically detect the latest FLAM file or use pre-loaded FLAM data"""
        if self.flam is not None:
            print("[INFO] Using pre-loaded FLAM data for simulation...")
            print(self.flam.head())
            self.run_flam_based_simulation()
        else:
            # Try to automatically get the latest FLAM file
            try:
                latest_flam_path = self.get_latest_flam_file()
                print(f"[INFO] Auto-detected latest FLAM file: {os.path.basename(latest_flam_path)}")
                self.flam = self.load_flam_file(latest_flam_path)
                self.run_flam_based_simulation()
            except FileNotFoundError:
                print("[INFO] No FLAM files found, running default Federated Learning Core...")
                self.federated_learning.run()

    def run_flam_based_simulation(self):
        print("[DEBUG] Parsed FLAM Columns:", self.flam.columns.tolist())
        
        # Initialize FL system only once
        if not hasattr(self, '_fl_initialized'):
            print("[INFO] Initializing FL system for first time...")
            self.federated_learning.initialize_data()
            self.federated_learning.initialize_model()
            self._fl_initialized = True

        for _, row in self.flam.iterrows():
            matrix_raw = row["federatedlearning_adjacencymatrix"]
            phase = str(row.get("phase", "TRAINING")).strip().upper()
            time_stamp = row.get("time_stamp", "Unknown")
            timestep = row.get("timestep", 1)
            round_num = row.get("round", self.current_round)
            aggregator_id = row.get("aggregator_id", 0)

            try:
                if isinstance(matrix_raw, str):
                    matrix = self.parse_adjacency_matrix(matrix_raw)
                else:
                    matrix = matrix_raw

                # Simplified display format focusing on essential information
                print(f"\nTime: {time_stamp}, Timestep: {timestep}, Round: {round_num}, Phase: {phase}")
                print(f"Aggregation Server: {aggregator_id}, Target Node: {aggregator_id}")
                
                # Display matrix
                for matrix_row in matrix:
                    print(",".join(map(str, matrix_row)))

                # Set topology and run FL round
                self.federated_learning.set_topology(matrix, aggregator_id)
                
                # Prepare simplified metadata for FL core
                flam_metadata = {
                    "phase": phase,
                    "timestep": timestep,
                    "round": round_num,
                    "aggregator_id": aggregator_id
                }
                
                # Run the FL round with simplified metadata
                self.federated_learning.run_flam_round(flam_metadata)

                # Update current round based on FLAM data
                if round_num != self.current_round:
                    self.current_round = round_num

            except Exception as e:
                print(f"[WARN] Error in FLAM round: {e}")
                continue

    def parse_adjacency_matrix(self, matrix_str):
        cleaned = (
            matrix_str.replace('\n', '')
                      .replace('\r', '')
                      .replace('\x00', '')
                      .replace('], [', '],[')
                      .strip().rstrip(',')
        )

        if not cleaned.startswith('[['):
            cleaned = f'[{cleaned}]'

        left, right = cleaned.count('['), cleaned.count(']')
        if left > right:
            cleaned += ']' * (left - right)
        elif right > left:
            cleaned = '[' * (right - left) + cleaned

        return ast.literal_eval(cleaned)

    def load_flam_file(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()

        if ext in ['.txt', '.csv']:
            return self._load_flam(file_path)
        elif ext == '.json':
            df = pd.read_json(file_path)
            df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
            return df
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    def _load_flam(self, file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()

        flam_entries = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            # Support both old "Timestep:" and new "Time:" formats
            if line.startswith("Timestep:") or line.startswith("Time:"):
                current_header = line
                
                # Determine matrix size dynamically
                # Look for the next non-empty line after header to determine matrix size
                matrix_start = i + 1
                matrix_lines = []
                j = matrix_start
                
                # Count consecutive non-empty lines that look like matrix data
                while j < len(lines) and lines[j].strip():
                    line_content = lines[j].strip()
                    # Check if line contains comma-separated integers (matrix data)
                    if ',' in line_content and all(c.isdigit() or c in ', ' for c in line_content):
                        matrix_lines.append(lines[j])
                        j += 1
                    else:
                        break
                
                # If no matrix lines found, skip this entry
                if not matrix_lines:
                    i += 1
                    continue
                
                entry = self.process_flam_block(current_header, matrix_lines)
                flam_entries.append(entry)
                i = j  # Move to next header
            else:
                i += 1

        return pd.DataFrame(flam_entries)

    def process_flam_block(self, header: str, matrix_lines: list) -> dict:
        import numpy as np

        header_parts = [h.strip() for h in header.split(',')]

        # Initialize default values
        time_stamp = None
        timestep = None
        round_num = 1
        aggregator_id = 0
        redistribution_server = None
        target_node = 0
        phase = 'TRAINING'
        phase_length = 1
        timestep_in_phase = 1
        current_connections = 0
        cumulative_connections = "0/0"
        connected_sats = []
        missing_sats = []
        target_sats = []
        phase_complete = False

        # Parse header fields dynamically
        for part in header_parts:
            if ':' in part:
                key, value = part.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                if key == "Time":
                    time_stamp = value
                elif key == "Timestep":
                    try:
                        timestep = int(value)
                    except ValueError:
                        timestep = 1
                elif key == "Round":
                    try:
                        round_num = int(value)
                    except ValueError:
                        round_num = 1
                elif key == "Phase":
                    phase = value.upper()
                elif key == "Aggregation Server":
                    try:
                        aggregator_id = int(value)
                    except ValueError:
                        aggregator_id = 0
                elif key == "Redistribution Server":
                    if value != "TBD":
                        try:
                            redistribution_server = int(value)
                        except ValueError:
                            redistribution_server = None
                    else:
                        redistribution_server = None
                elif key == "Target Node":
                    try:
                        target_node = int(value)
                    except ValueError:
                        target_node = 0
                elif key == "Phase Length":
                    try:
                        phase_length = int(value)
                    except ValueError:
                        phase_length = 1
                elif key == "Timestep in Phase":
                    try:
                        timestep_in_phase = int(value)
                    except ValueError:
                        timestep_in_phase = 1
                elif key == "Current Connections":
                    try:
                        current_connections = int(value)
                    except ValueError:
                        current_connections = 0
                elif key == "Cumulative Connections":
                    cumulative_connections = value
                elif key == "Connected Sats":
                    # Parse list format: [1, 3, 4, 6, 7]
                    if value.startswith('[') and value.endswith(']'):
                        sat_list_str = value[1:-1]  # Remove brackets
                        try:
                            connected_sats = [int(x.strip()) for x in sat_list_str.split(',') if x.strip()]
                        except ValueError:
                            connected_sats = []
                    else:
                        connected_sats = []
                elif key == "Missing Sats":
                    # Parse list format: [5] or []
                    if value.startswith('[') and value.endswith(']'):
                        sat_list_str = value[1:-1]  # Remove brackets
                        try:
                            missing_sats = [int(x.strip()) for x in sat_list_str.split(',') if x.strip()]
                        except ValueError:
                            missing_sats = []
                    else:
                        missing_sats = []
                elif key == "Target Sats":
                    # Parse list format: [1, 3, 4, 5, 6, 7]
                    if value.startswith('[') and value.endswith(']'):
                        sat_list_str = value[1:-1]  # Remove brackets
                        try:
                            target_sats = [int(x.strip()) for x in sat_list_str.split(',') if x.strip()]
                        except ValueError:
                            target_sats = []
                    else:
                        target_sats = []
                elif key == "Phase Complete":
                    phase_complete = value.lower() == 'true'

        # Handle backward compatibility for old format
        if time_stamp is None and timestep is not None:
            time_stamp = str(timestep)

        # Process matrix data
        matrix_data = []
        for line in matrix_lines:
            if line.strip():
                # Handle comma-separated format
                row = list(map(int, line.strip().split(',')))
                matrix_data.append(row)

        adjacency_matrix = np.array(matrix_data)
        
        # Determine satellite count from matrix size
        sat_count = len(adjacency_matrix) if len(adjacency_matrix) > 0 else 8

        return {
            'time_stamp': time_stamp,
            'timestep': timestep,
            'satellite_count': sat_count,
            'satellite_names': [f"sat_{i}" for i in range(sat_count)],
            'aggregator_flag': aggregator_id is not None,
            'aggregator_id': aggregator_id,
            'redistribution_server': redistribution_server,
            'target_node': target_node,
            'federatedlearning_adjacencymatrix': adjacency_matrix,
            'phase': phase,
            'round': round_num,
            'phase_length': phase_length,
            'timestep_in_phase': timestep_in_phase,
            'current_connections': current_connections,
            'cumulative_connections': cumulative_connections,
            'connected_sats': connected_sats,
            'missing_sats': missing_sats,
            'target_sats': target_sats,
            'phase_complete': phase_complete
        }


if __name__ == "__main__":
    print("[START] Initializing Federated Learning Handler...")

    fl_core = FederatedLearning()
    fl_core.set_num_clients(3)
    fl_core.set_num_rounds(1)

    if not hasattr(fl_core, 'reset_clients'):
        def reset_clients():
            fl_core.client_data = []
        fl_core.reset_clients = reset_clients

    handler = FLHandler(fl_core)

    # No longer hardcode file paths, let handler auto-detect the latest FLAM file
    print("[INFO] FL Handler will auto-detect latest FLAM file...")
    handler.run_module()
    print("\n[DONE] FL process complete.")
