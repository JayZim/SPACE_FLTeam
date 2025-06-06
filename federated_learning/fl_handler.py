"""
Filename: fl_handler.py
Description: Reads algorithm data to perform preprocessing before running federated learning round.
Author: Nicholas Paul Candra
Date: 2025-05-12
Version: 2.6
Python Version: 3.10+
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

        # Track timestep number
        timestep_number = 1

        for timestamp, group in self.flam.groupby("time_stamp"):
            matrix_row = group.iloc[0]
            matrix_raw = matrix_row["federatedlearning_adjacencymatrix"]
            phase = str(matrix_row.get("phase", "TRAINING")).strip().upper()

           
            try:
                aggregator_id = int(str(matrix_row.get("aggregator_id", 0)).strip())
            except ValueError:
                aggregator_id = 0

            try:
                if isinstance(matrix_raw, str):
                    matrix = self.parse_adjacency_matrix(matrix_raw)
                else:
                    matrix = matrix_raw

                # Modified display format: Time first, then Timestep number
                print(f"\nTime: {timestamp}, Timestep: {timestep_number}, Round: {self.current_round}, Target Node: {aggregator_id}, Phase: {phase}")
                for row in matrix:
                    print(",".join(map(str, row)))

                # Set topology but don't reinitialize everything
                self.federated_learning.set_topology(matrix, aggregator_id)
                
                # Run the FL round with the current phase
                self.federated_learning.run_flam_round({"phase": phase})

                if phase == "TRANSMITTING":
                    self.current_round += 1

                # Increment timestep number
                timestep_number += 1

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
                matrix_lines = lines[i+1:i+6]
                entry = self.process_flam_block(current_header, matrix_lines)
                flam_entries.append(entry)
                i += 6
            else:
                i += 1

        return pd.DataFrame(flam_entries)

    def process_flam_block(self, header: str, matrix_lines: list) -> dict:
        import numpy as np

        header_parts = [h.strip() for h in header.split(',')]

        # Handle both formats: "Time: YYYY-MM-DD HH:MM:SS, Timestep: X" or "Timestep: X"
        time_stamp = None
        timestep = None
        
        if header.startswith("Time:"):
            # New format: "Time: YYYY-MM-DD HH:MM:SS, Timestep: X, ..."
            time_stamp = header_parts[0].split(':', 1)[1].strip()
            timestep = int(header_parts[1].split(':')[1].strip())
        else:
            # Old format: "Timestep: X, ..."
            timestep = int(header_parts[0].split(':')[1].strip())
            time_stamp = timestep  # Use timestep as time_stamp for backward compatibility

        # Extract other header information based on position
        # New format: Time, Timestep, Round, Target Node, Phase
        # Old format: Timestep, Round, Target Node, Phase
        if header.startswith("Time:"):
            # New format indices
            round_idx = 2
            target_idx = 3
            phase_idx = 4
        else:
            # Old format indices
            round_idx = 1
            target_idx = 2
            phase_idx = 3

        # Extract round number
        round_num = 1  # Default
        if len(header_parts) > round_idx and "Round" in header_parts[round_idx]:
            try:
                round_num = int(header_parts[round_idx].split(':')[1].strip())
            except (ValueError, IndexError):
                round_num = 1

        # Extract aggregator_id (Target Node)
        aggregator_id = 0  # Default
        if len(header_parts) > target_idx and "Target Node" in header_parts[target_idx]:
            try:
                aggregator_id = int(header_parts[target_idx].split(':')[1].strip())
            except (ValueError, IndexError):
                aggregator_id = 0

        # Extract Phase
        phase = 'TRAINING'  # Default
        if len(header_parts) > phase_idx and "Phase" in header_parts[phase_idx]:
            try:
                phase = header_parts[phase_idx].split(':')[1].strip().upper()
            except (ValueError, IndexError):
                phase = 'TRAINING'

        # Process matrix data
        matrix_data = []
        for line in matrix_lines:
            if line.strip():
                cleaned_line = line.replace(",", " ").strip()
                row = list(map(int, cleaned_line.split()))
                matrix_data.append(row)

        adjacency_matrix = np.array(matrix_data)
        
        # Determine satellite count from matrix size
        sat_count = len(adjacency_matrix) if len(adjacency_matrix) > 0 else 4

        return {
            'time_stamp': time_stamp,
            'satellite_count': sat_count,
            'satellite_names': [f"sat_{i}" for i in range(sat_count)],  # Generate default names
            'aggregator_flag': aggregator_id is not None,
            'aggregator_id': aggregator_id,  
            'federatedlearning_adjacencymatrix': adjacency_matrix,
            'phase': phase,
            'round': round_num
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
