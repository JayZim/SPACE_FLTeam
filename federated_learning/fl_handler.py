"""
Filename: fl_handler.py
Description: Reads algorithm data to perform preprocessing before running federated learning round.
Author: Nicholas Paul Candra
Date: 2025-05-12
Version: 1.3
Python Version: 3.10+
"""

import sys
import os
import ast
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from interfaces.handler import Handler
from federated_learning.fl_core import FederatedLearning


class FLHandler(Handler):
    def __init__(self, fl_core: FederatedLearning):
        super().__init__()
        self.federated_learning = fl_core

    def parse_input(self, file):
        return self.parse_file(file)

    def extract_client_addresses(self, flam):
        return flam.get("client_addresses", []) if isinstance(flam, dict) else []

    def run_module(self):
        if self.flam is not None:
            print("[INFO] Parsing FLAM metadata for simulation...")
            self.run_flam_based_simulation()
        else:
            print("[INFO] Running default Federated Learning Core...")
            self.federated_learning.run()

    def run_flam_based_simulation(self):
        print("[DEBUG] Parsed FLAM Columns:", self.flam.columns.tolist())

        for timestamp, group in self.flam.groupby("time_stamp"):
            print(f"\n[FLAM ROUND] \nTime: {timestamp}")
            print(f"Satellite Count: {group['satellite_count'].iloc[0]}")

            aggregator_row = group[group["aggregator_flag"] == True]
            aggregator = "True" if not aggregator_row.empty else "None"
            print(f"Aggregator: {aggregator}")

            matrix_row = aggregator_row.iloc[0] if not aggregator_row.empty else group.iloc[0]
            matrix_str = matrix_row["federatedlearning_adjacencymatrix"]

            try:
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

                matrix = ast.literal_eval(cleaned)

                # Matrix display
                print("Adjacency Matrix:")
                sat_count = len(matrix)
                header = "   | " + " | ".join(f"S{i+1}" for i in range(sat_count))
                print(header)
                print("-" * len(header))
                for i, links in enumerate(matrix):
                    row = f"S{i+1} | " + " | ".join(str(val) for val in links)
                    print(row)

                print("\nConnections:")
                for i, links in enumerate(matrix):
                    connected = [j+1 for j, val in enumerate(links) if val == 1]
                    print(f" - Satellite {i+1} connects to: {connected if connected else 'None'}")

            except Exception as e:
                print(f"[WARN] Couldn't parse adjacency matrix: {e}")
                continue

            # Reset client data before each round
            self.federated_learning.reset_clients()

            self.federated_learning.run()


def load_flam_txt(filepath: str) -> pd.DataFrame:
    with open(filepath, 'r') as f:
        lines = f.readlines()

    flam_entries = []

    for line_num, line in enumerate(lines[1:], start=2):  # Skip header
        fields = line.strip().split(None, 5)

        if len(fields) < 6:
            print(f"[WARN] Skipping malformed line {line_num}: not enough fields")
            continue

        time_stamp = fields[0] + ' ' + fields[1]
        satellite_count = int(fields[2])
        satellite_name = fields[3]
        aggregator_flag = fields[4]
        matrix_raw = " ".join(fields[5:])  # All remaining fields are part of matrix

        # Remove 'True' or 'False' if they leaked into matrix_raw
        if matrix_raw.lower().startswith("true "):
            matrix_raw = matrix_raw[5:].strip()
        elif matrix_raw.lower().startswith("false "):
            matrix_raw = matrix_raw[6:].strip()

        # Normalize aggregator flag
        is_aggregator = False
        if str(aggregator_flag).lower() in ['true', '1', 't', 'y', 'yes']:
            is_aggregator = True
        elif str(aggregator_flag).lower() in ['false', '0', 'f', 'n', 'no', 'none']:
            is_aggregator = False
        else:
            try:
                is_aggregator = bool(int(aggregator_flag))
            except (ValueError, TypeError):
                is_aggregator = False

        flam_entries.append({
            "time_stamp": time_stamp,
            "satellite_count": satellite_count,
            "satellite_name": None if str(satellite_name).strip().lower() == "none" else satellite_name,
            "aggregator_flag": is_aggregator,
            "federatedlearning_adjacencymatrix": matrix_raw.strip()
        })

    return pd.DataFrame(flam_entries)


if __name__ == "__main__":
    print("[START] Initializing Federated Learning Handler...")

    fl_core = FederatedLearning()
    fl_core.set_num_clients(3)
    fl_core.set_num_rounds(1)

    # Inject reset_clients method to the instance if not defined
    if not hasattr(fl_core, 'reset_clients'):
        def reset_clients():
            fl_core.client_data = []
        fl_core.reset_clients = reset_clients

    handler = FLHandler(fl_core)

    handler.flam = load_flam_txt(
        r"C:\Users\Nicholas Paul Candra\GitHub\SPACE_FLTeam\FLAM.txt"
    )

    handler.run_module()
    print("\n[DONE] FL process complete.")
