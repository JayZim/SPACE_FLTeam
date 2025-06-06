"""
Filename: federated_learning.py
Description: Manage FLOWER Federated Learning epochs.
Author: Joshua Zimmerman
Date: 2025-05-07
Version: 1.0
Python Version: 3.10.0

Changelog:
- 2024-08-02: Initial creation.
- 2024-08-11: Added a Model Manager Class, exposed variables in standalone execution for: round count, client count and model/data set.
- 2024-08-11: Refactored to adhere to OOP principals
- 2024-09-16: Offloaded Model Manger to model.py, Refactored Standalone
- 2025-05-07: Remade file: Added TensorFlow + PyTorch conversion and saving functionality. Removed all Flower functionality
- 2025-05-09: Added processing time tracking for rounds and overall training

Usage:
Run this file directly to start a Multithreading instance of Tensorflow FL with the chosen number of clients rounds and model.

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from typing import List, Dict
import numpy as np
import sys
import os
import time
import threading
from datetime import datetime
import glob
from torch.utils.data import DataLoader, Subset, random_split
import torchvision

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from federated_learning.fl_output import FLOutput
from federated_learning.fl_visualization import FLVisualization

# Add path manager for universal path handling
try:
    from utilities.path_manager import get_synth_flams_dir
    use_path_manager = True
except ImportError:
    use_path_manager = False

class FederatedLearning:
    """Custom Federated Learning engine."""

    def __init__(self):
        self.num_rounds = 10
        self.num_clients = 5
        self.global_model = None
        self.client_data = []
        self.round_times = {}
        self.round_accuracies = []
        self.total_training_time = 0
        self.timestep = 0

    def set_num_rounds(self, rounds: int) -> None:
        """Set the number of federated learning rounds."""
        self.num_rounds = rounds

    def set_num_clients(self, clients: int) -> None:
        """Set the number of clients participating in federated learning."""
        self.num_clients = clients

    def set_model(self, model):
        """Set the model architecture for federated learning."""
        self.global_model = model

    def set_topology(self, matrix, aggregator_id):
        """Set network topology for this round"""
        self.current_topology = matrix
        self.current_aggregator = aggregator_id

    def reset_clients(self):
        """Reset client data for new simulation run."""
        self.client_data = []
        self.round_times = {}
        self.round_accuracies = []
        self.total_training_time = 0
        self.timestep = 0

    def initialize_data(self):
        """Initialize client data loaders"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        full_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        
        # Split dataset among clients
        client_datasets = random_split(full_dataset, [len(full_dataset) // self.num_clients] * self.num_clients)
        self.client_data = [DataLoader(dataset, batch_size=32, shuffle=True) for dataset in client_datasets]

    def initialize_model(self):
        """Initialize the global model"""
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc = nn.Linear(784, 10)

            def forward(self, x):
                x = x.view(x.size(0), -1)
                return self.fc(x)
        
        self.global_model = SimpleModel()

    def train_client(self, model, data_loader):
        """Train a client model on local data"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        
        correct = 0
        total = 0
        for _ in range(3):  # Local epochs
            for batch_idx, (data, target) in enumerate(data_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                # Calculate accuracy for this batch
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = correct / total if total > 0 else 0
        return model.state_dict(), accuracy

    def federated_averaging(self, client_models: List[dict]):
        """Perform federated averaging to update the global model."""
        global_dict = self.global_model.state_dict()
        for key in global_dict.keys():
            global_dict[key] = torch.stack([client_model[key].float() for client_model in client_models], 0).mean(0)
        self.global_model.load_state_dict(global_dict)

    def get_round_metrics(self) -> Dict:
        """Return round metrics for analysis"""
        return self.round_times
    
    def get_latest_flam_file(self):
        """Get the latest generated FLAM file path"""
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

    def run_flam_round(self, flam_entry):
        """Run a single FLAM-based round step based on the phase."""
        phase = flam_entry.get("phase", "TRAINING")
        print(f"\n[FLAM Phase: {phase}]")

        if phase == "TRAINING":
            client_models = []
            round_correct = 0
            round_total = 0
            start_time = time.time()

            for client_id, data_loader in enumerate(self.client_data):
                client_model = type(self.global_model)()
                client_model.load_state_dict(self.global_model.state_dict())
                print(f"Training client {client_id + 1}...")

                state_dict, accuracy = self.train_client(client_model, data_loader)
                client_models.append(state_dict)
                round_correct += accuracy * len(data_loader.dataset)
                round_total += len(data_loader.dataset)

            self._pending_client_models = client_models
            self._pending_round_correct = round_correct
            self._pending_round_total = round_total
            self._pending_round_start_time = start_time

        elif phase == "TRANSMITTING":
            if hasattr(self, "_pending_client_models"):
                self.federated_averaging(self._pending_client_models)

                round_time = time.time() - self._pending_round_start_time
                round_accuracy = self._pending_round_correct / self._pending_round_total if self._pending_round_total > 0 else 0

                print(f"Completed round with accuracy: {round_accuracy:.2%} in {round_time:.2f} seconds")

                # Clean up
                del self._pending_client_models
                del self._pending_round_correct
                del self._pending_round_total
                del self._pending_round_start_time
                
    def load_flam_schedule(self, flam_path):
        """Parse FLAM CSV and return a list of dicts with timestep, phase, etc."""
        schedule = []
        with open(flam_path, "r") as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("Timestep:"):
                # Parse header
                parts = line.split(", ")
                timestep = int(parts[0].split(":")[1])
                phase = parts[3].split(":")[1].strip()
                schedule.append({"timestep": timestep, "phase": phase})
                # Skip the adjacency matrix (5 lines)
                i += 6
            else:
                i += 1
        return schedule

    def run(self, flam_path=None):
        """Run the federated learning process, synchronized with FLAM phases."""
        self.initialize_data()
        self.initialize_model()
        total_start_time = time.time()
        round_accuracies = []

        # Load FLAM schedule - auto-detect latest if not provided
        flam_schedule = []
        if flam_path is None:
            try:
                flam_path = self.get_latest_flam_file()
                print(f"[INFO] Auto-detected latest FLAM file: {os.path.basename(flam_path)}")
            except FileNotFoundError as e:
                print(f"[ERROR] {e}")
                print("[INFO] Running without FLAM schedule...")
                return
        
        if flam_path:
            flam_schedule = self.load_flam_schedule(flam_path)
            print(f"Loaded FLAM schedule with {len(flam_schedule)} timesteps.")

        # Start the timestep counter in a background thread
        stop_event = threading.Event()
        def timestep_counter():
            while not stop_event.is_set():
                time.sleep(12) # Increment every 12 seconds
                self.timestep += 1
                print(f"Timestep: {self.timestep}")

        counter_thread = threading.Thread(target=timestep_counter)
        counter_thread.daemon = True
        counter_thread.start()

        try:
            flam_idx = 0
            round_num = 0
            while flam_idx < len(flam_schedule) and round_num < self.num_rounds:
                flam_entry = flam_schedule[flam_idx]
                # Wait until the global timestep matches the FLAM timestep exactly
                while self.timestep < flam_entry["timestep"]:
                    time.sleep(0.5)
                if self.timestep == flam_entry["timestep"]:
                    phase = flam_entry["phase"]
                    print(f"\nFLAM Timestep {flam_entry['timestep']} Phase: {phase}")

                    if phase == "TRAINING":
                        # Only train if we haven't already trained for this round
                        if not hasattr(self, "_pending_client_models"):
                            print(f"Training round {round_num + 1}...")
                            client_models = []
                            client_accuracies = []
                            round_correct = 0
                            round_total = 0
                            for client_id, data_loader in enumerate(self.client_data):
                                client_model = type(self.global_model)()
                                client_model.load_state_dict(self.global_model.state_dict())
                                print(f"Training client {client_id + 1}...")
                                state_dict, accuracy = self.train_client(client_model, data_loader)
                                client_models.append(state_dict)
                                client_accuracies.append(accuracy)
                                round_correct += accuracy * len(data_loader.dataset)
                                round_total += len(data_loader.dataset)
                            self._pending_client_models = client_models
                            self._pending_round_correct = round_correct
                            self._pending_round_total = round_total
                            self._pending_round_start_time = time.time()
                        else:
                            print(f"Already trained for round {round_num + 1}, waiting for TRANSMITTING phase...")
                    elif phase == "TRANSMITTING":
                        if hasattr(self, "_pending_client_models"):
                            self.federated_averaging(self._pending_client_models)
                            print("Global model updated via federated averaging.")
                            round_time = time.time() - self._pending_round_start_time
                            round_accuracy = self._pending_round_correct / self._pending_round_total if self._pending_round_total > 0 else 0
                            self.round_times[f"round_{round_num + 1}"] = round_time
                            round_accuracies.append(round_accuracy)
                            print(f"Round {round_num + 1} completed in {round_time:.2f} seconds with average accuracy {round_accuracy:.2%}.")
                            round_num += 1
                            # Clean up
                            del self._pending_client_models
                            del self._pending_round_correct
                            del self._pending_round_total
                            del self._pending_round_start_time
                flam_idx += 1

            self.total_training_time = time.time() - total_start_time
            self.round_accuracies = round_accuracies

            print(f"\nFederated learning process completed in {self.total_training_time:.2f} seconds.")
            print("\nRound-wise processing times and accuracies:")
            for idx, round_time in self.round_times.items():
                print(f"{idx}: {round_time:.2f} seconds, Accuracy: {round_accuracies[int(idx.split('_')[1]) - 1]:.2%}")
            print(f"Average round time: {self.total_training_time/self.num_rounds:.2f} seconds")
        finally:
            stop_event.set()
            counter_thread.join(timeout=2)
            print("Counter thread stopped.")
            print("Total timestep count:", self.timestep)

if __name__ == "__main__":
    """Standalone entry point for testing FederatedLearning."""
    # Default customization values
    num_rounds = 5
    num_clients = 3
    model_type = "SimpleCNN"
    data_set = "MNIST"

    # Create timestamped run directory under results_from_output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = os.path.join(os.path.dirname(__file__), "results_from_output")
    run_dir = os.path.join(results_root, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    # Initialize and run the FederatedLearning instance
    fl_instance = FederatedLearning()
    fl_instance.set_num_rounds(num_rounds)
    fl_instance.set_num_clients(num_clients)
    
    # Let it auto-detect the latest FLAM file instead of hardcoding
    fl_instance.run(flam_path=None)

    # Evaluate the model
    output = FLOutput()
    output.evaluate_model(fl_instance.global_model, fl_instance.total_training_time)

    # Add timing and accuracy metrics
    timing_metrics = fl_instance.get_round_metrics()
    for metric_name, value in timing_metrics.items():
        output.add_metric(metric_name, value)

    # Add round accuracies explicitly
    output.add_metric("round_accuracies", fl_instance.round_accuracies)

    # Save results into this run folder
    log_file = os.path.join(run_dir, f"results_{timestamp}.log")
    metrics_file = os.path.join(run_dir, f"metrics_{timestamp}.json")
    model_file = os.path.join(run_dir, f"model_{timestamp}.pt")

    # Log results and save files
    output.log_result(log_file)
    output.write_to_file(metrics_file, format="json")
    output.save_model(model_file)

    print("\nResults have been saved to:")
    print("Log file:", log_file)
    print("Metrics file:", metrics_file)
    print("Model file:", model_file)

    # Generate visualizations
    print("\nGenerating visualizations...")
    viz = FLVisualization(results_dir=run_dir)
    viz.visualize_from_json(metrics_file)
    print(f"Visualizations saved under {run_dir}")
