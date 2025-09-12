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
- 2025-10-04: Integrated FLAM file parsing and dynamic topology adaptation

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
import re

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from federated_learning.fl_output import FLOutput
from federated_learning.fl_visualization import FLVisualization
from federated_learning.model_evaluation import EnhancedModelEvaluationModule

# Add path manager for universal path handling
try:
    from utilities.path_manager import get_synth_flams_dir
    use_path_manager = True
except ImportError:
    use_path_manager = False

class FederatedLearning:
    """Custom Federated Learning engine."""

    # --- Parameter Server Object (Inner Class) ---
    class ParameterServer:
        def __init__(self, model):
            self.global_model = model
            self.updates = []

        def receive_update(self, client_state_dict):
            self.updates.append(client_state_dict)

        def aggregate(self):
            if not self.updates:
                return
            global_dict = self.global_model.state_dict()
            for key in global_dict.keys():
                global_dict[key] = torch.stack([update[key].float() for update in self.updates], 0).mean(0)
            self.global_model.load_state_dict(global_dict)
            self.updates = []

        def get_global_model(self):
            return self.global_model

    # --- Client Object (Inner Class) ---
    class Client:
        def __init__(self, client_id, model, data_loader):
            self.client_id = client_id
            self.model = model
            self.data_loader = data_loader

        def train(self, epochs=3):
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(self.model.parameters(), lr=0.01)
            correct = 0
            total = 0
            for _ in range(epochs):
                for data, target in self.data_loader:
                    optimizer.zero_grad()
                    output = self.model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            accuracy = correct / total if total > 0 else 0
            return self.model.state_dict(), accuracy

        def update_model(self, global_state_dict):
            self.model.load_state_dict(global_state_dict)

    def __init__(self, enable_model_evaluation=True):
        self.num_rounds = 10
        self.num_clients = 5
        self.global_model = None
        self.client_data = []
        self.round_times = {}
        self.round_accuracies = []
        self.total_training_time = 0
        self.model_evaluation_enabled = enable_model_evaluation
        self.model_eval_module = None
        self.selected_model_name = None
        self.model_evaluation_history = []
        
        # Initialize model evaluation module if enabled
        if self.model_evaluation_enabled:
            try:
                self.model_eval_module = EnhancedModelEvaluationModule()
                print("✓ Model evaluation module initialized")
            except Exception as e:
                print(f"Warning: Failed to initialize model evaluation module: {e}")
                self.model_evaluation_enabled = False

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

    def initialize_model(self, model_name=None, auto_select=True, interactive_mode=True):
        """Initialize the global model with optional model selection"""
        if self.model_evaluation_enabled and auto_select and model_name is None:
            try:
                # Show available models and let user choose
                if interactive_mode:
                    available_models = self.model_eval_module.registry.list_models()
                    print(f"\n{'='*60}")
                    print("Available Models for Federated Learning:")
                    print(f"{'='*60}")
                    
                    for i, model in enumerate(available_models, 1):
                        model_info = self.model_eval_module.registry.get_model(model)
                        print(f"{i}. {model}")
                        print(f"   Description: {model_info.description}")
                        print(f"   Category: {model_info.category}, Complexity: {model_info.complexity}")
                        print()
                    
                    print(f"{len(available_models) + 1}. Auto-select best model")
                    print(f"{len(available_models) + 2}. Use default model")
                    print(f"{'='*60}")
                    
                    while True:
                        try:
                            choice = input(f"Please select a model (1-{len(available_models) + 2}): ").strip()
                            choice_num = int(choice)
                            
                            if 1 <= choice_num <= len(available_models):
                                # User selected a specific model
                                selected_model_name = available_models[choice_num - 1]
                                model_info = self.model_eval_module.registry.get_model(selected_model_name)
                                self.global_model = model_info.model_class(**model_info.parameters)
                                self.selected_model_name = selected_model_name
                                print(f"✓ Using selected model: {selected_model_name}")
                                break
                            elif choice_num == len(available_models) + 1:
                                # User chose auto-select
                                if self.client_data:
                                    criterion = nn.CrossEntropyLoss()
                                    selected_name, selected_model = self.model_eval_module.select_model_for_fl(
                                        self.client_data[0], criterion
                                    )
                                    self.selected_model_name = selected_name
                                    self.global_model = selected_model
                                    print(f"✓ Auto-selected model: {selected_name}")
                                else:
                                    self.global_model = self._create_default_model()
                                    self.selected_model_name = "DefaultModel"
                                    print("✓ Using default model (no data available for auto-selection)")
                                break
                            elif choice_num == len(available_models) + 2:
                                # User chose default model
                                self.global_model = self._create_default_model()
                                self.selected_model_name = "DefaultModel"
                                print("✓ Using default model")
                                break
                            else:
                                print(f"Invalid choice. Please enter a number between 1 and {len(available_models) + 2}")
                        except ValueError:
                            print("Invalid input. Please enter a number.")
                        except KeyboardInterrupt:
                            print("\nOperation cancelled. Using default model.")
                            self.global_model = self._create_default_model()
                            self.selected_model_name = "DefaultModel"
                            break
                else:
                    # Non-interactive auto-select
                    if self.client_data:
                        criterion = nn.CrossEntropyLoss()
                        selected_name, selected_model = self.model_eval_module.select_model_for_fl(
                            self.client_data[0], criterion
                        )
                        self.selected_model_name = selected_name
                        self.global_model = selected_model
                        print(f"✓ Auto-selected model: {selected_name}")
                    else:
                        self.global_model = self._create_default_model()
                        self.selected_model_name = "DefaultModel"
            except Exception as e:
                print(f"Model selection failed: {e}. Using default model.")
                self.global_model = self._create_default_model()
                self.selected_model_name = "DefaultModel"
        elif self.model_evaluation_enabled and model_name:
            # Use specified model
            try:
                model_info = self.model_eval_module.registry.get_model(model_name)
                if model_info:
                    self.global_model = model_info.model_class(**model_info.parameters)
                    self.selected_model_name = model_name
                    print(f"✓ Using specified model: {model_name}")
                else:
                    raise ValueError(f"Model {model_name} not found in registry")
            except Exception as e:
                print(f"Failed to use specified model: {e}. Using default model.")
                self.global_model = self._create_default_model()
                self.selected_model_name = "DefaultModel"
        else:
            # Use default model
            self.global_model = self._create_default_model()
            self.selected_model_name = "DefaultModel"
    
    def _create_default_model(self):
        """Create a default model as fallback"""
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc = nn.Linear(784, 10)

            def forward(self, x):
                x = x.view(x.size(0), -1)
                return self.fc(x)
        
        return SimpleModel()

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
    
    def get_model_evaluation_summary(self) -> Dict:
        """Get summary of model evaluation results"""
        if not self.model_evaluation_enabled:
            return {"message": "Model evaluation not enabled"}
        
        return {
            "selected_model": self.selected_model_name,
            "evaluation_history": self.model_evaluation_history,
            "available_models": self.model_eval_module.registry.list_models() if self.model_eval_module else [],
            "model_evaluation_enabled": self.model_evaluation_enabled
        }
    
    def enable_model_evaluation(self):
        """Enable model evaluation functionality"""
        if not self.model_evaluation_enabled:
            try:
                self.model_eval_module = EnhancedModelEvaluationModule()
                self.model_evaluation_enabled = True
                print("✓ Model evaluation enabled")
            except Exception as e:
                print(f"Failed to enable model evaluation: {e}")
    
    def disable_model_evaluation(self):
        """Disable model evaluation functionality"""
        self.model_evaluation_enabled = False
        self.model_eval_module = None
        print("Model evaluation disabled")
    
    def list_available_models(self) -> List[str]:
        """List all available models in the registry"""
        if self.model_evaluation_enabled and self.model_eval_module:
            return self.model_eval_module.registry.list_models()
        return []
    
    def register_custom_model(self, name: str, model_class, parameters: Dict, 
                            description: str, category: str, complexity: str):
        """Register a custom model"""
        if self.model_evaluation_enabled and self.model_eval_module:
            self.model_eval_module.register_custom_model(
                name, model_class, parameters, description, category, complexity
            )
            print(f"✓ Custom model '{name}' registered")
        else:
            print("Model evaluation not enabled. Cannot register custom model.")
    
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
                
    # --- FLAM-related code (commented out) ---
    # The following method and all FLAM schedule/timestep logic are commented out.
    # They were used to synchronize FL rounds with an external FLAM schedule.

    # def load_flam_schedule(self, flam_path):
    #     """Parse FLAM CSV and return a list of dicts with timestep, phase, etc."""
    #     schedule = []
    #     with open(flam_path, "r") as f:
    #         lines = f.readlines()
    #     i = 0
    #     while i < len(lines):
    #         line = lines[i].strip()
    #         if line.startswith("Timestep:"):
    #             # Parse header
    #             parts = line.split(", ")
    #             timestep = int(parts[0].split(":")[1])
    #             phase = parts[3].split(":")[1].strip()
    #             schedule.append({"timestep": timestep, "phase": phase})
    #             # Skip the adjacency matrix (5 lines)
    #             i += 6
    #         else:
    #             i += 1
    #     return schedule

    # --- End FLAM-related code ---

    @staticmethod
    def parse_flam_file(flam_path):
        """
        Parse the FLAM CSV file and yield a dict for each timestep:
        {
            'timestep': int,
            'round': int,
            'target_node': int,
            'phase': str,
            'connected_sats': List[int],
            'missing_sats': List[int],
            'adjacency': List[List[int]]
        }
        """
        with open(flam_path, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("Time:"):
                # Parse header
                header = line
                timestep = int(re.search(r'Timestep: (\d+)', header).group(1))
                round_num = int(re.search(r'Round: (\d+)', header).group(1))
                target_node = int(re.search(r'Target Node: (\d+)', header).group(1))
                phase = re.search(r'Phase: ([A-Z]+)', header).group(1)
                connected_sats = eval(re.search(r'Connected Sats: (\[.*?\])', header).group(1))
                missing_sats = eval(re.search(r'Missing Sats: (\[.*?\])', header).group(1))
                adjacency = []
                for j in range(i+1, i+1+8):  # 8 clients/nodes
                    adjacency.append([int(x) for x in lines[j].strip().split(',')])
                yield {
                    'timestep': timestep,
                    'round': round_num,
                    'target_node': target_node,
                    'phase': phase,
                    'connected_sats': connected_sats,
                    'missing_sats': missing_sats,
                    'adjacency': adjacency
                }
                i += 8
            i += 1


    def run(self, flam_path=None, model_name=None, auto_select_model=True, interactive_mode=True):
        """
        Run the federated learning process using FLAM file for topology and participation.
        """
        self.initialize_data()
        self.initialize_model(model_name, auto_select_model, interactive_mode)
        total_start_time = time.time()
        round_accuracies = []

        # Initialize parameter server and clients
        server = self.ParameterServer(self.global_model)
        clients = [self.Client(i, type(self.global_model)(), self.client_data[i]) for i in range(self.num_clients)]

        flam_schedule = list(self.parse_flam_file(flam_path))
        print(f"Loaded FLAM schedule with {len(flam_schedule)} timesteps.")

        for flam_entry in flam_schedule:
            print(f"\n--- Timestep {flam_entry['timestep']} | Round {flam_entry['round']} ---")
            ps_client = flam_entry['target_node']
            phase = flam_entry['phase']
            in_range_clients = flam_entry['connected_sats']
            out_of_range_clients = flam_entry['missing_sats']

            print(f"Parameter Server for this timestep: Client {ps_client+1}")
            print(f"Phase: {phase}")
            print(f"In-range clients: {[f'Client {i+1}' for i in in_range_clients]}")
            print(f"Out-of-range clients: {[f'Client {i+1}' for i in out_of_range_clients]}")
            print(f"Parameter Server (Client {ps_client+1}) will aggregate this timestep.")

            # Distribute global model to all clients
            global_state = server.get_global_model().state_dict()
            for client in clients:
                client.update_model(global_state)

            # Only train during TRAINING phase
            round_accuracies_this = []
            if phase == "TRANSMITTING":
                for client_id, client in enumerate(clients):
                    if client_id in in_range_clients:
                        state_dict, acc = client.train()
                        server.receive_update(state_dict)
                        round_accuracies_this.append(acc)
                        print(f"Client {client.client_id+1} accuracy: {acc:.2%}")
                    else:
                        print(f"Client {client.client_id+1} skipped (out of range)")

                # Server aggregates updates from in-range clients
                server.aggregate()
                if round_accuracies_this:
                    avg_acc = sum(round_accuracies_this) / len(round_accuracies_this)
                    print(f"Timestep {flam_entry['timestep']} average in-range client accuracy: {avg_acc:.2%}")
                else:
                    print(f"Timestep {flam_entry['timestep']}: No clients in range to train.")

                self.round_times[f"timestep_{flam_entry['timestep']}"] = time.time() - total_start_time
                round_accuracies.append(avg_acc if round_accuracies_this else 0)

        self.total_training_time = time.time() - total_start_time
        self.round_accuracies = round_accuracies

        print(f"\nFederated learning process completed in {self.total_training_time:.2f} seconds.")
        print("\nTimestep-wise processing times and accuracies:")
        for idx, round_time in self.round_times.items():
            print(f"{idx}: {round_time:.2f} seconds, Accuracy: {round_accuracies[int(idx.split('_')[1]) - 1]:.2%}")
        print(f"Average timestep time: {self.total_training_time/len(self.round_times):.2f} seconds")

if __name__ == "__main__":
    """Standalone entry point for testing FederatedLearning."""
    # Default customization values
    num_rounds = 5
    num_clients = 8
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
    
    # Prompt user for FLAM file path
    flam_path = input("Enter the path to your FLAM file:\n").strip()
    if not os.path.isfile(flam_path):
        print(f"Error: FLAM file not found at '{flam_path}'")
        sys.exit(1)

    fl_instance.run(flam_path=flam_path)

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
