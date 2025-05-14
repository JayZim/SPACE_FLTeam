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
from datetime import datetime

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from federated_learning.fl_output import FLOutput

class FederatedLearning:
    """Custom Federated Learning engine."""

    def __init__(self):
        self.num_rounds = None
        self.num_clients = None
        self.global_model = None
        self.client_data = []
        self.round_times = {}  # Dictionary to store processing times for each round
        self.total_training_time = 0  # Total training time

    def set_num_rounds(self, rounds: int) -> None:
        self.num_rounds = rounds
        print(f"Number of rounds set to: {self.num_rounds}")

    def set_num_clients(self, clients: int) -> None:
        self.num_clients = clients
        print(f"Number of clients set to: {self.num_clients}")

    def initialize_data(self):
        """Split dataset among clients."""
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
        for i in range(self.num_clients):
            start = i * (len(dataset) // self.num_clients)
            end = (i + 1) * (len(dataset) // self.num_clients)
            subset = torch.utils.data.Subset(dataset, range(start, end))
            client_loader = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=True)
            self.client_data.append(client_loader)
        print("Client data initialized.")

    def initialize_model(self):
        """Define a simple PyTorch model."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc = nn.Linear(28 * 28, 10)

            def forward(self, x):
                x = x.view(-1, 28 * 28)
                return self.fc(x)

        self.global_model = SimpleModel()
        print("Global model initialized.")

    def train_client(self, model, data_loader):
        """Train a single client."""
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        for data, target in data_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        return model.state_dict()

    def federated_averaging(self, client_models: List[dict]):
        """Aggregate client models into a global model."""
        global_state = self.global_model.state_dict()
        for key in global_state.keys():
            global_state[key] = torch.stack([client[key] for client in client_models], dim=0).mean(dim=0)
        self.global_model.load_state_dict(global_state)
        print("Global model updated via federated averaging.")

    def get_round_metrics(self) -> Dict:
        """Get metrics for the current training session."""
        return {
            "round_times": self.round_times,
            "total_training_time": self.total_training_time,
            "average_round_time": self.total_training_time / self.num_rounds if self.num_rounds > 0 else 0,
            "timestamp": datetime.now().isoformat()
        }

    def run(self):
        """Run the federated learning process."""
        self.initialize_data()
        self.initialize_model()

        # Start total training time
        total_start_time = time.time()

        for round_num in range(self.num_rounds):
            print(f"Starting round {round_num + 1}...")

            # Start round time
            round_start_time = time.time()

            client_models = []
            for client_id, data_loader in enumerate(self.client_data):
                print(f"Training client {client_id + 1}...")
                client_model = self.train_client(self.global_model, data_loader)
                client_models.append(client_model)

            self.federated_averaging(client_models)

            # Calculate round time
            round_time = time.time() - round_start_time
            self.round_times[f"round_{round_num + 1}"] = round_time

            print(f"Round {round_num + 1} completed in {round_time:.2f} seconds.")

        # Calculate total training time
        self.total_training_time = time.time() - total_start_time
        print(f"\nFederated learning process completed in {self.total_training_time:.2f} seconds.")
        print("\nRound-wise processing times:")
        for round_num, round_time in self.round_times.items():
            print(f"{round_num}: {round_time:.2f} seconds")
        print(f"Average round time: {self.total_training_time/self.num_rounds:.2f} seconds")

if __name__ == "__main__":
    """Standalone entry point for testing FederatedLearning."""
    # Default customization values
    num_rounds = 5
    num_clients = 8
    model_type = "SimpleCNN"
    data_set = "MNIST"

    # Create timestamp for unique output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), "results_from_output")

    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Initialize and run the FederatedLearning instance
    fl_instance = FederatedLearning()
    fl_instance.set_num_rounds(num_rounds)
    fl_instance.set_num_clients(num_clients)
    fl_instance.run()

    # Evaluate the model
    output = FLOutput()
    output.evaluate_model(fl_instance.global_model, fl_instance.total_training_time)

    # Add timing metrics
    timing_metrics = fl_instance.get_round_metrics()
    for metric_name, value in timing_metrics.items():
        output.add_metric(metric_name, value)

    # Save results with timestamp
    log_file = os.path.join(results_dir, f"results_{timestamp}.log")
    metrics_file = os.path.join(results_dir, f"metrics_{timestamp}.json")
    model_file = os.path.join(results_dir, f"model_{timestamp}.pt")

    # Log results and save files
    output.log_result(log_file)
    output.write_to_file(metrics_file, format="json")
    output.save_model(model_file)

    print(f"\nResults have been saved to the following files:")
    print(f"Log file: {log_file}")
    print(f"Metrics file: {metrics_file}")
    print(f"Model file: {model_file}")
