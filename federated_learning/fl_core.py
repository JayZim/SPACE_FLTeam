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

Usage: 
Run this file directly to start a Multithreading instance of Tensorflow FL with the chosen number of clients rounds and model.

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from typing import List
import numpy as np

class FederatedLearning:
    """Custom Federated Learning engine."""

    def __init__(self):
        self.num_rounds = None
        self.num_clients = None
        self.global_model = None
        self.client_data = []

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

    def run(self):
        """Run the federated learning process."""
        self.initialize_data()
        self.initialize_model()

        for round_num in range(self.num_rounds):
            print(f"Starting round {round_num + 1}...")
            client_models = []
            for client_id, data_loader in enumerate(self.client_data):
                print(f"Training client {client_id + 1}...")
                client_model = self.train_client(self.global_model, data_loader)
                client_models.append(client_model)

            self.federated_averaging(client_models)
            print(f"Round {round_num + 1} completed.")

        print("Federated learning process completed.")

if __name__ == "__main__":
    """Standalone entry point for testing FederatedLearning."""
    # Default customization values
    num_rounds = 5
    num_clients = 3
    model_type = "SimpleCNN"
    data_set = "MNIST"

    # Initialize and run the FederatedLearning instance
    fl_instance = FederatedLearning()
    fl_instance.set_num_rounds(num_rounds)
    fl_instance.set_num_clients(num_clients)
    fl_instance.run()    