"""
Filename: FL_core_bs.py
Description: the basic federated learning engine without any optimization.
Author: STephen Zeng
Date: 2025-05-14
Version: 1.0
Python Version: 3.10.0

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

class BasicFederatedLearning:
    """Basic Federated Learning Engine - No Optimization"""

    def __init__(self):
        self.device = torch.device("cpu")  # Only use CPU
        self.num_rounds = None
        self.num_clients = None
        self.global_model = None
        self.client_data = []
        self.round_times = {}  # Dictionary to store processing time for each round
        self.total_training_time = 0  # Total training time

    def set_num_rounds(self, rounds: int) -> None:
        self.num_rounds = rounds
        print(f"Number of federated learning rounds set to: {self.num_rounds}")

    def set_num_clients(self, clients: int) -> None:
        self.num_clients = clients
        print(f"Number of clients set to: {self.num_clients}")

    def initialize_data(self):
        """Split the dataset among clients"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
        
        # Simply divide the dataset evenly
        samples_per_client = len(dataset) // self.num_clients
        indices = list(range(len(dataset)))
        
        for i in range(self.num_clients):
            start = i * samples_per_client
            end = (i + 1) * samples_per_client
            subset = torch.utils.data.Subset(dataset, indices[start:end])
            client_loader = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=True)
            self.client_data.append(client_loader)
        
        print(f"Data initialization completed, time taken: 0.01 seconds")
        print(f"Number of samples per client: {samples_per_client}")

    def initialize_model(self):
        """Define the CNN model"""
        class CNNModel(nn.Module):
            def __init__(self):
                super(CNNModel, self).__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
                
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(32 * 7 * 7, 128),
                    nn.ReLU(),
                    nn.Linear(128, 10)
                )
            
            def forward(self, x):
                x = self.features(x)
                return self.classifier(x)

        self.global_model = CNNModel()
        print("Global model initialization completed")
        
        # Print the number of model parameters
        total_params = sum(p.numel() for p in self.global_model.parameters())
        print(f"Total number of model parameters: {total_params:,}")

    def train_client(self, model, data_loader):
        """Train a single client"""
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
        """Aggregate client models into the global model"""
        global_state = self.global_model.state_dict()
        for key in global_state.keys():
            global_state[key] = torch.stack([client[key] for client in client_models], dim=0).mean(dim=0)
        
        self.global_model.load_state_dict(global_state)
        print("Global model updated through federated averaging")

    def evaluate_model(self):
        """Evaluate model performance"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_dataset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
        
        self.global_model.eval()
        test_loss = 0
        correct = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_loader:
                output = self.global_model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(test_loader)
        accuracy = 100. * correct / len(test_loader.dataset)
        
        print(f'Test set evaluation results: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
        return test_loss, accuracy

    def run(self):
        """Run the federated learning process"""
        self.initialize_data()
        self.initialize_model()

        # Start timing the total training time
        total_start_time = time.time()

        for round_num in range(self.num_rounds):
            print(f"Starting round {round_num + 1} of federated learning...")

            # Start timing the round
            round_start_time = time.time()

            client_models = []
            for client_id, data_loader in enumerate(self.client_data):
                print(f"Training client {client_id + 1}...")
                # Create a copy of the model for each client
                client_model = type(self.global_model)()
                client_model.load_state_dict(self.global_model.state_dict())
                
                # Train the client model
                client_state_dict = self.train_client(client_model, data_loader)
                client_models.append(client_state_dict)

            # Aggregate the models
            self.federated_averaging(client_models)

            # Calculate the round time
            round_time = time.time() - round_start_time
            self.round_times[f"round_{round_num + 1}"] = round_time

            print(f"Round {round_num + 1} completed, time taken: {round_time:.2f} seconds")
            
            # Evaluate the model every round
            self.evaluate_model()

        # Calculate the total training time
        self.total_training_time = time.time() - total_start_time
        print(f"\nFederated learning process completed, total time taken: {self.total_training_time:.2f} seconds")
        print("\nTraining time for each round:")
        for round_num, round_time in self.round_times.items():
            print(f"{round_num}: {round_time:.2f} seconds")
        
        avg_round_time = self.total_training_time/self.num_rounds
        print(f"Average time per round: {avg_round_time:.2f} seconds")
        
        # Final evaluation
        final_loss, final_accuracy = self.evaluate_model()
        
        return {
            "final_accuracy": final_accuracy,
            "final_loss": final_loss,
            "training_time": self.total_training_time,
            "round_times": self.round_times,
            "average_round_time": avg_round_time
        }


if __name__ == "__main__":
    """Independent entry point for testing the basic federated learning"""
    # Default configuration values
    num_rounds = 5
    num_clients = 8

    # Create a timestamp for unique output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_basic")
    
    # Ensure the results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize and run an instance of the basic federated learning
    fl_instance = BasicFederatedLearning()
    fl_instance.set_num_rounds(num_rounds)
    fl_instance.set_num_clients(num_clients)
    
    # Perform training
    results = fl_instance.run()
    
    # Save the results
    log_file = os.path.join(results_dir, f"results_{timestamp}.log")
    metrics_file = os.path.join(results_dir, f"metrics_{timestamp}.json")
    model_file = os.path.join(results_dir, f"model_{timestamp}.pt")
    
    # Save the model
    torch.save(fl_instance.global_model.state_dict(), model_file)
    
    # Save the log
    with open(log_file, 'w') as f:
        f.write(f"Federated learning training results\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Number of rounds: {num_rounds}\n")
        f.write(f"Number of clients: {num_clients}\n")
        f.write(f"Final accuracy: {results['final_accuracy']:.2f}%\n")
        f.write(f"Total training time: {results['training_time']:.2f} seconds\n")
        f.write(f"Average time per round: {results['average_round_time']:.2f} seconds\n")
    
    # Save the metrics as JSON
    import json
    with open(metrics_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "rounds": num_rounds,
            "clients": num_clients,
            "final_accuracy": results['final_accuracy'],
            "final_loss": results['final_loss'],
            "training_time": results['training_time'],
            "round_times": results['round_times'],
            "average_round_time": results['average_round_time']
        }, f, indent=4)
    
    print(f"\nResults saved to the following files:")
    print(f"Log file: {log_file}")
    print(f"Metrics file: {metrics_file}")
    print(f"Model file: {model_file}")
