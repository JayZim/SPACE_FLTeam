"""
Filename: FL_core_th.py
Description: the core federated learning engine with performance optimizations.
Author: STephen Zeng
Date: 2025-05-14
Version: 1.0
Python Version: 3.10.0

"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
from typing import List, Dict, Any
import numpy as np
import sys
import os
import time
import copy
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

class OptimizedFederatedLearning:
    """Optimized Federated Learning Engine - Includes multiple performance optimizations"""

    def __init__(self):
        # Automatically detect and use GPU (if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.num_rounds = None
        self.num_clients = None
        self.global_model = None
        self.client_loaders = []
        
        self.round_times = {}
        self.total_training_time = 0
        self.memory_usage = []
        
        # Configure parallel processing
        self.max_workers = min(os.cpu_count(), 8)
        print(f"Configuring parallel thread pool: {self.max_workers} worker threads")

    def set_num_rounds(self, rounds: int) -> None:
        self.num_rounds = rounds
        print(f"Number of federated learning rounds set to: {self.num_rounds}")

    def set_num_clients(self, clients: int) -> None:
        self.num_clients = clients
        print(f"Number of clients set to: {self.num_clients}")

    def initialize_data(self):
        """Optimized data initialization and splitting"""
        print("Initializing dataset...")
        start_time = time.time()
        
        # Data preprocessing and standardization
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Load MNIST dataset
        dataset = datasets.MNIST('~/.pytorch/MNIST_data/', 
                               download=True, 
                               train=True, 
                               transform=transform)
        
        # Use random_split to evenly split the dataset
        samples_per_client = len(dataset) // self.num_clients
        client_datasets = random_split(
            dataset, 
            [samples_per_client] * self.num_clients
        )
        
        # Create optimized DataLoader for each client
        for client_dataset in client_datasets:
            loader = DataLoader(
                client_dataset,
                batch_size=64,  # Larger batch size to improve throughput
                shuffle=True,
                num_workers=2,  # Multi-threaded data loading
                pin_memory=True if self.device.type == 'cuda' else False,  # GPU memory optimization
                persistent_workers=True if self.device.type == 'cuda' else False,  # Keep worker threads active
                prefetch_factor=2 if self.device.type == 'cuda' else None  # Preload data
            )
            self.client_loaders.append(loader)
        
        print(f"Data initialization completed, time taken: {time.time() - start_time:.2f} seconds")
        print(f"Number of samples per client: {samples_per_client}")

    def initialize_model(self):
        """Create the same CNN model as the basic version"""
        class CNNModel(nn.Module):
            def __init__(self):
                super(CNNModel, self).__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),  # Use inplace operation to reduce memory usage
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
                
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(32 * 7 * 7, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 10)
                )
            
            def forward(self, x):
                x = self.features(x)
                return self.classifier(x)

        # Move the model to GPU (if available)
        self.global_model = CNNModel().to(self.device)
        print("Global model initialization completed")
        
        # Print the number of model parameters
        total_params = sum(p.numel() for p in self.global_model.parameters())
        print(f"Total number of model parameters: {total_params:,}")

    def train_client(self, client_id: int, model_state: Dict[str, torch.Tensor], data_loader):
        """Optimized client training function"""
        # Deep copy the model to avoid interference
        local_model = copy.deepcopy(self.global_model)
        local_model.load_state_dict(model_state)
        local_model.train()
        
        # Use SGD with momentum and weight decay
        optimizer = optim.SGD(
            local_model.parameters(), 
            lr=0.01,
            momentum=0.9,  # Use momentum to accelerate convergence
            weight_decay=1e-4  # Use weight decay to prevent overfitting
        )
        criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
        
        # Mixed precision training (only available on CUDA)
        scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
        
        for epoch in range(1):
            for batch_idx, (data, target) in enumerate(data_loader):
                # Move data to the appropriate device
                data, target = data.to(self.device), target.to(self.device)
                
                # Optimized gradient zeroing
                optimizer.zero_grad(set_to_none=True)
                
                # Mixed precision training
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        output = local_model(data)
                        loss = criterion(output, target)
                    
                    # Use scaler to handle gradients
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard training
                    output = local_model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
            
            # Update learning rate
            scheduler.step()
        
        print(f"Training client {client_id + 1}...")
        return local_model.state_dict()

    def federated_averaging(self, client_models: List[Dict[str, torch.Tensor]]):
        """Optimized federated averaging aggregation"""
        global_state = self.global_model.state_dict()
        
        # Efficient aggregation
        for key in global_state.keys():
            # Use torch.stack to improve efficiency
            stacked = torch.stack([client_model[key] for client_model in client_models], dim=0)
            global_state[key] = torch.mean(stacked, dim=0)
        
        self.global_model.load_state_dict(global_state)
        print("Global model updated through federated averaging")
        return global_state

    def evaluate_model(self):
        """Optimized model evaluation"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_dataset = datasets.MNIST('~/.pytorch/MNIST_data/', 
                                    download=True, 
                                    train=False, 
                                    transform=transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=1000,  # Large batch evaluation
            shuffle=False,
            num_workers=2,  # Multi-threaded loading
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.global_model.eval()
        test_loss = 0
        correct = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                
                test_loss += criterion(output, target).item()
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(test_loader)
        accuracy = 100. * correct / len(test_loader.dataset)
        
        print(f'Test set evaluation results: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
        return test_loss, accuracy

    def run(self):
        """Run the optimized federated learning process"""
        self.initialize_data()
        self.initialize_model()
        
        # Start timing the total training time
        total_start_time = time.time()
        
        # Use a thread pool to train clients in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for round_num in range(self.num_rounds):
                print(f"\nStarting round {round_num + 1}...")
                round_start_time = time.time()
                
                # Get the current global model state
                global_state = self.global_model.state_dict()
                
                # Submit client training tasks in parallel
                futures = []
                for client_id, data_loader in enumerate(self.client_loaders):
                    future = executor.submit(
                        self.train_client,
                        client_id,
                        copy.deepcopy(global_state),  # Deep copy to avoid reference issues
                        data_loader
                    )
                    futures.append(future)
                
                # Collect all client training results
                client_models = [future.result() for future in futures]
                
                # Aggregate and update the global model
                self.federated_averaging(client_models)
                
                # Calculate round time
                round_time = time.time() - round_start_time
                self.round_times[f"round_{round_num + 1}"] = round_time
                
                # Record memory usage (only on GPU)
                if torch.cuda.is_available():
                    mem_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
                    mem_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
                    self.memory_usage.append({
                        "round": round_num + 1,
                        "allocated_MB": mem_allocated,
                        "reserved_MB": mem_reserved
                    })
                
                print(f"Round {round_num + 1} completed, time taken: {round_time:.2f} seconds")
                self.evaluate_model()
                
        # Calculate the total training time
        self.total_training_time = time.time() - total_start_time
        
        print(f"\nFederated learning process completed, total time taken: {self.total_training_time:.2f} seconds")
        print("\nTraining time for each round:")
        for round_num, round_time in self.round_times.items():
            print(f"{round_num}: {round_time:.2f} seconds")
        
        avg_round_time = sum(self.round_times.values()) / len(self.round_times)
        print(f"Average time per round: {avg_round_time:.2f} seconds")
        
        # Final evaluation
        final_loss, final_accuracy = self.evaluate_model()
        
        return {
            "final_accuracy": final_accuracy,
            "final_loss": final_loss,
            "training_time": self.total_training_time,
            "round_times": self.round_times,
            "average_round_time": avg_round_time,
            "memory_usage": self.memory_usage if torch.cuda.is_available() else None
        }


if __name__ == "__main__":
    """Independent entry point for testing the optimized federated learning"""
    # Default configuration values
    num_rounds = 5
    num_clients = 3

    # Create a timestamp for a unique output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_optimized")
    
    # Ensure the results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize and run the optimized federated learning instance
    fl_instance = OptimizedFederatedLearning()
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
        f.write(f"Optimized federated learning training results\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Number of rounds: {num_rounds}\n")
        f.write(f"Number of clients: {num_clients}\n")
        f.write(f"Final accuracy: {results['final_accuracy']:.2f}%\n")
        f.write(f"Total training time: {results['training_time']:.2f} seconds\n")
        f.write(f"Average time per round: {results['average_round_time']:.2f} seconds\n")
        f.write(f"Device: {fl_instance.device}\n")
    
    # Save metrics as JSON
    with open(metrics_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "rounds": num_rounds,
            "clients": num_clients,
            "final_accuracy": results['final_accuracy'],
            "final_loss": results['final_loss'],
            "training_time": results['training_time'],
            "round_times": results['round_times'],
            "average_round_time": results['average_round_time'],
            "device": str(fl_instance.device),
            "memory_usage": results.get("memory_usage")
        }, f, indent=4)
    
    print(f"\nResults saved to the following files:")
    print(f"Log file: {log_file}")
    print(f"Metrics file: {metrics_file}")
    print(f"Model file: {model_file}")
