"""
Filename: Fl_core_MT.py
Description: Multithreading Federated Learning implementation using PyTorch.
Author: Stephen Zeng
Python Version: 3.10.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from typing import List, Dict
import concurrent.futures
import time
from datetime import datetime
import os
import sys
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from federated_learning.fl_output import FLOutput

class SimpleModel(nn.Module):
    """增强型神经网络模型"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class FederatedLearning:
    def __init__(self):
        self.num_rounds = 5
        self.num_clients = 8
        self.global_model = SimpleModel()
        self.client_data = []
        self.round_times = {}
        self.total_training_time = 0
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)  # 优化线程池大小

    def initialize_data(self):
        """多线程数据加载"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, 
                               train=True, transform=transform)
        
        client_size = len(dataset) // self.num_clients
        futures = []
        
        # 并行处理数据分片
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i in range(self.num_clients):
                start = i * client_size
                end = (i+1) * client_size
                futures.append(
                    executor.submit(self._create_client_loader, dataset, start, end)
                )
            for future in concurrent.futures.as_completed(futures):
                self.client_data.append(future.result())
        print(f"Initialized {len(self.client_data)} client datasets")

    def _create_client_loader(self, dataset, start, end):
        """创建客户端数据加载器（线程安全）"""
        subset = torch.utils.data.Subset(dataset, range(start, end))
        return torch.utils.data.DataLoader(
            subset, 
            batch_size=64, 
            shuffle=True,
            num_workers=0,  # 多线程数据加载
            pin_memory=True
        )

    def _train_client(self, model_state, data_loader):
        """客户端训练线程"""
        model = copy.deepcopy(self.global_model)
        model.load_state_dict(model_state)
        model.train()
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)  # 优化学习率
        criterion = nn.CrossEntropyLoss()
        
        for data, target in data_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
        return model.state_dict()

    def federated_averaging(self, client_models):
        """多线程安全聚合"""
        global_state = self.global_model.state_dict()
        for key in global_state.keys():
            global_state[key] = torch.mean(
                torch.stack([client[key].float() for client in client_models]), 
                dim=0
            ).to(torch.float32)
        self.global_model.load_state_dict(global_state)

    def run(self):
        """多线程主循环"""
        self.initialize_data()
        
        total_start = time.time()
        
        for round_num in range(self.num_rounds):
            round_start = time.time()
            
            # 并行客户端训练
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        self._train_client,
                        copy.deepcopy(self.global_model.state_dict()),
                        loader
                    ) for loader in self.client_data
                ]
                client_models = [f.result() for f in futures]
            
            self.federated_averaging(client_models)
            
            # 记录时间
            elapsed = time.time() - round_start
            self.round_times[f"round_{round_num+1}"] = elapsed
            print(f"Round {round_num+1} completed in {elapsed:.2f}s")
            
        self.total_training_time = time.time() - total_start
        self._save_results()

    def _save_results(self):
        """保存训练结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(os.path.dirname(__file__), "results_from_output")
        os.makedirs(results_dir, exist_ok=True)
        
        output = FLOutput()
        output.evaluate_model(self.global_model, self.total_training_time)
        
        # 保存指标
        metrics = {
            "round_times": self.round_times,
            "average_round_time": self.total_training_time / self.num_rounds,
            "model_architecture": str(self.global_model)
        }
        output.write_to_file(
            os.path.join(results_dir, f"metrics_{timestamp}.json"),
            format="json"
        )
        
        # 保存模型
        torch.save(
            self.global_model.state_dict(),
            os.path.join(results_dir, f"model_{timestamp}.pt")
        )
        print("Results saved successfully")

if __name__ == "__main__":
    fl = FederatedLearning()
    fl.num_rounds = 5
    fl.num_clients = 8
    fl.run()