"""
Filename: Fl_core_OT2.py
Description: single thread implementation of Federated Learning.
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

class FederatedLearningOT:
    def __init__(self):
        self.num_rounds = 5
        self.num_clients = 8
        self.global_model = SimpleModel()
        self.client_data = []
        self.round_times = {}
        self.total_training_time = 0  # 移除线程池相关代码

    def initialize_data(self):
        """单线程数据加载"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True,
                               train=True, transform=transform)
        
        client_size = len(dataset) // self.num_clients
        
        # 单线程处理数据分片
        for i in range(self.num_clients):
            start = i * client_size
            end = (i+1) * client_size
            self.client_data.append(
                self._create_client_loader(dataset, start, end)
            )
        print(f"Initialized {len(self.client_data)} client datasets")

    def _create_client_loader(self, dataset, start, end):
        """创建客户端数据加载器（单线程版）"""
        subset = torch.utils.data.Subset(dataset, range(start, end))
        return torch.utils.data.DataLoader(
            subset, 
            batch_size=64,
            shuffle=True,
            num_workers=0,  # 禁用多线程数据加载
            pin_memory=True
        )

    def _train_client(self, model_state, loader):
        """
        训练单个客户端模型
        :param model_state: 全局模型的状态字典
        :param loader: 客户端数据加载器
        :return: 训练后的客户端模型的状态字典
        """
        model = SimpleModel()
        model.load_state_dict(model_state)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        model.train()
        for data, target in loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        return model.state_dict()

    def federated_averaging(self, client_models):
        """
        联邦平均算法，更新全局模型
        :param client_models: 客户端模型的状态字典列表
        """
        if not client_models:
            return

        global_state = self.global_model.state_dict()
        for key in global_state.keys():
            global_state[key] = sum([client[key] for client in client_models]) / len(client_models)
        self.global_model.load_state_dict(global_state)

    def run(self):
        """单线程主循环"""
        self.initialize_data()
        
        total_start = time.time()
        
        for round_num in range(self.num_rounds):
            round_start = time.time()
            
            # 单线程客户端训练
            client_models = []
            for loader in self.client_data:
                client_models.append(
                    self._train_client(
                        copy.deepcopy(self.global_model.state_dict()),
                        loader
                    )
                )
            
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
    fl = FederatedLearningOT()
    fl.num_rounds = 5
    fl.num_clients = 8 
    fl.run()