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

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from federated_learning.fl_output import FLOutput

class BasicFederatedLearning:
    """基础版联邦学习引擎 - 无优化"""

    def __init__(self):
        self.num_rounds = None
        self.num_clients = None
        self.global_model = None
        self.client_data = []
        self.round_times = {}  # 存储每轮处理时间的字典
        self.total_training_time = 0  # 总训练时间

    def set_num_rounds(self, rounds: int) -> None:
        self.num_rounds = rounds
        print(f"联邦学习轮数设置为: {self.num_rounds}")

    def set_num_clients(self, clients: int) -> None:
        self.num_clients = clients
        print(f"客户端数量设置为: {self.num_clients}")

    def initialize_data(self):
        """在客户端之间分割数据集"""
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
        
        # 简单均分数据集
        for i in range(self.num_clients):
            start = i * (len(dataset) // self.num_clients)
            end = (i + 1) * (len(dataset) // self.num_clients)
            subset = torch.utils.data.Subset(dataset, range(start, end))
            client_loader = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=True)
            self.client_data.append(client_loader)
        
        print("客户端数据初始化完成")

    def initialize_model(self):
        """定义简单PyTorch模型"""
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc = nn.Linear(28 * 28, 10)

            def forward(self, x):
                x = x.view(-1, 28 * 28)
                return self.fc(x)

        self.global_model = SimpleModel()
        print("全局模型初始化完成")

    def train_client(self, model, data_loader):
        """训练单个客户端"""
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
        """将客户端模型聚合到全局模型"""
        global_state = self.global_model.state_dict()
        for key in global_state.keys():
            global_state[key] = torch.stack([client[key] for client in client_models], dim=0).mean(dim=0)
        
        self.global_model.load_state_dict(global_state)
        print("通过联邦平均更新全局模型")

    def get_round_metrics(self) -> Dict:
        """获取当前训练会话的指标"""
        return {
            "round_times": self.round_times,
            "total_training_time": self.total_training_time,
            "average_round_time": self.total_training_time / self.num_rounds if self.num_rounds > 0 else 0,
            "timestamp": datetime.now().isoformat()
        }

    def evaluate_model(self):
        """评估模型性能"""
        transform = transforms.Compose([transforms.ToTensor()])
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
        
        print(f'\n测试集评估结果: 平均损失: {test_loss:.4f}, 准确率: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
        return test_loss, accuracy

    def run(self):
        """运行联邦学习过程"""
        self.initialize_data()
        self.initialize_model()

        # 开始总训练时间计时
        total_start_time = time.time()

        for round_num in range(self.num_rounds):
            print(f"开始第 {round_num + 1} 轮...")

            # 开始轮次计时
            round_start_time = time.time()

            client_models = []
            for client_id, data_loader in enumerate(self.client_data):
                print(f"训练客户端 {client_id + 1}...")
                # 为每个客户端创建模型副本
                client_model = type(self.global_model)()
                client_model.load_state_dict(self.global_model.state_dict())
                
                # 训练客户端模型
                client_state_dict = self.train_client(client_model, data_loader)
                client_models.append(client_state_dict)

            # 聚合模型
            self.federated_averaging(client_models)

            # 计算轮次时间
            round_time = time.time() - round_start_time
            self.round_times[f"round_{round_num + 1}"] = round_time

            print(f"第 {round_num + 1} 轮完成，耗时: {round_time:.2f} 秒")
            
            # 每轮评估模型
            if (round_num + 1) % 1 == 0:
                self.evaluate_model()

        # 计算总训练时间
        self.total_training_time = time.time() - total_start_time
        print(f"\n联邦学习过程完成，总耗时: {self.total_training_time:.2f} 秒")
        print("\n各轮训练时间:")
        for round_num, round_time in self.round_times.items():
            print(f"{round_num}: {round_time:.2f} 秒")
        
        avg_round_time = self.total_training_time/self.num_rounds
        print(f"平均每轮时间: {avg_round_time:.2f} 秒")
        
        # 最终评估
        final_loss, final_accuracy = self.evaluate_model()
        
        return {
            "final_accuracy": final_accuracy,
            "final_loss": final_loss,
            "training_time": self.total_training_time,
            "metrics": self.get_round_metrics()
        }


if __name__ == "__main__":
    """独立入口点用于测试基础版联邦学习"""
    # 默认配置值
    num_rounds = 5
    num_clients = 3

    # 创建时间戳用于唯一输出文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), "results_basic")
    
    # 确保结果目录存在
    os.makedirs(results_dir, exist_ok=True)
    
    # 初始化并运行基础版联邦学习实例
    fl_instance = BasicFederatedLearning()
    fl_instance.set_num_rounds(num_rounds)
    fl_instance.set_num_clients(num_clients)
    
    # 执行训练
    results = fl_instance.run()
    
    # 保存结果
    log_file = os.path.join(results_dir, f"results_{timestamp}.log")
    metrics_file = os.path.join(results_dir, f"metrics_{timestamp}.json")
    model_file = os.path.join(results_dir, f"model_{timestamp}.pt")
    
    # 保存模型
    torch.save(fl_instance.global_model.state_dict(), model_file)
    
    # 保存日志
    with open(log_file, 'w') as f:
        f.write(f"联邦学习训练结果\n")
        f.write(f"时间戳: {timestamp}\n")
        f.write(f"轮数: {num_rounds}\n")
        f.write(f"客户端数: {num_clients}\n")
        f.write(f"最终准确率: {results['final_accuracy']:.2f}%\n")
        f.write(f"总训练时间: {results['training_time']:.2f}秒\n")
        f.write(f"平均每轮时间: {results['metrics']['average_round_time']:.2f}秒\n")
    
    print(f"\n结果已保存到以下文件:")
    print(f"日志文件: {log_file}")
    print(f"指标文件: {metrics_file}")
    print(f"模型文件: {model_file}")
