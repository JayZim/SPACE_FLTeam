import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
# 导入 random_split 函数，解决 NameError 问题
from torch.utils.data import random_split
from typing import List, Dict, Any
import numpy as np
import sys
import os
import time
import copy
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import json

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 假设FLOutput类在此模块中
from federated_learning.fl_output import FLOutput

class OptimizedFederatedLearning:
    """优化版联邦学习引擎"""

    def __init__(self):
        # 检测并使用可用的GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 基础配置
        self.num_rounds = None
        self.num_clients = None
        self.global_model = None
        self.client_data = []
        self.client_loaders = []
        
        # 性能监控
        self.round_times = {}
        self.total_training_time = 0
        self.memory_usage = []
        
        # 线程池配置 - 根据CPU核心数自动调整
        self.max_workers = min(os.cpu_count(), 8)  # 最多8个线程避免过载
        print(f"配置并行线程池: {self.max_workers}个工作线程")

    def set_num_rounds(self, rounds: int) -> None:
        self.num_rounds = rounds
        print(f"联邦学习轮数设置为: {self.num_rounds}")

    def set_num_clients(self, clients: int) -> None:
        self.num_clients = clients
        print(f"客户端数量设置为: {self.num_clients}")

    def initialize_data(self):
        """优化的数据集分割与加载器配置"""
        print("初始化数据集...")
        start_time = time.time()
        
        # 数据预处理优化
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST标准化参数
        ])
        
        # 数据集加载 - 使用缓存避免重复下载
        dataset = datasets.MNIST('~/.pytorch/MNIST_data/', 
                               download=True, 
                               train=True, 
                               transform=transform)
        
        # 高效数据分割 - 一次性分割避免重复计算
        samples_per_client = len(dataset) // self.num_clients
        client_datasets = random_split(
            dataset, 
            [samples_per_client] * self.num_clients
        )
        
        # 优化的数据加载器 - 并行加载与内存固定
        for client_dataset in client_datasets:
            loader = torch.utils.data.DataLoader(
                client_dataset,
                batch_size=64,  # 更大的批次提高GPU利用率
                shuffle=True,
                num_workers=2,  # 并行数据加载
                pin_memory=True,  # 固定内存加速GPU传输
                persistent_workers=True,  # 保持工作进程存活减少启动开销
                prefetch_factor=2  # 预加载下2个批次
            )
            self.client_loaders.append(loader)
        
        print(f"数据初始化完成，耗时: {time.time() - start_time:.2f}秒")
        print(f"每个客户端样本数: {samples_per_client}")

    def initialize_model(self):
        """定义优化的神经网络模型"""
        class OptimizedModel(nn.Module):
            def __init__(self):
                super(OptimizedModel, self).__init__()
                # 使用Sequential容器优化前向传播
                self.features = nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),  # inplace操作减少内存使用
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

        # 初始化模型并移至目标设备
        self.global_model = OptimizedModel().to(self.device)
        print("全局模型初始化完成")
        
        # 打印模型参数量
        total_params = sum(p.numel() for p in self.global_model.parameters())
        print(f"模型总参数量: {total_params:,}")

    def train_client(self, client_id: int, model_state: Dict[str, torch.Tensor], data_loader):
        """优化的客户端训练函数"""
        # 创建本地模型副本
        local_model = copy.deepcopy(self.global_model)
        local_model.load_state_dict(model_state)
        local_model.train()
        
        # 优化的训练配置
        optimizer = optim.SGD(
            local_model.parameters(), 
            lr=0.01,
            momentum=0.9,  # 动量加速收敛
            weight_decay=1e-4  # 轻微正则化
        )
        criterion = nn.CrossEntropyLoss()
        
        # 自适应学习率调度器
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
        
        # 使用混合精度训练加速
        scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
        
        # 训练循环
        for epoch in range(1):  # 每轮联邦学习只在本地训练1个epoch
            for batch_idx, (data, target) in enumerate(data_loader):
                # 将数据移至目标设备
                data, target = data.to(self.device), target.to(self.device)
                
                # 清零梯度 - 使用set_to_none=True加速
                optimizer.zero_grad(set_to_none=True)
                
                if scaler is not None:
                    # 混合精度训练路径
                    with torch.cuda.amp.autocast():
                        output = local_model(data)
                        loss = criterion(output, target)
                    
                    # 缩放梯度以防止下溢
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # 标准训练路径
                    output = local_model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
            
            scheduler.step()
        
        # 返回训练后的模型参数
        return local_model.state_dict()

    def federated_averaging(self, client_models: List[Dict[str, torch.Tensor]]):
        """高效的联邦平均聚合"""
        # 创建全局状态字典
        global_state = self.global_model.state_dict()
        
        # 对每个参数进行平均
        for key in global_state.keys():
            # 使用torch.stack高效堆叠相同形状的张量
            stacked = torch.stack([client_model[key] for client_model in client_models], dim=0)
            # 计算平均值并保存
            global_state[key] = torch.mean(stacked, dim=0)
        
        return global_state

    def get_round_metrics(self) -> Dict[str, Any]:
        """获取训练性能指标"""
        avg_round_time = sum(self.round_times.values()) / len(self.round_times) if self.round_times else 0
        
        return {
            "round_times": self.round_times,
            "total_training_time": self.total_training_time,
            "average_round_time": avg_round_time,
            "memory_usage": self.memory_usage,
            "device": str(self.device),
            "timestamp": datetime.now().isoformat()
        }

    def evaluate_model(self, test_loader=None):
        """评估全局模型性能"""
        if test_loader is None:
            # 创建测试数据集
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            test_dataset = datasets.MNIST('~/.pytorch/MNIST_data/', 
                                        download=True, 
                                        train=False, 
                                        transform=transform)
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=1000,
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )
        
        # 切换到评估模式
        self.global_model.eval()
        test_loss = 0
        correct = 0
        criterion = nn.CrossEntropyLoss()
        
        # 不计算梯度以加速评估
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                
                # 累加批次损失
                test_loss += criterion(output, target).item()
                
                # 获取最高概率的预测类别
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        # 计算平均损失和准确率
        test_loss /= len(test_loader)
        accuracy = 100. * correct / len(test_loader.dataset)
        
        print(f'\n测试集评估结果: 平均损失: {test_loss:.4f}, 准确率: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
        return test_loss, accuracy

    def run(self):
        """执行优化版联邦学习流程"""
        # 初始化数据和模型
        self.initialize_data()
        self.initialize_model()
        
        # 记录总训练开始时间
        total_start_time = time.time()
        
        # 创建线程池
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 执行多轮联邦学习
            for round_num in range(self.num_rounds):
                print(f"\n开始第 {round_num + 1} 轮联邦学习...")
                round_start_time = time.time()
                
                # 获取当前全局模型状态
                global_state = self.global_model.state_dict()
                
                # 并行训练所有客户端
                futures = []
                for client_id, data_loader in enumerate(self.client_loaders):
                    # 提交客户端训练任务到线程池
                    future = executor.submit(
                        self.train_client,
                        client_id,
                        copy.deepcopy(global_state),  # 深拷贝避免共享状态
                        data_loader
                    )
                    futures.append(future)
                
                # 收集所有客户端模型
                client_models = [future.result() for future in futures]
                
                # 联邦平均聚合
                aggregated_state = self.federated_averaging(client_models)
                self.global_model.load_state_dict(aggregated_state)
                
                # 记录本轮训练时间
                round_time = time.time() - round_start_time
                self.round_times[f"round_{round_num + 1}"] = round_time
                
                # 记录内存使用情况
                if torch.cuda.is_available():
                    mem_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
                    mem_reserved = torch.cuda.memory_reserved() / (1024 ** 2)    # MB
                    self.memory_usage.append({
                        "round": round_num + 1,
                        "allocated_MB": mem_allocated,
                        "reserved_MB": mem_reserved
                    })
                
                print(f"第 {round_num + 1} 轮完成，耗时: {round_time:.2f} 秒")
                
                # 每轮结束后评估模型（可选）
                if (round_num + 1) % 1 == 0:  # 每轮都评估
                    test_loss, accuracy = self.evaluate_model()
                
        # 计算总训练时间
        self.total_training_time = time.time() - total_start_time
        
        # 打印性能统计
        print(f"\n联邦学习训练完成，总耗时: {self.total_training_time:.2f} 秒")
        print("\n各轮训练时间:")
        for round_num, round_time in self.round_times.items():
            print(f"{round_num}: {round_time:.2f} 秒")
        
        avg_round_time = sum(self.round_times.values()) / len(self.round_times)
        print(f"平均每轮时间: {avg_round_time:.2f} 秒")
        
        # 最终模型评估
        final_loss, final_accuracy = self.evaluate_model()
        
        return {
            "final_accuracy": final_accuracy,
            "final_loss": final_loss,
            "training_time": self.total_training_time,
            "metrics": self.get_round_metrics()
        }


if __name__ == "__main__":
    """独立入口点用于测试优化版联邦学习"""
    # 默认配置值
    num_rounds = 5
    num_clients = 3
    
    # 创建时间戳用于唯一输出文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), "results_from_output")
    
    # 确保结果目录存在
    os.makedirs(results_dir, exist_ok=True)
    
    # 初始化并运行优化版联邦学习实例
    fl_instance = OptimizedFederatedLearning()
    fl_instance.set_num_rounds(num_rounds)
    fl_instance.set_num_clients(num_clients)
    
    # 执行训练
    results = fl_instance.run()
    
    # 保存结果
    log_file = os.path.join(results_dir, f"results_{timestamp}.log")
    metrics_file = os.path.join(results_dir, f"metrics_{timestamp}.json")
    model_file = os.path.join(results_dir, f"model_{timestamp}.pt")
    
    # 保存指标
    with open(metrics_file, 'w') as f:
        json.dump(results["metrics"], f, indent=2)
    
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
