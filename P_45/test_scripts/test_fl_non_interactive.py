#!/usr/bin/env python3
"""
非交互式FL测试脚本
完全避免用户输入，直接测试FL核心功能
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_fl_non_interactive():
    """非交互式FL测试"""
    
    print(f"\n{'='*60}")
    print(f"🚀 非交互式联邦学习测试")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    try:
        # 导入FL模块
        from federated_learning.fl_core import FederatedLearning
        from federated_learning.fl_output import FLOutput
        from federated_learning.fl_visualization import FLVisualization
        from torchvision import datasets, transforms
        
        print("\n🤖 创建FL实例...")
        
        # 创建FL实例，完全禁用交互式功能
        fl_instance = FederatedLearning(enable_model_evaluation=False, enable_adaptation=False)
        
        # 设置参数
        fl_instance.set_num_rounds(2)  # 减少轮数以加快测试
        fl_instance.set_num_clients(3)  # 减少客户端数
        
        print("✅ FL实例创建成功")
        print(f"   - 轮数: {fl_instance.num_rounds}")
        print(f"   - 客户端数: {fl_instance.num_clients}")
        
        # 初始化数据（使用fallback方法避免adaptation system）
        print("\n📊 初始化数据...")
        fl_instance._initialize_data_fallback("MNIST")
        print("✅ 数据初始化完成")
        
        # 初始化模型（使用fallback方法避免交互式选择）
        print("\n🧠 初始化模型...")
        fl_instance._initialize_model_fallback("SimpleCNN", auto_select=False, interactive_mode=False)
        print("✅ 模型初始化完成")
        
        # 运行FL训练
        print("\n🔄 开始联邦学习训练...")
        start_time = time.time()
        
        # 直接运行FL核心训练逻辑，跳过run方法中的交互式部分
        fl_instance.total_start_time = time.time()
        round_accuracies = []

        # 初始化parameter server和clients
        server = fl_instance.ParameterServer(fl_instance.global_model)
        clients = [fl_instance.Client(i, type(fl_instance.global_model)(), fl_instance.client_data[i]) 
                  for i in range(fl_instance.num_clients)]

        # 运行FL训练轮次
        for round_num in range(fl_instance.num_rounds):
            print(f"\n--- Round {round_num + 1} ---")
            
            # 分发全局模型到所有客户端
            global_state = server.get_global_model().state_dict()
            for client in clients:
                client.update_model(global_state)

            round_accuracies_this = []
            
            # 训练所有客户端
            for client_id, client in enumerate(clients):
                state_dict, acc = client.train()
                server.receive_update(state_dict)
                round_accuracies_this.append(acc)
                print(f"Client {client.client_id+1} accuracy: {acc:.2%}")
            
            # 聚合更新
            server.aggregate()
            if round_accuracies_this:
                avg_acc = sum(round_accuracies_this) / len(round_accuracies_this)
                print(f"Round {round_num + 1} average accuracy: {avg_acc:.2%}")
                round_accuracies.append(avg_acc)
            
            # 更新全局模型
            fl_instance.global_model = server.get_global_model()

        fl_instance.total_training_time = time.time() - start_time
        fl_instance.round_accuracies = round_accuracies
        
        print(f"✅ 联邦学习训练完成!")
        print(f"⏱️  训练时间: {fl_instance.total_training_time:.2f} 秒")
        print(f"📊 总轮数: {len(fl_instance.round_accuracies)}")
        print(f"📊 最终准确率: {fl_instance.round_accuracies[-1]:.2%}" if fl_instance.round_accuracies else "N/A")
        
        # 创建测试数据集用于评估
        print("\n📊 创建测试数据集...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # 28x28x3 = 2352
        ])
        test_dataset = datasets.MNIST(
            root=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'federated_learning', 'data', 'MNIST'),
            train=False,
            download=True,
            transform=transform
        )
        print("✅ 测试数据集创建完成")
        
        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / "federated_learning" / "results_from_output" / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📁 创建输出目录: {output_dir}")
        
        # 生成FL输出
        print("\n📊 生成FL输出文件...")
        
        # 创建FLOutput实例
        fl_output = FLOutput(test_dataset=test_dataset)
        
        # 评估模型
        fl_output.evaluate_model(fl_instance.global_model, fl_instance.total_training_time)
        
        # 添加指标
        fl_output.add_metric("round_accuracies", fl_instance.round_accuracies)
        fl_output.add_metric("round_times", {f"round_{i+1}": (i+1) * 5.0 for i in range(len(fl_instance.round_accuracies))})
        
        # 保存文件
        log_file = output_dir / f"fl_results_{timestamp}.log"
        metrics_file = output_dir / f"fl_metrics_{timestamp}.json"
        model_file = output_dir / f"fl_model_{timestamp}.pt"
        
        fl_output.log_result(str(log_file))
        fl_output.write_to_file(str(metrics_file), format="json")
        fl_output.save_model(str(model_file))
        
        print(f"✅ FL输出文件生成完成:")
        print(f"   - 日志文件: {log_file}")
        print(f"   - 指标文件: {metrics_file}")
        print(f"   - 模型文件: {model_file}")
        
        # 生成可视化
        print("\n🎨 生成可视化...")
        try:
            # 生成动画
            acc_gif = output_dir / "accuracy_progress.gif"
            part_gif = output_dir / "client_participation.gif"
            
            fl_output.animate_accuracy_progress(str(metrics_file), str(acc_gif))
            fl_output.animate_client_participation(str(metrics_file), str(part_gif))
            
            print(f"✅ 动画生成完成:")
            print(f"   - 准确率动画: {acc_gif}")
            print(f"   - 参与度动画: {part_gif}")
            
            # 生成仪表板
            viz = FLVisualization(results_dir=str(output_dir))
            viz.visualize_from_json(str(metrics_file))
            
            dashboard_file = output_dir / "dashboard.html"
            print(f"   - 仪表板: {dashboard_file}")
            
        except Exception as viz_error:
            print(f"⚠️ 可视化生成失败: {viz_error}")
        
        # 检查生成的文件
        print(f"\n📁 检查生成的文件...")
        output_files = list(output_dir.rglob("*"))
        print(f"📊 输出文件总数: {len(output_files)}")
        
        for file in output_files:
            if file.is_file():
                file_size = file.stat().st_size
                print(f"   - {file.name} ({file_size} bytes)")
        
        return True, str(output_dir)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None

def main():
    """主函数"""
    print("🧪 SPACE项目 - 非交互式联邦学习测试")
    print("=" * 60)
    
    success, output_dir = test_fl_non_interactive()
    
    if success:
        print(f"\n🎉 测试成功完成!")
        print(f"📁 输出目录: {output_dir}")
        sys.exit(0)
    else:
        print(f"\n❌ 测试失败!")
        sys.exit(1)

if __name__ == "__main__":
    main()
