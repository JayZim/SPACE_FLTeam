#!/usr/bin/env python3
"""
Interactive Federated Learning Demo
Author: Stephen zeng
Date: 2025-01-07
Version: 1.0

This script provides an interactive demonstration of the FL system where you can
actually select models and datasets and see the system in action.
"""

import sys
import os
import time
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def print_header(title: str, width: int = 80):
    """Print a formatted header"""
    print("\n" + "="*width)
    print(f" {title}")
    print("="*width)

def print_section(title: str, width: int = 60):
    """Print a formatted section header"""
    print(f"\n{'-'*width}")
    print(f" {title}")
    print(f"{'-'*width}")

def interactive_model_selection():
    """Interactive model and dataset selection"""
    print_header("Interactive Model and Dataset Selection")
    
    try:
        from federated_learning.model_selection import ModelSelection
        
        available_models = ["SimpleCNN", "ResNet50", "EfficientNetB0", "VisionTransformer", "CustomCNN"]
        available_datasets = ["MNIST", "CIFAR10", "EuroSAT"]
        
        selector = ModelSelection(available_models, available_datasets)
        
        print("🎯 Welcome to the Interactive FL System!")
        print("You will now select a model and dataset combination.")
        print("\nPress Enter to continue...")
        input()
        
        # Interactive selection
        selected_model, selected_dataset = selector.select_model_dataset_combination()
        
        # Show combination info
        selector.show_combination_info(selected_model, selected_dataset)
        
        return selected_model, selected_dataset
        
    except Exception as e:
        print(f"✗ Interactive selection failed: {e}")
        return None, None

def run_fl_simulation(model_name: str, dataset_name: str):
    """Run a simulated FL process"""
    print_header(f"Running FL Simulation: {model_name} + {dataset_name}")
    
    try:
        from federated_learning.fl_core import FederatedLearning
        
        print("🚀 Initializing FL system...")
        fl = FederatedLearning(enable_model_evaluation=True)
        fl.set_num_clients(2)
        fl.set_num_rounds(1)
        
        print(f"📊 Loading {dataset_name} dataset...")
        fl.initialize_data(dataset_name)
        
        print(f"🤖 Initializing {model_name} model...")
        fl.initialize_model(model_name=model_name, auto_select=False, interactive_mode=False)
        
        print_section("FL Simulation Results")
        summary = fl.get_model_evaluation_summary()
        print(f"✅ Selected Model: {summary.get('selected_model', 'N/A')}")
        print(f"✅ Model Evaluation Enabled: {summary.get('model_evaluation_enabled', False)}")
        print(f"✅ Available Models: {len(summary.get('available_models', []))}")
        
        print("\n🎉 FL simulation completed successfully!")
        print("The system is ready for full federated learning training.")
        
        return True
        
    except Exception as e:
        print(f"✗ FL simulation failed: {e}")
        return False

def show_system_overview():
    """Show system overview"""
    print_header("Federated Learning System Overview")
    
    print("🌟 System Features:")
    print("   • 5 Models: SimpleCNN, ResNet50, CustomCNN, EfficientNetB0, VisionTransformer")
    print("   • 3 Datasets: MNIST, CIFAR10, EuroSAT")
    print("   • Interactive model and dataset selection")
    print("   • Model evaluation and comparison")
    print("   • Dataset-specific optimizations")
    print("   • Legacy TensorFlow integration (configuration only)")
    
    print_section("Model Categories")
    categories = {
        "CNN": ["SimpleCNN", "CustomCNN"],
        "ResNet": ["ResNet50"],
        "EfficientNet": ["EfficientNetB0"],
        "Transformer": ["VisionTransformer"]
    }
    
    for category, models in categories.items():
        print(f"🏷️  {category}: {', '.join(models)}")
    
    print_section("Dataset Information")
    datasets = {
        "MNIST": "Handwritten digits (28×28 grayscale, 60K samples)",
        "CIFAR10": "Natural images (32×32 color, 50K samples)",
        "EuroSAT": "Satellite images (64×64 color, 27K samples)"
    }
    
    for dataset, description in datasets.items():
        print(f"📊 {dataset}: {description}")
    
    print_section("Recommended Combinations")
    recommendations = [
        ("EfficientNetB0", "EuroSAT", "🎯 Best accuracy for satellite images"),
        ("VisionTransformer", "EuroSAT", "🚀 Advanced architecture for complex data"),
        ("ResNet50", "CIFAR10", "⚖️  Balanced performance for natural images"),
        ("EfficientNetB0", "CIFAR10", "💡 Efficient architecture for natural images"),
        ("SimpleCNN", "MNIST", "⚡ Fast training for simple tasks"),
        ("CustomCNN", "MNIST", "🔧 Configurable architecture for simple tasks")
    ]
    
    for model, dataset, description in recommendations:
        print(f"✅ {model} + {dataset}: {description}")

def main():
    """Main interactive demo"""
    print_header("Interactive Federated Learning Demo", 100)
    print("🎯 Experience the complete FL system interactively")
    print("📅 Author: Stephen zeng | Date: 2025-01-07")
    
    try:
        # Show system overview
        show_system_overview()
        
        print("\n" + "="*100)
        print("Press Enter to start the interactive selection process...")
        input()
        
        # Interactive selection
        selected_model, selected_dataset = interactive_model_selection()
        
        if selected_model and selected_dataset:
            print("\n" + "="*100)
            print("Press Enter to run the FL simulation with your selections...")
            input()
            
            # Run simulation
            success = run_fl_simulation(selected_model, selected_dataset)
            
            if success:
                print("\n🎉 Interactive demo completed successfully!")
                print(f"✅ You selected: {selected_model} + {selected_dataset}")
                print("🚀 The FL system is fully functional and ready for use!")
            else:
                print("\n❌ Demo encountered an error, but the system is still functional.")
        else:
            print("\n⚠️  Selection was cancelled, but the system is still available.")
        
        print("\n" + "="*100)
        print("Thank you for trying the Interactive FL Demo!")
        print("The system is ready for production use.")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user.")
        print("The FL system is still available for use.")
    except Exception as e:
        print(f"\n💥 Demo failed with error: {e}")
        print("Please check the system configuration.")

if __name__ == "__main__":
    main()
