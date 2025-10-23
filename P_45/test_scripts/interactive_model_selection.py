#!/usr/bin/env python3
"""
Interactive Model and Dataset Selection Script
Allow users to select different model and dataset combinations for testing
"""

import os
import sys
import json
from datetime import datetime

# Add project root directory to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

def display_available_options():
    """Display available model and dataset options"""
    print("\n" + "="*80)
    print("🤖 Available Model and Dataset Options")
    print("="*80)
    
    # Available models
    models = {
        "1": {"name": "SimpleCNN", "description": "Simple Convolutional Neural Network - Fast training, suitable for small datasets"},
        "2": {"name": "ResNet50", "description": "ResNet50 Deep Residual Network - Balanced performance and speed"},
        "3": {"name": "CustomCNN", "description": "Custom Convolutional Neural Network - Customizable architecture"},
        "4": {"name": "EfficientNetB0", "description": "EfficientNetB0 - High-performance model with best accuracy"},
        "5": {"name": "VisionTransformer", "description": "Vision Transformer - Latest technology with highest accuracy"}
    }
    
    print("\n📊 Available Models:")
    for key, model in models.items():
        print(f"  {key}. {model['name']} - {model['description']}")
    
    # Available datasets
    datasets = {
        "1": {"name": "MNIST", "description": "Handwritten Digit Recognition (28x28 grayscale) - 10 classes, 60K training samples"},
        "2": {"name": "CIFAR10", "description": "Natural Image Classification (32x32 color) - 10 classes, 50K training samples"},
        "3": {"name": "EuroSAT", "description": "Satellite Image Classification (64x64 color) - 10 land use classes, 27K samples"}
    }
    
    print("\n🗄️  Available Datasets:")
    for key, dataset in datasets.items():
        print(f"  {key}. {dataset['name']} - {dataset['description']}")
    
    return models, datasets

def get_user_selection():
    """Get user selection"""
    models, datasets = display_available_options()
    
    print("\n" + "="*80)
    print("🎯 Recommended Combinations")
    print("="*80)
    print("• EuroSAT + EfficientNetB0/VisionTransformer (Best accuracy)")
    print("• CIFAR10 + ResNet50/EfficientNetB0 (Balanced performance)")
    print("• MNIST + SimpleCNN/CustomCNN (Fast training)")
    
    print("\n" + "="*80)
    print("📝 Please Select Your Configuration")
    print("="*80)
    
    # Select dataset
    while True:
        try:
            dataset_choice = input("\n🗄️  Please select dataset (1-3): ").strip()
            if dataset_choice in datasets:
                selected_dataset = datasets[dataset_choice]
                break
            else:
                print("❌ Invalid selection, please enter 1-3")
        except KeyboardInterrupt:
            print("\n\n👋 Selection cancelled")
            sys.exit(0)
    
    # Select model
    while True:
        try:
            model_choice = input("\n🤖 Please select model (1-5): ").strip()
            if model_choice in models:
                selected_model = models[model_choice]
                break
            else:
                print("❌ Invalid selection, please enter 1-5")
        except KeyboardInterrupt:
            print("\n\n👋 Selection cancelled")
            sys.exit(0)
    
    # Select number of rounds
    while True:
        try:
            rounds = input("\n🔄 Please enter number of training rounds (recommended 3-10): ").strip()
            rounds = int(rounds)
            if 1 <= rounds <= 50:
                break
            else:
                print("❌ Please enter a number between 1-50")
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\n👋 Selection cancelled")
            sys.exit(0)
    
    # Select number of clients
    while True:
        try:
            clients = input("\n👥 Please enter number of clients (recommended 4-8): ").strip()
            clients = int(clients)
            if 1 <= clients <= 20:
                break
            else:
                print("❌ Please enter a number between 1-20")
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\n👋 Selection cancelled")
            sys.exit(0)
    
    return selected_dataset, selected_model, rounds, clients

def run_test_with_selection(dataset, model, rounds, clients, timesteps=100):
    """Run test with selected configuration"""
    print("\n" + "="*80)
    print("🚀 Starting Test Run")
    print("="*80)
    print(f"📊 Dataset: {dataset['name']}")
    print(f"🤖 Model: {model['name']}")
    print(f"🔄 Rounds: {rounds}")
    print(f"👥 Clients: {clients}")
    print(f"⏱️  Timesteps: {timesteps}")
    print("="*80)
    
    # Build command
    cmd = f"python3 main.py flomps TLEs/SatCount8.tle --timesteps {timesteps} --model-type {model['name']} --data-set {dataset['name']} --num-rounds {rounds} --num-clients {clients}"
    
    print(f"\n🔧 Executing command: {cmd}")
    print("\n⏳ Running test...")
    
    # Record start time
    start_time = datetime.now()
    
    # Execute command
    import subprocess
    try:
        result = subprocess.run(cmd, shell=True, cwd=project_root, capture_output=False)
        
        # Record end time
        end_time = datetime.now()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"\n✅ Test completed successfully!")
            print(f"⏱️  Total time: {duration}")
            
            # Save test configuration
            save_test_config(dataset, model, rounds, clients, timesteps, duration)
            
        else:
            print(f"\n❌ Test failed, return code: {result.returncode}")
            
    except Exception as e:
        print(f"\n❌ Execution error: {e}")

def save_test_config(dataset, model, rounds, clients, timesteps, duration):
    """Save test configuration to file"""
    config = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset,
        "model": model,
        "rounds": rounds,
        "clients": clients,
        "timesteps": timesteps,
        "duration": str(duration),
        "status": "success"
    }
    
    # Ensure directory exists
    results_dir = os.path.join(project_root, "P_45", "test_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save configuration file
    config_file = os.path.join(results_dir, f"interactive_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Test configuration saved to: {config_file}")

def main():
    """Main function"""
    print("🎯 SPACE Project Interactive Model and Dataset Selection")
    print("="*80)
    print("This script will help you select different model and dataset combinations for testing")
    print("You can test different configurations to find the best combination for your needs")
    
    try:
        # Get user selection
        dataset, model, rounds, clients = get_user_selection()
        
        # Confirm selection
        print("\n" + "="*80)
        print("✅ Confirm Your Selection")
        print("="*80)
        print(f"📊 Dataset: {dataset['name']} - {dataset['description']}")
        print(f"🤖 Model: {model['name']} - {model['description']}")
        print(f"🔄 Training rounds: {rounds}")
        print(f"👥 Number of clients: {clients}")
        print(f"⏱️  Timesteps: 100 (8 satellite configuration)")
        
        confirm = input("\n❓ Confirm to start test? (y/N): ").strip().lower()
        if confirm in ['y', 'yes']:
            run_test_with_selection(dataset, model, rounds, clients)
        else:
            print("👋 Test cancelled")
            
    except KeyboardInterrupt:
        print("\n\n👋 Operation cancelled")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")

if __name__ == "__main__":
    main()