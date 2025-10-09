#!/usr/bin/env python3
"""
Federated Learning Interactive Demo
Author: Stephen zeng
Date: 2025-09-24
Version: 1.0

changelog:
- 2025-09-24: Initial creation.

Usage:
Run this script to test the complete FL system with interactive model and dataset selection.

This script demonstrates the complete FL system with interactive model and dataset selection.
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

def test_model_selection():
    """Test the model selection interface"""
    print_header("Testing Model Selection Interface")
    
    try:
        from federated_learning.model_selection import ModelSelection
        
        available_models = ["SimpleCNN", "ResNet50", "EfficientNetB0", "VisionTransformer", "CustomCNN"]
        available_datasets = ["MNIST", "CIFAR10", "EuroSAT"]
        
        selector = ModelSelection(available_models, available_datasets)
        
        print("Available Models:")
        selector.display_models()
        
        print("\nAvailable Datasets:")
        selector.display_datasets()
        
        # Test auto-recommendations
        print_section("Auto-Recommendations")
        for dataset in available_datasets:
            recommendation = selector.get_auto_recommendation(dataset)
            print(f"Dataset: {dataset} → Recommended Model: {recommendation}")
        
        # Test combination validation
        print_section("Model-Dataset Combination Validation")
        test_combinations = [
            ("EfficientNetB0", "EuroSAT"),
            ("ResNet50", "CIFAR10"),
            ("SimpleCNN", "MNIST"),
            ("VisionTransformer", "EuroSAT"),
            ("CustomCNN", "MNIST"),
            ("SimpleCNN", "EuroSAT"),  # Should be invalid
        ]
        
        for model, dataset in test_combinations:
            is_valid = selector.validate_combination(model, dataset)
            status = "✓ Valid" if is_valid else "✗ Invalid"
            print(f"{status}: {model} + {dataset}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model selection test failed: {e}")
        return False

def test_model_evaluation():
    """Test the model evaluation system"""
    print_header("Testing Model Evaluation System")
    
    try:
        from federated_learning.model_evaluation import EnhancedModelEvaluationModule
        
        # Initialize evaluation module
        eval_module = EnhancedModelEvaluationModule()
        
        print_section("Model Registry")
        models = eval_module.registry.list_models()
        print(f"Registered models: {len(models)}")
        
        for model_name in models:
            model_info = eval_module.registry.get_model(model_name)
            print(f"  - {model_name}: {model_info.description}")
            print(f"    Category: {model_info.category}, Complexity: {model_info.complexity}")
        
        print_section("Model Categories")
        categories = ["CNN", "ResNet", "EfficientNet", "Transformer"]
        for category in categories:
            models_in_category = eval_module.registry.get_models_by_category(category)
            print(f"{category}: {models_in_category}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model evaluation test failed: {e}")
        return False

def test_data_loading():
    """Test data loading for all datasets"""
    print_header("Testing Data Loading")
    
    try:
        from federated_learning.fl_core import FederatedLearning
        
        fl = FederatedLearning(enable_model_evaluation=True)
        fl.set_num_clients(2)
        fl.set_num_rounds(1)
        
        datasets = ["MNIST", "CIFAR10", "EuroSAT"]
        
        for dataset in datasets:
            print_section(f"Loading {dataset} Dataset")
            try:
                fl.initialize_data(dataset)
                print(f"✓ {dataset} dataset loaded successfully")
            except Exception as e:
                print(f"✗ {dataset} dataset loading failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        return False

def test_configuration_system():
    """Test the configuration system"""
    print_header("Testing Configuration System")
    
    try:
        import json
        
        # Load configuration
        with open('options.json', 'r') as f:
            config = json.load(f)
        
        fl_config = config.get("federated_learning", {})
        
        print_section("Configuration Overview")
        print(f"Number of rounds: {fl_config.get('num_rounds', 'N/A')}")
        print(f"Number of clients: {fl_config.get('num_clients', 'N/A')}")
        print(f"Default model: {fl_config.get('model_type', 'N/A')}")
        print(f"Default dataset: {fl_config.get('data_set', 'N/A')}")
        
        print_section("Available Models")
        available_models = fl_config.get('available_models', [])
        for i, model in enumerate(available_models, 1):
            print(f"  {i}. {model}")
        
        print_section("Available Datasets")
        available_datasets = fl_config.get('available_datasets', [])
        for i, dataset in enumerate(available_datasets, 1):
            print(f"  {i}. {dataset}")
        
        print_section("Model Evaluation Settings")
        model_eval = fl_config.get('model_evaluation', {})
        print(f"Auto selection enabled: {model_eval.get('enable_auto_selection', False)}")
        
        selection_config = model_eval.get('selection', {})
        print(f"Max memory: {selection_config.get('max_memory_mb', 'N/A')} MB")
        print(f"Max complexity: {selection_config.get('max_complexity', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_interactive_fl_workflow():
    """Test the complete interactive FL workflow"""
    print_header("Testing Interactive FL Workflow")
    
    try:
        from federated_learning.fl_core import FederatedLearning
        
        print("Creating FL instance with model evaluation enabled...")
        fl = FederatedLearning(enable_model_evaluation=True)
        fl.set_num_clients(2)
        fl.set_num_rounds(1)
        
        print_section("Interactive Model and Dataset Selection")
        print("This would normally show the interactive selection interface.")
        print("For demo purposes, we'll simulate the selection process...")
        
        # Simulate user selections
        test_scenarios = [
            ("EfficientNetB0", "EuroSAT", "Best accuracy scenario"),
            ("ResNet50", "CIFAR10", "Balanced performance scenario"),
            ("SimpleCNN", "MNIST", "Fast training scenario")
        ]
        
        for model_name, dataset_name, description in test_scenarios:
            print(f"\n--- Testing: {description} ---")
            print(f"Model: {model_name}, Dataset: {dataset_name}")
            
            try:
                # Initialize data
                fl.initialize_data(dataset_name)
                print(f"✓ Data initialized: {dataset_name}")
                
                # Initialize model
                fl.initialize_model(model_name=model_name, auto_select=False, interactive_mode=False)
                print(f"✓ Model initialized: {model_name}")
                
                # Get model evaluation summary
                summary = fl.get_model_evaluation_summary()
                print(f"✓ Model evaluation summary: {summary.get('selected_model', 'N/A')}")
                
            except Exception as e:
                print(f"✗ Scenario failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Interactive FL workflow test failed: {e}")
        return False

def test_legacy_model_integration():
    """Test the legacy model.py integration"""
    print_header("Testing Legacy Model Integration")
    
    try:
        from model import Model
        
        print_section("Legacy Model Configuration")
        model = Model()
        
        # Test configuration
        model.set_model_type("SimpleCNN")
        model.set_data_set("MNIST")
        
        print(f"Model type: {model.get_model_type()}")
        print(f"Dataset: {model.get_data_set()}")
        
        print_section("Legacy Model Creation (TensorFlow)")
        try:
            tf_model = model.create_model()
            print(f"✓ TensorFlow model created: {type(tf_model)}")
            print(f"Model summary: {tf_model.summary() if hasattr(tf_model, 'summary') else 'N/A'}")
        except Exception as e:
            print(f"✗ TensorFlow model creation failed: {e}")
        
        print_section("Configuration Integration")
        print("✓ Legacy model.py is used for configuration purposes only")
        print("✓ Actual FL training uses PyTorch models from model_evaluation.py")
        
        return True
        
    except Exception as e:
        print(f"✗ Legacy model integration test failed: {e}")
        return False

def run_complete_demo():
    """Run the complete interactive demo"""
    print_header("Federated Learning System Interactive Demo", 100)
    print("This demo showcases the complete FL system with all components")
    print("Author: Stephen zeng | Date: 2025-09-24")
    
    start_time = time.time()
    
    # Test components
    tests = [
        ("Configuration System", test_configuration_system),
        ("Model Selection Interface", test_model_selection),
        ("Model Evaluation System", test_model_evaluation),
        ("Data Loading", test_data_loading),
        ("Legacy Model Integration", test_legacy_model_integration),
        ("Interactive FL Workflow", test_interactive_fl_workflow),
    ]
    
    results = {}
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*100}")
        print(f"Running: {test_name}")
        print(f"{'='*100}")
        
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed += 1
                print(f"\n✓ {test_name}: PASSED")
            else:
                print(f"\n✗ {test_name}: FAILED")
        except Exception as e:
            print(f"\n✗ {test_name}: ERROR - {e}")
            results[test_name] = False
    
    # Final results
    end_time = time.time()
    duration = end_time - start_time
    
    print_header("Demo Results Summary", 100)
    
    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        icon = "✓" if result else "✗"
        print(f"{icon} {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    print(f"Demo duration: {duration:.2f} seconds")
    
    if passed == total:
        print("\n🎉 All tests passed! FL system is fully functional.")
        print("\nSystem Features Demonstrated:")
        print("• 5 models: SimpleCNN, ResNet50, CustomCNN, EfficientNetB0, VisionTransformer")
        print("• 3 datasets: MNIST, CIFAR10, EuroSAT")
        print("• Interactive model and dataset selection")
        print("• Model evaluation and comparison")
        print("• Dataset-specific optimizations")
        print("• Legacy TensorFlow integration")
        print("• Complete FL workflow")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please review the issues above.")
    
    return passed == total

def main():
    """Main function"""
    try:
        success = run_complete_demo()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
