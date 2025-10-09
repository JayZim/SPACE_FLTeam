"""
Model Selection Module for Federated Learning
Author: Stephen zeng
Date: 2025-09-24
Version: 1.0

changelog:
- 2025-09-24: Initial creation.

Usage:
This module provides model selection capabilities for federated learning,
including interactive selection interface and model recommendation system.
"""

import sys
import os
from typing import List, Dict, Any, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ModelSelection:
    """Interactive interface for model and dataset selection"""
    
    def __init__(self, available_models: List[str], available_datasets: List[str]):
        self.available_models = available_models
        self.available_datasets = available_datasets
        
        # Model descriptions
        self.model_descriptions = {
            "SimpleCNN": "Simple Convolutional Neural Network - Fast, lightweight, good for MNIST",
            "ResNet50": "ResNet50 - Deep residual network, good for complex images",
            "EfficientNetB0": "EfficientNet-B0 - Efficient architecture, optimized for EuroSAT",
            "VisionTransformer": "Vision Transformer - Modern attention-based architecture",
            "CustomCNN": "Custom CNN with configurable filters - Flexible architecture"
        }
        
        # Dataset descriptions
        self.dataset_descriptions = {
            "MNIST": "Handwritten digits (28x28 grayscale) - 10 classes, 60K training samples",
            "CIFAR10": "Natural images (32x32 color) - 10 classes, 50K training samples",
            "EuroSAT": "Satellite images (64x64 color) - 10 land use classes, 27K samples"
        }
    
    def display_models(self) -> None:
        """Display available models with descriptions"""
        print(f"\n{'='*80}")
        print("Available Models:")
        print(f"{'='*80}")
        
        for i, model in enumerate(self.available_models, 1):
            description = self.model_descriptions.get(model, "No description available")
            print(f"{i}. {model}")
            print(f"   {description}")
            print()
    
    def display_datasets(self) -> None:
        """Display available datasets with descriptions"""
        print(f"\n{'='*80}")
        print("Available Datasets:")
        print(f"{'='*80}")
        
        for i, dataset in enumerate(self.available_datasets, 1):
            description = self.dataset_descriptions.get(dataset, "No description available")
            print(f"{i}. {dataset}")
            print(f"   {description}")
            print()
    
    def select_model(self) -> str:
        """Interactive model selection"""
        self.display_models()
        
        while True:
            try:
                choice = input(f"Please select a model (1-{len(self.available_models)}): ").strip()
                choice_num = int(choice)
                
                if 1 <= choice_num <= len(self.available_models):
                    selected_model = self.available_models[choice_num - 1]
                    print(f"✓ Selected model: {selected_model}")
                    return selected_model
                else:
                    print(f"Invalid choice. Please enter a number between 1 and {len(self.available_models)}")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\nOperation cancelled.")
                return self.available_models[0]  # Return default
    
    def select_dataset(self) -> str:
        """Interactive dataset selection"""
        self.display_datasets()
        
        while True:
            try:
                choice = input(f"Please select a dataset (1-{len(self.available_datasets)}): ").strip()
                choice_num = int(choice)
                
                if 1 <= choice_num <= len(self.available_datasets):
                    selected_dataset = self.available_datasets[choice_num - 1]
                    print(f"✓ Selected dataset: {selected_dataset}")
                    return selected_dataset
                else:
                    print(f"Invalid choice. Please enter a number between 1 and {len(self.available_datasets)}")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\nOperation cancelled.")
                return self.available_datasets[0]  # Return default
    
    def select_model_dataset_combination(self) -> Tuple[str, str]:
        """Select both model and dataset with recommendations"""
        print(f"\n{'='*80}")
        print("Federated Learning Model and Dataset Selection")
        print(f"{'='*80}")
        
        # Show recommendations
        print("\nRecommended Combinations:")
        print("• EuroSAT + EfficientNetB0/VisionTransformer (Best accuracy)")
        print("• CIFAR10 + ResNet50/EfficientNetB0 (Balanced performance)")
        print("• MNIST + SimpleCNN/CustomCNN (Fast training)")
        print()
        
        # Select dataset first
        print("Step 1: Select Dataset")
        selected_dataset = self.select_dataset()
        
        # Show model recommendations based on dataset
        print(f"\nStep 2: Select Model for {selected_dataset}")
        if selected_dataset == "EuroSAT":
            print("Recommended: EfficientNetB0 or VisionTransformer")
        elif selected_dataset == "CIFAR10":
            print("Recommended: ResNet50 or EfficientNetB0")
        elif selected_dataset == "MNIST":
            print("Recommended: SimpleCNN or CustomCNN")
        
        selected_model = self.select_model()
        
        print(f"\n✓ Final Selection:")
        print(f"  Model: {selected_model}")
        print(f"  Dataset: {selected_dataset}")
        
        return selected_model, selected_dataset
    
    def get_auto_recommendation(self, dataset: str) -> str:
        """Get automatic model recommendation based on dataset"""
        recommendations = {
            "EuroSAT": "EfficientNetB0",
            "CIFAR10": "ResNet50", 
            "MNIST": "SimpleCNN"
        }
        return recommendations.get(dataset, "SimpleCNN")
    
    def validate_combination(self, model: str, dataset: str) -> bool:
        """Validate if model-dataset combination is supported"""
        # Define supported combinations
        supported_combinations = {
            "SimpleCNN": ["MNIST", "CIFAR10"],
            "ResNet50": ["MNIST", "CIFAR10", "EuroSAT"],
            "EfficientNetB0": ["MNIST", "CIFAR10", "EuroSAT"],
            "VisionTransformer": ["MNIST", "CIFAR10", "EuroSAT"],
            "CustomCNN": ["MNIST", "CIFAR10"]
        }
        
        return dataset in supported_combinations.get(model, [])
    
    def show_combination_info(self, model: str, dataset: str) -> None:
        """Show information about the selected combination"""
        print(f"\n{'='*60}")
        print(f"Selected Combination: {model} + {dataset}")
        print(f"{'='*60}")
        
        # Show expected performance
        performance_info = {
            ("EfficientNetB0", "EuroSAT"): "Expected accuracy: 85-95%",
            ("VisionTransformer", "EuroSAT"): "Expected accuracy: 80-90%",
            ("ResNet50", "CIFAR10"): "Expected accuracy: 75-85%",
            ("EfficientNetB0", "CIFAR10"): "Expected accuracy: 80-90%",
            ("SimpleCNN", "MNIST"): "Expected accuracy: 95-99%",
            ("CustomCNN", "MNIST"): "Expected accuracy: 95-99%"
        }
        
        expected_perf = performance_info.get((model, dataset), "Performance varies")
        print(f"Expected Performance: {expected_perf}")
        
        # Show training characteristics
        if dataset == "EuroSAT":
            print("Training Notes: Uses data augmentation, Adam optimizer, 5 local epochs")
        elif dataset == "CIFAR10":
            print("Training Notes: Uses data augmentation, SGD with momentum, 4 local epochs")
        else:
            print("Training Notes: Standard training, SGD optimizer, 3 local epochs")
        
        print(f"{'='*60}")

def main():
    """Test the model selection interface"""
    available_models = ["SimpleCNN", "ResNet50", "EfficientNetB0", "VisionTransformer", "CustomCNN"]
    available_datasets = ["MNIST", "CIFAR10", "EuroSAT"]
    
    selector = ModelSelection(available_models, available_datasets)
    
    print("Testing Model Selection Interface")
    model, dataset = selector.select_model_dataset_combination()
    
    if selector.validate_combination(model, dataset):
        selector.show_combination_info(model, dataset)
    else:
        print(f"Warning: {model} + {dataset} combination may not be optimal")

if __name__ == "__main__":
    main()
