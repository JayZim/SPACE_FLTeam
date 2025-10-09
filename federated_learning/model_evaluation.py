"""
Model Evaluation Module for Federated Learning
Author: Stephen zeng
Date: 2025-09-24
Version: 2.0

Changelog:
- 2025-09-05: Initial creation of Enhanced Model Evaluation Module
- 2025-09-24: Refactored to focus on model evaluation, separated model selection

Usage:
This module provides comprehensive model evaluation capabilities for federated learning,
including model registry, performance evaluation, and model comparison.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import json
import os
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """Data class to store model evaluation metrics"""
    accuracy: float
    loss: float
    training_time: float
    inference_time: float
    memory_usage: float
    convergence_rounds: int
    stability_score: float
    efficiency_score: float

@dataclass
class ModelInfo:
    """Data class to store model information"""
    name: str
    model_class: type
    parameters: Dict[str, Any]
    description: str
    category: str  # e.g., "CNN", "ResNet", "Transformer"
    complexity: str  # e.g., "Low", "Medium", "High"

class ModelRegistry:
    """Registry for managing available models"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.models: Dict[str, ModelInfo] = {}
            self._register_default_models()
            ModelRegistry._initialized = True
    
    def _register_default_models(self):
        """Register default models available in the system"""
        # Simple CNN
        self.register_model(
            name="SimpleCNN",
            model_class=self._create_simple_cnn,
            parameters={"input_shape": (28, 28, 1), "num_classes": 10},
            description="Simple Convolutional Neural Network for MNIST",
            category="CNN",
            complexity="Low"
        )
        
        # ResNet50
        self.register_model(
            name="ResNet50",
            model_class=self._create_resnet50,
            parameters={"input_shape": (32, 32, 3), "num_classes": 10},
            description="ResNet50 adapted for MNIST classification",
            category="ResNet",
            complexity="High"
        )
        
        # Custom CNN
        self.register_model(
            name="CustomCNN",
            model_class=self._create_custom_cnn,
            parameters={"input_shape": (28, 28, 1), "num_classes": 10, "filters": [32, 64, 128]},
            description="Custom CNN with configurable filters",
            category="CNN",
            complexity="Medium"
        )
        
        # EfficientNet-B0 (EuroSAT optimized)
        self.register_model(
            name="EfficientNetB0",
            model_class=self._create_efficientnet_b0,
            parameters={"input_shape": (64, 64, 3), "num_classes": 10},
            description="EfficientNet-B0 optimized for EuroSAT dataset",
            category="EfficientNet",
            complexity="Medium"
        )
        
        # Vision Transformer (ViT) (Modern architecture)
        self.register_model(
            name="VisionTransformer",
            model_class=self._create_vision_transformer,
            parameters={"input_shape": (64, 64, 3), "num_classes": 10, "patch_size": 16, "embed_dim": 192},
            description="Vision Transformer for modern image classification",
            category="Transformer",
            complexity="High"
        )
    
    def register_model(self, name: str, model_class: type, parameters: Dict[str, Any], 
                      description: str, category: str, complexity: str):
        """Register a new model in the registry"""
        model_info = ModelInfo(
            name=name,
            model_class=model_class,
            parameters=parameters,
            description=description,
            category=category,
            complexity=complexity
        )
        self.models[name] = model_info
        logger.info(f"Registered model: {name}")
    
    def get_model(self, name: str) -> Optional[ModelInfo]:
        """Get model information by name"""
        return self.models.get(name)
    
    def list_models(self) -> List[str]:
        """List all available model names"""
        return list(self.models.keys())
    
    def get_models_by_category(self, category: str) -> List[str]:
        """Get models by category"""
        return [name for name, info in self.models.items() if info.category == category]
    
    def _create_simple_cnn(self, **kwargs):
        """Create Simple CNN model"""
        class SimpleCNN(nn.Module):
            def __init__(self, input_shape=(28, 28, 1), num_classes=10):
                super(SimpleCNN, self).__init__()
                self.conv1 = nn.Conv2d(input_shape[2], 32, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.fc1 = nn.Linear(64 * 7 * 7, 128)
                self.fc2 = nn.Linear(128, num_classes)
                self.dropout = nn.Dropout(0.5)
                
            def forward(self, x):
                x = self.pool(torch.relu(self.conv1(x)))
                x = self.pool(torch.relu(self.conv2(x)))
                x = x.view(-1, 64 * 7 * 7)
                x = torch.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return x
        
        return SimpleCNN(**kwargs)
    
    def _create_resnet50(self, **kwargs):
        """Create ResNet50 model"""
        import torchvision.models as models
        
        class ResNet50MNIST(nn.Module):
            def __init__(self, input_shape=(32, 32, 3), num_classes=10):
                super(ResNet50MNIST, self).__init__()
                self.resnet = models.resnet50(pretrained=True)
                # Modify first layer for 3-channel input
                self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
                # Modify last layer for number of classes
                self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
                
            def forward(self, x):
                return self.resnet(x)
        
        return ResNet50MNIST(**kwargs)
    
    def _create_custom_cnn(self, **kwargs):
        """Create Custom CNN model"""
        class CustomCNN(nn.Module):
            def __init__(self, input_shape=(28, 28, 1), num_classes=10, filters=[32, 64, 128]):
                super(CustomCNN, self).__init__()
                self.filters = filters
                self.conv_layers = nn.ModuleList()
                self.bn_layers = nn.ModuleList()
                
                # Create convolutional layers
                in_channels = input_shape[2]
                for i, out_channels in enumerate(filters):
                    self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
                    self.bn_layers.append(nn.BatchNorm2d(out_channels))
                    in_channels = out_channels
                
                # Calculate flattened size
                self.flattened_size = filters[-1] * (input_shape[0] // (2 ** len(filters))) ** 2
                self.fc = nn.Linear(self.flattened_size, num_classes)
                self.dropout = nn.Dropout(0.3)
                
            def forward(self, x):
                for conv, bn in zip(self.conv_layers, self.bn_layers):
                    x = torch.relu(bn(conv(x)))
                    x = torch.max_pool2d(x, 2)
                x = x.view(x.size(0), -1)
                x = self.dropout(x)
                x = self.fc(x)
                return x
        
        return CustomCNN(**kwargs)
    
    def _create_efficientnet_b0(self, **kwargs):
        """Create EfficientNet-B0 model optimized for EuroSAT"""
        import torchvision.models as models
        
        class EfficientNetB0EuroSAT(nn.Module):
            def __init__(self, input_shape=(64, 64, 3), num_classes=10):
                super(EfficientNetB0EuroSAT, self).__init__()
                # Load pre-trained EfficientNet-B0
                self.efficientnet = models.efficientnet_b0(pretrained=True)
                
                # Modify the classifier for our number of classes
                self.efficientnet.classifier = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(self.efficientnet.classifier[1].in_features, num_classes)
                )
                
                # Data augmentation will be handled by transforms, not in the model
                # This is more efficient and follows PyTorch best practices
                
            def forward(self, x):
                return self.efficientnet(x)
        
        return EfficientNetB0EuroSAT(**kwargs)
    
    def _create_vision_transformer(self, **kwargs):
        """Create Vision Transformer model"""
        class VisionTransformer(nn.Module):
            def __init__(self, input_shape=(64, 64, 3), num_classes=10, patch_size=16, embed_dim=192, 
                         num_heads=3, num_layers=6, mlp_ratio=4.0):
                super(VisionTransformer, self).__init__()
                
                self.patch_size = patch_size
                self.embed_dim = embed_dim
                self.num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
                
                # Patch embedding
                self.patch_embed = nn.Conv2d(input_shape[2], embed_dim, 
                                           kernel_size=patch_size, stride=patch_size)
                
                # Position embedding
                self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
                
                # Class token
                self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
                
                # Transformer encoder
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=int(embed_dim * mlp_ratio),
                    dropout=0.1,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                
                # Classification head
                self.norm = nn.LayerNorm(embed_dim)
                self.head = nn.Linear(embed_dim, num_classes)
                
                # Initialize weights
                self.apply(self._init_weights)
                
            def _init_weights(self, m):
                if isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
            
            def forward(self, x):
                B = x.shape[0]
                
                # Patch embedding
                x = self.patch_embed(x)  # (B, embed_dim, H/patch_size, W/patch_size)
                x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
                
                # Add class token
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
                
                # Add position embedding
                x = x + self.pos_embed
                
                # Transformer encoder
                x = self.transformer(x)
                
                # Classification
                x = self.norm(x[:, 0])  # Use class token
                x = self.head(x)
                
                return x
        
        return VisionTransformer(**kwargs)

class ModelEvaluator:
    """Evaluator for model performance assessment"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.evaluation_history: List[Dict[str, Any]] = []
    
    def evaluate_model(self, model: nn.Module, data_loader, criterion: nn.Module, 
                      num_epochs: int = 3) -> ModelMetrics:
        """Evaluate a model and return comprehensive metrics"""
        model = model.to(self.device)
        model.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training metrics
        start_time = time.time()
        total_loss = 0.0
        correct = 0
        total = 0
        convergence_rounds = 0
        loss_history = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                epoch_total += target.size(0)
                epoch_correct += (predicted == target).sum().item()
            
            avg_loss = epoch_loss / len(data_loader)
            accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0
            
            loss_history.append(avg_loss)
            
            # Check for convergence
            if len(loss_history) > 1 and abs(loss_history[-1] - loss_history[-2]) < 0.001:
                convergence_rounds = epoch + 1
                break
        
        training_time = time.time() - start_time
        
        # Calculate inference time
        model.eval()
        inference_start = time.time()
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(self.device)
                _ = model(data)
                break  # Only test one batch
        inference_time = time.time() - inference_start
        
        # Calculate memory usage (approximate)
        memory_usage = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024  # MB
        
        # Calculate stability score (based on loss variance)
        stability_score = 1.0 / (1.0 + np.var(loss_history)) if len(loss_history) > 1 else 1.0
        
        # Calculate efficiency score (accuracy per training time)
        efficiency_score = accuracy / training_time if training_time > 0 else 0
        
        metrics = ModelMetrics(
            accuracy=accuracy,
            loss=avg_loss,
            training_time=training_time,
            inference_time=inference_time,
            memory_usage=memory_usage,
            convergence_rounds=convergence_rounds if convergence_rounds > 0 else num_epochs,
            stability_score=stability_score,
            efficiency_score=efficiency_score
        )
        
        # Store evaluation history
        self.evaluation_history.append({
            "model_name": model.__class__.__name__,
            "metrics": metrics,
            "timestamp": time.time()
        })
        
        return metrics
    
    def compare_models(self, model_metrics: Dict[str, ModelMetrics]) -> Dict[str, Any]:
        """Compare multiple models and return ranking"""
        if not model_metrics:
            return {}
        
        # Create comparison matrix
        comparison = {
            "accuracy_ranking": sorted(model_metrics.items(), key=lambda x: x[1].accuracy, reverse=True),
            "efficiency_ranking": sorted(model_metrics.items(), key=lambda x: x[1].efficiency_score, reverse=True),
            "stability_ranking": sorted(model_metrics.items(), key=lambda x: x[1].stability_score, reverse=True),
            "memory_ranking": sorted(model_metrics.items(), key=lambda x: x[1].memory_usage),
            "convergence_ranking": sorted(model_metrics.items(), key=lambda x: x[1].convergence_rounds)
        }
        
        # Calculate overall score (weighted combination)
        overall_scores = {}
        for name, metrics in model_metrics.items():
            score = (
                0.3 * metrics.accuracy +
                0.2 * metrics.efficiency_score +
                0.2 * metrics.stability_score +
                0.15 * (1.0 / (1.0 + metrics.memory_usage / 100)) +  # Normalize memory usage
                0.15 * (1.0 / (1.0 + metrics.convergence_rounds / 10))  # Normalize convergence
            )
            overall_scores[name] = score
        
        comparison["overall_ranking"] = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
        
        return comparison

class ModelSelector:
    """Intelligent model selection based on evaluation results"""
    
    def __init__(self, registry: ModelRegistry, evaluator: ModelEvaluator):
        self.registry = registry
        self.evaluator = evaluator
        self.selection_history: List[Dict[str, Any]] = []
    
    def select_best_model(self, data_loader, criterion: nn.Module, 
                         selection_criteria: str = "overall",
                         constraints: Optional[Dict[str, Any]] = None) -> Tuple[str, nn.Module]:
        """Select the best model based on evaluation criteria"""
        
        if constraints is None:
            constraints = {}
        
        # Filter models based on constraints
        candidate_models = self._filter_models_by_constraints(constraints)
        
        if not candidate_models:
            raise ValueError("No models match the specified constraints")
        
        # Evaluate all candidate models
        model_metrics = {}
        for model_name in candidate_models:
            model_info = self.registry.get_model(model_name)
            if model_info:
                model = model_info.model_class(**model_info.parameters)
                metrics = self.evaluator.evaluate_model(model, data_loader, criterion)
                model_metrics[model_name] = metrics
        
        # Compare models and select best
        comparison = self.evaluator.compare_models(model_metrics)
        
        if selection_criteria == "overall":
            best_model_name = comparison["overall_ranking"][0][0]
        elif selection_criteria == "accuracy":
            best_model_name = comparison["accuracy_ranking"][0][0]
        elif selection_criteria == "efficiency":
            best_model_name = comparison["efficiency_ranking"][0][0]
        elif selection_criteria == "stability":
            best_model_name = comparison["stability_ranking"][0][0]
        else:
            best_model_name = comparison["overall_ranking"][0][0]
        
        # Create the selected model
        model_info = self.registry.get_model(best_model_name)
        selected_model = model_info.model_class(**model_info.parameters)
        
        # Record selection
        selection_record = {
            "selected_model": best_model_name,
            "criteria": selection_criteria,
            "constraints": constraints,
            "comparison": comparison,
            "timestamp": time.time()
        }
        self.selection_history.append(selection_record)
        
        logger.info(f"Selected model: {best_model_name} based on {selection_criteria} criteria")
        
        return best_model_name, selected_model
    
    def _filter_models_by_constraints(self, constraints: Dict[str, Any]) -> List[str]:
        """Filter models based on constraints"""
        candidate_models = []
        
        for model_name, model_info in self.registry.models.items():
            # Check category constraint
            if "category" in constraints and model_info.category != constraints["category"]:
                continue
            
            # Check complexity constraint
            if "complexity" in constraints and model_info.complexity != constraints["complexity"]:
                continue
            
            # Check memory constraint (approximate)
            if "max_memory_mb" in constraints:
                # This is a rough estimate - actual memory usage depends on data
                estimated_memory = self._estimate_model_memory(model_info)
                if estimated_memory > constraints["max_memory_mb"]:
                    continue
            
            candidate_models.append(model_name)
        
        return candidate_models
    
    def _estimate_model_memory(self, model_info: ModelInfo) -> float:
        """Estimate model memory usage in MB"""
        # This is a rough estimation based on model complexity
        complexity_multiplier = {
            "Low": 1.0,
            "Medium": 2.0,
            "High": 4.0
        }
        
        base_memory = 10.0  # Base memory in MB
        multiplier = complexity_multiplier.get(model_info.complexity, 1.0)
        
        return base_memory * multiplier

class EnhancedModelEvaluationModule:
    """Main module that orchestrates model evaluation and selection"""
    
    def __init__(self, device: str = "cpu"):
        self.registry = ModelRegistry()
        self.evaluator = ModelEvaluator(device)
        self.selector = ModelSelector(self.registry, self.evaluator)
        self.device = device
    
    def register_custom_model(self, name: str, model_class: type, parameters: Dict[str, Any],
                            description: str, category: str, complexity: str):
        """Register a custom model"""
        self.registry.register_model(name, model_class, parameters, description, category, complexity)
    
    def evaluate_all_models(self, data_loader, criterion: nn.Module, 
                           num_epochs: int = 3) -> Dict[str, ModelMetrics]:
        """Evaluate all registered models"""
        model_metrics = {}
        
        for model_name in self.registry.list_models():
            model_info = self.registry.get_model(model_name)
            if model_info:
                logger.info(f"Evaluating model: {model_name}")
                model = model_info.model_class(**model_info.parameters)
                metrics = self.evaluator.evaluate_model(model, data_loader, criterion, num_epochs)
                model_metrics[model_name] = metrics
        
        return model_metrics
    
    def select_model_for_fl(self, data_loader, criterion: nn.Module,
                           fl_constraints: Optional[Dict[str, Any]] = None) -> Tuple[str, nn.Module]:
        """Select the best model for federated learning"""
        if fl_constraints is None:
            fl_constraints = {
                "max_memory_mb": 200,  # Increased memory limit for new models
                "complexity": "Medium",  # Balance between performance and efficiency
                "preferred_categories": ["CNN", "EfficientNet", "ResNet", "Transformer"]
            }
        
        return self.selector.select_best_model(
            data_loader, criterion, 
            selection_criteria="overall",
            constraints=fl_constraints
        )
    
    def get_model_recommendations(self, data_loader, criterion: nn.Module) -> Dict[str, Any]:
        """Get model recommendations with detailed analysis"""
        # Evaluate all models
        model_metrics = self.evaluate_all_models(data_loader, criterion)
        
        # Compare models
        comparison = self.evaluator.compare_models(model_metrics)
        
        # Generate recommendations
        recommendations = {
            "best_overall": comparison["overall_ranking"][0] if comparison["overall_ranking"] else None,
            "best_accuracy": comparison["accuracy_ranking"][0] if comparison["accuracy_ranking"] else None,
            "best_efficiency": comparison["efficiency_ranking"][0] if comparison["efficiency_ranking"] else None,
            "best_stability": comparison["stability_ranking"][0] if comparison["stability_ranking"] else None,
            "detailed_metrics": model_metrics,
            "comparison_matrix": comparison
        }
        
        return recommendations
    
    def save_evaluation_report(self, filepath: str):
        """Save evaluation report to file"""
        report = {
            "evaluation_history": self.evaluator.evaluation_history,
            "selection_history": self.selector.selection_history,
            "available_models": {
                name: {
                    "description": info.description,
                    "category": info.category,
                    "complexity": info.complexity,
                    "parameters": info.parameters
                }
                for name, info in self.registry.models.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Evaluation report saved to: {filepath}")

# Example usage and testing
if __name__ == "__main__":
    # Create the enhanced model evaluation module
    eval_module = EnhancedModelEvaluationModule()
    
    # Print available models
    print("Available models:")
    for model_name in eval_module.registry.list_models():
        model_info = eval_module.registry.get_model(model_name)
        print(f"- {model_name}: {model_info.description} ({model_info.category}, {model_info.complexity})")
    
    print("\nEnhanced Model Evaluation Module initialized successfully!")
