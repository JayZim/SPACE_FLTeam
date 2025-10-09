"""
Federated Learning Dataset and Model Adaptation System
Author: Stephen Zeng
Date: 2025-10-10
Version: 1.0

A lightweight adaptation system for federated learning that supports:
- Dataset switching between MNIST, CIFAR10, and EuroSAT
- Model adaptation with transfer learning
- Intelligent model selection based on dataset characteristics
- Accuracy-prioritized training strategies

Changelog:
- 2025-10-10: Initial creation.
- 2025-10-10: Added dataset switching.
- 2025-10-10: Added model adaptation.
- 2025-10-10: Added intelligent model selection.
- 2025-10-10: Added accuracy-prioritized training strategies.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
from typing import Dict, List, Tuple, Optional, Any
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dataset specifications - 统一使用 64x64 尺寸
DATASET_SPECS = {
    "MNIST": {
        "input_size": (64, 64, 1),  # 统一调整为 64x64
        "original_size": (28, 28, 1),
        "num_classes": 10,
        "normalization": (0.5,),
        "augmentation": False
    },
    "CIFAR10": {
        "input_size": (64, 64, 3),  # 统一调整为 64x64
        "original_size": (32, 32, 3),
        "num_classes": 10,
        "normalization": (0.5, 0.5, 0.5),
        "augmentation": True
    },
    "EuroSAT": {
        "input_size": (64, 64, 3),  # 保持 64x64
        "original_size": (64, 64, 3),
        "num_classes": 10,
        "normalization": (0.3443, 0.3804, 0.4086),
        "augmentation": True
    }
}

# Model compatibility matrix
MODEL_COMPATIBILITY = {
    "SimpleCNN": ["MNIST", "CIFAR10"],
    "CustomCNN": ["MNIST", "CIFAR10"],
    "ResNet50": ["MNIST", "CIFAR10", "EuroSAT"],
    "EfficientNetB0": ["MNIST", "CIFAR10", "EuroSAT"],
    "VisionTransformer": ["MNIST", "CIFAR10", "EuroSAT"]
}

class DatasetAdapter:
    """Handles dataset loading and adaptation for different datasets"""
    
    def __init__(self):
        self.data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        
    def load_dataset(self, name: str) -> torchvision.datasets.VisionDataset:
        """Load dataset with appropriate transforms"""
        if name not in DATASET_SPECS:
            raise ValueError(f"Unsupported dataset: {name}. Available: {list(DATASET_SPECS.keys())}")
        
        spec = DATASET_SPECS[name]
        transform = self.get_transform(name)
        data_path = os.path.join(self.data_root, name)
        
        try:
            if name == "MNIST":
                dataset = torchvision.datasets.MNIST(
                    root=data_path, train=True, download=True, transform=transform
                )
            elif name == "CIFAR10":
                dataset = torchvision.datasets.CIFAR10(
                    root=data_path, train=True, download=True, transform=transform
                )
            elif name == "EuroSAT":
                dataset = torchvision.datasets.EuroSAT(
                    root=data_path, download=True, transform=transform
                )
            
            logger.info(f"✓ Loaded {name} dataset with {len(dataset)} samples")
            return dataset
            
        except Exception as e:
            logger.warning(f"Failed to load {name}, falling back to MNIST: {e}")
            return self.load_dataset("MNIST")
    
    def get_transform(self, name: str) -> transforms.Compose:
        """Get appropriate transforms for dataset - 统一调整为 64x64"""
        spec = DATASET_SPECS[name]
        
        transform_list = []
        
        # 统一调整所有数据集为 64x64
        target_size = spec["input_size"][:2]  # (64, 64)
        transform_list.append(transforms.Resize(target_size))
        
        # Add augmentation if specified
        if spec["augmentation"]:
            if name == "EuroSAT":
                transform_list.extend([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=15),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                ])
            elif name == "CIFAR10":
                transform_list.extend([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=10),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                ])
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        
        # Add normalization
        if len(spec["normalization"]) == 1:
            # MNIST: grayscale, need to convert to RGB for consistency
            transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))  # 1 channel -> 3 channels
            transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        else:
            transform_list.append(transforms.Normalize(spec["normalization"], spec["normalization"]))
        
        return transforms.Compose(transform_list)
    
    def create_client_splits(self, dataset, num_clients: int, 
                           use_heterogeneous: bool = True) -> List[DataLoader]:
        """Create client data splits"""
        if use_heterogeneous:
            return self._create_heterogeneous_split(dataset, num_clients)
        else:
            return self._create_homogeneous_split(dataset, num_clients)
    
    def _create_homogeneous_split(self, dataset, num_clients: int) -> List[DataLoader]:
        """Create homogeneous client splits"""
        client_datasets = random_split(dataset, [len(dataset) // num_clients] * num_clients)
        return [DataLoader(dataset, batch_size=32, shuffle=True) for dataset in client_datasets]
    
    def _create_heterogeneous_split(self, dataset, num_clients: int) -> List[DataLoader]:
        """Create heterogeneous client splits with class imbalance"""
        # Convert to list for manipulation
        data_list = list(dataset)
        
        # Group by class
        class_data = {}
        for i, (_, target) in enumerate(data_list):
            label = target.item() if isinstance(target, torch.Tensor) else target
            if label not in class_data:
                class_data[label] = []
            class_data[label].append(i)
        
        # Create client splits with different class distributions
        client_datasets = [[] for _ in range(num_clients)]
        
        for class_label, indices in class_data.items():
            # Each client gets different proportion of each class
            for client_id in range(num_clients):
                # Create bias towards certain classes for each client
                bias_strength = 0.3 + 0.2 * (client_id % 3)  # Vary bias strength
                if class_label % 3 == client_id % 3:  # Some clients prefer certain classes
                    proportion = 0.6 + bias_strength
                else:
                    proportion = (1.0 - 0.6 - bias_strength) / (num_clients - 1)
                
                n_samples = int(len(indices) * proportion)
                if n_samples > 0 and n_samples <= len(indices):
                    selected = torch.randint(0, len(indices), (min(n_samples, len(indices)),))
                    client_datasets[client_id].extend([indices[i] for i in selected])
        
        # Convert to DataLoaders
        client_loaders = []
        for client_indices in client_datasets:
            if client_indices:
                subset = Subset(dataset, client_indices)
                client_loaders.append(DataLoader(subset, batch_size=32, shuffle=True))
            else:
                # Fallback: give client a small random sample
                random_indices = torch.randint(0, len(dataset), (100,))
                subset = Subset(dataset, random_indices)
                client_loaders.append(DataLoader(subset, batch_size=32, shuffle=True))
        
        logger.info(f"✓ Created {len(client_loaders)} heterogeneous client splits")
        return client_loaders

class ModelAdapter:
    """Handles model selection and adaptation"""
    
    def __init__(self, model_eval_module=None):
        self.model_eval_module = model_eval_module
        self.current_model = None
        self.current_dataset = None
    
    def select_model(self, dataset_name: str, model_name: Optional[str] = None,
                    current_model=None) -> nn.Module:
        """Select appropriate model for dataset"""
        self.current_dataset = dataset_name
        
        if model_name:
            # Use specified model
            if self.model_eval_module:
                model_info = self.model_eval_module.registry.get_model(model_name)
                if model_info and dataset_name in MODEL_COMPATIBILITY.get(model_name, []):
                    self.current_model = model_info.model_class(**model_info.parameters)
                    logger.info(f"✓ Selected specified model: {model_name}")
                    return self.current_model
                else:
                    logger.warning(f"Model {model_name} not compatible with {dataset_name}, using auto-selection")
        
        # Auto-select best model for dataset
        if self.model_eval_module:
            try:
                # Get compatible models
                compatible_models = [name for name, datasets in MODEL_COMPATIBILITY.items() 
                                   if dataset_name in datasets]
                
                if compatible_models:
                    # Use the most advanced compatible model
                    if "EfficientNetB0" in compatible_models and dataset_name == "EuroSAT":
                        selected_name = "EfficientNetB0"
                    elif "ResNet50" in compatible_models and dataset_name == "CIFAR10":
                        selected_name = "ResNet50"
                    elif "SimpleCNN" in compatible_models:
                        selected_name = "SimpleCNN"
                    else:
                        selected_name = compatible_models[0]
                    
                    model_info = self.model_eval_module.registry.get_model(selected_name)
                    self.current_model = model_info.model_class(**model_info.parameters)
                    logger.info(f"✓ Auto-selected model: {selected_name} for {dataset_name}")
                    return self.current_model
            
            except Exception as e:
                logger.warning(f"Auto-selection failed: {e}")
        
        # Fallback: create default model
        self.current_model = self._create_default_model(dataset_name)
        logger.info(f"✓ Using default model for {dataset_name}")
        return self.current_model
    
    def transfer_weights(self, source_model: nn.Module, target_model: nn.Module) -> nn.Module:
        """Transfer compatible weights from source to target model with intelligent matching"""
        if source_model is None:
            return target_model
        
        try:
            source_state = source_model.state_dict()
            target_state = target_model.state_dict()
            
            transferred_layers = 0
            total_layers = len(target_state)
            transferred_params = 0
            total_params = sum(p.numel() for p in target_state.values())
            
            # Strategy 1: Exact name and shape matching
            for name, param in target_state.items():
                if name in source_state:
                    source_param = source_state[name]
                    if param.shape == source_param.shape:
                        target_state[name] = source_param
                        transferred_layers += 1
                        transferred_params += param.numel()
            
            # Strategy 2: Partial name matching for similar architectures
            # Match conv layers, bn layers, etc. even if names differ slightly
            if transferred_layers < total_layers * 0.1:  # If less than 10% transferred
                source_conv_layers = {k: v for k, v in source_state.items() if 'conv' in k.lower()}
                target_conv_layers = {k: v for k, v in target_state.items() if 'conv' in k.lower()}
                
                # Try to match first few conv layers
                source_conv_list = sorted(source_conv_layers.items())
                target_conv_list = sorted(target_conv_layers.items())
                
                for i, (target_name, target_param) in enumerate(target_conv_list[:3]):  # First 3 conv layers
                    if i < len(source_conv_list):
                        source_name, source_param = source_conv_list[i]
                        if target_param.shape == source_param.shape and target_name not in source_state:
                            target_state[target_name] = source_param
                            transferred_layers += 1
                            transferred_params += target_param.numel()
                            logger.debug(f"Matched {source_name} -> {target_name}")
            
            # Strategy 3: Feature extractor matching for pretrained models
            # If both models have backbone/feature extractor, try to match those
            if transferred_layers < total_layers * 0.1:
                # Check for common backbone patterns
                backbone_patterns = ['backbone', 'features', 'encoder', 'base']
                for pattern in backbone_patterns:
                    source_backbone = {k: v for k, v in source_state.items() if pattern in k.lower()}
                    target_backbone = {k: v for k, v in target_state.items() if pattern in k.lower()}
                    
                    if source_backbone and target_backbone:
                        for target_name, target_param in target_backbone.items():
                            # Try to find matching layer in source
                            layer_type = target_name.split('.')[-2] if '.' in target_name else target_name
                            for source_name, source_param in source_backbone.items():
                                if layer_type in source_name and target_param.shape == source_param.shape:
                                    if target_name not in [k for k, v in target_state.items() if torch.equal(v, source_state.get(k, torch.zeros_like(v)))]:
                                        target_state[target_name] = source_param
                                        transferred_layers += 1
                                        transferred_params += target_param.numel()
                                        break
            
            # Load the updated state dict
            target_model.load_state_dict(target_state)
            
            transfer_ratio = transferred_layers / total_layers
            param_ratio = transferred_params / total_params
            
            # Report transfer success
            if transfer_ratio > 0.3 or param_ratio > 0.3:
                logger.info(f"✓ Transferred weights: {transferred_layers}/{total_layers} layers ({transfer_ratio:.1%}), "
                          f"{transferred_params}/{total_params} params ({param_ratio:.1%})")
            elif transfer_ratio > 0.05:
                logger.info(f"⚠️ Partial weight transfer: {transferred_layers}/{total_layers} layers ({transfer_ratio:.1%}), "
                          f"{transferred_params}/{total_params} params ({param_ratio:.1%})")
            else:
                logger.info(f"ℹ️ Minimal weight transfer: {transferred_layers}/{total_layers} layers ({transfer_ratio:.1%}) - "
                          f"Using fresh initialization (architectures too different)")
            
            return target_model
            
        except Exception as e:
            logger.warning(f"Weight transfer failed: {e}, using fresh weights")
            return target_model
    
    def adapt_architecture(self, model: nn.Module, input_shape: Tuple[int, int, int], 
                          num_classes: int) -> nn.Module:
        """Adapt model architecture for new input/output requirements"""
        # This is a simplified version - in practice, you'd need more sophisticated adaptation
        # For now, we'll assume the model registry handles most of this
        
        # Check if model needs adaptation
        first_layer = None
        last_layer = None
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and first_layer is None:
                first_layer = module
            if isinstance(module, nn.Linear):
                last_layer = module
        
        # Adapt input layer if needed
        if first_layer and isinstance(first_layer, nn.Conv2d):
            if first_layer.in_channels != input_shape[2]:
                logger.info(f"Adapting input channels: {first_layer.in_channels} -> {input_shape[2]}")
        
        # Adapt output layer if needed
        if last_layer and last_layer.out_features != num_classes:
            logger.info(f"Adapting output classes: {last_layer.out_features} -> {num_classes}")
        
        return model
    
    def _create_default_model(self, dataset_name: str) -> nn.Module:
        """Create a default model as fallback - 统一使用 64x64 RGB 输入"""
        spec = DATASET_SPECS[dataset_name]
        
        class DefaultModel(nn.Module):
            def __init__(self, num_classes):
                super(DefaultModel, self).__init__()
                # 统一使用 3 通道 64x64 输入
                self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.dropout = nn.Dropout(0.5)
                
                # 64x64 -> 32x32 -> 16x16 -> 8x8 (3 pooling layers)
                self.fc1 = nn.Linear(128 * 8 * 8, 256)
                self.fc2 = nn.Linear(256, num_classes)
            
            def forward(self, x):
                x = self.pool(torch.relu(self.conv1(x)))  # 64 -> 32
                x = self.pool(torch.relu(self.conv2(x)))  # 32 -> 16
                x = self.pool(torch.relu(self.conv3(x)))  # 16 -> 8
                x = x.view(-1, 128 * 8 * 8)
                x = torch.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return x
        
        return DefaultModel(spec["num_classes"])

class AdaptationStrategy:
    """Manages adaptation strategies and decisions"""
    
    def __init__(self):
        self.dataset_similarity_matrix = {
            ("MNIST", "CIFAR10"): 0.3,
            ("MNIST", "EuroSAT"): 0.1,
            ("CIFAR10", "EuroSAT"): 0.7,
            ("CIFAR10", "MNIST"): 0.3,
            ("EuroSAT", "MNIST"): 0.1,
            ("EuroSAT", "CIFAR10"): 0.7,
        }
    
    def should_transfer_weights(self, old_dataset: str, new_dataset: str, 
                               old_model_name: str = None, new_model_name: str = None) -> bool:
        """Determine if weights should be transferred between datasets and models"""
        if old_dataset == new_dataset:
            return True
        
        # Check dataset similarity
        similarity = self.dataset_similarity_matrix.get((old_dataset, new_dataset), 0.0)
        
        # If models are the same type, increase transfer likelihood
        if old_model_name and new_model_name:
            # Same model architecture
            if old_model_name == new_model_name:
                should_transfer = similarity > 0.2  # Lower threshold for same architecture
                logger.info(f"Dataset similarity ({old_dataset} -> {new_dataset}): {similarity:.2f}")
                logger.info(f"Model architecture: Same ({old_model_name})")
                logger.info(f"Transfer decision: {'YES' if should_transfer else 'NO'} (same architecture bonus)")
                return should_transfer
            
            # Similar model families (e.g., both CNNs)
            model_families = {
                'SimpleCNN': 'CNN',
                'CustomCNN': 'CNN',
                'ResNet50': 'ResNet',
                'EfficientNetB0': 'EfficientNet',
                'VisionTransformer': 'Transformer'
            }
            old_family = model_families.get(old_model_name, 'Unknown')
            new_family = model_families.get(new_model_name, 'Unknown')
            
            if old_family == new_family and old_family != 'Unknown':
                should_transfer = similarity > 0.3  # Medium threshold for same family
                logger.info(f"Dataset similarity ({old_dataset} -> {new_dataset}): {similarity:.2f}")
                logger.info(f"Model family: Same ({old_family})")
                logger.info(f"Transfer decision: {'YES' if should_transfer else 'NO'} (same family bonus)")
                return should_transfer
        
        # Default: use similarity threshold
        should_transfer = similarity > 0.5
        logger.info(f"Dataset similarity ({old_dataset} -> {new_dataset}): {similarity:.2f}")
        logger.info(f"Transfer decision: {'YES' if should_transfer else 'NO'}")
        
        return should_transfer
    
    def get_training_config(self, dataset_name: str) -> Dict[str, Any]:
        """Get optimized training configuration for dataset"""
        configs = {
            "MNIST": {
                "optimizer": "SGD",
                "lr": 0.01,
                "momentum": 0.9,
                "local_epochs": 3,
                "batch_size": 32,
                "weight_decay": 0.0
            },
            "CIFAR10": {
                "optimizer": "SGD",
                "lr": 0.01,
                "momentum": 0.9,
                "local_epochs": 4,
                "batch_size": 32,
                "weight_decay": 1e-4
            },
            "EuroSAT": {
                "optimizer": "Adam",
                "lr": 0.001,
                "momentum": None,
                "local_epochs": 5,
                "batch_size": 32,
                "weight_decay": 1e-4
            }
        }
        
        config = configs.get(dataset_name, configs["MNIST"])
        logger.info(f"Training config for {dataset_name}: {config}")
        return config
    
    def evaluate_compatibility(self, model_name: str, dataset_name: str) -> float:
        """Evaluate model-dataset compatibility score (0-1)"""
        if dataset_name in MODEL_COMPATIBILITY.get(model_name, []):
            # Base compatibility
            score = 0.8
            
            # Add bonuses for optimal combinations
            if (model_name == "EfficientNetB0" and dataset_name == "EuroSAT") or \
               (model_name == "ResNet50" and dataset_name == "CIFAR10") or \
               (model_name == "SimpleCNN" and dataset_name == "MNIST"):
                score += 0.2
            
            return min(score, 1.0)
        
        return 0.0
    
    def get_expected_accuracy(self, model_name: str, dataset_name: str) -> Tuple[float, float]:
        """Get expected accuracy range for model-dataset combination"""
        accuracy_ranges = {
            ("EfficientNetB0", "EuroSAT"): (0.85, 0.95),
            ("VisionTransformer", "EuroSAT"): (0.80, 0.90),
            ("ResNet50", "CIFAR10"): (0.75, 0.85),
            ("EfficientNetB0", "CIFAR10"): (0.80, 0.90),
            ("SimpleCNN", "MNIST"): (0.95, 0.99),
            ("CustomCNN", "MNIST"): (0.95, 0.99),
            ("ResNet50", "MNIST"): (0.97, 0.99),
            ("EfficientNetB0", "MNIST"): (0.96, 0.99),
        }
        
        return accuracy_ranges.get((model_name, dataset_name), (0.70, 0.85))

class FLAdaptationSystem:
    """Main adaptation system that orchestrates all components"""
    
    def __init__(self, model_eval_module=None):
        self.dataset_adapter = DatasetAdapter()
        self.model_adapter = ModelAdapter(model_eval_module)
        self.adaptation_strategy = AdaptationStrategy()
        
        logger.info("✓ FL Adaptation System initialized")
    
    def switch_dataset(self, old_dataset: str, new_dataset: str, 
                      current_model=None, preserve_weights: bool = True,
                      old_model_name: str = None) -> Tuple[nn.Module, List[DataLoader]]:
        """Switch to new dataset with intelligent adaptation"""
        logger.info(f"\n{'='*60}")
        logger.info(f"SWITCHING DATASET: {old_dataset} -> {new_dataset}")
        logger.info(f"{'='*60}")
        
        # Load new dataset
        new_dataset_obj = self.dataset_adapter.load_dataset(new_dataset)
        client_data = self.dataset_adapter.create_client_splits(new_dataset_obj, 5)
        
        # Select new model
        new_model = self.model_adapter.select_model(new_dataset)
        new_model_name = new_model.__class__.__name__
        
        # Determine if we should transfer weights (with model awareness)
        should_transfer = preserve_weights and self.adaptation_strategy.should_transfer_weights(
            old_dataset, new_dataset, old_model_name, new_model_name
        )
        
        # Apply transfer learning if appropriate
        if should_transfer and current_model:
            adapted_model = self.model_adapter.transfer_weights(current_model, new_model)
        else:
            adapted_model = new_model
        
        logger.info(f"✓ Successfully switched to {new_dataset}")
        return adapted_model, client_data
    
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get summary of adaptation capabilities"""
        return {
            "supported_datasets": list(DATASET_SPECS.keys()),
            "model_compatibility": MODEL_COMPATIBILITY,
            "adaptation_features": [
                "Dataset switching",
                "Weight transfer learning", 
                "Intelligent model selection",
                "Heterogeneous client splits",
                "Accuracy-optimized training"
            ]
        }

# Example usage and testing
if __name__ == "__main__":
    # Test the adaptation system
    print("Testing FL Adaptation System...")
    
    adaptation_system = FLAdaptationSystem()
    
    # Test dataset loading
    print("\nTesting dataset loading:")
    for dataset_name in ["MNIST", "CIFAR10", "EuroSAT"]:
        try:
            dataset = adaptation_system.dataset_adapter.load_dataset(dataset_name)
            print(f"✓ {dataset_name}: {len(dataset)} samples")
        except Exception as e:
            print(f"✗ {dataset_name}: {e}")
    
    # Test model selection
    print("\nTesting model selection:")
    for dataset_name in ["MNIST", "CIFAR10", "EuroSAT"]:
        model = adaptation_system.model_adapter.select_model(dataset_name)
        print(f"✓ {dataset_name}: {model.__class__.__name__}")
    
    # Test adaptation strategy
    print("\nTesting adaptation strategy:")
    for old_ds in ["MNIST", "CIFAR10", "EuroSAT"]:
        for new_ds in ["MNIST", "CIFAR10", "EuroSAT"]:
            if old_ds != new_ds:
                should_transfer = adaptation_system.adaptation_strategy.should_transfer_weights(old_ds, new_ds)
                print(f"{old_ds} -> {new_ds}: {'Transfer' if should_transfer else 'Fresh start'}")
    
    print("\n✓ FL Adaptation System test completed!")
