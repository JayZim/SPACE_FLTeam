# Federated Learning System Architecture

## System Overview

The FL system has been reorganized with clear separation of concerns between model selection and model evaluation modules.

## Architecture Components

### 1. Model Selection Module (`federated_learning/model_selection.py`)
**Purpose**: Interactive model and dataset selection interface

**Key Features**:
- Interactive command-line interface for model/dataset selection
- Model-dataset combination validation
- Automatic recommendations based on dataset
- Performance expectations display
- Training characteristics information

**Classes**:
- `ModelSelection`: Main class for interactive selection

**Supported Models**:
- SimpleCNN (Low complexity)
- ResNet50 (High complexity)
- CustomCNN (Medium complexity)
- EfficientNetB0 (Medium complexity)
- VisionTransformer (High complexity)

**Supported Datasets**:
- MNIST (28x28 grayscale, 60K samples)
- CIFAR10 (32x32 color, 50K samples)
- EuroSAT (64x64 color, 27K samples)

### 2. Model Evaluation Module (`federated_learning/model_evaluation.py`)
**Purpose**: Comprehensive model evaluation and comparison

**Key Features**:
- Model registry with singleton pattern
- Performance evaluation metrics
- Model comparison and ranking
- Memory usage estimation
- Convergence analysis
- Stability and efficiency scoring

**Classes**:
- `ModelRegistry`: Singleton registry for available models
- `ModelEvaluator`: Performance evaluation engine
- `ModelSelector`: Intelligent model selection
- `EnhancedModelEvaluationModule`: Main orchestration class

**Evaluation Metrics**:
- Accuracy
- Loss
- Training time
- Inference time
- Memory usage
- Convergence rounds
- Stability score
- Efficiency score

### 3. Core FL Engine (`federated_learning/fl_core.py`)
**Purpose**: Main federated learning execution engine

**Key Features**:
- Multi-dataset support (MNIST, CIFAR10, EuroSAT)
- Model selection integration
- FLAM file parsing
- Client training with dataset-specific optimizations
- Parameter server aggregation

**Dataset-Specific Optimizations**:
- **EuroSAT**: Adam optimizer, data augmentation, 5 local epochs
- **CIFAR10**: SGD with momentum, data augmentation, 4 local epochs
- **MNIST**: Standard SGD, 3 local epochs

### 4. Data Structure
```
federated_learning/data/
├── MNIST/           # Handwritten digit dataset
├── CIFAR10/         # Natural image dataset
├── EuroSAT/         # Satellite image dataset
└── README.md        # Dataset documentation
```

### 5. Legacy Components
- **`model.py`**: Legacy TensorFlow-based configuration interface
  - Used for backward compatibility and configuration purposes
  - Contains TensorFlow model definitions (not used in actual FL training)
  - Actual FL training uses PyTorch models from `model_evaluation.py`
  - Maintained for configuration interface in `fl_config.py`

### 6. Configuration System (`options.json`)
**Key Settings**:
- Available models and datasets
- Model evaluation parameters
- Dataset-specific optimization settings
- Memory and complexity constraints

## System Flow

1. **Initialization**: Load configuration and initialize modules
2. **Model Selection**: Interactive or automatic model/dataset selection
3. **Data Loading**: Load and preprocess selected dataset
4. **Model Evaluation**: Evaluate model performance (if enabled)
5. **Federated Training**: Execute FL rounds with selected model
6. **Results**: Generate performance metrics and visualizations

## Integration Points

### Model Selection → FL Core
- `fl_core.py` imports `ModelSelection` for interactive selection
- Selected model and dataset passed to FL execution

### Model Evaluation → FL Core
- `fl_core.py` uses `EnhancedModelEvaluationModule` for model evaluation
- Evaluation results integrated into FL workflow

### Configuration → All Modules
- `options.json` provides configuration for all modules
- Centralized settings management

## Key Improvements

1. **Separation of Concerns**: Model selection and evaluation are now separate modules
2. **Singleton Pattern**: Prevents duplicate model registration
3. **Data Organization**: All datasets centralized in FL module
4. **Comprehensive Testing**: Full system validation with `check_fl_system.py`
5. **Interactive Interface**: User-friendly model/dataset selection
6. **Performance Optimization**: Dataset-specific training configurations

## Usage Examples

### Interactive Selection
```python
from federated_learning.model_selection import ModelSelection

selector = ModelSelection(available_models, available_datasets)
model, dataset = selector.select_model_dataset_combination()
```

### Model Evaluation
```python
from federated_learning.model_evaluation import EnhancedModelEvaluationModule

eval_module = EnhancedModelEvaluationModule()
model_name, model = eval_module.select_model_for_fl(data_loader, criterion)
```

### FL Execution
```python
from federated_learning.fl_core import FederatedLearning

fl = FederatedLearning(enable_model_evaluation=True)
fl.run(interactive_mode=True, dataset_name="EuroSAT")
```

## System Validation

The `check_fl_system.py` script performs comprehensive validation:
- File structure verification
- Module import testing
- Model registry validation
- Model selection testing
- Data structure checking
- Configuration validation
- End-to-end workflow testing

All checks passed successfully, confirming the system is working correctly.
