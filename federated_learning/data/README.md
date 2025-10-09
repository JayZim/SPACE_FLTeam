# Federated Learning Dataset Directory

This directory contains the 3 datasets supported by the federated learning system:

## Dataset Structure

```
federated_learning/data/
├── MNIST/           # Handwritten digit dataset
│   └── raw/        # Raw data files
├── CIFAR10/         # Natural image dataset
│   └── cifar-10-batches-py/  # CIFAR-10 data files
└── EuroSAT/         # Satellite image dataset
    └── (auto-created after download)
```

## Dataset Details

### 1. MNIST
- **Description**: Handwritten digit recognition dataset
- **Image Size**: 28x28 grayscale images
- **Number of Classes**: 10 (0-9)
- **Training Samples**: 60,000
- **Test Samples**: 10,000
- **Usage**: Simple model testing and benchmarking

### 2. CIFAR-10
- **Description**: Natural image classification dataset
- **Image Size**: 32x32 color images
- **Number of Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Training Samples**: 50,000
- **Test Samples**: 10,000
- **Usage**: Medium complexity model testing

### 3. EuroSAT
- **Description**: Satellite image land use classification dataset
- **Image Size**: 64x64 color images
- **Number of Classes**: 10 (AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake)
- **Training Samples**: 27,000
- **Usage**: Complex model testing and EuroSAT optimization

## Data Loading

The system automatically loads datasets from their respective subdirectories. If a dataset doesn't exist, the system will automatically download it.

## Notes

- All datasets are stored within the FL module for easy management and deployment
- Dataset paths are relative to the `federated_learning` directory
- Supports automatic download and caching mechanisms
