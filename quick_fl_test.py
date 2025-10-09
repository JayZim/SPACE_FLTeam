#!/usr/bin/env python3
"""
Quick FL training test - non-interactive mode
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from federated_learning.fl_core import FederatedLearning

print("=" * 80)
print("Quick FL training test")
print("=" * 80)

# Create FL system
print("\n1. Initialize FL system...")
fl = FederatedLearning(enable_adaptation=True)

# Load data
print("2. Load MNIST dataset...")
fl.initialize_data("MNIST", use_heterogeneous=True)

# Initialize model
print("3. Initialize SimpleCNN model...")
fl.initialize_model(auto_select=False, interactive_mode=False)

# Set training parameters
print("4. Set training parameters (2 rounds)...")
fl.set_num_rounds(2)

# Start training
print("\n" + "=" * 80)
print("Start training...")
print("=" * 80 + "\n")

try:
    fl.run()
    
    print("\n" + "=" * 80)
    print("✅ Training completed!")
    print("=" * 80)
    
    if fl.round_accuracies:
        print(f"\nTraining results:")
        for i, acc in enumerate(fl.round_accuracies, 1):
            print(f"  Round {i}: {acc:.4f}")
        print(f"\nFinal accuracy: {fl.round_accuracies[-1]:.4f}")
    
        print("\n🎉 FL system running normally!")
    
except Exception as e:
    print(f"\n✗ Training failed: {e}")
    import traceback
    traceback.print_exc()

