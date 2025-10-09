#!/usr/bin/env python3
"""
FL training test - unified 64x64 system
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from federated_learning.fl_core import FederatedLearning

def test_fl_training():
    """Test basic FL training"""
    print("=" * 80)
    print("FL training test - unified 64x64 system")
    print("=" * 80)
    
    # Create FL system
    fl = FederatedLearning(enable_adaptation=True)
    print("✓ FL system initialized successfully\n")
    
    # Initialize MNIST dataset
    print("Load MNIST dataset...")
    fl.initialize_data("MNIST", use_heterogeneous=True)
    print("✓ MNIST dataset loaded successfully\n")
    
    # Initialize model
    print("Initialize SimpleCNN model...")
    fl.initialize_model(auto_select=False, interactive_mode=False)
    print(f"✓ Model initialized successfully: {fl.selected_model_name}\n")
    
    # Set training parameters
    fl.set_num_rounds(3)  # Train 3 rounds
    print("✓ Set training rounds: 3\n")
    
    # Start training
    print("=" * 80)
    print("Start FL training...")
    print("=" * 80)
    
    try:
        fl.run()
        print("\n" + "=" * 80)
        print("✅ FL training completed!")
        print("=" * 80)
        
        # Display training results
        if fl.round_accuracies:
            print(f"\nTraining results:")
            for i, acc in enumerate(fl.round_accuracies, 1):
                print(f"  Round {i}: Accuracy = {acc:.4f}")
                print(f"\nFinal accuracy: {fl.round_accuracies[-1]:.4f}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ FL training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fl_training()
    
    if success:
        print("\n🎉 FL training test successful! System running normally.")
    else:
        print("\n⚠️ FL training test failed, please check the logs.")

