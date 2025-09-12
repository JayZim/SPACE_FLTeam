#!/usr/bin/env python3
"""
Interactive Model Selection Test
Author: Stephen zeng
Date: 2025-09-05
Version: 1.0

This script tests the interactive model selection functionality
of the Enhanced Model Evaluation Module.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from federated_learning.fl_core import FederatedLearning

def test_interactive_model_selection():
    """Test interactive model selection"""
    print("=" * 60)
    print("Interactive Model Selection Test")
    print("=" * 60)
    
    # Create FL instance with model evaluation enabled
    print("Creating FL instance with model evaluation...")
    fl = FederatedLearning(enable_model_evaluation=True)
    fl.set_num_rounds(2)  # Reduced rounds for testing
    fl.set_num_clients=3  # Reduced clients for testing
    
    print("✓ FL instance created with model evaluation enabled")
    
    # List available models
    print("\nAvailable models:")
    available_models = fl.list_available_models()
    for model in available_models:
        print(f"  - {model}")
    
    # Run FL with interactive model selection
    print("\n" + "="*60)
    print("Starting Federated Learning with Interactive Model Selection")
    print("="*60)
    
    try:
        # This will show the interactive menu
        fl.run(flam_path=None, auto_select_model=True, interactive_mode=True)
        
        print("\n" + "="*60)
        print("Federated Learning Completed Successfully!")
        print("="*60)
        
        # Get results
        eval_summary = fl.get_model_evaluation_summary()
        print(f"Selected model: {eval_summary['selected_model']}")
        print(f"Model evaluation enabled: {eval_summary['model_evaluation_enabled']}")
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nError during FL execution: {e}")
        import traceback
        traceback.print_exc()

def test_non_interactive_mode():
    """Test non-interactive mode (auto-selection)"""
    print("\n" + "=" * 60)
    print("Non-Interactive Mode Test (Auto-Selection)")
    print("=" * 60)
    
    # Create FL instance
    fl = FederatedLearning(enable_model_evaluation=True)
    fl.set_num_rounds(1)  # Very short test
    fl.set_num_clients=2
    
    try:
        # This will auto-select without user interaction
        fl.run(flam_path=None, auto_select_model=True, interactive_mode=False)
        
        eval_summary = fl.get_model_evaluation_summary()
        print(f"Auto-selected model: {eval_summary['selected_model']}")
        
    except Exception as e:
        print(f"Error in non-interactive mode: {e}")

def test_specific_model():
    """Test using a specific model"""
    print("\n" + "=" * 60)
    print("Specific Model Test")
    print("=" * 60)
    
    # Create FL instance
    fl = FederatedLearning(enable_model_evaluation=True)
    fl.set_num_rounds(1)
    fl.set_num_clients=2
    
    # List available models
    available_models = fl.list_available_models()
    if available_models:
        # Use the first available model
        model_name = available_models[0]
        print(f"Using specific model: {model_name}")
        
        try:
            fl.run(flam_path=None, model_name=model_name, auto_select_model=False)
            
            eval_summary = fl.get_model_evaluation_summary()
            print(f"Used model: {eval_summary['selected_model']}")
            
        except Exception as e:
            print(f"Error with specific model: {e}")
    else:
        print("No models available for testing")

def main():
    """Run all tests"""
    print("Enhanced Model Evaluation - Interactive Selection Tests")
    print("=" * 60)
    
    try:
        # Test 1: Interactive model selection
        test_interactive_model_selection()
        
        # Test 2: Non-interactive mode
        test_non_interactive_mode()
        
        # Test 3: Specific model
        test_specific_model()
        
        print("\n" + "=" * 60)
        print("All tests completed! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nTests failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
