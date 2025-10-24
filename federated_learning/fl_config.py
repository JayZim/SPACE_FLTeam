"""
Filename: fl_config.py
Description: Converts Federated Learning JSON options to local data types and configures core Federated Learning module via setters. 
Author: Kimsakona Sok & Stephen zeng
Date: 2025-09-24
Version: 1.0
Python Version: 

Changelog:
- 2024-08-02: Initial creation.
- 2024-09-16: Supported standalone execution, refactored Model Setters
- 2025-05-15: Added JSON file reading, and printing the options out to see the connection
- 2025-09-24: Added Enhanced Model Evaluation Module integration
Usage: 
- Import this module and call read_options_from_file() with a JSON config file path.
- Alternatively, run the file directly to test configuration loading.

# NOTE: model.py is a legacy TensorFlow-based configuration interface
# It provides model and dataset configuration but uses TensorFlow while FL core uses PyTorch
# Currently kept for backward compatibility but actual FL training uses PyTorch models from model_evaluation.py
"""

import json
from dataclasses import dataclass
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from federated_learning.fl_core import FederatedLearning
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interfaces.output import Output

@dataclass
class AlgorithmOptions:
    key: str
    node_processing_time: int
    search_depth: int

class Config:
    # Constructor, accepts core module
    def __init__(self, federated_learning: FederatedLearning = None):
        self.fl_core = federated_learning
        # self.model = Model()  # Removed: model_simple.py is no longer needed
        self.options = None
        self.model_options = None
        
        # Set default values for standalone usage
        if self.fl_core is None:
            self.model_type = "SimpleCNN"
            self.data_set = "MNIST"
            self.num_rounds = 10
            self.num_clients = 5
        else:
            # Use values from fl_core if available
            self.model_type = getattr(self.fl_core, 'model_type', "SimpleCNN")
            self.data_set = getattr(self.fl_core, 'data_set', "MNIST")
            self.num_rounds = getattr(self.fl_core, 'num_rounds', 10)
            self.num_clients = getattr(self.fl_core, 'num_clients', 5)

    def read_options(self, options: dict):
        self.options = options
        self.set_federated_learning_model()
        self.set_federated_learning()
        
        # Enable model evaluation if configured
        if self.options.get("model_evaluation", {}).get("enable_auto_selection", False):
            self.fl_core.enable_model_evaluation()

    def read_options_from_file(self, file_path: str):
        with open(file_path, 'r') as file:
            full_config = json.load(file)
        
        options = full_config.get("federated_learning", {})
        self.read_options(options)

    def set_federated_learning(self) -> None:
        self.fl_core.set_num_rounds(self.options["num_rounds"])
        self.fl_core.set_num_clients(self.options["num_clients"])

        if hasattr(self.fl_core, 'print_config_summary'):
            self.fl_core.print_config_summary(
                model_type=self.options["model_type"],
                data_set=self.options["data_set"]
            )
        print ("FL setters called")

    def set_federated_learning_model(self) -> None:
        # NOTE: Model configuration is now handled directly in fl_core.py
        # The actual PyTorch models are created in initialize_model() using model_evaluation.py
        # No need for legacy model configuration object
        
        # Set model type and dataset directly on fl_core for configuration
        self.fl_core.model_type = self.options["model_type"]
        self.fl_core.data_set = self.options["data_set"]
        
        # Store available options for interactive selection
        if hasattr(self.fl_core, 'available_models'):
            self.fl_core.available_models = self.options.get("available_models", ["SimpleCNN", "ResNet50","EfficientNetB0", "VisionTransformer", "CustomCNN"])
        if hasattr(self.fl_core, 'available_datasets'):
            self.fl_core.available_datasets = self.options.get("available_datasets", ["MNIST", "CIFAR10", "EuroSAT"])
        
        print ("MODEL setters called")

if __name__ == "__main__":
    from federated_learning.fl_core import FederatedLearning

    fl_instance = FederatedLearning()
    config = Config(fl_instance)

    config.read_options_from_file("options.json")
