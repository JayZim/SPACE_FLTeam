"""
Filename: algorithm_config.py
Description: Converts Algorithm JSON options to local data types and configures core Algorithm module via setters. 
Initial Creator: Elysia Guglielmo (System Architect)
Author: Yuganya Perumal, stephen zeng

Date: 2025-05-14
Version: 1.0
Python Version: 3.10

Changelog:
- 2024-07-11: Initial creation.
- 2024-08-24: Reading JSON Options satellite_names to configure the setters.
- 2024-10-03: satellite_names not set by reading JSON Options.
- 2025-05-14: Added Algorithm Tuner by Stephen ZENG

Usage: 

"""
import json
from dataclasses import dataclass, field

from interfaces.config import Config
from flomps_algorithm.algorithm_core import Algorithm

@dataclass
class AlgorithmOptions:
    key: str
    # 添加调优器选项
    tuner_enabled: bool = False
    tuning_iterations: int = 10
    tuning_parameters: dict = field(default_factory=dict)


class AlgorithmConfig(Config):

    # Constructor, accepts core module
    def __init__(self, algorithm:Algorithm):
        self.algorithm = algorithm
        self.options = None

    # Traverse JSON options to check for nested objects
    def read_options(self, options):
        self.options = options
        self.set_algorithm()
        self.set_algorithm_tuner()  # set algorithm tuner if enabled

    def read_options_from_file(self, file):
        return super().read_options_from_file(file)

    def set_algorithm(self):
        self.algorithm.set_output_to_file(self.options["module_settings"]["output_to_file"])
    
    def set_algorithm_tuner(self):  
        if "tuner" in self.options and self.options["tuner"]["enabled"]:  
            from flomps_algorithm.algorithm_tuner import AlgorithmTuner  
            tuner = AlgorithmTuner(self.algorithm)  
            tuner.set_tuning_parameters(self.options["tuner"]["parameters"])  
            self.algorithm.set_tuner(tuner)
