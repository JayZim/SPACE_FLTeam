"""
Filename: algorithm_config.py
Description: Converts Algorithm JSON options to local data types and configures core Algorithm module via setters. 
Initial Creator: Elysia Guglielmo (System Architect)
Author: Yuganya Perumal
Date: 2024-07-11
Version: 1.0
Python Version: 

Changelog:
- 2024-07-11: Initial creation.
- 2024-08-24: Reading JSON Options satellite_names to configure the setters.
- 2024-10-03: satellite_names not set by reading JSON Options.

Usage: 

"""
import json
from dataclasses import dataclass

from interfaces.config import Config
from flomps_algorithm.algorithm_core import Algorithm

@dataclass
class AlgorithmOptions:
    key: str


class AlgorithmConfig(Config):

    # Constructor, accepts core module
    def __init__(self, algorithm:Algorithm):
        self.algorithm = algorithm
        self.options = None

    # Traverse JSON options to check for nested objects
    def read_options(self, options):
        self.options = options
        self.set_algorithm()

    def read_options_from_file(file):
         return super().read_options_from_file()

    def set_algorithm(self):
        self.algorithm.set_output_to_file(self.options["module_settings"]["output_to_file"])

        # Set server selection parameters
        if "server_selection" in self.options:
            server_selection = self.options["server_selection"]
            self.algorithm.set_connect_to_all_satellites(
                server_selection.get("connect_to_all_satellites", False)
            )
            self.algorithm.set_max_lookahead(
                server_selection.get("max_lookahead", 20)
            )
            self.algorithm.set_minimum_connected_satellites(
                server_selection.get("minimum_connected_satellites", 5)
            )

