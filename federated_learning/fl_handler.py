"""
Filename: fl_handler.py
Description: Reads algorithm data to perform preprocessing before running federated learning round.
Author: Nicholas Paul Candra
Date: 2025-05-12
Version: 1.0
Python Version: 3.10+

Changelog:
- 2024-08-02: Initial creation.
- 2024-01-31: Core Integration Setup for FedAvg
- 2025-05-12: Modified the code into TensorFlow + Pytorch

Todo/notes:
- Model Integration +
Usage:
- Used to initialize, parse, and run the federated learning process based on input data.
"""

from interfaces.handler import Handler
from federated_learning.fl_core import FederatedLearning


class FLHandler(Handler):
    def _init_(self, fl_core: FederatedLearning):
        self.federated_learning = fl_core
        self.flam = None

    def parse_data(self, data):
        if data is not None:
            self.flam = data
        else:
            raise FileNotFoundError("FL attempted to parse a FLAM, but it was empty.")
        
        return super().parse_data()

    def parse_file(self, file):
        return super().parse_file(file)

    def extract_client_addresses(self, flam):       
        return flam.get("client_addresses", [])

    def run_module(self):
        
        if self.flam is None:
            # If no FLAM data was provided, run default FL
            self.federated_learning.run()
        else:
            # Set FLAM data and establish connections
            self.federated_learning.set_flam(self.flam)
            client_addresses = self.extract_client_addresses(self.flam)
            self.federated_learning.establish_connection(client_addresses)
            self.federated_learning.run()