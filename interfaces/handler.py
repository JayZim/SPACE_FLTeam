"""
Filename: handler.py
Description: Handler interface for all component input classes intended to parse data from an expected format. 
Author: Nicholas Paul Candra
Date: 2025-05-14
Version: 1.0
Python Version: 

Changelog:
- 2024-06-29: Initial creation.

Usage: 
Implement this interface by assigning this type to a derived class and defining parse_input() method.
Example:
    class SpecialHandler(Handler)
        parse_input(file):
            # functionality here
"""

import abc
import os
import pandas as pd

class Handler(abc.ABC):
    def __init__(self, federated_learning=None):
        self.federated_learning = federated_learning
        self.flam = None

    @abc.abstractmethod
    def parse_input(self, file):
        """Abstract method to be implemented by subclasses."""
        pass

    def parse_data(self, data):
        if data is not None:
            self.flam = data
            if self.federated_learning:
                self.federated_learning.set_flam(self.flam)
        else:
            raise FileNotFoundError("FL attempted to parse a FLAM, but was empty.")

    def parse_file(self, file):
        if not os.path.isfile(file):
            raise FileNotFoundError(f"The file '{file}' does not exist.")
        try:
            flam_data = pd.read_csv(file)
            self.flam = flam_data
            if self.federated_learning:
                self.federated_learning.set_flam(self.flam)
            return True
        except Exception as e:
            raise ValueError(f"Error parsing FLAM file: {str(e)}")

