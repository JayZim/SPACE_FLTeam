"""
Filename: handler.py
Description: Handler interface for all component input classes intended to parse data from an expected format.
Author: Elysia Guglielmo
Date: 2024-06-29
Version: 1.0
"""

import abc
import os
import pandas as pd

class Handler(abc.ABC):
    def __init__(self):
        self.flam = None
        self.federated_learning = None

    @abc.abstractmethod
    def parse_input(self, file):
        pass

    def parse_data(self, data):
        if data is not None:
            self.flam = data
        else:
            raise FileNotFoundError("FL attempted to parse a FLAM, but it was empty.")
        return self.flam

    def parse_file(self, file):
        if not os.path.isfile(file):
            raise FileNotFoundError(f"The file '{file}' does not exist.")
        try:
            flam_data = pd.read_csv(file)  # Adjust to .read_json if needed
            self.flam = flam_data

            # REMOVED: self.federated_learning.set_flam(self.flam)

            return True
        except Exception as e:
            raise ValueError(f"Error parsing FLAM file: {str(e)}")
