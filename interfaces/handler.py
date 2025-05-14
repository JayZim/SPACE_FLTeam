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
