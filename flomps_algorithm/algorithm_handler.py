"""
Filename: algorithm_handler.py
Description: Reads satellite link data to perform preprocessing before satellite communication algorithm execution. 
Initial Creator: Elysia Guglielmo (System Architect)
Author: Yuganya Perumal
Date: 2024-06-29
Version: 1.0
Python Version: 

Changelog:
- 2024-06-29: Initial creation.
- 2024-08-24: getters and setters for satellite names.
- 2024-09-09: setting output instead of return and algorithm core to handle algorithm steps.
- 2024-09-22: Implemented Input validation.
- 2024-10-03: Implemented auto generation of satellite names if satellite simulator component does not provide one.
- 2024-10-14: Modified read_adjacency_matrices() method to accomodate modified version of output.txt from sat sim component

Usage: 
Instantiate AlgorithmHandler and assign Algorithm. 
Then run start_algorithm() with adjacency matrices. 
"""
from interfaces.handler import Handler
from flomps_algorithm.algorithm_core import Algorithm
import numpy as npy
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
class AlgorithmHandler(Handler):

    # Constructor
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.adjacency_matrices = []
        self.sat_names = []
    
    def parse_input(self, file):
        # Implementation of abstract method - delegates to parse_file
        return self.parse_file(file)
    
    @staticmethod 
    def read_adjacency_matrices(file_name):
        # Check if file exist.
        if not os.path.isfile(file_name):
            raise FileNotFoundError(f"The file '{file_name}' does not exist.")
        
        adjacency_matrices = []
        satellite_names =[]

        with open(file_name, 'r') as file:
            # Inner function implementaion to skip blank lines
            def next_non_empty_line():
                while True:
                    line = file.readline()
                    if not line:
                        return None  # End of file
                    line = line.strip()
                    if line:  # Skip empty lines
                        return line

            # Read number of satellites
            first_line = next_non_empty_line()
            if first_line.startswith("Number of satellites:"):
                first_line = next_non_empty_line()

            # Read satellite names
            if first_line and "," in first_line:
                satellite_names = [sat_name.strip() for sat_name in first_line.split(",")]
            else:
                buffered_line = first_line

            # Read Adjacency Matrices
            while True:
                line = buffered_line if 'buffered_line' in locals() else next_non_empty_line()
                if not line:
                    break #Reached End of file

                # Check if line starts with string 'Time: '
                if not line.startswith("Time: "):
                    raise ValueError(f"line expected to start with 'Time: ', but read: {line}")

                # Extract the timestamp from line starting with string "Time: "
                if line.startswith("Time: "):
                    time_stamp = line.split("Time: ")[-1].strip()
                
                # Reset the buffer for the next iteration
                if 'buffered_line' in locals():
                    del buffered_line

                # read adjacency matrix for current time stamp.
                matrix = []
                while True:
                    row_line = next_non_empty_line()
                    if not row_line or row_line.startswith("Time: "):
                        buffered_line = row_line
                        break # end of matrix
                    # Check if the matrix row structure is as expected.
                    try:
                        matrix_row = list(map(int, row_line.split()))
                    except ValueError:
                        raise ValueError(f"Matrix row contains unexpected values: {row_line}")

                    # add to matrix after validation
                    matrix.append(matrix_row)  

                    # Check if the matrix rows have consistent lengths
                    if len(matrix_row) != len(matrix[0]):
                        raise ValueError(f"row length is different in the matrix: {row_line}")  
                    # Check if matrix is empty
                    if not matrix:
                        raise ValueError("The matrix is empty.")   
        
                adjacency_matrices.append((time_stamp, npy.array(matrix)))
        return satellite_names, adjacency_matrices

    # Validate incoming adjacency matrices satisfy being square and symmetry.
    def validate_adjacency_matrices(self, adjacency_matrices):
        for index, (timestamp, each_matrix) in enumerate(adjacency_matrices):
            # Check if the Adjacency matrix is empty
            if len(each_matrix) == 0 or len(each_matrix[0]) == 0:
                raise ValueError(f"Adjacency Matrix at time '{timestamp}' is empty.")

            # Check if the Adjacency matrix is square
            if len(each_matrix) != len(each_matrix[0]):
                raise ValueError(f"Adjacency Matrix at time '{timestamp}' is not square: {len(each_matrix)}x{len(each_matrix[0])}")

            # Check if the Adjacency matrix is symmetric
            if not npy.array_equal(npy.array(each_matrix), npy.array(each_matrix).T):
                raise ValueError(f"Adjacency Matrix at time '{timestamp}' is not symmetric.")
        return True

    # Automatically generates satellite names based on the adjacency matrix size, 
    # if satellite names not provided by Satellite Simulator Componenet
    def auto_generate_satellite_names(self,adjacency_matrix_size):
        self.sat_names = [f"Satellite {k+1}" for k in range(adjacency_matrix_size)]
        self.send_satNames(self.sat_names)

    def parse_data(self, data):
        if data is None:
            raise ValueError("No incoming Adjacency Matrix, unable to proceed.")
        else:
            # Handle different types of input data
            if hasattr(data, 'matrices') and hasattr(data, 'satellite_names'):
                # SatSimResult object
                self.adjacency_matrices = data.matrices 
                self.sat_names = data.satellite_names 
            elif isinstance(data, list):
                # Direct list of adjacency matrices
                self.adjacency_matrices = data
                self.sat_names = []  # Will be auto-generated
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
                
            if self.validate_adjacency_matrices(self.adjacency_matrices):
                if not self.sat_names:
                    no_of_rows = self.adjacency_matrices[0][1].shape[0]
                    self.auto_generate_satellite_names(no_of_rows)
                else:
                    self.send_satNames(self.sat_names)
                self.send_adjmatrices(self.adjacency_matrices)

    def parse_file(self, file_name):
        result_read = self.read_adjacency_matrices(file_name)
        if len(result_read) == 1:
            self.adjacency_matrices = result_read[0]
        else:
            self.sat_names, self.adjacency_matrices = result_read
        if self.validate_adjacency_matrices(self.adjacency_matrices):
            if not self.sat_names:
                no_of_rows = self.adjacency_matrices[0][1].shape[0]
                self.auto_generate_satellite_names(no_of_rows)
            else:
                self.send_satNames(self.sat_names)
            self.send_adjmatrices(self.adjacency_matrices)

    def send_adjmatrices(self, adj_matrices):
        self.algorithm.set_adjacency_matrices(adj_matrices)

    def send_satNames(self, sat_names):
        self.algorithm.set_satellite_names(sat_names)

    def run_module(self):
        self.algorithm.start_algorithm_steps() 
        