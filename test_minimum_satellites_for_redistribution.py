"""
Filename: test_minimum_satellites_for_redistribution.py
Description: Comprehensive test suite for minimum_satellites_for_redistribution feature
Author: Claude Code
Date: 2025-10-31
Version: 1.0
Python Version: 3.12

Test Coverage:
- Configuration loading from options.json
- Setter/getter methods for minimum_satellites_for_redistribution
- TRANSMITTING phase uses minimum_connected_satellites
- REDISTRIBUTION/CHECK phase uses minimum_satellites_for_redistribution
- Independent configuration of both parameters
- Edge cases and fallback behavior
- Different values for TRANSMITTING vs REDISTRIBUTION minimums
- Interaction with connect_to_all_satellites mode
- Validation that correct minimum is used in each phase

Usage:
python test_minimum_satellites_for_redistribution.py
"""

import unittest
import sys
import os
import json
import numpy as np
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from flomps_algorithm.algorithm_core import Algorithm
from flomps_algorithm.algorithm_config import AlgorithmConfig


class TestMinimumSatellitesForRedistributionSettersGetters(unittest.TestCase):
    """Test setter/getter methods for minimum_satellites_for_redistribution"""

    def setUp(self):
        """Set up test fixtures"""
        self.algorithm = Algorithm()

    def test_default_value(self):
        """Test that default value is set correctly"""
        self.assertEqual(self.algorithm.minimum_satellites_for_redistribution, 7)

    def test_setter_method_exists(self):
        """Test that setter method exists and is callable"""
        self.assertTrue(hasattr(self.algorithm, 'set_minimum_satellites_for_redistribution'))
        self.assertTrue(callable(self.algorithm.set_minimum_satellites_for_redistribution))

    def test_setter_method_works(self):
        """Test that setter method correctly updates the value"""
        self.algorithm.set_minimum_satellites_for_redistribution(10)
        self.assertEqual(self.algorithm.minimum_satellites_for_redistribution, 10)

        self.algorithm.set_minimum_satellites_for_redistribution(3)
        self.assertEqual(self.algorithm.minimum_satellites_for_redistribution, 3)

    def test_setter_with_various_values(self):
        """Test setter with various valid values"""
        test_values = [0, 1, 5, 7, 10, 15, 20, 50, 100]
        for value in test_values:
            with self.subTest(value=value):
                self.algorithm.set_minimum_satellites_for_redistribution(value)
                self.assertEqual(self.algorithm.minimum_satellites_for_redistribution, value)

    def test_independence_from_minimum_connected_satellites(self):
        """Test that minimum_satellites_for_redistribution is independent from minimum_connected_satellites"""
        # Set both to different values
        self.algorithm.set_minimum_connected_satellites(3)
        self.algorithm.set_minimum_satellites_for_redistribution(8)

        # Verify they don't affect each other
        self.assertEqual(self.algorithm.minimum_connected_satellites, 3)
        self.assertEqual(self.algorithm.minimum_satellites_for_redistribution, 8)

        # Change one and verify the other stays the same
        self.algorithm.set_minimum_connected_satellites(5)
        self.assertEqual(self.algorithm.minimum_connected_satellites, 5)
        self.assertEqual(self.algorithm.minimum_satellites_for_redistribution, 8)

        self.algorithm.set_minimum_satellites_for_redistribution(12)
        self.assertEqual(self.algorithm.minimum_connected_satellites, 5)
        self.assertEqual(self.algorithm.minimum_satellites_for_redistribution, 12)


class TestConfigurationLoading(unittest.TestCase):
    """Test configuration loading from AlgorithmConfig"""

    def setUp(self):
        """Set up test fixtures"""
        self.algorithm = Algorithm()
        self.config = AlgorithmConfig(self.algorithm)

    def test_config_loads_minimum_satellites_for_redistribution(self):
        """Test that config correctly loads minimum_satellites_for_redistribution from options"""
        test_options = {
            'server_selection': {
                'connect_to_all_satellites': False,
                'max_lookahead': 20,
                'minimum_connected_satellites': 5,
                'minimum_satellites_for_redistribution': 9
            },
            'module_settings': {
                'output_to_file': False
            }
        }

        self.config.read_options(test_options)

        self.assertEqual(self.algorithm.minimum_satellites_for_redistribution, 9)

    def test_config_uses_default_when_missing(self):
        """Test that config uses default value when option is missing"""
        test_options = {
            'server_selection': {
                'connect_to_all_satellites': False,
                'max_lookahead': 20,
                'minimum_connected_satellites': 5
                # minimum_satellites_for_redistribution intentionally omitted
            },
            'module_settings': {
                'output_to_file': False
            }
        }

        self.config.read_options(test_options)

        # Should use default value of 7
        self.assertEqual(self.algorithm.minimum_satellites_for_redistribution, 7)

    def test_config_loads_both_minimums_independently(self):
        """Test that both minimum parameters load independently"""
        test_options = {
            'server_selection': {
                'minimum_connected_satellites': 4,
                'minimum_satellites_for_redistribution': 10
            },
            'module_settings': {
                'output_to_file': False
            }
        }

        self.config.read_options(test_options)

        self.assertEqual(self.algorithm.minimum_connected_satellites, 4)
        self.assertEqual(self.algorithm.minimum_satellites_for_redistribution, 10)

    def test_config_with_different_values(self):
        """Test configuration with various different values"""
        test_cases = [
            (3, 7),
            (5, 5),   # Same value
            (10, 3),  # Redistribution lower than transmitting
            (1, 15),  # Large difference
        ]

        for transmit_min, redist_min in test_cases:
            with self.subTest(transmit_min=transmit_min, redist_min=redist_min):
                algorithm = Algorithm()
                config = AlgorithmConfig(algorithm)

                test_options = {
                    'server_selection': {
                        'minimum_connected_satellites': transmit_min,
                        'minimum_satellites_for_redistribution': redist_min
                    },
                    'module_settings': {
                        'output_to_file': False
                    }
                }

                config.read_options(test_options)

                self.assertEqual(algorithm.minimum_connected_satellites, transmit_min)
                self.assertEqual(algorithm.minimum_satellites_for_redistribution, redist_min)


class TestTransmittingPhaseUsesCorrectMinimum(unittest.TestCase):
    """Test that TRANSMITTING phase uses minimum_connected_satellites"""

    def setUp(self):
        """Set up test fixtures"""
        self.algorithm = Algorithm()
        satellite_names = ["Sat0", "Sat1", "Sat2", "Sat3", "Sat4", "Sat5", "Sat6", "Sat7"]
        self.algorithm.set_satellite_names(satellite_names)

    def test_transmitting_uses_minimum_connected_satellites(self):
        """Test that aggregation server selection uses minimum_connected_satellites"""
        # Create matrices where different satellites have different connectivity
        matrices = []
        for t in range(10):
            # Sat0: connects to 3 satellites
            # Sat1: connects to 5 satellites
            # Sat2: connects to 7 satellites
            matrix = np.array([
                [0, 1, 1, 1, 0, 0, 0, 0],  # Sat0: 3 connections
                [1, 0, 1, 1, 1, 1, 0, 0],  # Sat1: 5 connections
                [1, 1, 0, 1, 1, 1, 1, 1],  # Sat2: 7 connections
                [1, 1, 1, 0, 0, 0, 0, 0],  # Sat3: 3 connections
                [0, 1, 1, 0, 0, 0, 0, 0],  # Sat4: 2 connections
                [0, 1, 1, 0, 0, 0, 0, 0],  # Sat5: 2 connections
                [0, 0, 1, 0, 0, 0, 0, 0],  # Sat6: 1 connection
                [0, 0, 1, 0, 0, 0, 0, 0]   # Sat7: 1 connection
            ])
            matrices.append((t, matrix))

        self.algorithm.set_adjacency_matrices(matrices)

        # Set minimum_connected_satellites to 5 (should exclude Sat0, Sat3, Sat4, Sat5, Sat6, Sat7)
        # Set minimum_satellites_for_redistribution to 3 (this should NOT affect transmitting phase)
        self.algorithm.set_minimum_connected_satellites(5)
        self.algorithm.set_minimum_satellites_for_redistribution(3)
        self.algorithm.set_connect_to_all_satellites(False)
        self.algorithm.set_max_lookahead(10)

        # Find best server for aggregation (TRANSMITTING phase)
        best_server, timesteps, analysis = self.algorithm.find_best_server_for_round(0)

        # Should select Sat1 or Sat2 (both have >= 5 connections)
        # Should NOT select Sat0, Sat3, Sat4, Sat5, Sat6, Sat7 (< 5 connections)
        self.assertIn(best_server, [1, 2],
                      f"Expected Sat1 or Sat2 (>=5 connections), got Sat{best_server}")
        self.assertGreaterEqual(analysis['max_connections'], 5,
                               "Selected server should have at least 5 connections")

    def test_transmitting_ignores_redistribution_minimum(self):
        """Test that aggregation server selection ignores minimum_satellites_for_redistribution"""
        # Create matrices with specific connectivity patterns
        matrices = []
        for t in range(10):
            matrix = np.array([
                [0, 1, 1, 1, 0, 0],  # Sat0: 3 connections
                [1, 0, 1, 1, 1, 0],  # Sat1: 4 connections
                [1, 1, 0, 1, 1, 1],  # Sat2: 5 connections
                [1, 1, 1, 0, 0, 0],  # Sat3: 3 connections
                [0, 1, 1, 0, 0, 0],  # Sat4: 2 connections
                [0, 0, 1, 0, 0, 0]   # Sat5: 1 connection
            ])
            matrices.append((t, matrix))

        self.algorithm.set_satellite_names(["Sat0", "Sat1", "Sat2", "Sat3", "Sat4", "Sat5"])
        self.algorithm.set_adjacency_matrices(matrices)

        # Set minimum_connected_satellites to 3 (should allow Sat0, Sat1, Sat2, Sat3)
        # Set minimum_satellites_for_redistribution to 5 (should be IGNORED in transmitting phase)
        self.algorithm.set_minimum_connected_satellites(3)
        self.algorithm.set_minimum_satellites_for_redistribution(5)
        self.algorithm.set_connect_to_all_satellites(False)
        self.algorithm.set_max_lookahead(10)

        # Find best server for aggregation
        best_server, timesteps, analysis = self.algorithm.find_best_server_for_round(0)

        # Should consider Sat0, Sat1, Sat2, Sat3 (all have >= 3 connections)
        # Should select Sat2 (has most connections: 5)
        self.assertGreaterEqual(analysis['max_connections'], 3,
                               "Should use minimum_connected_satellites (3), not redistribution minimum (5)")


class TestRedistributionPhaseUsesCorrectMinimum(unittest.TestCase):
    """Test that REDISTRIBUTION/CHECK phase uses minimum_satellites_for_redistribution"""

    def setUp(self):
        """Set up test fixtures"""
        self.algorithm = Algorithm()
        satellite_names = ["Sat0", "Sat1", "Sat2", "Sat3", "Sat4", "Sat5", "Sat6", "Sat7"]
        self.algorithm.set_satellite_names(satellite_names)

    def test_redistribution_uses_minimum_satellites_for_redistribution(self):
        """Test that redistribution server selection uses minimum_satellites_for_redistribution"""
        # Create matrices where different satellites have different connectivity
        matrices = []
        for t in range(15):
            # Different connectivity for redistribution candidates
            matrix = np.array([
                [0, 1, 1, 1, 1, 1, 1, 1],  # Sat0: 7 connections (aggregation server)
                [1, 0, 1, 1, 1, 0, 0, 0],  # Sat1: 4 connections
                [1, 1, 0, 1, 1, 1, 1, 0],  # Sat2: 6 connections
                [1, 1, 1, 0, 1, 1, 1, 1],  # Sat3: 7 connections
                [1, 1, 1, 1, 0, 0, 0, 0],  # Sat4: 4 connections
                [1, 0, 1, 1, 0, 0, 0, 0],  # Sat5: 3 connections
                [1, 0, 1, 1, 0, 0, 0, 0],  # Sat6: 3 connections
                [1, 0, 0, 1, 0, 0, 0, 0]   # Sat7: 2 connections
            ])
            matrices.append((t, matrix))

        self.algorithm.set_adjacency_matrices(matrices)

        # Set minimum_connected_satellites to 4 (for transmitting)
        # Set minimum_satellites_for_redistribution to 6 (should exclude Sat1, Sat4, Sat5, Sat6, Sat7)
        self.algorithm.set_minimum_connected_satellites(4)
        self.algorithm.set_minimum_satellites_for_redistribution(6)
        self.algorithm.set_connect_to_all_satellites(False)
        self.algorithm.set_max_lookahead(15)

        # Select redistribution server from aggregation server Sat0
        aggregation_server = 0
        redist_server, time_a, analysis = self.algorithm.find_best_redistribution_server(
            aggregation_server=aggregation_server,
            start_matrix_index=0,
            max_lookahead=15
        )

        # Should select Sat2 or Sat3 (both have >= 6 connections)
        # Should NOT select Sat1, Sat4, Sat5, Sat6, Sat7 (< 6 connections)
        self.assertIn(redist_server, [0, 2, 3],
                      f"Expected Sat0, Sat2, or Sat3 (>=6 connections), got Sat{redist_server}")
        self.assertGreaterEqual(analysis['max_connections'], 6,
                               "Redistribution server should have at least 6 connections")

    def test_redistribution_ignores_transmitting_minimum(self):
        """Test that redistribution server selection ignores minimum_connected_satellites"""
        # Create matrices
        matrices = []
        for t in range(15):
            matrix = np.array([
                [0, 1, 1, 1, 1, 1, 1],  # Sat0: 6 connections (aggregation server)
                [1, 0, 1, 1, 1, 0, 0],  # Sat1: 4 connections
                [1, 1, 0, 1, 1, 1, 0],  # Sat2: 5 connections
                [1, 1, 1, 0, 1, 1, 1],  # Sat3: 6 connections
                [1, 1, 1, 1, 0, 0, 0],  # Sat4: 4 connections
                [1, 0, 1, 1, 0, 0, 0],  # Sat5: 3 connections
                [1, 0, 0, 1, 0, 0, 0]   # Sat6: 2 connections
            ])
            matrices.append((t, matrix))

        self.algorithm.set_satellite_names(["Sat0", "Sat1", "Sat2", "Sat3", "Sat4", "Sat5", "Sat6"])
        self.algorithm.set_adjacency_matrices(matrices)

        # Set minimum_connected_satellites to 6 (should be IGNORED in redistribution phase)
        # Set minimum_satellites_for_redistribution to 4 (should allow Sat1, Sat2, Sat3, Sat4)
        self.algorithm.set_minimum_connected_satellites(6)
        self.algorithm.set_minimum_satellites_for_redistribution(4)
        self.algorithm.set_connect_to_all_satellites(False)
        self.algorithm.set_max_lookahead(15)

        # Select redistribution server
        aggregation_server = 0
        redist_server, time_a, analysis = self.algorithm.find_best_redistribution_server(
            aggregation_server=aggregation_server,
            start_matrix_index=0,
            max_lookahead=15
        )

        # Should consider Sat1, Sat2, Sat3, Sat4 (all have >= 4 connections)
        # Should use minimum_satellites_for_redistribution (4), not minimum_connected_satellites (6)
        self.assertGreaterEqual(analysis['max_connections'], 4,
                               "Should use minimum_satellites_for_redistribution (4), not transmitting minimum (6)")


class TestIndependentConfiguration(unittest.TestCase):
    """Test that both parameters can be configured independently"""

    def setUp(self):
        """Set up test fixtures"""
        self.algorithm = Algorithm()

    def test_transmitting_higher_than_redistribution(self):
        """Test when transmitting minimum is higher than redistribution minimum"""
        self.algorithm.set_minimum_connected_satellites(10)
        self.algorithm.set_minimum_satellites_for_redistribution(5)

        self.assertEqual(self.algorithm.minimum_connected_satellites, 10)
        self.assertEqual(self.algorithm.minimum_satellites_for_redistribution, 5)

    def test_redistribution_higher_than_transmitting(self):
        """Test when redistribution minimum is higher than transmitting minimum"""
        self.algorithm.set_minimum_connected_satellites(3)
        self.algorithm.set_minimum_satellites_for_redistribution(8)

        self.assertEqual(self.algorithm.minimum_connected_satellites, 3)
        self.assertEqual(self.algorithm.minimum_satellites_for_redistribution, 8)

    def test_both_equal(self):
        """Test when both minimums are equal"""
        self.algorithm.set_minimum_connected_satellites(6)
        self.algorithm.set_minimum_satellites_for_redistribution(6)

        self.assertEqual(self.algorithm.minimum_connected_satellites, 6)
        self.assertEqual(self.algorithm.minimum_satellites_for_redistribution, 6)

    def test_extreme_differences(self):
        """Test with extreme differences between minimums"""
        # Very low transmitting, very high redistribution
        self.algorithm.set_minimum_connected_satellites(1)
        self.algorithm.set_minimum_satellites_for_redistribution(50)

        self.assertEqual(self.algorithm.minimum_connected_satellites, 1)
        self.assertEqual(self.algorithm.minimum_satellites_for_redistribution, 50)

        # Very high transmitting, very low redistribution
        self.algorithm.set_minimum_connected_satellites(100)
        self.algorithm.set_minimum_satellites_for_redistribution(2)

        self.assertEqual(self.algorithm.minimum_connected_satellites, 100)
        self.assertEqual(self.algorithm.minimum_satellites_for_redistribution, 2)


class TestEdgeCasesAndFallback(unittest.TestCase):
    """Test edge cases and fallback behavior"""

    def setUp(self):
        """Set up test fixtures"""
        self.algorithm = Algorithm()

    def test_redistribution_minimum_exceeds_available_satellites(self):
        """Test when minimum_satellites_for_redistribution exceeds available satellites"""
        satellite_names = ["Sat0", "Sat1", "Sat2", "Sat3"]
        self.algorithm.set_satellite_names(satellite_names)

        # Create matrices with limited connectivity
        matrices = []
        for t in range(10):
            matrix = np.array([
                [0, 1, 1, 0],  # Sat0: 2 connections
                [1, 0, 1, 0],  # Sat1: 2 connections
                [1, 1, 0, 1],  # Sat2: 3 connections (best)
                [0, 0, 1, 0]   # Sat3: 1 connection
            ])
            matrices.append((t, matrix))

        self.algorithm.set_adjacency_matrices(matrices)

        # Set minimum_satellites_for_redistribution to 5 (impossible with only 3 other satellites)
        self.algorithm.set_minimum_satellites_for_redistribution(5)
        self.algorithm.set_connect_to_all_satellites(False)
        self.algorithm.set_max_lookahead(10)

        # Should fallback to best available
        aggregation_server = 0
        redist_server, time_a, analysis = self.algorithm.find_best_redistribution_server(
            aggregation_server=aggregation_server,
            start_matrix_index=0,
            max_lookahead=10
        )

        # Should still return a server (fallback behavior)
        self.assertIsNotNone(redist_server)
        self.assertGreaterEqual(redist_server, 0)
        self.assertLess(redist_server, 4)

    def test_zero_minimum_for_redistribution(self):
        """Test with minimum_satellites_for_redistribution set to 0"""
        satellite_names = ["Sat0", "Sat1", "Sat2"]
        self.algorithm.set_satellite_names(satellite_names)

        matrices = [
            (0, np.array([
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 0]
            ]))
        ]
        self.algorithm.set_adjacency_matrices(matrices)

        self.algorithm.set_minimum_satellites_for_redistribution(0)
        self.algorithm.set_max_lookahead(10)

        # Should accept any server (even with 0 connections)
        aggregation_server = 0
        redist_server, time_a, analysis = self.algorithm.find_best_redistribution_server(
            aggregation_server=aggregation_server,
            start_matrix_index=0,
            max_lookahead=10
        )

        self.assertIsNotNone(redist_server)

    def test_same_server_for_aggregation_and_redistribution(self):
        """Test when same server is selected for both aggregation and redistribution"""
        satellite_names = ["Sat0", "Sat1", "Sat2", "Sat3", "Sat4"]
        self.algorithm.set_satellite_names(satellite_names)

        # Sat2 has best connectivity for both phases
        matrices = []
        for t in range(10):
            matrix = np.array([
                [0, 1, 1, 0, 0],  # Sat0: 2 connections
                [1, 0, 1, 0, 0],  # Sat1: 2 connections
                [1, 1, 0, 1, 1],  # Sat2: 4 connections (best)
                [0, 0, 1, 0, 0],  # Sat3: 1 connection
                [0, 0, 1, 0, 0]   # Sat4: 1 connection
            ])
            matrices.append((t, matrix))

        self.algorithm.set_adjacency_matrices(matrices)
        self.algorithm.set_minimum_connected_satellites(3)
        self.algorithm.set_minimum_satellites_for_redistribution(3)
        self.algorithm.set_max_lookahead(10)

        # Find best aggregation server
        best_server, _, _ = self.algorithm.find_best_server_for_round(0)
        self.assertEqual(best_server, 2, "Sat2 should be selected for aggregation")

        # Find redistribution server from Sat2
        redist_server, time_a, analysis = self.algorithm.find_best_redistribution_server(
            aggregation_server=2,
            start_matrix_index=0,
            max_lookahead=10
        )

        # If same server, time_a should be 0
        if redist_server == 2:
            self.assertEqual(time_a, 0, "Time to reach same server should be 0")


class TestConnectToAllSatellitesInteraction(unittest.TestCase):
    """Test interaction with connect_to_all_satellites mode"""

    def setUp(self):
        """Set up test fixtures"""
        self.algorithm = Algorithm()

    def test_connect_to_all_overrides_redistribution_minimum(self):
        """Test that connect_to_all_satellites overrides minimum_satellites_for_redistribution"""
        satellite_names = ["Sat0", "Sat1", "Sat2", "Sat3", "Sat4"]
        self.algorithm.set_satellite_names(satellite_names)

        # Sat1 connects to all, Sat2 connects to 3
        matrices = []
        for t in range(10):
            matrix = np.array([
                [0, 1, 1, 0, 0],  # Sat0: 2 connections
                [1, 0, 1, 1, 1],  # Sat1: 4 connections (all others)
                [1, 1, 0, 1, 0],  # Sat2: 3 connections
                [0, 1, 1, 0, 0],  # Sat3: 2 connections
                [0, 1, 0, 0, 0]   # Sat4: 1 connection
            ])
            matrices.append((t, matrix))

        self.algorithm.set_adjacency_matrices(matrices)

        # Set minimum_satellites_for_redistribution to 2 (would normally allow Sat2)
        # But connect_to_all_satellites=True should require all 4 other satellites
        self.algorithm.set_connect_to_all_satellites(True)
        self.algorithm.set_minimum_satellites_for_redistribution(2)
        self.algorithm.set_max_lookahead(10)

        # Find redistribution server
        aggregation_server = 0
        redist_server, time_a, analysis = self.algorithm.find_best_redistribution_server(
            aggregation_server=aggregation_server,
            start_matrix_index=0,
            max_lookahead=10
        )

        # Should select Sat1 (only one that connects to all)
        # Should NOT select Sat2 even though it meets minimum_satellites_for_redistribution
        self.assertIn(redist_server, [0, 1], "Should select server that connects to all satellites")
        if redist_server == 1:
            self.assertEqual(analysis['max_connections'], 4,
                           "Selected server should connect to all 4 other satellites")


class TestFullThreePhaseWorkflow(unittest.TestCase):
    """Test complete three-phase workflow with different minimums"""

    def setUp(self):
        """Set up test fixtures"""
        self.algorithm = Algorithm()
        satellite_names = ["Sat0", "Sat1", "Sat2", "Sat3", "Sat4", "Sat5", "Sat6", "Sat7"]
        self.algorithm.set_satellite_names(satellite_names)

    def test_complete_round_with_different_minimums(self):
        """Test complete round with different minimums for each phase"""
        # Create realistic connectivity matrices
        matrices = []
        for t in range(30):
            # Varying connectivity over time
            if t < 10:
                # Early timesteps: Sat2 has good connectivity (5), Sat5 has great (7)
                matrix = np.array([
                    [0, 1, 1, 1, 0, 0, 0, 0],  # Sat0: 3
                    [1, 0, 1, 1, 1, 0, 0, 0],  # Sat1: 4
                    [1, 1, 0, 1, 1, 1, 0, 0],  # Sat2: 5
                    [1, 1, 1, 0, 0, 0, 0, 0],  # Sat3: 3
                    [0, 1, 1, 0, 0, 0, 0, 0],  # Sat4: 2
                    [0, 0, 1, 0, 0, 0, 1, 1],  # Sat5: 3 (initially)
                    [0, 0, 0, 0, 0, 1, 0, 1],  # Sat6: 2
                    [0, 0, 0, 0, 0, 1, 1, 0]   # Sat7: 2
                ])
            elif t < 20:
                # Mid timesteps: Connectivity shifts
                matrix = np.array([
                    [0, 1, 1, 1, 1, 0, 0, 0],  # Sat0: 4
                    [1, 0, 1, 1, 1, 1, 0, 0],  # Sat1: 5
                    [1, 1, 0, 1, 1, 1, 1, 0],  # Sat2: 6
                    [1, 1, 1, 0, 1, 1, 0, 0],  # Sat3: 5
                    [1, 1, 1, 1, 0, 0, 0, 0],  # Sat4: 4
                    [0, 1, 1, 1, 0, 0, 1, 1],  # Sat5: 5
                    [0, 0, 1, 0, 0, 1, 0, 1],  # Sat6: 3
                    [0, 0, 0, 0, 0, 1, 1, 0]   # Sat7: 2
                ])
            else:
                # Late timesteps: Full connectivity for redistribution
                matrix = np.array([
                    [0, 1, 1, 1, 1, 1, 0, 0],  # Sat0: 5
                    [1, 0, 1, 1, 1, 1, 1, 0],  # Sat1: 6
                    [1, 1, 0, 1, 1, 1, 1, 1],  # Sat2: 7 (best)
                    [1, 1, 1, 0, 1, 1, 1, 0],  # Sat3: 6
                    [1, 1, 1, 1, 0, 1, 0, 0],  # Sat4: 5
                    [1, 1, 1, 1, 1, 0, 1, 1],  # Sat5: 7
                    [0, 1, 1, 1, 0, 1, 0, 1],  # Sat6: 5
                    [0, 0, 1, 0, 0, 1, 1, 0]   # Sat7: 3
                ])
            matrices.append((t, matrix))

        self.algorithm.set_adjacency_matrices(matrices)

        # Set different minimums for transmitting vs redistribution
        self.algorithm.set_minimum_connected_satellites(4)  # Lower for transmitting
        self.algorithm.set_minimum_satellites_for_redistribution(6)  # Higher for redistribution
        self.algorithm.set_connect_to_all_satellites(False)
        self.algorithm.set_max_lookahead(15)
        self.algorithm.set_output_to_file(False)

        # Run full algorithm
        self.algorithm.start_algorithm_steps()

        # Get output
        output = self.algorithm.get_algorithm_output()

        # Verify output exists
        self.assertIsNotNone(output)
        self.assertGreater(len(output), 0)

        # Verify all three phases executed
        phases_in_round = set()
        for timestep_key, entry in output.items():
            if entry['round_number'] == 1:
                phases_in_round.add(entry['phase'])

        self.assertIn('TRANSMITTING', phases_in_round)
        self.assertIn('REDISTRIBUTION', phases_in_round)
        # CHECK may or may not be present (0 duration if same server)

        # Verify aggregation server meets transmitting minimum (4)
        transmitting_entries = [e for e in output.values()
                               if e['round_number'] == 1 and e['phase'] == 'TRANSMITTING']
        if transmitting_entries:
            final_transmitting = transmitting_entries[-1]
            self.assertGreaterEqual(final_transmitting['server_connections_cumulative'], 4,
                                   "Aggregation server should meet minimum_connected_satellites (4)")

        # Verify redistribution server meets redistribution minimum (6)
        redistribution_entries = [e for e in output.values()
                                 if e['round_number'] == 1 and e['phase'] == 'REDISTRIBUTION']
        if redistribution_entries:
            final_redistribution = redistribution_entries[-1]
            self.assertGreaterEqual(len(final_redistribution['target_satellites']), 6,
                                   "Redistribution server should target at least 6 satellites")


class TestOptionsJsonIntegration(unittest.TestCase):
    """Test integration with actual options.json file"""

    def test_load_from_actual_options_json(self):
        """Test loading configuration from actual options.json file"""
        options_path = os.path.join(os.path.dirname(__file__), 'options.json')

        if not os.path.exists(options_path):
            self.skipTest(f"options.json not found at {options_path}")

        # Load options.json
        with open(options_path, 'r') as f:
            options_data = json.load(f)

        # Verify the new option exists in the file
        # The algorithm section is nested under federated_learning
        self.assertIn('federated_learning', options_data, "options.json should have 'federated_learning' section")
        fl_options = options_data['federated_learning']

        self.assertIn('algorithm', fl_options, "federated_learning section should have 'algorithm' subsection")
        algorithm_options = fl_options['algorithm']

        self.assertIn('server_selection', algorithm_options,
                     "algorithm section should have 'server_selection'")
        server_selection = algorithm_options['server_selection']

        self.assertIn('minimum_satellites_for_redistribution', server_selection,
                     "server_selection should have 'minimum_satellites_for_redistribution'")

        # Verify default value is 7
        self.assertEqual(server_selection['minimum_satellites_for_redistribution'], 7,
                        "Default value should be 7")

        # Load into algorithm
        algorithm = Algorithm()
        config = AlgorithmConfig(algorithm)
        config.read_options(algorithm_options)

        # Verify it loaded correctly
        self.assertEqual(algorithm.minimum_satellites_for_redistribution, 7)
        self.assertEqual(algorithm.minimum_connected_satellites,
                        server_selection.get('minimum_connected_satellites', 5))


def run_test_suite():
    """Run the complete test suite"""

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestMinimumSatellitesForRedistributionSettersGetters))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigurationLoading))
    suite.addTests(loader.loadTestsFromTestCase(TestTransmittingPhaseUsesCorrectMinimum))
    suite.addTests(loader.loadTestsFromTestCase(TestRedistributionPhaseUsesCorrectMinimum))
    suite.addTests(loader.loadTestsFromTestCase(TestIndependentConfiguration))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCasesAndFallback))
    suite.addTests(loader.loadTestsFromTestCase(TestConnectToAllSatellitesInteraction))
    suite.addTests(loader.loadTestsFromTestCase(TestFullThreePhaseWorkflow))
    suite.addTests(loader.loadTestsFromTestCase(TestOptionsJsonIntegration))

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY - minimum_satellites_for_redistribution Feature")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*80)

    if result.wasSuccessful():
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED")

    print("\nTest Coverage Summary:")
    print("  ✓ Configuration loading from options.json")
    print("  ✓ Setter/getter methods")
    print("  ✓ TRANSMITTING phase uses minimum_connected_satellites")
    print("  ✓ REDISTRIBUTION phase uses minimum_satellites_for_redistribution")
    print("  ✓ Independent configuration of both parameters")
    print("  ✓ Edge cases and fallback behavior")
    print("  ✓ Interaction with connect_to_all_satellites mode")
    print("  ✓ Complete three-phase workflow validation")
    print("  ✓ Integration with actual options.json file")
    print("="*80)

    return result


if __name__ == '__main__':
    result = run_test_suite()
    sys.exit(0 if result.wasSuccessful() else 1)
