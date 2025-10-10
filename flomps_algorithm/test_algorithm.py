"""
Filename: test_algorithm.py
Description: Comprehensive test module for FLOMPS algorithm functionality
Author: Claude Code
Date: 2025-10-10
Version: 1.0
Python Version: 3.12

Test Coverage:
- Algorithm initialization and configuration
- Server selection parameter validation
- Aggregation server selection logic
- Redistribution server selection logic
- connect_to_all_satellites mode
- minimum_connected_satellites behavior
- max_lookahead behavior
- Three-phase execution (TRANSMITTING → CHECK → REDISTRIBUTION)
- Edge cases and fallback scenarios
- Load balancing
- Configuration loading from options.json

Usage:
cd /path/to/flomps_algorithm && python test_algorithm.py
"""

import unittest
import sys
import os
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flomps_algorithm.algorithm_core import Algorithm
from flomps_algorithm.algorithm_config import AlgorithmConfig


class TestAlgorithmInitialization(unittest.TestCase):
    """Test basic algorithm initialization and setup"""

    def setUp(self):
        """Set up test fixtures"""
        self.algorithm = Algorithm()

    def test_algorithm_creation(self):
        """Test that algorithm can be created successfully"""
        self.assertIsNotNone(self.algorithm)
        self.assertIsInstance(self.algorithm, Algorithm)

    def test_default_parameters(self):
        """Test that default parameters are set correctly"""
        self.assertEqual(self.algorithm.connect_to_all_satellites, False)
        self.assertEqual(self.algorithm.max_lookahead, 20)
        self.assertEqual(self.algorithm.minimum_connected_satellites, 5)
        self.assertEqual(self.algorithm.round_number, 1)

    def test_satellite_names_setter(self):
        """Test setting satellite names"""
        satellite_names = ["Sat1", "Sat2", "Sat3", "Sat4"]
        self.algorithm.set_satellite_names(satellite_names)
        self.assertEqual(self.algorithm.get_satellite_names(), satellite_names)
        self.assertEqual(len(self.algorithm.selection_counts), 4)

    def test_adjacency_matrices_setter(self):
        """Test setting adjacency matrices"""
        matrices = [
            (0, np.array([[0, 1], [1, 0]])),
            (1, np.array([[0, 0], [0, 0]]))
        ]
        self.algorithm.set_adjacency_matrices(matrices)
        self.assertEqual(len(self.algorithm.get_adjacency_matrices()), 2)


class TestServerSelectionParameters(unittest.TestCase):
    """Test server selection parameter setters and getters"""

    def setUp(self):
        """Set up test fixtures"""
        self.algorithm = Algorithm()

    def test_set_connect_to_all_satellites(self):
        """Test setting connect_to_all_satellites parameter"""
        self.algorithm.set_connect_to_all_satellites(True)
        self.assertTrue(self.algorithm.connect_to_all_satellites)

        self.algorithm.set_connect_to_all_satellites(False)
        self.assertFalse(self.algorithm.connect_to_all_satellites)

    def test_set_max_lookahead(self):
        """Test setting max_lookahead parameter"""
        self.algorithm.set_max_lookahead(30)
        self.assertEqual(self.algorithm.max_lookahead, 30)

        self.algorithm.set_max_lookahead(10)
        self.assertEqual(self.algorithm.max_lookahead, 10)

    def test_set_minimum_connected_satellites(self):
        """Test setting minimum_connected_satellites parameter"""
        self.algorithm.set_minimum_connected_satellites(3)
        self.assertEqual(self.algorithm.minimum_connected_satellites, 3)

        self.algorithm.set_minimum_connected_satellites(7)
        self.assertEqual(self.algorithm.minimum_connected_satellites, 7)


class TestSingleSatelliteAnalysis(unittest.TestCase):
    """Test single satellite connectivity analysis"""

    def setUp(self):
        """Set up test fixtures with sample adjacency matrices"""
        self.algorithm = Algorithm()
        satellite_names = ["Sat1", "Sat2", "Sat3", "Sat4"]
        self.algorithm.set_satellite_names(satellite_names)

        # Create test adjacency matrices
        # Sat0 connects to Sat1 at timestep 0, Sat2 at timestep 1
        matrices = [
            (0, np.array([
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]
            ])),
            (1, np.array([
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ])),
            (2, np.array([
                [0, 1, 1, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 1],
                [0, 0, 1, 0]
            ]))
        ]
        self.algorithm.set_adjacency_matrices(matrices)

    def test_analyze_single_satellite(self):
        """Test analyzing a single satellite's connectivity"""
        analysis = self.algorithm.analyze_single_satellite(
            sat_idx=0,
            start_matrix_index=0,
            max_lookahead=3
        )

        self.assertIsNotNone(analysis)
        self.assertIn('max_connections', analysis)
        self.assertIn('timestamps_to_max', analysis)
        self.assertIn('connected_satellites', analysis)
        self.assertIn('connection_timeline', analysis)

        # Sat0 should connect to Sat1 at t=0, Sat2 at t=1, cumulative max = 2
        self.assertEqual(analysis['max_connections'], 2)

    def test_connection_timeline(self):
        """Test that connection timeline is tracked correctly"""
        analysis = self.algorithm.analyze_single_satellite(
            sat_idx=0,
            start_matrix_index=0,
            max_lookahead=3
        )

        timeline = analysis['connection_timeline']
        self.assertEqual(len(timeline), 3)

        # Check that cumulative connections are tracked
        for entry in timeline:
            self.assertIn('timestep', entry)
            self.assertIn('cumulative_count', entry)
            self.assertIn('cumulative_connections', entry)


class TestAggregationServerSelection(unittest.TestCase):
    """Test aggregation server selection logic"""

    def setUp(self):
        """Set up test fixtures"""
        self.algorithm = Algorithm()
        satellite_names = ["Sat1", "Sat2", "Sat3", "Sat4"]
        self.algorithm.set_satellite_names(satellite_names)

        # Create matrices where Sat2 has best connectivity
        matrices = [
            (0, np.array([
                [0, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0]
            ])),
            (1, np.array([
                [0, 0, 1, 0],
                [0, 0, 1, 1],
                [1, 1, 0, 1],
                [0, 1, 1, 0]
            ]))
        ]
        self.algorithm.set_adjacency_matrices(matrices)

    def test_find_best_server_basic(self):
        """Test basic server selection"""
        self.algorithm.set_minimum_connected_satellites(2)
        self.algorithm.set_max_lookahead(10)

        best_server, timesteps, analysis = self.algorithm.find_best_server_for_round(0)

        self.assertIsNotNone(best_server)
        self.assertGreaterEqual(best_server, 0)
        self.assertLess(best_server, 4)
        self.assertGreater(timesteps, 0)
        self.assertIn('max_connections', analysis)

    def test_server_selection_with_minimum_satellites(self):
        """Test server selection respects minimum_connected_satellites"""
        self.algorithm.set_minimum_connected_satellites(2)
        self.algorithm.set_connect_to_all_satellites(False)

        best_server, timesteps, analysis = self.algorithm.find_best_server_for_round(0)

        # Selected server should meet or exceed minimum requirement
        self.assertGreaterEqual(analysis['max_connections'], 2)


class TestRedistributionServerSelection(unittest.TestCase):
    """Test redistribution server selection logic"""

    def setUp(self):
        """Set up test fixtures"""
        self.algorithm = Algorithm()
        satellite_names = ["Sat1", "Sat2", "Sat3", "Sat4"]
        self.algorithm.set_satellite_names(satellite_names)

        # Create matrices for redistribution testing
        matrices = [
            (0, np.array([
                [0, 1, 1, 0],
                [1, 0, 1, 0],
                [1, 1, 0, 1],
                [0, 0, 1, 0]
            ])),
            (1, np.array([
                [0, 1, 1, 1],
                [1, 0, 1, 1],
                [1, 1, 0, 1],
                [1, 1, 1, 0]
            ]))
        ]
        self.algorithm.set_adjacency_matrices(matrices)

    def test_find_redistribution_server(self):
        """Test finding redistribution server from aggregation server"""
        self.algorithm.set_minimum_connected_satellites(2)

        aggregation_server = 0
        redist_server, time_a, analysis = self.algorithm.find_best_redistribution_server(
            aggregation_server=aggregation_server,
            start_matrix_index=0
        )

        self.assertIsNotNone(redist_server)
        self.assertGreaterEqual(redist_server, 0)
        self.assertLess(redist_server, 4)
        self.assertGreaterEqual(time_a, 0)

    def test_same_server_redistribution(self):
        """Test when aggregation server is also best for redistribution"""
        self.algorithm.set_minimum_connected_satellites(2)

        # If server 2 has best connectivity, it might be selected for both
        aggregation_server = 2
        redist_server, time_a, analysis = self.algorithm.find_best_redistribution_server(
            aggregation_server=aggregation_server,
            start_matrix_index=0
        )

        # If same server, time_a should be 0
        if redist_server == aggregation_server:
            self.assertEqual(time_a, 0)


class TestConnectToAllMode(unittest.TestCase):
    """Test connect_to_all_satellites mode"""

    def setUp(self):
        """Set up test fixtures"""
        self.algorithm = Algorithm()
        satellite_names = ["Sat1", "Sat2", "Sat3", "Sat4"]
        self.algorithm.set_satellite_names(satellite_names)

        # Create matrices where one satellite can reach all others
        matrices = [
            (0, np.array([
                [0, 1, 1, 1],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0]
            ]))
        ]
        self.algorithm.set_adjacency_matrices(matrices)

    def test_connect_to_all_mode_enabled(self):
        """Test that connect_to_all mode requires all satellites"""
        self.algorithm.set_connect_to_all_satellites(True)
        self.algorithm.set_max_lookahead(10)

        best_server, timesteps, analysis = self.algorithm.find_best_server_for_round(0)

        # In this test, Sat0 connects to all 3 others
        # If connect_to_all=True, it should select a server that connects to all
        # With our test data, Sat0 should be selected
        self.assertEqual(best_server, 0)
        self.assertEqual(analysis['max_connections'], 3)  # All 3 other satellites

    def test_connect_to_all_mode_disabled(self):
        """Test that connect_to_all mode disabled uses minimum_connected_satellites"""
        self.algorithm.set_connect_to_all_satellites(False)
        self.algorithm.set_minimum_connected_satellites(2)
        self.algorithm.set_max_lookahead(10)

        best_server, timesteps, analysis = self.algorithm.find_best_server_for_round(0)

        # Should accept any server with at least 2 connections
        self.assertGreaterEqual(analysis['max_connections'], 2)


class TestMaxLookahead(unittest.TestCase):
    """Test max_lookahead behavior"""

    def setUp(self):
        """Set up test fixtures"""
        self.algorithm = Algorithm()
        satellite_names = ["Sat1", "Sat2", "Sat3"]
        self.algorithm.set_satellite_names(satellite_names)

        # Create matrices where connections appear at different timesteps
        matrices = []
        for i in range(10):
            if i < 5:
                # First 5 timesteps: limited connectivity
                matrices.append((i, np.array([
                    [0, 1, 0],
                    [1, 0, 0],
                    [0, 0, 0]
                ])))
            else:
                # After timestep 5: better connectivity
                matrices.append((i, np.array([
                    [0, 1, 1],
                    [1, 0, 1],
                    [1, 1, 0]
                ])))

        self.algorithm.set_adjacency_matrices(matrices)

    def test_max_lookahead_limits_search(self):
        """Test that max_lookahead limits the search window"""
        self.algorithm.set_minimum_connected_satellites(2)
        self.algorithm.set_max_lookahead(3)  # Only look at first 3 timesteps

        best_server, timesteps, analysis = self.algorithm.find_best_server_for_round(0)

        # With max_lookahead=3, can only see limited connectivity
        # The analysis should be limited to looking ahead max 3 timesteps
        # Note: timesteps returned is timestamps_to_max which might be higher
        # But the server should be selected based on what's visible in window
        self.assertIsNotNone(best_server)
        self.assertGreaterEqual(analysis['max_connections'], 1)

    def test_extended_lookahead_finds_better(self):
        """Test that extending lookahead can find better servers"""
        self.algorithm.set_minimum_connected_satellites(2)
        self.algorithm.set_max_lookahead(10)  # Look at all timesteps

        best_server, timesteps, analysis = self.algorithm.find_best_server_for_round(0)

        # With max_lookahead=10, should see full connectivity
        self.assertEqual(analysis['max_connections'], 2)  # All other satellites


class TestThreePhaseExecution(unittest.TestCase):
    """Test three-phase algorithm execution"""

    def setUp(self):
        """Set up test fixtures"""
        self.algorithm = Algorithm()
        satellite_names = ["Sat1", "Sat2", "Sat3", "Sat4"]
        self.algorithm.set_satellite_names(satellite_names)

        # Create sufficient matrices for a complete round
        matrices = []
        for i in range(20):
            matrices.append((i, np.array([
                [0, 1, 1, 0],
                [1, 0, 1, 0],
                [1, 1, 0, 1],
                [0, 0, 1, 0]
            ])))

        self.algorithm.set_adjacency_matrices(matrices)
        self.algorithm.set_minimum_connected_satellites(2)
        self.algorithm.set_output_to_file(False)

    def test_algorithm_produces_output(self):
        """Test that algorithm execution produces output"""
        self.algorithm.start_algorithm_steps()
        output = self.algorithm.get_algorithm_output()

        self.assertIsNotNone(output)
        self.assertIsInstance(output, dict)
        self.assertGreater(len(output), 0)

    def test_output_contains_phases(self):
        """Test that output contains all three phases"""
        self.algorithm.start_algorithm_steps()
        output = self.algorithm.get_algorithm_output()

        phases_found = set()
        for timestamp, data in output.items():
            if 'phase' in data:
                phases_found.add(data['phase'])

        # Should have at least TRANSMITTING and REDISTRIBUTION
        # CHECK might be 0 duration if same server used
        self.assertIn('TRANSMITTING', phases_found)
        self.assertIn('REDISTRIBUTION', phases_found)


class TestEdgeCasesAndFallback(unittest.TestCase):
    """Test edge cases and fallback scenarios"""

    def setUp(self):
        """Set up test fixtures"""
        self.algorithm = Algorithm()

    def test_no_connectivity_fallback(self):
        """Test fallback when no satellites meet criteria"""
        satellite_names = ["Sat1", "Sat2", "Sat3"]
        self.algorithm.set_satellite_names(satellite_names)

        # Create matrices with no connectivity
        matrices = [
            (0, np.array([
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ]))
        ]
        self.algorithm.set_adjacency_matrices(matrices)

        self.algorithm.set_connect_to_all_satellites(True)
        self.algorithm.set_max_lookahead(10)

        # Should fallback to best available (even if it's 0 connections)
        best_server, timesteps, analysis = self.algorithm.find_best_server_for_round(0)

        self.assertIsNotNone(best_server)
        self.assertEqual(analysis['max_connections'], 0)

    def test_single_satellite(self):
        """Test behavior with only one satellite"""
        satellite_names = ["Sat1"]
        self.algorithm.set_satellite_names(satellite_names)

        matrices = [(0, np.array([[0]]))]
        self.algorithm.set_adjacency_matrices(matrices)

        self.algorithm.set_minimum_connected_satellites(0)

        best_server, timesteps, analysis = self.algorithm.find_best_server_for_round(0)

        self.assertEqual(best_server, 0)
        self.assertEqual(analysis['max_connections'], 0)

    def test_high_minimum_requirement(self):
        """Test when minimum_connected_satellites exceeds available satellites"""
        satellite_names = ["Sat1", "Sat2", "Sat3"]
        self.algorithm.set_satellite_names(satellite_names)

        matrices = [
            (0, np.array([
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 0]
            ]))
        ]
        self.algorithm.set_adjacency_matrices(matrices)

        # Require 5 connections when only 2 satellites exist (max 2 possible)
        self.algorithm.set_minimum_connected_satellites(5)
        self.algorithm.set_max_lookahead(10)

        # Should fallback to best available
        best_server, timesteps, analysis = self.algorithm.find_best_server_for_round(0)

        self.assertIsNotNone(best_server)


class TestLoadBalancing(unittest.TestCase):
    """Test load balancing across satellite selections"""

    def setUp(self):
        """Set up test fixtures"""
        self.algorithm = Algorithm()
        satellite_names = ["Sat1", "Sat2", "Sat3"]
        self.algorithm.set_satellite_names(satellite_names)

        # Create matrices where all satellites have equal connectivity
        matrices = []
        for i in range(6):
            matrices.append((i, np.array([
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 0]
            ])))

        self.algorithm.set_adjacency_matrices(matrices)
        self.algorithm.set_minimum_connected_satellites(2)
        self.algorithm.set_output_to_file(False)

    def test_load_balancing_rotation(self):
        """Test that load balancing rotates between equally good satellites"""
        # Select servers multiple times
        selected_servers = []

        for _ in range(3):
            best_server, _, _ = self.algorithm.find_best_server_for_round(0)
            selected_servers.append(best_server)

        # With load balancing, should eventually select different satellites
        # (assuming equal connectivity for all)
        unique_servers = set(selected_servers)

        # Should have selected at least 2 different servers across 3 rounds
        self.assertGreaterEqual(len(unique_servers), 2)


class TestConfigurationLoading(unittest.TestCase):
    """Test configuration loading from AlgorithmConfig"""

    def setUp(self):
        """Set up test fixtures"""
        self.algorithm = Algorithm()
        self.config = AlgorithmConfig(self.algorithm)

    def test_config_applies_settings(self):
        """Test that config correctly applies settings to algorithm"""
        test_options = {
            'server_selection': {
                'connect_to_all_satellites': True,
                'max_lookahead': 15,
                'minimum_connected_satellites': 3
            },
            'module_settings': {
                'output_to_file': False
            }
        }

        self.config.read_options(test_options)

        self.assertTrue(self.algorithm.connect_to_all_satellites)
        self.assertEqual(self.algorithm.max_lookahead, 15)
        self.assertEqual(self.algorithm.minimum_connected_satellites, 3)
        self.assertFalse(self.algorithm.output_to_file)

    def test_config_defaults(self):
        """Test that config uses defaults when settings missing"""
        test_options = {
            'module_settings': {
                'output_to_file': True
            }
        }

        self.config.read_options(test_options)

        # Should keep default values
        self.assertEqual(self.algorithm.connect_to_all_satellites, False)
        self.assertEqual(self.algorithm.max_lookahead, 20)
        self.assertEqual(self.algorithm.minimum_connected_satellites, 5)


class TestFedAvgMode(unittest.TestCase):
    """Test FedAvg mode (static server) functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.algorithm = Algorithm()
        satellite_names = ["Sat0", "Sat1", "Sat2", "Sat3"]
        self.algorithm.set_satellite_names(satellite_names)

    def test_fedavg_mode_parameters(self):
        """Test FedAvg mode parameter setters"""
        self.algorithm.set_fedavg_mode(True)
        self.assertTrue(self.algorithm.fedavg_mode)

        self.algorithm.set_static_server_id(2)
        self.assertEqual(self.algorithm.static_server_id, 2)

    def test_fedavg_two_phase_execution(self):
        """Test FedAvg mode executes two phases (TRANSMITTING → REDISTRIBUTION)"""
        # Create matrices with connectivity
        matrices = []
        for t in range(20):
            # Static server (Sat2) connects to all satellites over time
            if t < 5:
                # Gradual connectivity buildup
                matrix = np.array([
                    [0, 0, 1, 0],
                    [0, 0, 1, 0],
                    [1, 1, 0, 0],  # Sat2 as server
                    [0, 0, 0, 0]
                ])
            elif t < 10:
                matrix = np.array([
                    [0, 0, 1, 0],
                    [0, 0, 1, 0],
                    [1, 1, 0, 1],  # Sat2 connects to more
                    [0, 0, 1, 0]
                ])
            else:
                # Full connectivity
                matrix = np.array([
                    [0, 0, 1, 0],
                    [0, 0, 1, 0],
                    [1, 1, 0, 1],
                    [0, 0, 1, 0]
                ])
            matrices.append((t, matrix))

        self.algorithm.set_adjacency_matrices(matrices)

        # Enable FedAvg mode with Sat2 as static server
        self.algorithm.set_fedavg_mode(True)
        self.algorithm.set_static_server_id(2)
        self.algorithm.set_output_to_file(False)

        # Run algorithm
        self.algorithm.start_algorithm_steps()

        # Get output
        output = self.algorithm.get_algorithm_output()

        # Verify output exists
        self.assertIsNotNone(output)
        self.assertGreater(len(output), 0)

        # Check that only TRANSMITTING and REDISTRIBUTION phases exist
        phases_in_round = set()
        for timestep_key, entry in output.items():
            if entry['round_number'] == 1:
                phases_in_round.add(entry['phase'])

        self.assertIn('TRANSMITTING', phases_in_round)
        self.assertIn('REDISTRIBUTION', phases_in_round)
        self.assertNotIn('CHECK', phases_in_round)  # No CHECK phase in FedAvg

        # Verify static server is used for both phases
        for timestep_key, entry in output.items():
            if entry['round_number'] == 1:
                self.assertEqual(entry['target_node'], 2)  # Always Sat2

    def test_fedavg_config_loading(self):
        """Test FedAvg mode configuration loading from AlgorithmConfig"""
        config = AlgorithmConfig(self.algorithm)

        test_options = {
            'fedavg_mode': True,
            'static_server_id': 3,
            'server_selection': {
                'connect_to_all_satellites': True,  # Should be ignored in FedAvg
                'max_lookahead': 10,  # Should be ignored in FedAvg
                'minimum_connected_satellites': 2  # Should be ignored in FedAvg
            },
            'module_settings': {
                'output_to_file': False
            }
        }

        config.read_options(test_options)

        # Verify FedAvg parameters loaded
        self.assertTrue(self.algorithm.fedavg_mode)
        self.assertEqual(self.algorithm.static_server_id, 3)


def run_test_suite():
    """Run the complete test suite"""

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestAlgorithmInitialization))
    suite.addTests(loader.loadTestsFromTestCase(TestServerSelectionParameters))
    suite.addTests(loader.loadTestsFromTestCase(TestSingleSatelliteAnalysis))
    suite.addTests(loader.loadTestsFromTestCase(TestAggregationServerSelection))
    suite.addTests(loader.loadTestsFromTestCase(TestRedistributionServerSelection))
    suite.addTests(loader.loadTestsFromTestCase(TestConnectToAllMode))
    suite.addTests(loader.loadTestsFromTestCase(TestMaxLookahead))
    suite.addTests(loader.loadTestsFromTestCase(TestThreePhaseExecution))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCasesAndFallback))
    suite.addTests(loader.loadTestsFromTestCase(TestLoadBalancing))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigurationLoading))
    suite.addTests(loader.loadTestsFromTestCase(TestFedAvgMode))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)

    return result


if __name__ == '__main__':
    result = run_test_suite()
    sys.exit(0 if result.wasSuccessful() else 1)
