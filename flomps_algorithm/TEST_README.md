# FLOMPS Algorithm Test Suite

Comprehensive test module for validating all functionality of the FLOMPS algorithm.

## Overview

**File:** `test_algorithm.py`
**Test Coverage:** 25 tests across 11 test classes
**Success Rate:** 100% (25/25 passing)

## Running the Tests

```bash
# From project root
cd /Users/ash/Desktop/SPACE_FLTeam
python flomps_algorithm/test_algorithm.py

# Or from flomps_algorithm directory
cd flomps_algorithm
python test_algorithm.py
```

## Test Classes

### 1. TestAlgorithmInitialization (4 tests)
Tests basic algorithm setup and initialization.

-  `test_algorithm_creation` - Verify algorithm instantiation
-  `test_default_parameters` - Check default configuration values
-  `test_satellite_names_setter` - Test satellite name configuration
-  `test_adjacency_matrices_setter` - Test adjacency matrix setup

### 2. TestServerSelectionParameters (3 tests)
Tests parameter setters and getters.

-  `test_set_connect_to_all_satellites` - Verify connect_to_all parameter
-  `test_set_max_lookahead` - Verify max_lookahead parameter
-  `test_set_minimum_connected_satellites` - Verify minimum satellites parameter

### 3. TestSingleSatelliteAnalysis (2 tests)
Tests individual satellite connectivity analysis.

-  `test_analyze_single_satellite` - Verify single satellite analysis
-  `test_connection_timeline` - Check cumulative connection tracking

### 4. TestAggregationServerSelection (2 tests)
Tests aggregation server selection logic.

-  `test_find_best_server_basic` - Basic server selection
-  `test_server_selection_with_minimum_satellites` - Minimum satellite requirement

### 5. TestRedistributionServerSelection (2 tests)
Tests redistribution server selection logic.

-  `test_find_redistribution_server` - Two-hop redistribution selection
-  `test_same_server_redistribution` - Same server for aggregation and redistribution

### 6. TestConnectToAllMode (2 tests)
Tests connect_to_all_satellites mode.

-  `test_connect_to_all_mode_enabled` - Connect to all satellites requirement
-  `test_connect_to_all_mode_disabled` - Minimum satellites mode

### 7. TestMaxLookahead (2 tests)
Tests max_lookahead parameter behavior.

-  `test_max_lookahead_limits_search` - Verify lookahead window limits
-  `test_extended_lookahead_finds_better` - Extended search benefits

### 8. TestThreePhaseExecution (2 tests)
Tests three-phase algorithm execution.

-  `test_algorithm_produces_output` - Verify output generation
-  `test_output_contains_phases` - Check TRANSMITTING, CHECK, REDISTRIBUTION phases

### 9. TestEdgeCasesAndFallback (3 tests)
Tests edge cases and fallback scenarios.

-  `test_no_connectivity_fallback` - Fallback with no connectivity
-  `test_single_satellite` - Single satellite scenario
-  `test_high_minimum_requirement` - Requirement exceeds available satellites

### 10. TestLoadBalancing (1 test)
Tests load balancing across satellites.

-  `test_load_balancing_rotation` - Verify server rotation

### 11. TestConfigurationLoading (2 tests)
Tests configuration loading from AlgorithmConfig.

-  `test_config_applies_settings` - Config application
-  `test_config_defaults` - Default value handling

## Test Features

### What's Tested

1. **Initialization & Setup**
   - Algorithm object creation
   - Default parameter values
   - Satellite names configuration
   - Adjacency matrix setup

2. **Server Selection Parameters**
   - `connect_to_all_satellites` (bool)
   - `max_lookahead` (int)
   - `minimum_connected_satellites` (int)

3. **Connectivity Analysis**
   - Single satellite analysis
   - Cumulative connection tracking
   - Connection timeline generation
   - Max connectivity detection

4. **Aggregation Server Selection**
   - Best server selection logic
   - Multi-criteria selection (connectivity, speed, load balancing)
   - Minimum satellite requirements
   - Lookahead window constraints

5. **Redistribution Server Selection**
   - Two-hop optimization (A + B)
   - Same server optimization
   - Minimum connectivity requirements
   - Time optimization

6. **Operating Modes**
   - Connect-to-all mode (requires all satellites)
   - Minimum satellites mode (requires minimum threshold)
   - Max lookahead constraints
   - Extended search with fallback

7. **Three-Phase Execution**
   - TRANSMITTING phase
   - CHECK phase
   - REDISTRIBUTION phase
   - Output data structure

8. **Edge Cases**
   - No connectivity scenarios
   - Single satellite systems
   - High minimum requirements
   - Fallback scenarios

9. **Load Balancing**
   - Server rotation
   - Selection count tracking
   - Fair distribution

10. **Configuration**
    - AlgorithmConfig integration
    - options.json loading
    - Default value handling

## Test Data

Tests use synthetic adjacency matrices to simulate various connectivity scenarios:

- **Full connectivity**: All satellites can reach each other
- **Partial connectivity**: Limited connections between satellites
- **No connectivity**: Isolated satellites
- **Time-varying connectivity**: Connections appear at different timesteps

## Expected Behavior

### Server Selection Criteria

**Priority Order:**
1. **Connectivity requirement**: Must meet minimum/all satellite requirement
2. **Lookahead window**: Must achieve within max_lookahead timesteps
3. **Max connections**: Prefer satellites with highest connectivity
4. **Speed**: Prefer satellites that connect faster
5. **Load balancing**: Prefer less frequently selected satellites

### Fallback Behavior

When no satellite meets criteria:
1. Extend search to 2x max_lookahead
2. If still not found, select best available (even if below minimum)
3. Always returns a valid server

## Success Criteria

All tests must pass for algorithm to be considered production-ready:

```
======================================================================
TEST SUMMARY
======================================================================
Tests run: 25
Successes: 25
Failures: 0
Errors: 0
======================================================================
```

## Integration with CI/CD

This test suite can be integrated into continuous integration pipelines:

```bash
# Exit with non-zero status on failure
python flomps_algorithm/test_algorithm.py
echo $?  # 0 if all tests pass, 1 if any fail
```

## Adding New Tests

To add new tests:

1. Create a new test class inheriting from `unittest.TestCase`
2. Add test methods starting with `test_`
3. Add the test class to `run_test_suite()` function
4. Run the complete suite to verify

Example:
```python
class TestNewFeature(unittest.TestCase):
    def setUp(self):
        self.algorithm = Algorithm()

    def test_new_functionality(self):
        # Test implementation
        self.assertTrue(True)

# Add to run_test_suite()
suite.addTests(loader.loadTestsFromTestCase(TestNewFeature))
```

## Related Files

- `algorithm_core.py` - Main algorithm implementation
- `algorithm_config.py` - Configuration loader
- `algorithm_handler.py` - Handler for running algorithm
- `algorithm_output.py` - Output formatter
- `/options.json` - Configuration file

## Documentation

For detailed algorithm documentation, see:
- `ALGORITHM_CORE_README_UPDATED.md`
- `3phaseImplementation.md`
