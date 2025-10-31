# Minimum Satellites for Redistribution Feature

## Overview

This feature adds independent configuration of minimum satellite requirements for the TRANSMITTING and REDISTRIBUTION phases of the FLOMPS algorithm.

## Motivation

Previously, both the TRANSMITTING (aggregation) and REDISTRIBUTION phases used the same `minimum_connected_satellites` parameter. This new feature allows operators to set different thresholds for each phase, providing more flexibility in satellite constellation management.

**Use Case Example:**
- Set a lower minimum (e.g., 5) for TRANSMITTING to allow faster aggregation
- Set a higher minimum (e.g., 7) for REDISTRIBUTION to ensure broader model distribution

## Implementation Details

### Configuration

**File:** `options.json`

```json
{
  "federated_learning": {
    "algorithm": {
      "server_selection": {
        "connect_to_all_satellites": false,
        "max_lookahead": 20,
        "minimum_connected_satellites": 5,          // Used for TRANSMITTING phase
        "minimum_satellites_for_redistribution": 7  // Used for REDISTRIBUTION/CHECK phases
      }
    }
  }
}
```

### Code Changes

#### 1. Algorithm Core (`flomps_algorithm/algorithm_core.py`)

**Added instance variable (line 95):**
```python
self.minimum_satellites_for_redistribution = 7
```

**Added setter method (lines 126-127):**
```python
def set_minimum_satellites_for_redistribution(self, minimum_satellites_for_redistribution):
    self.minimum_satellites_for_redistribution = minimum_satellites_for_redistribution
```

**Updated `find_best_redistribution_server()` method (lines 385-394):**
```python
# Determine required connections based on settings
if self.connect_to_all_satellites:
    required_connections = num_satellites - 1  # All satellites except self
else:
    required_connections = self.minimum_satellites_for_redistribution  # Changed from minimum_connected_satellites

print(f"\n=== Finding Redistribution Server (from Aggregation Server {aggregation_server}) ===")
print(f"Configuration: connect_to_all={self.connect_to_all_satellites}, "
      f"min_satellites_redist={self.minimum_satellites_for_redistribution}, max_lookahead={max_lookahead}")
print(f"Required connections: {required_connections}/{num_satellites-1}")
```

#### 2. Algorithm Config (`flomps_algorithm/algorithm_config.py`)

**Added configuration reader (lines 63-65):**
```python
self.algorithm.set_minimum_satellites_for_redistribution(
    server_selection.get("minimum_satellites_for_redistribution", 7)
)
```

## Phase Behavior

### TRANSMITTING Phase
- Uses: `minimum_connected_satellites`
- Purpose: Select aggregation server that can collect models from client satellites
- Configured via: `server_selection.minimum_connected_satellites` (default: 5)

### CHECK Phase
- Uses: `minimum_satellites_for_redistribution`
- Purpose: Select redistribution server based on redistribution requirements
- The CHECK phase selects the redistribution server, so it uses redistribution minimums

### REDISTRIBUTION Phase
- Uses: `minimum_satellites_for_redistribution`
- Purpose: Distribute global model to client satellites
- Configured via: `server_selection.minimum_satellites_for_redistribution` (default: 7)

## Configuration Options

### Independent Settings
Both parameters can be configured independently:

```json
{
  "minimum_connected_satellites": 3,          // Low for faster aggregation
  "minimum_satellites_for_redistribution": 10  // High for broad distribution
}
```

```json
{
  "minimum_connected_satellites": 10,         // High for quality aggregation
  "minimum_satellites_for_redistribution": 3   // Low for faster redistribution
}
```

```json
{
  "minimum_connected_satellites": 5,          // Same for both phases
  "minimum_satellites_for_redistribution": 5
}
```

### Interaction with `connect_to_all_satellites`

When `connect_to_all_satellites: true`, both minimum settings are overridden:
- TRANSMITTING: Must connect to ALL satellites
- REDISTRIBUTION: Must connect to ALL satellites

### Default Values
- `minimum_connected_satellites`: 5
- `minimum_satellites_for_redistribution`: 7

## Testing

### Test File
`test_minimum_satellites_for_redistribution.py`

### Test Coverage
- **23 tests total** - All passing ✓

#### Test Classes:
1. `TestMinimumSatellitesForRedistributionSettersGetters` (5 tests)
   - Default value verification
   - Setter/getter functionality
   - Independence from `minimum_connected_satellites`

2. `TestConfigurationLoading` (4 tests)
   - Loading from options.json
   - Default value handling
   - Independent configuration

3. `TestTransmittingPhaseUsesCorrectMinimum` (2 tests)
   - Verifies TRANSMITTING uses `minimum_connected_satellites`
   - Verifies TRANSMITTING ignores `minimum_satellites_for_redistribution`

4. `TestRedistributionPhaseUsesCorrectMinimum` (2 tests)
   - Verifies REDISTRIBUTION uses `minimum_satellites_for_redistribution`
   - Verifies REDISTRIBUTION ignores `minimum_connected_satellites`

5. `TestIndependentConfiguration` (4 tests)
   - Tests various configuration combinations
   - Extreme differences handling

6. `TestEdgeCasesAndFallback` (3 tests)
   - Exceeding available satellites
   - Zero minimum
   - Same server for both phases

7. `TestConnectToAllSatellitesInteraction` (1 test)
   - Verifies `connect_to_all_satellites` overrides minimums

8. `TestFullThreePhaseWorkflow` (1 test)
   - Complete round execution with different minimums
   - Output validation

9. `TestOptionsJsonIntegration` (1 test)
   - Loading from actual `options.json` file
   - Configuration verification

### Running Tests

```bash
python test_minimum_satellites_for_redistribution.py
```

**Expected Output:**
```
================================================================================
TEST SUMMARY - minimum_satellites_for_redistribution Feature
================================================================================
Tests run: 23
Successes: 23
Failures: 0
Errors: 0
================================================================================
✓ ALL TESTS PASSED!
```

## Example Usage

### Scenario 1: Quick Aggregation, Thorough Distribution
```json
{
  "server_selection": {
    "minimum_connected_satellites": 3,
    "minimum_satellites_for_redistribution": 8
  }
}
```

**Result:**
- TRANSMITTING phase completes faster (only needs 3 connections)
- REDISTRIBUTION ensures model reaches more satellites (requires 8 connections)

### Scenario 2: Quality Aggregation, Fast Distribution
```json
{
  "server_selection": {
    "minimum_connected_satellites": 10,
    "minimum_satellites_for_redistribution": 4
  }
}
```

**Result:**
- TRANSMITTING collects from many satellites (requires 10 connections)
- REDISTRIBUTION completes faster (only needs 4 connections)

### Scenario 3: Balanced Approach
```json
{
  "server_selection": {
    "minimum_connected_satellites": 6,
    "minimum_satellites_for_redistribution": 6
  }
}
```

**Result:**
- Both phases use same threshold (6 connections)

## Fallback Behavior

If no satellite meets the minimum requirements:
1. Algorithm extends search to ALL remaining timesteps
2. If still no match, selects best available satellite
3. Logs warning message indicating fallback behavior

## Compatibility

- **Backward Compatible:** Yes
- **Default Behavior:** If `minimum_satellites_for_redistribution` is not specified, defaults to 7
- **Existing Configurations:** Continue to work without modification

## Files Modified

1. `/Users/ash/Desktop/SPACE_FLTeam/options.json` - Added configuration option
2. `/Users/ash/Desktop/SPACE_FLTeam/flomps_algorithm/algorithm_core.py` - Core implementation
3. `/Users/ash/Desktop/SPACE_FLTeam/flomps_algorithm/algorithm_config.py` - Configuration reader

## Files Created

1. `/Users/ash/Desktop/SPACE_FLTeam/test_minimum_satellites_for_redistribution.py` - Test suite

## Future Enhancements

Potential improvements:
- Add validation to ensure minimums don't exceed total satellites
- Add dynamic adjustment based on constellation size
- Add metrics/logging for tracking minimum usage patterns
- Consider adding separate minimums for CHECK phase (currently uses redistribution minimum)

