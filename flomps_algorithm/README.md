# FLOMPS Algorithm Module

This directory contains the core implementation of the FLOMPS (Federated Learning Over Moving Parameter Server) algorithm for satellite swarm federated learning.

## Overview

The FLOMPS algorithm implements a three-phase approach to federated learning in dynamic satellite networks, where connectivity changes over time due to orbital motion. The algorithm intelligently selects parameter servers based on multi-criteria optimization to minimize communication time while ensuring model distribution across the satellite constellation.

## Files

### Core Implementation
- **`algorithm_core.py`** - Main algorithm implementation with three-phase execution
- **`algorithm_handler.py`** - Handler for processing adjacency matrices and running algorithm
- **`algorithm_config.py`** - Configuration loader that reads from options.json
- **`algorithm_output.py`** - Output formatter that generates FLAM CSV files

### Configuration & Testing
- **`test_algorithm.py`** - Comprehensive test suite (25 tests, 100% coverage)
- **`TEST_README.md`** - Detailed test documentation
- **`README.md`** - This file

### Documentation
- **`ALGORITHM_CORE_README_UPDATED.md`** - Detailed algorithm documentation
- **`3phaseImplementation.md`** - Three-phase implementation details

## Three-Phase Algorithm

The FLOMPS algorithm executes federated learning rounds in three distinct phases:

### Phase 1: TRANSMITTING (Model Aggregation)
**Purpose:** Collect local models from client satellites to aggregation server

**Process:**
1. Analyze all satellites for connectivity potential
2. Select aggregation server using multi-criteria optimization:
   - Primary: Highest maximum connections
   - Secondary: Fastest to achieve maximum connections
   - Tertiary: Load balancing (least frequently selected)
3. Wait for server to connect to target satellites
4. Aggregate local models into global model
5. Complete when all target satellites connected or timesteps exhausted
6. No artificial timeout - continues until natural completion

**Output:** Global model at aggregation server

### Phase 2: CHECK (Server Transfer)
**Purpose:** Transfer global model to optimal redistribution server

**Process:**
1. Evaluate all satellites for redistribution capability
2. Calculate two-hop time: A + B
   - A = Time for aggregation server to reach candidate
   - B = Time for candidate to reach all clients
3. Select redistribution server with minimum total time
4. Transfer global model (if different from aggregation server)

**Output:** Global model at redistribution server

### Phase 3: REDISTRIBUTION (Model Distribution)
**Purpose:** Distribute global model from redistribution server to all clients

**Process:**
1. Redistribution server broadcasts global model
2. Track cumulative connections to client satellites
3. Complete when all target satellites receive model or timesteps exhausted
4. No artificial timeout - continues until natural completion

**Output:** All clients updated with global model

### Phase Duration Tracking
Each phase tracks:
- `phase_length`: Total duration of phase (updated retroactively)
- `timestep_in_phase`: Current position within phase
- `phase_complete`: Boolean flag indicating completion

## Server Selection Algorithm

### Configurable Parameters

The algorithm supports three key configuration parameters (set in `/options.json`):

```json
{
  "algorithm": {
    "server_selection": {
      "connect_to_all_satellites": false,
      "max_lookahead": 20,
      "minimum_connected_satellites": 5
    }
  }
}
```

#### 1. connect_to_all_satellites (boolean)
- **true**: Server must connect to ALL satellites (n-1 connections)
- **false**: Server must meet minimum_connected_satellites threshold
- **Default:** false

#### 2. max_lookahead (integer)
- Maximum timesteps to look ahead when evaluating satellites during server selection
- Server must achieve required connections within this window
- If no valid server found, algorithm searches through ALL remaining timesteps
- Does NOT apply to actual communication phases (TRANSMITTING/REDISTRIBUTION)
- **Default:** 20 timesteps

#### 3. minimum_connected_satellites (integer)
- Minimum number of satellites server must connect to (when connect_to_all=false)
- Both minimum requirement AND lookahead window must be satisfied
- If unmet, algorithm falls back to best available server
- **Default:** 5 satellites

### Selection Criteria

**Step 1: Filter Valid Candidates**
- Must meet connectivity requirement (all satellites OR minimum satellites)
- Must achieve requirement within max_lookahead timesteps

**Step 2: Rank Valid Candidates**
1. **Primary:** Maximum connections (prefer higher)
2. **Secondary:** Speed to required connections (prefer faster)
3. **Tertiary:** Load balancing (prefer less frequently selected)

**Step 3: Fallback (if no valid candidates)**
1. Search through ALL remaining timesteps (not just 2x max_lookahead)
2. If still impossible to meet requirements:
   - Display clear IMPOSSIBLE message
   - Select satellite with most connections (best available)
3. Always returns a valid server

### Multi-Criteria Optimization

The algorithm uses a hierarchical selection process:

```
For each satellite:
  Analyze connectivity over lookahead window
  Track cumulative connections (persistent model)
  Calculate time to required connections

Filter satellites:
  Keep only those meeting requirements within lookahead

Select best:
  1. Highest max_connections
  2. If tied: Fastest to reach required connections
  3. If tied: Least frequently selected (load balancing)
```

## Cumulative Connectivity Model

Unlike traditional approaches that reset connections each timestep, FLOMPS uses a **cumulative connectivity model** that better reflects satellite communication:

- Once a satellite connects, the connection persists for the duration of the round
- This models realistic store-and-forward communication
- Satellites accumulate connections over time
- The server tracks which satellites have been reached, not just current connections

**Example:**
```
Timestep 1: Server connects to [Sat1, Sat2]
Timestep 2: Server connects to [Sat3]
Timestep 3: Server connects to [Sat1]

Cumulative connections: [Sat1, Sat2, Sat3] = 3 satellites reached
```

## Usage

### Running the Algorithm

**From algorithm handler:**
```bash
cd /path/to/flomps_algorithm
python algorithm_handler.py ../sat_sim/output/sat_sim_YYYYMMDD_HHMMSS.txt
```

**From workflow:**
```bash
python main.py flomps --input-file TLEs/SatCount8.tle
```

**Direct generation:**
```bash
python generate_flam_csv.py TLEs/SatCount8.tle
```

### Configuration

Edit `/options.json` to configure server selection:

```json
{
  "algorithm": {
    "server_selection": {
      "connect_to_all_satellites": false,
      "max_lookahead": 20,
      "minimum_connected_satellites": 5
    },
    "module_settings": {
      "output_to_file": true
    }
  }
}
```

### Programmatic Usage

```python
from flomps_algorithm.algorithm_core import Algorithm

# Create algorithm instance
algorithm = Algorithm()

# Configure parameters
algorithm.set_connect_to_all_satellites(False)
algorithm.set_max_lookahead(20)
algorithm.set_minimum_connected_satellites(5)

# Set satellite names
satellite_names = ["Sat1", "Sat2", "Sat3", "Sat4"]
algorithm.set_satellite_names(satellite_names)

# Set adjacency matrices from SatSim
algorithm.set_adjacency_matrices(adjacency_matrices)

# Run algorithm
algorithm.start_algorithm_steps()

# Get output
output = algorithm.get_algorithm_output()
```

## Output Format

The algorithm generates FLAM (Federated Learning Adjacency Matrix) CSV files in `synth_FLAMs/`:

### Filename Convention
```
flam_<n>n_<t>t_flomps_3phase_YYYY-MM-DD_HH-MM-SS.csv
```
- `n` = number of satellites
- `t` = number of timesteps
- `flomps_3phase` = algorithm type
- `YYYY-MM-DD_HH-MM-SS` = generation timestamp

### CSV Format
```csv
Timestep: 1, Round: 1, Target Node: 2, Phase: TRANSMITTING
0,0,1,0
0,0,1,0
1,1,0,1
0,0,1,0

Timestep: 7, Round: 1, Target Node: 2, Phase: CHECK
...

Timestep: 8, Round: 1, Target Node: 2, Phase: REDISTRIBUTION
...
```

Each block contains:
- Header: Timestep, Round number, Target node (server), Phase
- Adjacency matrix: n×n matrix showing active connections

## Testing

### Running Tests

```bash
cd /Users/ash/Desktop/SPACE_FLTeam
python flomps_algorithm/test_algorithm.py
```

### Test Coverage

**25 tests across 11 test classes:**
- Algorithm initialization (4 tests)
- Server selection parameters (3 tests)
- Single satellite analysis (2 tests)
- Aggregation server selection (2 tests)
- Redistribution server selection (2 tests)
- Connect-to-all mode (2 tests)
- Max lookahead behavior (2 tests)
- Three-phase execution (2 tests)
- Edge cases and fallback (3 tests)
- Load balancing (1 test)
- Configuration loading (2 tests)

**Success Rate:** 100% (25/25 passing)

See `TEST_README.md` for detailed test documentation.

## Key Features

### 1. Dynamic Server Selection
- Adapts to changing satellite connectivity
- Multi-criteria optimization
- Configurable requirements

### 2. Two-Hop Optimization
- Redistribution server selection minimizes total time
- Evaluates aggregation-to-redistribution + redistribution-to-clients
- Can use same server for both aggregation and redistribution if optimal

### 3. Load Balancing
- Tracks selection frequency per satellite
- Rotates between equally-qualified servers
- Prevents server overload

### 4. Flexible Configuration
- Three independent parameters
- Can enforce strict requirements or use best-effort
- Supports small to large satellite constellations

### 5. Robust Fallback
- Searches ALL remaining timesteps when no valid candidates found
- Selects server with most connections when requirements impossible
- Clear IMPOSSIBLE messages when strict requirements can't be met
- Always selects a valid server (graceful degradation)

## Algorithm Performance

### Time Complexity
- Satellite analysis: O(n × t) where n = satellites, t = lookahead
- Server selection: O(n²) for comparison
- Total per round: O(n² × t)

### Space Complexity
- Adjacency matrices: O(n² × T) where T = total timesteps
- Analysis cache: O(n × t)
- Output data: O(T)

### Scalability
- Tested with 4-40 satellites
- Recommended max_lookahead: 10-30 timesteps
- Large constellations (>40 satellites) may require optimization

## Dependencies

- Python 3.12
- NumPy 1.26.4
- Standard library: json, random, sys

## Integration

The algorithm integrates with:
- **SatSim**: Receives adjacency matrices
- **Federated Learning**: Outputs FLAM schedules
- **Module Factory**: Instantiated via factory pattern
- **CLI**: Accessible through main.py workflows

## Configuration Architecture

### Three-Tier Hierarchy
1. `settings.json` - Global system settings (which config file to use)
2. `options.json` - Module-specific configuration (algorithm parameters)
3. Command-line arguments - Runtime overrides

**Priority:** CLI args > options.json > defaults

### AlgorithmConfig Flow
```
options.json → AlgorithmConfig.read_options()
           → algorithm.set_*() methods
           → Algorithm instance configured
```

## Common Scenarios

### Scenario 1: Require All Satellites
```json
{
  "connect_to_all_satellites": true,
  "max_lookahead": 30
}
```
Server must connect to all satellites within 30 timesteps.

### Scenario 2: Best Effort (Default)
```json
{
  "connect_to_all_satellites": false,
  "max_lookahead": 20,
  "minimum_connected_satellites": 5
}
```
Server must connect to at least 5 satellites within 20 timesteps.

### Scenario 3: Quick Rounds
```json
{
  "connect_to_all_satellites": false,
  "max_lookahead": 10,
  "minimum_connected_satellites": 3
}
```
Prioritize speed over coverage.

### Scenario 4: Maximum Coverage
```json
{
  "connect_to_all_satellites": false,
  "max_lookahead": 50,
  "minimum_connected_satellites": 15
}
```
Allow longer wait for more connections.

## Troubleshooting

### No valid servers found
**Cause:** Requirements too strict for satellite constellation
**Solution:** Reduce minimum_connected_satellites or increase max_lookahead

### Always falling back
**Cause:** Connectivity pattern doesn't support requirements within lookahead
**Solution:** Check TLE file for realistic orbits, adjust parameters

### Same server selected repeatedly
**Cause:** Only one satellite meets requirements
**Solution:** Relax connect_to_all or minimum_connected_satellites

### Rounds take too long
**Cause:** max_lookahead too high, waiting for many connections
**Solution:** Reduce minimum_connected_satellites or max_lookahead

## Version History

- **v2.0** (2025-10-10): Added configurable server selection parameters
- **v1.5** (2025-10-03): Implemented three-phase round structure
- **v1.2** (2025-09-12): Optimized function hierarchy
- **v1.1** (2025-09-05): Added cumulative connectivity model
- **v1.0** (2024-09-21): Initial load balancing implementation

## Authors

- Elysia Guglielmo (System Architect)
- Yuganya Perumal (Algorithm Implementation)
- Gagandeep Singh (Three-Phase Design)
- Claude Code (Configuration System & Testing)

## License

Part of SPACE (Satellite Federated Learning Project)

## References

For more details, see:
- `ALGORITHM_CORE_README_UPDATED.md` - Detailed algorithm documentation
- `3phaseImplementation.md` - Three-phase design specification
- `TEST_README.md` - Test suite documentation
- `PROJECT_ARCHITECTURE_GUIDE.md` - System architecture
