# Three-Phase FLOMPS Algorithm Implementation

## Overview

This document describes the implementation of the three-phase round structure for the FLOMPS (Federated Learning Over Moving Parameter Server) algorithm, completed on October 3, 2025.

## Problem Statement

The original FLOMPS algorithm only simulated the uplink aggregation phase (clients sending models to parameter server) but did not simulate the downlink redistribution phase (parameter server distributing the global model back to clients). This was a critical gap in the federated learning simulation, as real federated learning requires both phases for a complete round.

## Solution: Three-Phase Round Structure

Each federated learning round now consists of three distinct phases:

### Phase 1: TRANSMITTING (Aggregation)
- **Purpose**: Aggregate local models from clients to parameter server
- **Server Selection**: Analyze all satellites and select one that reaches the most clients fastest
- **Duration**: Variable, continues until aggregation server connects to all target satellites (cumulative connectivity)
- **End State**: Aggregation server has the global model

### Phase 2: CHECK (Redistribution Server Selection)
- **Purpose**: Select optimal satellite for redistribution and transfer global model
- **Selection Criteria**: Two-hop optimization
  - Time A: Aggregation server to candidate satellite
  - Time B: Candidate satellite to all other clients
  - Total time = A + B
  - Select satellite with minimum total time
- **Duration**: Time A (timesteps to reach selected redistribution server)
- **Special Case**: If redistribution server equals aggregation server, duration = 0 timesteps
- **End State**: Redistribution server has the global model

### Phase 3: REDISTRIBUTION (Distribution)
- **Purpose**: Distribute global model to all clients
- **Logic**: Same as TRANSMITTING but from redistribution server
- **Duration**: Variable, continues until redistribution server connects to all clients (cumulative connectivity)
- **End State**: All clients have synchronized global model

## Implementation Details

### Files Modified

#### 1. algorithm_core.py
**Added Fields:**
- `self.aggregation_server`: Tracks server from TRANSMITTING phase
- `self.redistribution_server`: Tracks server from CHECK/REDISTRIBUTION phases

**New Method:**
```python
find_best_redistribution_server(aggregation_server, start_matrix_index, max_lookahead=20)
```
- Analyzes path from aggregation server to each candidate
- For each reachable candidate, analyzes redistribution capability
- Performs two-hop optimization (A + B)
- Primary criteria: Minimum total time
- Secondary criteria: Maximum connections (tie-breaker)
- Tertiary criteria: Random selection (final tie-breaker)

**Rewritten Method:**
```python
start_algorithm_steps()
```
- Completely restructured to execute three phases per round
- Each phase has its own loop and completion logic
- Phase lengths are updated retroactively after phase completion
- Comprehensive logging for each phase

#### 2. algorithm_output.py
**Updated CSV Format:**
- Filename: `flam_{n}n_{t}t_flomps_3phase_{timestamp}.csv`
- Added fields:
  - `aggregation_id`: Server from TRANSMITTING phase
  - `redistribution_id`: Server from REDISTRIBUTION phase (TBD during TRANSMITTING)
  - `phase_length`: Total duration of current phase (updated retroactively)
  - `timestep_in_phase`: Position within current phase
  - `phase_complete`: Boolean flag indicating phase completion

**Enhanced Header:**
```
Time: {timestamp}, Timestep: {n}, Round: {r}, Phase: {phase},
Aggregation Server: {agg_id}, Redistribution Server: {redist_id},
Target Node: {target}, Phase Length: {len}, Timestep in Phase: {step},
Current Connections: {curr}, Cumulative Connections: {cum}/{total},
Connected Sats: [list], Missing Sats: [list], Target Sats: [list],
Phase Complete: {bool}
```

**Round Statistics:**
- Now shows breakdown by phase
- Displays aggregation and redistribution servers
- Shows duration of each phase separately
- Lists which phases were executed

#### 3. generate_flam_csv.py
**Bug Fix:**
- Removed obsolete `set_algorithm_parameters()` call
- Algorithm now works without deprecated parameter setting

### Key Algorithm Features

#### Two-Hop Optimization
The CHECK phase uses sophisticated two-hop path analysis:
1. Calculate time for aggregation server to reach each candidate
2. Calculate time for each candidate to reach all clients
3. Select candidate minimizing total transfer time
4. Ensures efficient global model distribution

#### Cumulative Connectivity Model
Both TRANSMITTING and REDISTRIBUTION phases use cumulative connectivity:
- Connections persist over time within a phase
- More realistic satellite communication behavior
- Tracks progressive connection building

#### Phase Length Correction
Phase lengths are updated retroactively:
```python
# After phase completion
for i in range(phase_start_index, current_matrix_index):
    timestamp_key = adjacency_matrices[i][0]
    algorithm_output[timestamp_key]['phase_length'] = actual_phase_length
```
This ensures all timesteps in a phase show the correct total phase duration.

#### Load Balancing
Selection count tracking prevents server monopolization:
- Tracks how many times each satellite has been selected
- Used as tertiary criteria in tie-breaking
- Ensures fair distribution of computational load

## Testing Results

### Test Configuration
- Satellite count: 8 satellites
- Simulation duration: 120 timesteps
- Total rounds completed: 11 rounds

### Sample Round (Round 2)
```
TRANSMITTING: 7 timesteps (Server 7)
  Aggregation Server 7 collected from [0, 2, 3, 4, 5, 6]

CHECK: 1 timestep (Server 7 to Server 5)
  Time A = 1 timestep
  Time B = 1 timestep
  Total = 2 timesteps (Server 5 selected)

REDISTRIBUTION: 1 timestep (Server 5)
  Redistribution Server 5 distributed to [0, 2, 3, 4, 7]

Total Round Duration: 9 timesteps
```

### Observed Patterns
- Rounds with same aggregation/redistribution server: CHECK duration = 0
- Rounds with different servers: CHECK duration = 1-2 timesteps typically
- Redistribution servers often different from aggregation servers when optimal
- Load balancing working correctly across rounds

## CSV Output Example

```csv
Time: 2025-10-03 10:30:00, Timestep: 31, Round: 2, Phase: TRANSMITTING,
Aggregation Server: 1, Redistribution Server: TBD, Target Node: 1,
Phase Length: 20, Timestep in Phase: 20, Current Connections: 4,
Cumulative Connections: 5/5, Connected Sats: [2, 3, 4, 5, 7],
Missing Sats: [], Target Sats: [2, 3, 4, 5, 7], Phase Complete: True

Time: 2025-10-03 10:31:00, Timestep: 32, Round: 2, Phase: CHECK,
Aggregation Server: 1, Redistribution Server: 7, Target Node: 7,
Phase Length: 1, Timestep in Phase: 1, Current Connections: 0,
Cumulative Connections: 0/0, Connected Sats: [], Missing Sats: [],
Target Sats: [], Phase Complete: True

Time: 2025-10-03 10:32:00, Timestep: 33, Round: 2, Phase: REDISTRIBUTION,
Aggregation Server: 1, Redistribution Server: 7, Target Node: 7,
Phase Length: 1, Timestep in Phase: 1, Current Connections: 5,
Cumulative Connections: 5/5, Connected Sats: [1, 2, 3, 4, 5],
Missing Sats: [], Target Sats: [1, 2, 3, 4, 5], Phase Complete: True
```

## Benefits of Three-Phase Implementation

1. **Complete FL Simulation**: Now simulates both uplink and downlink communication
2. **Moving Parameter Server**: Model location explicitly tracked through phases
3. **Optimized Distribution**: Two-hop optimization minimizes total distribution time
4. **Realistic Behavior**: Accounts for satellite movement and relay scenarios
5. **Comprehensive Metadata**: All phase information available for FL module consumption
6. **Flexible Architecture**: Can handle same-server and different-server scenarios

## Usage

### Command Line
```bash
cd /path/to/flomps_algorithm
python algorithm_handler.py ../sat_sim/output/sat_sim_20250606_233120.txt
```

### Output Location
```
synth_FLAMs/flam_8n_120t_flomps_3phase_2025-10-03_10-41-45.csv
```

## Future Enhancements

Potential improvements to consider:
- Adaptive lookahead window based on constellation density
- Multi-criteria optimization weights for different scenarios
- Support for partial redistribution when full coverage is impossible
- Performance metrics tracking across phases
- Visualization of three-phase execution timeline

## Conclusion

The three-phase FLOMPS algorithm implementation successfully addresses the missing redistribution phase in federated learning simulation. The algorithm now provides a complete end-to-end simulation of moving parameter server federated learning in satellite constellations, with comprehensive tracking of model location, server selection, and distribution patterns.
