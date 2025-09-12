# FLOMPS Algorithm Core

## Overview

### Current Function Architecture
```
Algorithm Class
├── Data Management
│   ├── set_satellite_names() ✓
│   ├── set_adjacency_matrices() ✓
│   └── get_algorithm_output() ✓
├── Analysis Engine
│   ├── analyze_all_satellites() ✓
│   └── analyze_single_satellite() ✓
├── Server Selection
│   └── find_best_server_for_round() ✓
└── Main Execution
    └── start_algorithm_steps() ✓
```

---

## Core Functionality

### What FLOMPS Does
FLOMPS is designed to perform federated learning across a constellation of satellites where:
- **Satellites move in orbits** with dynamic, intermittent connectivity
- **Communication windows are limited** and constantly changing
- **Server selection is critical** for efficient federated learning rounds
- **Load balancing is essential** to prevent satellite resource exhaustion

### Algorithm Philosophy
The algorithm uses a **comprehensive analysis-based approach** where:
1. **Multi-Criteria Selection**: Chooses servers based on maximum connectivity, speed, and load balancing
2. **Cumulative Connectivity**: Models realistic satellite behavior where connections persist over time
3. **Predictive Analysis**: Analyzes all satellites before making selection decisions
4. **Round-Based Processing**: Organizes federated learning into efficient communication rounds

---

## Class Structure

### `Algorithm` Class

The main class that orchestrates the entire FLOMPS algorithm process.

#### **Initialization (`__init__`)**
```python
def __init__(self):
```
**Purpose**: Sets up the algorithm with default values and tracking variables.

**Key Attributes**:
- `satellite_names`: List of satellite identifiers
- `adjacency_matrices`: Time-series connectivity data from satellite simulation
- `selection_counts`: Load balancing tracker (how often each satellite serves)
- `round_number`: Current federated learning round
- `output_to_file`: Whether to save results to files

---

## Function Interactions & Data Flow

### **Main Algorithm Flow**
```
start_algorithm_steps() [MAIN ORCHESTRATOR]
    ↓
    find_best_server_for_round()
    ↓
    analyze_all_satellites()
    ↓
    analyze_single_satellite() [for each satellite]
    ↓
    [Multi-criteria server selection]
    ↓
    [Round execution with selected server]
    ↓
    [Repeat until all data processed]
```

---

### 2. **Analysis Engine (Core Innovation)**

#### `analyze_all_satellites(start_matrix_index, max_lookahead=20)`
**Purpose**: Comprehensive analysis of entire satellite constellation to find connectivity potential.

**Algorithm**:
1. **Iterates through all satellites** in the constellation
2. **Calls analyze_single_satellite()** for detailed analysis of each
3. **Provides comparative overview** of all satellites' capabilities
4. **Returns analysis dictionary** for server selection decision

**Key Features**:
- **Parallel analysis**: Evaluates all satellites simultaneously
- **Detailed logging**: Shows connectivity potential for each satellite
- **Comprehensive data**: Provides complete picture for optimal decision making

#### `analyze_single_satellite(sat_idx, start_matrix_index, max_lookahead=20)`
**Purpose**: Deep analysis of individual satellite's connectivity potential over time window.

**Cumulative Connectivity Model**:
- **Persistent Connections**: Once connected, satellites stay connected for the round
- **Progressive Building**: Accumulates connections over time
- **Realistic Modeling**: Matches actual satellite communication behavior

**Algorithm**:
1. **Initialize target set**: All satellites except the analyzed one
2. **Scan future timestamps**: Look ahead up to `max_lookahead`
3. **Accumulate connections**: Add newly connected satellites to cumulative set
4. **Track timeline**: Record connection progression over time
5. **Calculate metrics**: Maximum connections and time to achieve them

**Returns**:
```python
{
    'max_connections': int,           # Maximum satellites this one can connect to
    'timestamps_to_max': int,         # Time needed to achieve max connectivity
    'connected_satellites': set,      # Set of satellites it can connect to
    'connection_timeline': list,      # Detailed progression timeline
    'satellite_idx': int,             # Satellite index
    'satellite_name': str             # Satellite identifier
}
```

---

### 3. **Server Selection (Multi-Criteria Optimization)**

#### `find_best_server_for_round(start_matrix_index=0)`
**Purpose**: Select the optimal satellite to serve as federated learning coordinator using sophisticated multi-criteria analysis.

**Selection Hierarchy**:
1. **Primary Criteria**: **Maximum Connectivity**
   - Selects satellite that can connect to the most other satellites
   - Ensures maximum participation in federated learning round

2. **Secondary Criteria**: **Speed to Maximum**
   - Among satellites with equal max connectivity, picks the fastest
   - Minimizes round duration for efficiency

3. **Tertiary Criteria**: **Load Balancing**
   - Among equally performing satellites, picks least frequently selected
   - Ensures fair distribution of computational load

**Algorithm Flow**:
```python
# Step 1: Comprehensive Analysis
satellite_analysis = self.analyze_all_satellites(start_matrix_index)

# Step 2: Multi-Criteria Selection
for each satellite:
    if higher_max_connections:
        new_best = satellite  # Primary criteria
    elif same_max_connections and faster:
        new_best = satellite  # Secondary criteria
    elif same_performance and less_frequently_selected:
        new_best = satellite  # Tertiary criteria

# Step 3: Selection & Tracking
self.selection_counts[best_server] += 1
return best_server, predicted_time, analysis_data
```

**Key Features**:
- **Intelligent reasoning**: Shows why each satellite was or wasn't selected
- **Predictive capability**: Estimates round duration before execution
- **Load balancing**: Prevents server monopolization
- **Comprehensive output**: Provides detailed selection rationale

**Returns**: `(server_index, estimated_timestamps, analysis_data)`

---

### 4. **Main Algorithm Execution**

#### `start_algorithm_steps()`
**Purpose**: Execute the complete FLOMPS federated learning algorithm with round-based processing.

**Algorithm Flow**:
1. **Round Initialization**
   - Call `find_best_server_for_round()` for optimal server selection
   - Set round parameters and tracking variables

2. **Connectivity Monitoring**
   - Track cumulative connections for selected server
   - Continue until server connects to ALL target satellites
   - Generate federated learning output for each timestamp

3. **Round Completion**
   - Detect when target connectivity is achieved
   - Update all timestamps in round with final metrics
   - Calculate success rate and performance statistics

4. **Safety Mechanisms**
   - **Timeout protection**: Prevent infinite rounds (20 timestamp max)
   - **Data validation**: Handle edge cases and sparse connectivity
   - **Progress tracking**: Detailed logging and feedback

**Output Structure**:
Each timestamp generates comprehensive federated learning data:
```python
{
    'satellite_count': int,
    'satellite_names': list,
    'selected_satellite': str,
    'aggregator_id': int,
    'federatedlearning_adjacencymatrix': matrix,
    'aggregator_flag': bool,
    'round_number': int,
    'phase': str,
    'target_node': int,
    'round_length': int,
    'timestep_in_round': int,
    'server_connections_current': int,
    'server_connections_cumulative': int,
    'target_connections': int,
    'connected_satellites': list,
    'missing_satellites': list,
    'target_satellites': list,
    'round_complete': bool
}
```

---

## Key Algorithm Features

### **1. Multi-Criteria Optimization**
- **Hierarchical decision making**: Maximum connectivity → Speed → Load balancing
- **Intelligent selection**: Explains reasoning for each server choice
- **Adaptive behavior**: Adjusts to changing satellite constellation conditions

### **2. Comprehensive Analysis**
- **Full constellation evaluation**: Analyzes every satellite before selection
- **Predictive capability**: Estimates performance before round execution
- **Detailed insights**: Provides timeline and connectivity progression data

### **3. Cumulative Connectivity Model**
- **Realistic satellite behavior**: Connections persist over time
- **Progressive round building**: Satellites connect incrementally
- **Accurate performance prediction**: Better round planning and optimization

### **4. Load Balancing**
- **Selection count tracking**: Monitors satellite usage across rounds
- **Fair distribution**: Prevents server monopolization
- **Balanced workload**: Ensures constellation-wide participation

---

## Performance Characteristics

### **Optimization Goals**
1. **Maximize connectivity**: Select servers that can reach most satellites
2. **Minimize round duration**: Achieve target connectivity as quickly as possible
3. **Ensure fairness**: Distribute server responsibilities evenly
4. **Handle sparse connectivity**: Robust operation with limited satellite visibility


### **Recent Test Results** (September 2025)
- **Constellation Size**: 8 satellites
- **Total Rounds**: 38 rounds completed
- **Total Timesteps**: 100 (full dataset)
- **Success Rate**: High efficiency with optimal server selections
- **Load Balancing**: Even distribution across capable satellites

---

## Integration Points

### **Input**: Satellite Simulation Data
- **Source**: `sat_sim` component
- **Format**: Time-series adjacency matrices
- **Content**: Satellite connectivity patterns over time

### **Output**: Federated Learning Data
- **Destination**: `federated_learning` component
- **Format**: FLAM (Federated Learning Adjacency Matrix)
- **Content**: Optimized server selections and connectivity data

### **Configuration**: Algorithm Parameters
- **Analysis window**: `max_lookahead` parameter (default: 20 timestamps)
- **Timeout limits**: Maximum round duration (default: 20 timestamps)
- **Load balancing**: Selection count tracking and penalty system

---

## Usage Example

```python
# Initialize algorithm
algorithm = Algorithm()

# Configure satellite constellation
algorithm.set_satellite_names(['Sat1', 'Sat2', 'Sat3', 'Sat4'])

# Load connectivity data from satellite simulator
algorithm.set_adjacency_matrices(connectivity_data)

# Execute FLOMPS algorithm
algorithm.start_algorithm_steps()

# Retrieve results
results = algorithm.get_algorithm_output()
```

---
