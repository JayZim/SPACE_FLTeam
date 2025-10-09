"""
Filename: module_factory.py
Description: Creates modules for simulation pipeline
Author: Elysia Guglielmo
Date: 2024-08-011
Version: 1.0
Python Version: 

Changelog:
- 2024-08-11: Initial creation.

Usage: Access a module and its I/O interfaces by calling the relevant create function

"""
from enum import Enum
from typing import NamedTuple

from sat_sim.sat_sim import SatSim
from sat_sim.sat_sim_config import SatSimConfig
from sat_sim.sat_sim_handler import SatSimHandler
from sat_sim.sat_sim_output import SatSimOutput

from flomps_algorithm.algorithm_core import Algorithm
from flomps_algorithm.algorithm_config import AlgorithmConfig
from flomps_algorithm.algorithm_handler import AlgorithmHandler
from flomps_algorithm.algorithm_output import AlgorithmOutput

from federated_learning.fl_core import FederatedLearning
from federated_learning.fl_config import FLConfig
from federated_learning.fl_handler import FLHandler
from federated_learning.fl_output import FLOutput

class ModuleKey(Enum):
    SAT_SIM = "sat_sim"
    ALGORITHM = "algorithm"
    FL = "federated_learning"
    MODEL = "model"

MODULE_MAPPING = {
        'sat_sim': 'sat_sim',
        'algorithm': 'algorithm',
        'fl': 'federated_learning',
        'model': 'model'
    }

# Instantiate relevant classes

# Instantiate Handlers, Configs and Cores
# (Outputs are instantiated within Cores)
class SatSimModule(NamedTuple):
    config: SatSimConfig
    handler: SatSimHandler
    output: SatSimOutput

class AlgorithmModule(NamedTuple):
    config: AlgorithmConfig
    handler: AlgorithmHandler
    output: AlgorithmOutput

class FLModule(NamedTuple):
    config: FLConfig
    handler: FLHandler
    output: FLOutput

def create_single_instance(module_key: ModuleKey):
    if module_key == ModuleKey.SAT_SIM:
        return create_sat_sim_module()
    elif module_key == ModuleKey.ALGORITHM:
        return create_algorithm_module()
    elif module_key == ModuleKey.FL:
        return create_fl_module()
    elif module_key == ModuleKey.MODEL:
        # TODO: Model class
        pass
    else:
        raise ValueError("Invalid ModuleKey")


def create_sat_sim_module() -> SatSimModule:
    sat_sim = SatSim()
    sat_sim_config = SatSimConfig(sat_sim)
    sat_sim_handler = SatSimHandler(sat_sim)

    return SatSimModule(sat_sim_config, sat_sim_handler, sat_sim.output)

def create_algorithm_module() -> AlgorithmModule:
    algorithm = Algorithm()
    algorithm_config = AlgorithmConfig(algorithm)
    algorithm_handler = AlgorithmHandler(algorithm)
    
    return AlgorithmModule(algorithm_config, algorithm_handler, algorithm.output)

def create_fl_module() -> FLModule:
    fl = FederatedLearning()
    fl_config = FLConfig(fl)
    fl_handler = FLHandler(fl)
    fl_output = FLOutput()

    return FLModule(fl_config, fl_handler, fl_output)

def create_fl_module_with_model_evaluation() -> FLModule:
    """Create FL module with model evaluation enabled"""
    fl = FederatedLearning(enable_model_evaluation=True)
    fl_config = FLConfig(fl)
    fl_handler = FLHandler(fl)
    fl_output = FLOutput()

    return FLModule(fl_config, fl_handler, fl_output)