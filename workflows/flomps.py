import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import module_factory

def build_modules(options):
    sat_sim_module = module_factory.create_sat_sim_module()
    sat_sim_module.config.read_options(options["sat_sim"])
        
    algorithm_module = module_factory.create_algorithm_module()
    algorithm_module.config.read_options(options["algorithm"])

    fl_module = module_factory.create_fl_module()
    fl_module.config.read_options(options["federated_learning"])

    return sat_sim_module, algorithm_module, fl_module

def run(input_file, options):
    # Create Modules
    sat_sim_module, algorithm_module, fl_module = build_modules(options)
    
    # Simulation Process
    sat_sim_module.handler.parse_file(input_file)
    sat_sim_module.handler.run_module()
    matrices = sat_sim_module.output.get_result()
    print(matrices)
        
    algorithm_module.handler.parse_data(matrices)
    algorithm_module.handler.run_module()
    flam = algorithm_module.output.get_result()
    print(flam)

    fl_module.handler.parse_data(flam)
    fl_module.handler.run_module()

def run_with_tuning(input_file, options):  
    # Create Modules  
    sat_sim_module, algorithm_module, fl_module = build_modules(options)  
      
    # Simulation Process  
    sat_sim_module.handler.parse_file(input_file)  
    sat_sim_module.handler.run_module()  
    matrices = sat_sim_module.output.get_result()  
      
    # if "tuner" in options["algorithm"] and options["algorithm"]["tuner"]["enabled"]:
    if "tuner" in options["algorithm"] and options["algorithm"]["tuner"]["enabled"]:  
        # assume AlgorithmConfig has a tuner attribute 
        tuner = algorithm_module.handler.algorithm.tuner  
        best_params, best_perf = tuner.optimize(matrices.matrices)  
        print(f"Tuning completed. Best parameters: {best_params}")  
        print(f"Performance metrics: {best_perf}")  
      
    # countinue with the rest of the algorithm process
    algorithm_module.handler.parse_data(matrices)  
    algorithm_module.handler.run_module()  
    flam = algorithm_module.output.get_result()  
      
    fl_module.handler.parse_data(flam)  
    fl_module.handler.run_module()
