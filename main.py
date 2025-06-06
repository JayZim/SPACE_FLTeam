"""
Filename: main.py
Description: Simulation Manager
Author: Elysia Guglielmo
Date: 2024-08-11
Version: 2.0

Changelog:
- 2024-08-08: Initial creation.
- 2024-08-19: CLI Argparse to JSON system for Module Config.
- 2024-09-01: Run standalone module functionality from CLI + Argparse refactoring.
- 2024-09-15: Global settings, custom JSON options file feature, bug fixes
- 2024-09-22: Integrated FLOMPS simulation workflow.
- 2024-10-28: Refactored for future development and simulation workflows.

Usage: 
python main.py <workflow> <input-file> <standalone-module(?)> <module-args>
python main.py settings <global-args>

TEST COMMENT FOR GIT TOMFOOLERY---)))
"""

import sys
import argparse
import json

import module_factory
from module_factory import ModuleKey
import cli_args

from workflows import flomps

def set_config_file(file):
    """Provide a JSON file to use for options configuration."""
    with open('settings.json') as f:
        options = json.load(f)
        options['config_file'] = file

    with open('settings.json', 'w') as f:
        json.dump(options, f, indent=4)
    
def get_config_file():
    """Get the current JSON file in use for options configuration."""
    config_file = ''
    with open('settings.json') as f:
        options = json.load(f)
        config_file = options['config_file']

    return config_file

def read_options_file():
    """Parse the current JSON file in use for options configuration."""
    file = get_config_file()
    with open(file) as f:
        options = json.load(f)
        
    return options

def write_options_file(options):
    """Update the JSON options file with new options. Must maintain original structure."""
    with open('options.json', 'w') as f:
        json.dump(options, f, indent=4)

def setup_parser():
    """Prepares the main parser with subparsers for simulation workflows."""
    parser = argparse.ArgumentParser(description='Run FLOMPS Simulation Suite')

    subparsers = parser.add_subparsers(dest='command', help="Available commands")

    settings_parser = subparsers.add_parser('settings', help="System settings")
    cli_args.setup_settings_parser(settings_parser)

    flomps_parser = subparsers.add_parser('flomps', help="Run a FLOMPS simulation")
    cli_args.setup_flomps_parser(flomps_parser)

    # Continue with command subparsers here, one per workflow

    return parser

def separate_args(parser, args_for_config):
    """
    Separate module arguments from simulation options and positional arguments.

    Return options arguments and module arguments separately.
    """
    for group in parser._action_groups:
        print(f"Group: {group.title}")
        for action in group._group_actions:
            print(f" - {action.dest}")

    def get_group_arg_keys(parser, group_title):
        group = next(group for group in parser._action_groups if group.title == group_title)
        keys = [a.dest for a in group._group_actions]
        return keys
    
    options_arg_keys = get_group_arg_keys(parser, "options")
    positional_arg_keys = get_group_arg_keys(parser, "positional arguments")

    options_args = {k: v for k, v in args_for_config.items() if k in options_arg_keys}
    module_args = {k: v for k, v in args_for_config.items() if k not in options_arg_keys and k not in positional_arg_keys}

    return options_args, module_args

def check_standalone_module_flag(args):
    """
    Arguments to run standalone modules contain the 'only' keyword at the end e.g. --sat-sim-only.

    Check if the user is specifying to run in standalone mode.
    """
    flags = [k[:-5] for k in args.keys() if 'only' in k]

    if len(flags) > 1:
        raise ValueError("Cannot set more than one module to run standalone")

    if len(flags) == 0:
        return None
    
    single_module = flags[0]

    # error for mapping failure
    if single_module not in module_factory.MODULE_MAPPING:
        raise ValueError(f"Invalid module specified for standalone operation: {single_module}")

    try:
        return ModuleKey(module_factory.MODULE_MAPPING[single_module])
    except ValueError:
        raise ValueError(f"Invalid module specified for standalone operation: {single_module}")

def check_args_match_module(parser, args_for_config, module_key: ModuleKey):
    """Validate whether user-provided arguments match the module to run standalone."""
    group = next(g for g in parser._action_groups if g.title == module_key)

    # Get dest (names) for all actions for comparison
    group_actions_keys = []
    for action in group._group_actions:
        group_actions_keys.append(action.dest)

    mismatch = any(a not in group_actions_keys for a in args_for_config.keys())

    if mismatch:
        raise ValueError(f"Invalid command found for module: {module_key.value}")


def check_standalone_config(parser, args_for_config):
    """Check if user has specified to run a module in standalone mode."""
    try:
        options_args, module_args = separate_args(parser, args_for_config)
        print("OPTIONS ARGS")
        print(options_args)
        print("MODULE ARGS")
        print(module_args)

        single_module_key = check_standalone_module_flag(options_args)
        # Ensure only options for that module were selected
        if single_module_key:
            if module_args:
                check_args_match_module(parser, module_args, single_module_key)
            return single_module_key
        else:
            return None
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
        

def update_options_with_args(parser, args_for_config, options):
    """Update the JSON options file with specified arguments for simulation."""
    # Apply incoming args to update options.json
    for group in parser._action_groups:
        # Skip default Argparser groups
        if group.title == "positional arguments" or group.title == "options":
            continue

        # Select args pertaining to group
        print(f'ARG GROUP: {group.title}')
        group_arg_keys = [a.dest for a in group._group_actions]
        options_to_update = {k: v for k, v in args_for_config.items() if k in group_arg_keys}

        # Apply new settings to JSON options of relevant module
        options[group.title.value].update(options_to_update)

    return options

def run_standalone_module(single_module_key, input_file, options):
    """Sequence to run through a single module."""
    module = module_factory.create_single_instance(single_module_key)
    module.config.read_options(options[single_module_key.value])
    module.handler.parse_file(input_file)
    module.handler.run_module()
    output = module.output.get_result()
    print(output)

def log_options(options):
    """Show current JSON options configuration."""
    print(get_config_file())
    print(options)

def read_settings_cli(options, args):
    """Process arguments for global program settings."""
    if args.show_options:
        log_options(options)
        return
    if args.options_file:
        set_config_file(args.options_file)
        return

def read_modules_cli(parser, args, options):
    """Process arguments for simulation with SPACE modules."""
    args_for_config = {k: v for k, v in args.__dict__.items() if v is not None and v is not False}
    args_for_config.pop("command")
    print(args_for_config)

    # Check whether user wishes to run a module standalone
    single_module_key = check_standalone_config(parser, args_for_config)
    print(single_module_key)

    updated_options = update_options_with_args(parser, args_for_config, options)
    write_options_file(updated_options)

    # Get input file
    input_file = args.input_file

    # Input file ready to be used by entry module
    return single_module_key, input_file


if __name__ == "__main__":
    options = read_options_file()
    parser = setup_parser()

    # Only get arg keys with specified values
    args = parser.parse_args()

    subparsers_action = parser._subparsers._group_actions[0]
    
    if args.command == 'settings':
        read_settings_cli(options, args)
    else:
        parser = subparsers_action.choices[args.command]
        single_module_key, input_file = read_modules_cli(parser, args, options)

        # Check which module was selected
        if single_module_key is not None:
            # Run standalone module
            run_standalone_module(single_module_key, input_file, options)
        else:

            # --- Simulation Workflows --- #

            if args.command == 'flomps':
                # Extract custom parameters if provided
                timesteps = getattr(args, 'timesteps', None)
                custom_duration = getattr(args, 'custom_duration', None)
                
                # Pass custom parameters to FLOMPS workflow
                flomps.run(input_file, options, timesteps=timesteps, custom_duration=custom_duration)

            # Continue if statements here for additional workflows...

   