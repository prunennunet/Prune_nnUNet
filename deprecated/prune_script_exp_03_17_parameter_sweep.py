import yaml
import subprocess
import copy
from itertools import product
import os
import logging
from pathlib import Path

from helper.execute_cmd import execute_command

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("parameter_sweep.log"),
        logging.StreamHandler()
    ]
)


def modify_and_run_config(config_path):
    """Modify the config file with different parameter combinations and run the command."""
    config_path = Path(config_path)

    # Load the config
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Create a backup of the original config
    backup_path = config_path.with_suffix(config_path.suffix + '.backup')
    with open(backup_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    logging.info(f"Created backup of original config at {backup_path}")

    # Define the parameter ranges
    # For abs_values, we want [0, 0.001, 0.002, ..., 0.009, 0.01, 0.02, ..., 0.09, 0.01]
    # abs_values = []
    # for exp in [-3, -2, -1]:
    #     for i in range(1, 10):
    #         abs_values.append(i * 10 ** exp)
    # abs_values = [0] + abs_values  # Add 0 at the beginning
    # abs_values = [0.01, 0.02]
    abs_values = [i / 1000 for i in range(0, 21)]
    # print(abs_values)

    prune_bias_values = [True]
    prune_weights_values = [True]
    prune_layers_values = [['conv']]  # None for null
    exclude_prune_layers_values = [[None]]


    # Generate all combinations but filter out invalid ones where both prune_bias and prune_weights are False
    raw_combinations = list(product(abs_values, prune_bias_values, prune_weights_values, prune_layers_values, exclude_prune_layers_values))
    combinations = [combo for combo in raw_combinations if not (combo[1] is False and combo[2] is False)]

    total_combinations = len(combinations)
    logging.info(f"Starting parameter sweep with {total_combinations} combinations")

    # For each combination
    for idx, (abs_val, prune_bias, prune_weights, prune_layers, exclude_prune_layers) in enumerate(combinations):
        # Log the current combination
        logging.info(f"Running combination {idx + 1}/{total_combinations}:")
        logging.info(f"  abs_val = {abs_val}")
        logging.info(f"  prune_bias = {prune_bias}")
        logging.info(f"  prune_weights = {prune_weights}")
        logging.info(f"  prune_layers = {prune_layers}")
        logging.info(f"  exclude_prune_layers = {exclude_prune_layers}")

        # Make a deep copy of the original config
        modified_config = copy.deepcopy(config)

        # Modify the parameters
        modified_config['prune']['prune_parameters']['max_val'] = abs_val
        modified_config['prune']['prune_parameters']['min_val'] = -abs_val
        modified_config['prune']['prune_parameters']['prune_bias'] = prune_bias
        modified_config['prune']['prune_parameters']['prune_weights'] = prune_weights
        modified_config['prune']['prune_parameters']['prune_layers'] = prune_layers
        modified_config['prune']['prune_parameters']['prune_exclude_layers'] = exclude_prune_layers

        # Save the modified config
        with open(config_path, 'w') as file:
            yaml.dump(modified_config, file, default_flow_style=False)

        # Execute the command
        cmd = ["python3", "-m", "prune_nnunet.prune_pred_eval", str(config_path), "-p", "-e"]
        execute_command(cmd)

        logging.info(f"Finished combination {idx + 1}/{total_combinations}")

    # Restore the original config
    backup_path.replace(config_path)
    logging.info(f"Restored original config from {backup_path}")
    logging.info("Parameter sweep completed")


if __name__ == "__main__":
    config_path = "prune_nnunet/config.yaml"

    try:
        modify_and_run_config(config_path)
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        # Try to restore the backup if it exists
        backup_path = Path(config_path).with_suffix(Path(config_path).suffix + '.backup')
        if backup_path.exists():
            backup_path.replace(Path(config_path))
            logging.info(f"Restored original config from {backup_path}")