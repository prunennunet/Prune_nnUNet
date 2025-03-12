from pprint import pprint

import yaml



def read_config(yaml_file):
    """Read configuration from YAML file."""
    try:
        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Config file '{yaml_file}' not found")
    except yaml.YAMLError as e:
        raise e


def get_prune_config(config):
    if 'prune' not in config:
        raise ValueError("Error: 'prune' section not found in config file")
    prune_config = config['prune']

    model_folder = prune_config.get('model_folder', None)
    fold = prune_config.get('fold', None)
    checkpoint_name = prune_config.get('checkpoint_name', None)

    if model_folder is None or fold is None or checkpoint_name is None:
        raise ValueError("Error: 'model_folder', 'fold' and 'checkpoint_name' are required in the 'prune' section")

    if isinstance(fold, int):
        # If fold is a single integer, convert it to a tuple
        fold = (fold,)
        prune_config['fold_tuple'] = fold
    elif isinstance(fold, (list, tuple)):
        # If fold is already a container, verify it has exactly one integer
        if len(fold) != 1 or not isinstance(fold[0], int):
            raise ValueError("Error: 'fold' must contain exactly one integer")
    else:
        # If fold is neither an int nor a container
        raise ValueError("Error: 'fold' must be an integer or a container with one integer")

    return prune_config


