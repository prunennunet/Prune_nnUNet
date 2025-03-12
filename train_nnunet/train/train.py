import sys


def get_train_cmd(config):
    train_config = config['train']

    # Check for required parameters
    if 'dataset_name_or_id' not in train_config:
        print("Error: 'dataset_name_or_id' is required in the config file")
        sys.exit(1)

    if 'configuration' not in train_config:
        print("Error: 'configuration' is required in the config file")
        sys.exit(1)

    folds = train_config.get('fold')
    if not isinstance(folds, list): # handle folds as list
        folds = [folds] # may also contain just one int/str in the list
    for fold in folds:
        try:
            fold_int = int(fold)
            if fold_int < 0 or fold_int > 4:
                print(f"Warning: Fold {fold} is out of the expected range (0-4)")
        except ValueError:
            if fold != 'all':
                raise ValueError(f"Fold {fold} is neither a valid integer nor 'all'")

    cmds = []
    for fold in folds:
        cmds.append(build_train_cmd(train_config, fold))
    return cmds


def build_train_cmd(config, fold):
    """Build the command to execute based on configuration with specific fold."""
    # Required arguments
    cmd = ['nnUNetv2_train']
    cmd.append(str(config['dataset_name_or_id']))
    cmd.append(str(config['configuration']))
    cmd.append(str(fold))

    # Optional arguments with values
    if 'tr' in config and str(config['tr']) != 'nnUNetTrainer':
        cmd.extend(['-tr', str(config['tr'])])

    if 'p' in config and str(config['p']) != 'nnUNetPlans':
        cmd.extend(['-p', str(config['p'])])

    if 'pretrained_weights' in config and config['pretrained_weights'] is not None:
        cmd.extend(['-pretrained_weights', str(config['pretrained_weights'])])

    if 'num_gpus' in config and str(config['num_gpus']) != '1':
        cmd.extend(['-num_gpus', str(config['num_gpus'])])

    if 'device' in config and str(config['device']) != 'cuda':
        cmd.extend(['-device', str(config['device'])])

    # Boolean flags (no values)
    if config.get('npz', False):
        cmd.append('--npz')

    if config.get('c', False):
        cmd.append('--c')

    if config.get('val', False):
        cmd.append('--val')

    if config.get('val_best', False):
        cmd.append('--val_best')

    if config.get('disable_checkpointing', False):
        cmd.append('--disable_checkpointing')
    return cmd
