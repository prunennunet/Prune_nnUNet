import sys
import os


def process_fold_config(folds):
    # Ensure folds is a list of lists
    if not isinstance(folds, list):
        folds = [[folds]]  # Convert single value directly to a list of lists
    else:
        # Check if folds is already a list of lists
        has_list = any(isinstance(item, list) for item in folds)
        if not has_list:
            folds = [folds]  # Wrap the entire list in another list
        else:
            # Handle mixed case where some elements are lists and others are not
            folds = [item if isinstance(item, list) else [item] for item in folds]

    # Now validate each fold value
    for fold_group in folds:
        for fold in fold_group:
            try:
                fold_int = int(fold)
                if fold_int < 0 or fold_int > 4:
                    print(f"Warning: Fold {fold} is out of the expected range (0-4)")
            except ValueError:
                if fold != 'all':
                    raise ValueError(f"Fold {fold} is neither a valid integer nor 'all'")
    return folds


def get_predict_cmd(config):
    train_config = config['train']
    predict_config = config['predict']

    # Check for required parameters
    if 'input_folder' not in predict_config:
        print("Error: 'input_folder' is required in the predict config")
        sys.exit(1)

    if 'output_folder' not in predict_config:
        print("Error: 'output_folder' is required in the predict config")
        sys.exit(1)

    # Either model_folder or (dataset_name_or_id and configuration) must be present
    if 'model_folder' not in predict_config:
        if 'dataset_name_or_id' not in train_config:
            print("Error: Either 'model_folder' or 'dataset_name_or_id' must be specified")
            sys.exit(1)
        if 'configuration' not in train_config:
            print("Error: 'configuration' is required when using dataset_name_or_id")
            sys.exit(1)

    # Make sure output directory exists
    os.makedirs(predict_config['output_folder'], exist_ok=True)

    folds = process_fold_config(predict_config.get('fold'))

    print(f"Predicting on folds: {folds}")

    cmds = []
    for fold in folds:
        cmds.append(build_predict_cmd(predict_config, train_config, fold))
    return cmds


def create_output_path(predict_config: dict, fold: list):
    path = str(predict_config['output_folder'])
    if len(fold) == 1:
        # Create a subfolder for this specific fold
        fold_subdir = os.path.join(path, f"fold_{fold[0]}")
        os.makedirs(fold_subdir, exist_ok=True)
        print(f"Using single fold {fold[0]}, saving results to {fold_subdir}")
        # Update the output folder to the fold-specific subdirectory
        output_path = fold_subdir
    else:
        # Create a subfolder for the ensemble of folds
        ensemble_name = "ensemble_" + "_".join(str(f) for f in fold)
        ensemble_subdir = os.path.join(path, ensemble_name)
        os.makedirs(ensemble_subdir, exist_ok=True)
        print(f"Using ensemble of folds {fold}, saving results to {ensemble_subdir}")
        # Update the output folder to the ensemble subdirectory
        output_path = ensemble_subdir

    # Determine which checkpoint is being used and create appropriate subdir
    checkpoint_name = predict_config.get('checkpoint_name', 'checkpoint_final.pth')
    if 'checkpoint_best' in checkpoint_name:
        model_type_dir = 'best_model'
    else:  # Default to 'final_model' for 'checkpoint_final' or any other checkpoint
        model_type_dir = 'final_model'

    # Create and update the output path to include the model type subdirectory
    output_path = os.path.join(output_path, model_type_dir)
    os.makedirs(output_path, exist_ok=True)
    print(f"Using checkpoint {checkpoint_name}, saving results to {output_path}")
    return output_path


def build_predict_cmd(predict_config, train_config, fold):
    """Build the predict command based on configuration."""
    # Start with the base command, which depends on whether we're using model_folder or dataset approach
    if 'model_folder' in predict_config:
        cmd = ['nnUNetv2_predict_from_modelfolder']
    else:
        cmd = ['nnUNetv2_predict']

    # Create output folder based on single fold or ensemble of multiple folds
    output_folder = create_output_path(predict_config, fold)

    # Required arguments
    cmd.extend(['-i', str(predict_config['input_folder'])])
    cmd.extend(['-o', output_folder])

    # Check which style of command we need to build
    if 'model_folder' in predict_config:
        # We're using the model_folder method (predict_entry_point_modelfolder)
        cmd.extend(['-m', str(predict_config['model_folder'])])
    else:
        # We're using the dataset method (predict_entry_point)
        cmd.extend(['-d', str(train_config['dataset_name_or_id'])])

        if 'configuration' in train_config:
            cmd.extend(['-c', str(train_config['configuration'])])

        if 'tr' in train_config and train_config['tr'] != 'nnUNetTrainer':
            cmd.extend(['-tr', str(train_config['tr'])])

        if 'p' in train_config and train_config['p'] != 'nnUNetPlans':
            cmd.extend(['-p', str(train_config['p'])])

    # Add fold arguments to the command
    cmd.extend(['-f'] + [str(f) for f in fold])

    # Optional arguments with values
    if 'step_size' in predict_config and str(predict_config['step_size']) != '0.5':
        cmd.extend(['-step_size', str(predict_config['step_size'])])

    if 'checkpoint_name' in predict_config and predict_config['checkpoint_name'] != 'checkpoint_final.pth':
        cmd.extend(['-chk', str(predict_config['checkpoint_name'])])

    if 'num_processes_preprocessing' in predict_config and str(predict_config['num_processes_preprocessing']) != '3':
        cmd.extend(['-npp', str(predict_config['num_processes_preprocessing'])])

    if 'num_processes_segmentation_export' in predict_config and str(predict_config['num_processes_segmentation_export']) != '3':
        cmd.extend(['-nps', str(predict_config['num_processes_segmentation_export'])])

    if 'device' in predict_config and str(predict_config['device']) != 'cuda':
        cmd.extend(['-device', str(predict_config['device'])])

    if 'prev_stage_predictions' in predict_config and predict_config['prev_stage_predictions'] is not None:
        cmd.extend(['-prev_stage_predictions', str(predict_config['prev_stage_predictions'])])

    if 'num_parts' in predict_config and str(predict_config['num_parts']) != '1':
        cmd.extend(['-num_parts', str(predict_config['num_parts'])])

    if 'part_id' in predict_config and str(predict_config['part_id']) != '0':
        cmd.extend(['-part_id', str(predict_config['part_id'])])

    # Boolean flags
    if predict_config.get('disable_tta', False):
        cmd.append('--disable_tta')

    if predict_config.get('verbose', False):
        cmd.append('--verbose')

    if predict_config.get('save_probabilities', False):
        cmd.append('--save_probabilities')

    if predict_config.get('continue_prediction', False):
        cmd.append('--continue_prediction')

    if predict_config.get('disable_progress_bar', False):
        cmd.append('--disable_progress_bar')

    if predict_config.get('return_intermediates', False):
        cmd.append('--return_intermediates')

    return cmd