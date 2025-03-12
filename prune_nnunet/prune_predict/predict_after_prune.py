import os
import sys

import torch
from torch.nn.utils import prune

from helper.data_format import format_scientific
from helper.load_model import load_predictor_from_folder
from prune_nnunet.prune_analysis.verify_prune import verify_pruning, count_zero_parameters
from prune_nnunet.prune_method.find_function import get_pruning_function
from train_nnunet.predict.predict import process_fold_config


def create_output_path(config: dict, fold: list):
    path = str(config['output_folder'])

    if 'prune_method' in config:
        path = os.path.join(path, config['prune_method'])

    if 'prune_parameters' in config and config['prune_parameters']:
        # Create a string with parameters in format "key1_value1__key2_value2"
        param_parts = []
        for key, value in config['prune_parameters'].items():
            # Format numbers in scientific notation (Xe-Y)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                # Convert value to scientific notation
                value_str = format_scientific(value)
                # Scientific notation formatting already gives the X.XXe±Y format
            elif isinstance(value, list):
                # Convert each element to string and join with hyphens
                elements = [str(item) for item in value]
                value_str = "-".join(elements)
            else:
                value_str = str(value)

            param_parts.append(f"{key}_{value_str}")
        param_dir = "__".join(param_parts)

        path = os.path.join(path, param_dir)

    if len(fold) == 1:
        # Create a subfolder for this specific fold
        fold_subdir = os.path.join(path, f"fold_{fold[0]}")
        print(f"Using single fold {fold[0]}, saving results to {fold_subdir}")
        # Update the output folder to the fold-specific subdirectory
        output_path = fold_subdir
    else:
        # Create a subfolder for the ensemble of folds
        ensemble_name = "ensemble_" + "_".join(str(f) for f in fold)
        ensemble_subdir = os.path.join(path, ensemble_name)
        print(f"Using ensemble of folds {fold}, saving results to {ensemble_subdir}")
        # Update the output folder to the ensemble subdirectory
        output_path = ensemble_subdir

    # Determine which checkpoint is being used and create appropriate subdir
    checkpoint_name = config.get('checkpoint_name', 'checkpoint_final.pth')
    if 'checkpoint_best' in checkpoint_name:
        model_type_dir = 'best_model'
    else:  # Default to 'final_model' for 'checkpoint_final' or any other checkpoint
        model_type_dir = 'final_model'

    # Create and update the output path to include the model type subdirectory
    output_path = os.path.join(output_path, model_type_dir)
    os.makedirs(output_path, exist_ok=True)
    print(f"Using checkpoint {checkpoint_name}, saving results to {output_path}")
    return output_path


def predict_after_prune(config: dict):
    # Check whether all required parameters are provided correctly
    prune_config = config['prune']
    if 'model_folder' not in prune_config:
        print("Error: 'model_folder' is required in the config file for pruning.")
        sys.exit(1)
    if 'checkpoint_name' not in prune_config:
        print("Error: 'checkpoint_name' is required in the config file for pruning.")
        sys.exit(1)
    folds = process_fold_config(prune_config.get('fold'))

    # Prune & Predict with the model trained in each fold
    pred_dirs = []
    for fold in folds:
        # Load the predictor and get the model weights
        fold_tuple = tuple(fold)
        predictor = load_predictor_from_folder(prune_config['model_folder'], fold_tuple, prune_config['checkpoint_name'])
        model = predictor.network

        # Find the function to the corresponding pruning method
        prune_func = get_pruning_function(prune_config['prune_method'])
        prune_params = prune_config.get('prune_parameters', {})
        prune_func(model, **prune_params)
        _ = verify_pruning(model)

        # ================== Start the prediction here =====================
        output_folder = create_output_path(prune_config, fold)
        pred_dirs.append(output_folder)

        torch.save(model.state_dict(), os.path.join(output_folder, "pruned_model_with_masks.pth"))

        for name, module in model.named_modules():
            if hasattr(module, 'weight_orig'):
                print(f"Removing pruning parameterization for {module}")
                prune.remove(module, 'weight')
            if hasattr(module, 'bias_orig'):
                prune.remove(module, 'bias')
                print(f"Removing pruning parameterization for {module}")

        torch.save(model.state_dict(), os.path.join(output_folder, "pruned_model_standard.pth"))

        # Update the list_of_parameters with the pruned weights
        if len(fold) == 1:
            for i in range(len(fold)):
                predictor.list_of_parameters[i] = model.state_dict()
                print(f"Updated predictor.list_of_parameters with pruned weights for fold {fold[i]}")
        else:
            '''
            TODO: When multiple folds are used, weights for different folds are stored in model.list_of_parameters,
            this is not yet adapted, to be done.
            '''
            print("Error: Multiple folds are not yet supported for prediction after pruning.")
            sys.exit(1)

        # Perform prediction with the modified model
        predictor.predict_from_files(
            prune_config['input_folder'],
            output_folder,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=8,
            num_processes_segmentation_export=8,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0,
        )

        weight_stats, bias_stats, total_stats = count_zero_parameters(predictor.network, output_folder)

    return pred_dirs
