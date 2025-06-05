import copy
import os
import sys

import torch
from torch.nn.utils import prune

from helper.data_format import format_scientific
from helper.load_model import load_predictor_from_folder
from prune_nnunet.prune_analysis.verify_prune import verify_pruning_model, count_zero_parameters_model, \
    analyze_pruning_masks_model, write_pruning_mask_analysis, write_zero_parameter_analysis
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
    print(f"Pruning for folds: {folds}")

    # Prune & Predict with the model trained in each fold
    pred_dirs = []
    for fold in folds:
        # For "analyze" stats
        analyze_weight_pruned_sum = 0
        analyze_weight_total_sum = 0

        analyze_bias_pruned_sum = 0
        analyze_bias_total_sum = 0

        analyze_total_pruned_sum = 0
        analyze_total_params_sum = 0

        # For "zero" stats
        zero_weight_pruned_sum = 0
        zero_weight_total_sum = 0

        zero_bias_pruned_sum = 0
        zero_bias_total_sum = 0

        zero_total_pruned_sum = 0
        zero_total_params_sum = 0

        # Load the predictor and get the model weights
        fold_tuple = tuple(fold)
        print(f"We are loading for folds: {fold_tuple}")
        predictor = load_predictor_from_folder(prune_config['model_folder'], fold_tuple, prune_config['checkpoint_name'])

        # Find the function to the corresponding pruning method
        prune_func = get_pruning_function(prune_config['prune_method'])
        prune_params = prune_config.get('prune_parameters', {})

        output_folder = create_output_path(prune_config, fold)
        pred_dirs.append(output_folder)

        for single_fold in range(len(fold)):
            print(f"Pruning fold {fold[single_fold]} model")
            # model = predictor.network
            model = copy.deepcopy(predictor.network) 
            model.load_state_dict(predictor.list_of_parameters[single_fold])
            prune_func(model, **prune_params)
            torch.save(model.state_dict(), os.path.join(output_folder, f"pruned_with_mask_fold_{fold[single_fold]}.pth"))

            w_stats, b_stats, t_stats = analyze_pruning_masks_model(
                model, output_folder, single_fold
            )
            # Unpack
            w_pruned, w_total, w_prop = w_stats
            b_pruned, b_total, b_prop = b_stats
            t_pruned, t_total, t_prop = t_stats

            analyze_weight_pruned_sum += w_pruned
            analyze_weight_total_sum += w_total
            analyze_bias_pruned_sum += b_pruned
            analyze_bias_total_sum += b_total
            analyze_total_pruned_sum += t_pruned
            analyze_total_params_sum += t_total

            for name, module in model.named_modules():
                if hasattr(module, 'weight_orig'):
                    print(f"Removing pruning parameterization for {module}")
                    prune.remove(module, 'weight')
                if hasattr(module, 'bias_orig'):
                    prune.remove(module, 'bias')
                    print(f"Removing pruning parameterization for {module}")
            # _ = verify_pruning_model(model)

            torch.save(model.state_dict(), os.path.join(output_folder, f"pruned_without_mask_fold_{fold[single_fold]}.pth"))

            w_stats_z, b_stats_z, t_stats_z = count_zero_parameters_model(
                model, output_folder, single_fold
            )
            # Unpack
            w_pruned_z, w_total_z, w_prop_z = w_stats_z
            b_pruned_z, b_total_z, b_prop_z = b_stats_z
            t_pruned_z, t_total_z, t_prop_z = t_stats_z

            zero_weight_pruned_sum += w_pruned_z
            zero_weight_total_sum += w_total_z
            zero_bias_pruned_sum += b_pruned_z
            zero_bias_total_sum += b_total_z
            zero_total_pruned_sum += t_pruned_z
            zero_total_params_sum += t_total_z

            # Update the list_of_parameters with the pruned weights
            predictor.list_of_parameters[single_fold] = model.state_dict()

        write_pruning_mask_analysis(
            analyze_weight_pruned_sum,
            analyze_weight_total_sum,
            analyze_bias_pruned_sum,
            analyze_bias_total_sum,
            analyze_total_pruned_sum,
            analyze_total_params_sum,
            output_folder
        )

        write_zero_parameter_analysis(
            zero_weight_pruned_sum,
            zero_weight_total_sum,
            zero_bias_pruned_sum,
            zero_bias_total_sum,
            zero_total_pruned_sum,
            zero_total_params_sum,
            output_folder
        )

        # Perform prediction with the modified model
        if prune_config['return_intermediates']:
            predictor.set_return_intermediates_true()

        predictor.predict_from_files(
            prune_config['input_folder'],
            output_folder,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=3,
            num_processes_segmentation_export=3,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0,
        )

    return pred_dirs


# def predict_after_prune(config: dict):
#     # Check whether all required parameters are provided correctly
#     prune_config = config['prune']
#     if 'model_folder' not in prune_config:
#         print("Error: 'model_folder' is required in the config file for pruning.")
#         sys.exit(1)
#     if 'checkpoint_name' not in prune_config:
#         print("Error: 'checkpoint_name' is required in the config file for pruning.")
#         sys.exit(1)
#     folds = process_fold_config(prune_config.get('fold'))
#     print(f"Pruning for folds: {folds}")
#
#     # Prune & Predict with the model trained in each fold
#     pred_dirs = []
#     for fold in folds:
#         # Load the predictor and get the model weights
#         fold_tuple = tuple(fold)
#         predictor = load_predictor_from_folder(prune_config['model_folder'], fold_tuple, prune_config['checkpoint_name'])
#         model = predictor.network
#
#         # Find the function to the corresponding pruning method
#         prune_func = get_pruning_function(prune_config['prune_method'])
#         prune_params = prune_config.get('prune_parameters', {})
#         prune_func(model, **prune_params)
#
#         # ================== Start the prediction here =====================
#         output_folder = create_output_path(prune_config, fold)
#         pred_dirs.append(output_folder)
#
#         torch.save(model.state_dict(), os.path.join(output_folder, "pruned_model_with_masks.pth"))
#
#         weight_stats, bias_stats, total_stats = analyze_pruning_masks_model(model, output_folder)
#
#         for name, module in model.named_modules():
#             if hasattr(module, 'weight_orig'):
#                 print(f"Removing pruning parameterization for {module}")
#                 prune.remove(module, 'weight')
#             if hasattr(module, 'bias_orig'):
#                 prune.remove(module, 'bias')
#                 print(f"Removing pruning parameterization for {module}")
#         _ = verify_pruning_model(model)
#
#         torch.save(model.state_dict(), os.path.join(output_folder, "pruned_model_standard.pth"))
#
#         # Update the list_of_parameters with the pruned weights
#         if len(fold) == 1:
#             for i in range(len(fold)):
#                 predictor.list_of_parameters[i] = model.state_dict()
#                 print(f"Updated predictor.list_of_parameters with pruned weights for fold {fold[i]}")
#         else:
#             '''
#             TODO: When multiple folds are used, weights for different folds are stored in model.list_of_parameters,
#             this is not yet adapted, to be done.
#             '''
#             print("Error: Multiple folds are not yet supported for prediction after pruning.")
#             sys.exit(1)
#
#         # Perform prediction with the modified model
#         predictor.predict_from_files(
#             prune_config['input_folder'],
#             output_folder,
#             save_probabilities=False,
#             overwrite=True,
#             num_processes_preprocessing=3,
#             num_processes_segmentation_export=3,
#             folder_with_segs_from_prev_stage=None,
#             num_parts=1,
#             part_id=0,
#         )
#
#         weight_stats, bias_stats, total_stats = count_zero_parameters_model(model, output_folder)
#
#     return pred_dirs
