import sys
import os
import glob
from train_nnunet.predict.predict import process_fold_config


def get_evaluate_cmd(config):
    train_config = config['train']
    evaluate_config = config['evaluate']
    predict_config = config['predict']

    # Check for required parameters
    if 'gt_folder' not in evaluate_config:
        print("Error: 'gt_folder' is required in the evaluate config")
        sys.exit(1)

    # For prediction folder, use the one from the prediction config if available
    prediction_folder = evaluate_config.get('pred_folder', None)
    # If prediction_folder is not specified in the config, derive it from evaluate_config
    if prediction_folder is None and 'output_folder' in predict_config:
        prediction_folder = predict_config['output_folder']
    if prediction_folder is None:
        print("Error: Neither pred_folder nor output_folder specified in evaluate and predict config")
        sys.exit(1)

    folds = process_fold_config(predict_config.get('fold'))

    print(f"Evaluating on folds: {folds}")

    cmds = []
    for fold in folds:
        cmds.append(build_eval_cmd(evaluate_config, predict_config, train_config, fold,
                                   prediction_folder))
    return cmds


def get_pred_folder(eval_config, pred_config, folder, fold):
    if len(fold) == 1:
        # Prediction results are in a subdirectory for single fold
        fold_subdir = os.path.join(folder, f"fold_{fold[0]}")
        if os.path.exists(fold_subdir):
            prediction_folder = fold_subdir
            print(f"Using predictions from single fold directory: {prediction_folder}")
        else:
            print(f"Error: Could not find predictions for fold {fold[0]} in {folder}")
            sys.exit(1)
    else:
        # Create a subfolder for the ensemble of folds
        ensemble_name = "ensemble_" + "_".join(str(f) for f in fold)
        ensemble_subdir = os.path.join(folder, ensemble_name)
        if os.path.exists(ensemble_subdir):
            prediction_folder = ensemble_subdir
            print(f"Using predictions from ensemble of folds: {prediction_folder}")
        else:
            print(f"Error: Could not find predictions for ensemble of folds {fold} in {folder}")
            sys.exit(1)

    if eval_config.get('checkpoint_name', None) is None:
        checkpoint_name = pred_config.get('checkpoint_name', None)
        if checkpoint_name is None:
            print("Error: checkpoint_name not specified in evaluate or predict config")
            sys.exit(1)
        if 'checkpoint_best' in checkpoint_name:
            model_type_dir = 'best_model'
        else:  # Default to 'final_model' for 'checkpoint_final' or any other checkpoint
            model_type_dir = 'final_model'
        # Create and update the output path to include the model type subdirectory
        model_subdir = os.path.join(prediction_folder, model_type_dir)
        if os.path.exists(model_subdir):
            prediction_folder = model_subdir
            print(f"Using checkpoint {checkpoint_name}, saving results to {prediction_folder}")
        else:
            print(f"Error: Could not find checkpoint {checkpoint_name} in {prediction_folder}")
            sys.exit(1)

    return prediction_folder


def get_model_folder(eval_config, predict_config, train_config):
    # Get dataset_id from training config
    dataset_id = None
    if 'dataset_name_or_id' in train_config:
        dataset_id = train_config['dataset_name_or_id']

    # If model_folder is specified directly, use it
    model_folder = predict_config.get('model_folder')

    if not model_folder and dataset_id:
        # Find dataset folder based on dataset_id
        dataset_folder = find_dataset_folder(dataset_id, base_dir='nnUNet_results')
        if dataset_folder:
            # Get trainer, plans, and configuration from config or use defaults
            trainer = train_config.get('tr', 'FlexibleTrainerV1')
            plans = train_config.get('p', 'nnUNetPlans')
            configuration = train_config.get('configuration', '2d')

            # Find model folder
            model_folder = find_model_folder(dataset_folder, trainer, plans, configuration)

    if not model_folder:
        print(f"Error: Could not find model folder.")
        sys.exit(1)

    print(f"Using model folder: {model_folder}")

    return model_folder


def build_eval_cmd(eval_config, predict_config, train_config, fold, prediction_folder):
    """Build the nnUNetv2_evaluate command."""
    cmd = ['nnUNetv2_evaluate_folder']

    # Add required arguments
    cmd.append(eval_config['gt_folder'])

    prediction_folder = get_pred_folder(eval_config, predict_config, prediction_folder, fold)
    cmd.append(prediction_folder)

    model_folder = get_model_folder(eval_config, predict_config, train_config)
    dataset_json = os.path.join(model_folder, "dataset.json")
    if not os.path.exists(dataset_json):
        print(f"Error: dataset.json not found at {dataset_json}")
        sys.exit(1)
    cmd.extend(['-djfile', dataset_json])

    plans_json = os.path.join(model_folder, "plans.json")
    if not os.path.exists(plans_json):
        print(f"Error: plans.json not found at {plans_json}")
        sys.exit(1)
    cmd.extend(['-pfile', plans_json])

    # Set output file if provided
    output_file = eval_config.get('output_file')
    if output_file is not None:
        cmd.extend(['-o', output_file])

    # Get number of processes
    num_processes = eval_config.get('num_processes')
    if num_processes is not None:
        cmd.extend(['-np', str(num_processes)])

    # Get chill option
    chill = eval_config.get('chill', False)
    if chill:
        cmd.append('--chill')

    return cmd


def find_dataset_folder(dataset_id, base_dir="nnUNet_results"):
    """Find folder containing the given dataset ID."""
    # If nnUNet_results is an environment variable, use it
    if "nnUNet_results" in os.environ:
        base_dir = os.environ["nnUNet_results"]

    # Ensure the base directory exists
    full_base_dir = os.path.abspath(base_dir)
    if not os.path.exists(full_base_dir):
        print(f"Error: Base directory {full_base_dir} does not exist")
        return None

    # Format the dataset ID with leading zeros to ensure it's 3 digits
    dataset_id_str = f"{int(dataset_id):03d}"

    # Look for folders matching Dataset{dataset_id}*
    pattern = os.path.join(full_base_dir, f"Dataset{dataset_id_str}*")
    matching_dirs = glob.glob(pattern)

    if not matching_dirs:
        print(f"Error: No directory matching pattern {pattern} found")
        return None

    # Return the first matching directory
    print(f"Found dataset directory: {matching_dirs[0]}")
    return matching_dirs[0]


def find_model_folder(dataset_folder, trainer="FlexibleTrainerV1", plans="nnUNetPlans", configuration="2d"):
    """Find the model folder within the dataset folder."""
    if not dataset_folder or not os.path.exists(dataset_folder):
        return None

    # Try exact match first
    pattern = os.path.join(dataset_folder, f"{trainer}__{plans}__{configuration}")
    if os.path.exists(pattern):
        return pattern

    # If exact match not found, try a more flexible pattern
    pattern = os.path.join(dataset_folder, f"{trainer}*{plans}*{configuration}")
    matching_dirs = glob.glob(pattern)

    if not matching_dirs:
        print(f"Error: No model directory found in {dataset_folder}")
        return None

    # Return the first matching directory
    print(f"Found model directory: {matching_dirs[0]}")
    return matching_dirs[0]
