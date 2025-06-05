import cmd
import os
import sys


def eval_after_prune_pred(config: dict, pred_dir: str):
    """Build the nnUNetv2_evaluate command."""
    prune_config = config['prune']

    cmd = ['nnUNetv2_evaluate_folder']
    

    # Add required arguments
    cmd.append(prune_config['gt_folder'])
    cmd.append(pred_dir)

    print(f"We are evaluating {pred_dir} with {prune_config['gt_folder']}")

    # If model_folder is specified directly, use it
    model_folder = prune_config.get('model_folder', None)
    if model_folder is None:
        print("Error: 'model_folder' is required in the config file for evaluation.")
        sys.exit(1)

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
    output_file = prune_config.get('output_file', None)
    if output_file is not None:
        cmd.extend(['-o', output_file])

    # Get number of processes
    num_processes = prune_config.get('num_processes', None)
    if num_processes is not None:
        cmd.extend(['-np', str(num_processes)])

    # Get chill option
    chill = prune_config.get('chill', False)
    if chill:
        cmd.append('--chill')

    return cmd
