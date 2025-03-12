import yaml
import subprocess
import os
import itertools
import time
import re

# Parameters to sweep
folds = [0, 1, 2, 3, 4]
min_vals = [0e-0]
prune_biases = [False]

# Path to the config file
config_path = 'train_nnunet/config.yaml'


# Function to update the config file
def update_config(fold, min_val, max_val, prune_bias):
    # Load the current config
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Update the parameters
    config['prune']['fold'] = fold
    config['prune']['prune_parameters']['min_val'] = min_val
    config['prune']['prune_parameters']['max_val'] = max_val
    config['prune']['prune_parameters']['prune_bias'] = prune_bias

    # Save the updated config
    with open(config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

    return config


# Function to read the output folder from the config file
def get_output_folder_from_config():
    try:
        # Read the updated config file
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        # Get the output folder path
        if 'prune' in config and 'predict' in config['prune'] and 'output_folder' in config['prune']['predict']:
            output_folder = config['prune']['predict']['output_folder']
            if output_folder:
                print(f"Successfully read output folder from config: {output_folder}")
                return output_folder

        raise ValueError("Could not find output_folder in config file")
    except Exception as e:
        raise Exception(f"Error reading output folder from config: {e}")


# Function to run the experiment
def run_experiment(fold, min_val, max_val, prune_bias):
    # Update the config
    update_config(fold, min_val, max_val, prune_bias)

    # Run prune_test.py and capture its output
    print(f"Running prune_test.py with fold={fold}, min_val={min_val}, max_val={max_val}, prune_bias={prune_bias}")
    try:
        # First check which Python command is available
        python_cmd = 'python3' if os.system('which python3 > /dev/null') == 0 else 'python'
        print(f"Using Python command: {python_cmd}")

        result = subprocess.run([python_cmd, 'prune_test.py'], capture_output=True, text=True)

        # Check if the command was successful
        if result.returncode != 0:
            print("ERROR: prune_test.py failed with the following error:")
            print(result.stderr)
            print("\nStandard output:")
            print(result.stdout)
            raise Exception(f"prune_test.py exited with status {result.returncode}")

        # Print the output for debugging
        print("Output from prune_test.py:")
        print(result.stdout)
    except FileNotFoundError:
        print(f"ERROR: Could not find prune_test.py in the current directory. Current working directory: {os.getcwd()}")
        raise

    # Determine the output folder
    output_folder = get_output_folder_from_config()
    print(f"Detected output folder: {output_folder}")

    # Run the evaluation command
    eval_cmd = [
        'nnUNetv2_evaluate_folder',
        'nnUNet_raw/Dataset027_ACDC/labelsTs',
        output_folder,
        '-djfile',
        'nnUNet_results_prune_test/Dataset027_ACDC/FlexibleTrainerV1__nnUNetPlans__2d/dataset.json',
        '-pfile',
        'nnUNet_results_prune_test/Dataset027_ACDC/FlexibleTrainerV1__nnUNetPlans__2d/plans.json'
    ]

    print(f"Running evaluation command: {' '.join(eval_cmd)}")
    try:
        eval_result = subprocess.run(eval_cmd, capture_output=True, text=True)

        # Check if the command was successful
        if eval_result.returncode != 0:
            print("ERROR: Evaluation command failed with the following error:")
            print(eval_result.stderr)
            print("\nStandard output:")
            print(eval_result.stdout)
            raise Exception(f"Evaluation command exited with status {eval_result.returncode}")
    except FileNotFoundError:
        print(
            "ERROR: Could not find nnUNetv2_evaluate_folder. Make sure it's in your PATH or you're in the correct environment.")
        raise

    # Print the evaluation output
    print("Evaluation output:")
    print(eval_result.stdout)

    # Log the results
    log_results(fold, min_val, max_val, prune_bias, output_folder, eval_result.stdout)


# Function to log the results
def log_results(fold, min_val, max_val, prune_bias, output_folder, eval_output):
    log_file = "prune_experiments_log.txt"

    with open(log_file, 'a') as file:
        file.write(f"Experiment completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write(f"Parameters: fold={fold}, min_val={min_val}, max_val={max_val}, prune_bias={prune_bias}\n")
        file.write(f"Output folder: {output_folder}\n")

        # Try to extract evaluation metrics from the output
        # Assuming the evaluation prints metrics in a format like "Metric: value"
        metrics = re.findall(r"([A-Za-z_]+):\s*([\d.]+)", eval_output)
        if metrics:
            file.write("Evaluation metrics:\n")
            for metric, value in metrics:
                file.write(f"  {metric}: {value}\n")

        file.write("-" * 50 + "\n")


# Main function to run all experiments
def main():
    # Initialize the log file
    log_file = "prune_experiments_log.txt"

    with open(log_file, 'w') as file:
        file.write(f"Pruning experiments started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write("=" * 50 + "\n\n")

    # Create all parameter combinations
    param_combinations = []
    for fold in folds:
        for min_val in min_vals:
            max_val = abs(min_val)  # Opposite of min_val (same absolute value but positive)
            for prune_bias in prune_biases:
                param_combinations.append((fold, min_val, max_val, prune_bias))

    total_experiments = len(param_combinations)
    print(f"Starting {total_experiments} experiments")

    for i, (fold, min_val, max_val, prune_bias) in enumerate(param_combinations):
        print(f"Experiment {i + 1}/{total_experiments}")
        print(f"Parameters: fold={fold}, min_val={min_val}, max_val={max_val}, prune_bias={prune_bias}")

        try:
            run_experiment(fold, min_val, max_val, prune_bias)
            print(f"Experiment {i + 1}/{total_experiments} completed successfully")
        except Exception as e:
            print(f"Error in experiment {i + 1}/{total_experiments}: {e}")
            with open(log_file, 'a') as file:
                file.write(
                    f"ERROR in experiment: fold={fold}, min_val={min_val}, max_val={max_val}, prune_bias={prune_bias}\n")
                file.write(f"Error message: {str(e)}\n")
                file.write("-" * 50 + "\n")

        print("-" * 50)


if __name__ == "__main__":
    main()