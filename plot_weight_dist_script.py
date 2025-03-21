from pprint import pprint
import sys

from helper.config_manager import ConfigManager
from helper.read_config import read_config
from model_analysis.plot_weight_distribution import load_model_weights, analyze_network_layers, \
    analyze_parameter_distributions


def main(config: dict):
    plot_config = config.get('plot', {})

    folds = plot_config.get('fold', 0)
    if not isinstance(folds, list): # handle folds as list
        folds = [folds] # may also contain just one int/str in the list
    # Check whether all values in fold meet requirements
    for fold in folds:
        try:
            fold_int = int(fold)
            if fold_int < 0 or fold_int > 4:
                print(f"Warning: Fold {fold} is out of the expected range (0-4)")
        except ValueError:
            if fold != 'all':
                raise ValueError(f"Fold {fold} is neither a valid integer nor 'all'")

    # Get required values from loaded config
    dataset_name_or_id = plot_config.get('dataset_name_or_id', 27)
    base_dir = plot_config.get('base_dir', 'nnUNet_results')
    output_dir = plot_config.get('output_dir', 'parameter_distributions')
    trainer_plan_version = plot_config.get('trainer_plan_version', 'nnUNetTrainer__nnUNetPlans__2d')
    checkpoint_name = plot_config.get('checkpoint_name', 'checkpoint_final.pth')

    # Print configuration
    print(f"Parameter Analysis Configuration:")
    print(f"Dataset ID: {dataset_name_or_id}")
    print(f"Base Directory: {base_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Model Version: {trainer_plan_version}")
    print(f"Checkpoint: {checkpoint_name}")
    print(f"Folds: {folds}")

    # Analyze weights for each fold
    for fold in folds:
        print(f"\nAnalyzing fold {fold}")

        # Load weights
        try:
            state_dict = load_model_weights(base_dir, dataset_name_or_id, fold, trainer_plan_version, checkpoint_name)
            # TODO: The function below is kinda hardcoded
            filtered_layers = analyze_network_layers(state_dict)
            print("\nLayer counts per component:")
            for component, stages in filtered_layers.items():
                print(f"{component}: {len(stages)} stages")

            print("\nDetailed layer structure:")
            pprint(filtered_layers, indent=4)

            fold_output_dir = analyze_parameter_distributions(state_dict, filtered_layers, output_dir, fold, plot_config)

            print(f"\nParameter distribution analysis complete")
            print(f"Plots have been saved to: {fold_output_dir}")

            # Print the types of plots generated
            print("\nGenerated plots:")
            if plot_config.get('analysis', {}).get('layer_wise', True):
                print("1. Layer-wise distributions:")
                print("   - Individual distributions for conv, norm, transpconv, and seg layers")
            if plot_config.get('analysis', {}).get('stage_wise', True):
                print("2. Stage-wise distributions:")
                print("   - Parameter distributions across different stages")
            if plot_config.get('analysis', {}).get('component_wise', True):
                print("3. Component-wise distributions:")
                print("   - Comparison between different layer types")

        except FileNotFoundError as e:
            print(f"Could not find weights for fold {fold}")
            expected_path = f"{base_dir}/Dataset{dataset_name_or_id:03d}_ACDC/{trainer_plan_version}/fold_{fold}/{checkpoint_name}"
            print(f"Expected path: {expected_path}")
        except Exception as e:
            print(f"Error analyzing fold {fold}: {str(e)}")
            import traceback
            traceback.print_exc()



if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 plot_weight_dist_script.py train_nnunet/config.yaml")
        sys.exit(1)

    # First argument after script name should be the config file
    config_path = sys.argv[1]
    # Initialize config manager and create backup
    config_manager = ConfigManager(config_path)
    config_manager.backup()
    config_file = config_manager.read_config()

    main(config_file)
