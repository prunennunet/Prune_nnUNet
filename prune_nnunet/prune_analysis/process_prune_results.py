import os.path
import sys
from pprint import pprint

from helper.config_manager import ConfigManager
from prune_nnunet.prune_method.find_function import get_process_function, get_save_data_function


def main():
    if len(sys.argv) <= 1:
        print("Usage: python3 -m prune_nnunet.prune_analysis.process_prune_results prune_nnunet/config.yaml")
        sys.exit(1)

    # First argument after script name should be the config file
    config_path = sys.argv[1]
    # Initialize config manager and create backup
    config_manager = ConfigManager(config_path)
    config_manager.backup()
    config_file = config_manager.read_config()
    pprint(config_file)

    prune_config = config_file.get('prune', {})

    base_dir = prune_config.get('output_folder', '')
    if not os.path.exists(base_dir):
        print(f"Error: output_folder '{base_dir}' does not exist")
        sys.exit(1)
    print(f"Starting analysis of directory: {base_dir}")

    process_func = get_process_function(prune_config['prune_method'])
    print(f"Using process function: {process_func.__name__}")
    results = process_func(base_dir)
    print(results)

    json_file = os.path.join(base_dir, prune_config.get('prune_method', ''), "pruning_analysis_results.json")
    csv_file = os.path.join(base_dir, prune_config.get('prune_method', ''), "pruning_analysis_results.csv")

    save_data_func = get_save_data_function(prune_config['prune_method'])
    print(f"Using save data function: {save_data_func.__name__}")
    save_data_func(results, json_file, csv_file)

    print(f"Processed {len(results)} model directories")
    print(f"Results saved to {json_file} and {csv_file}")



if __name__ == "__main__":
    main()