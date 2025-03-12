import sys
from pprint import pprint

from helper.config_manager import ConfigManager
from helper.read_config import read_config
from prune_nnunet.prune_eval.eval_after_prune import eval_after_prune_pred
from prune_nnunet.prune_predict.predict_after_prune import predict_after_prune
from helper.execute_cmd import execute_command


def predict_eval(config_file, eval_flag):
    pred_after_prune_dirs = predict_after_prune(config_file)
    if eval_flag:
        for pred_after_prune_dir in pred_after_prune_dirs:
            cmd = eval_after_prune_pred(config_file, pred_after_prune_dir)
            execute_command(cmd)
    return


if __name__ == '__main__':
    if len(sys.argv) <= 2:
        print("Usage: python3 -m prune_nnunet.prune_pred_eval prune_nnunet/config.yaml -p -e")
        sys.exit(1)

    # First argument after script name should be the config file
    config_path = sys.argv[1]
    # Initialize config manager and create backup
    config_manager = ConfigManager(config_path)
    config_manager.backup()
    config_file = read_config(config_path)
    pprint(config_file)

    # Check for flags in the remaining arguments
    predict_flag = '-p' in sys.argv[2:]
    eval_flag = '-e' in sys.argv[2:]

    if not predict_flag:
        sys.exit("At least -p must be specified")

    if config_file.get('prune') is None:
        sys.exit("Error: 'prune' must be specified in the config file.")

    # Call functions based on flags
    predict_eval(config_file, eval_flag)

    config_manager.update_config(config_file)
    print("Workflow completed. Original config file was backed up at:", config_manager.backup_path)