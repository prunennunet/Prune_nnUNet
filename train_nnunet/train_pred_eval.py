import subprocess
import sys

from helper.execute_cmd import execute_command
from helper.read_config import read_config
from train_nnunet.predict.predict import get_predict_cmd
from train_nnunet.train.train import get_train_cmd
from train_nnunet.evaluation.eval import get_evaluate_cmd
from helper.config_manager import ConfigManager


def train(config: dict):
    cmds = get_train_cmd(config)
    for cmd in cmds:
        execute_command(cmd)
    return


def predict(config: dict):
    cmds = get_predict_cmd(config)
    for cmd in cmds:
        execute_command(cmd)
    return


def evaluate(config: dict):
    cmds = get_evaluate_cmd(config)
    for cmd in cmds:
        execute_command(cmd)
    return


if __name__ == '__main__':
    if len(sys.argv) <= 2:
        print("Usage: python3 -m train_nnunet.train_pred_eval train_nnunet/config.yaml -t -p -e")
        sys.exit(1)

    # First argument after script name should be the config file
    config_path = sys.argv[1]
    # Initialize config manager and create backup
    config_manager = ConfigManager(config_path)
    config_manager.backup()
    config_file = read_config(config_path)

    # Check for flags in the remaining arguments
    train_flag = '-t' in sys.argv[2:]
    predict_flag = '-p' in sys.argv[2:]
    eval_flag = '-e' in sys.argv[2:]

    if not (train_flag or predict_flag or eval_flag):
        sys.exit("At least one of -t, -p, or -e must be specified")

    if config_file.get('train') is None or config_file.get('predict') is None or config_file.get('evaluate') is None:
        sys.exit("Error: 'train', 'predict', and 'evaluate' must all be specified in the config file.")

    # Call functions based on flags
    if train_flag:
        train(config_file)

    if predict_flag:
        predict(config_file)

    if eval_flag:
        evaluate(config_file)

    config_manager.update_config(config_file)
    print("Workflow completed. Original config file was backed up at:", config_manager.backup_path)
