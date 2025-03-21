from pathlib import Path

import torch
from torch.nn.utils import prune
import json
import re


class FlowPruningMethod(prune.BasePruningMethod):
    """
    Structured pruning method that selectively prunes half of the channel filters
    based on the specified data flow direction.

    This pruning method is designed for U-Net architectures with encoder and decoder flows.
    When the specified flow is 'encoder', it prunes the first half of the channel filters.
    When the specified flow is 'decoder', it prunes the second half of the channel filters.

    Attributes:
        del_flow (str): The flow direction to eliminate - either 'encoder' or 'decoder'
    """
    PRUNING_TYPE = 'structured'  # Indicates this prunes structured groups (channels) rather than individual weights

    def __init__(self, eliminate_data_flow):
        """
        Initialize the FlowPruningMethod.

        Args:
            eliminate_data_flow (str): The flow direction to eliminate - must be either
                                     'encoder' or 'decoder'
        """
        if eliminate_data_flow not in ['encoder', 'decoder', None]:
            raise ValueError(f"Invalid value for eliminate_data_flow: {eliminate_data_flow}. "
                             "Expected 'encoder', 'decoder', or None.")
        self.del_flow = eliminate_data_flow

    def compute_mask(self, t, default_mask):
        """
        Computes a binary mask for the input tensor based on the specified data flow.

        For encoder pruning: Sets the first half of channels to 0 (pruned)
        For decoder pruning: Sets the second half of channels to 0 (pruned)

        Args:
            t (torch.Tensor): The tensor to prune, with channels as the first dimension
            default_mask (torch.Tensor): The default mask with the same shape as t

        Returns:
            torch.Tensor: A binary mask where 1 indicates keeping the value and 0 indicates pruning it
        """
        # Create a mask with the same shape as the input tensor (starting with all ones)
        mask = torch.ones_like(default_mask)

        # If del_flow is None, return the mask with all ones (no pruning)
        if self.del_flow is None:
            return mask

        # Calculate the number of channels (assuming the channel dimension is the 0th dimension)
        num_channels = t.shape[0]
        half_channels = num_channels // 2

        # Determine which half to prune based on del_flow
        if self.del_flow == 'encoder':
            # Prune the first half of channels if we're eliminating encoder flow
            mask[:half_channels] = 0
        elif self.del_flow == 'decoder':
            # Prune the second half of channels if we're eliminating decoder flow
            mask[half_channels:] = 0
        else:
            raise ValueError(f"Invalid value for del_flow: {self.del_flow}. Expected 'encoder' or 'decoder'.")

        return mask


def prune_flow_values(module, name, eliminate_data_flow):
    """Prunes weights in the specified parameter that fall within [min_val, max_val]"""
    FlowPruningMethod.apply(module, name, eliminate_data_flow)
    return module


def apply_flow_pruning_to_model(model, eliminate_data_flow, prune_weights: bool = True, prune_bias: bool = False,
                                prune_layers: list = None):
    """
    Apply flow-based pruning to specifically named layers in a model.

    This function applies the FlowPruningMethod to selected layers in the model.
    For each layer, it will prune either the first half (encoder flow) or
    second half (decoder flow) of the channels based on eliminate_data_flow.

    Args:
        model: The PyTorch model to prune
        eliminate_data_flow: The flow direction to eliminate - either 'encoder' or 'decoder'
        prune_weights: Whether to prune the weight parameters (default: True)
        prune_bias: Whether to also prune the bias parameters (default: False)
        prune_layers: List of exact layer names to prune. If None, all layers will be pruned (default: None)

    Returns:
        The pruned model
    """
    pruned_count = 0
    total_count = 0

    # Early return if eliminate_data_flow is None to avoid unnecessary iteration
    if eliminate_data_flow is None:
        print("No pruning applied as eliminate_data_flow is None")
        return model

    for name, module in model.named_modules():
        # Only prune layers that are specifically named in prune_layers
        should_prune = prune_layers is None  # Prune all if no specific layers provided

        if prune_layers is not None and name in prune_layers:
            should_prune = True
            print("=" * 60)
            print(f"Pruning {name} based on layer name")
            print("=" * 60)

        # Only process the module if it matches one of the specified names
        if should_prune:
            # Prune weights if specified and if the module has weights
            if prune_weights and hasattr(module, 'weight') and module.weight is not None:
                # Count weights before pruning
                original_weight = module.weight.data.clone()
                total_layer = original_weight.numel()
                total_count += total_layer

                # Apply flow pruning to weights
                prune_flow_values(module, 'weight', eliminate_data_flow)

                # Count pruned weights
                if hasattr(module, 'weight_mask'):
                    pruned_layer = (module.weight_mask == 0).sum().item()
                    pruned_count += pruned_layer
                    print(
                        f"{name}.weight: pruned {pruned_layer}/{total_layer} weights ({pruned_layer / total_layer:.2%})")

            # Prune bias if specified and if the module has bias
            if prune_bias and hasattr(module, 'bias') and module.bias is not None:
                original_bias = module.bias.data.clone()
                total_bias = original_bias.numel()
                total_count += total_bias

                # Apply flow pruning to bias
                prune_flow_values(module, 'bias', eliminate_data_flow)

                # Count pruned bias values
                if hasattr(module, 'bias_mask'):
                    pruned_bias = (module.bias_mask == 0).sum().item()
                    pruned_count += pruned_bias
                    print(
                        f"{name}.bias: pruned {pruned_bias}/{total_bias} bias values ({pruned_bias / total_bias:.2%})")

    # Print overall statistics
    if total_count > 0:
        print(f"Total pruned: {pruned_count}/{total_count} parameters ({pruned_count / total_count:.2%})")

    return model


def process_flow_pruning_results(root_dir):
    """
    Process all directories and extract required information for flow pruning experiments.

    This function walks through the directory structure to extract performance metrics
    and pruning statistics for different flow pruning configurations.

    Args:
        root_dir: Path to the root directory containing the experiment results

    Returns:
        A list of dictionaries containing the extracted information for each experiment
    """
    results = []
    root_dir_path = Path(root_dir)

    # Walk through pruning methods first (e.g., FlowPruning)
    for pruning_method_dir in root_dir_path.iterdir():
        if not pruning_method_dir.is_dir():
            continue

        pruning_method = pruning_method_dir.name

        # Walk through each pruning configuration
        for config_dir in pruning_method_dir.iterdir():
            if not config_dir.is_dir():
                continue

            # Parse the configuration directory name
            config_name = config_dir.name
            config_parts = config_name.split('__')

            # Initialize parameters
            eliminate_data_flow = None
            prune_bias = None
            prune_layers = ""
            prune_weights = None

            # Extract parameters from each part
            for part in config_parts:
                if part.startswith("eliminate_data_flow_"):
                    flow_value = part.split("eliminate_data_flow_")[1]
                    # Convert "None" string to None
                    eliminate_data_flow = None if flow_value.lower() == "none" else flow_value
                elif part.startswith("prune_bias_"):
                    prune_bias_value = part.split("prune_bias_")[1]
                    prune_bias = prune_bias_value.lower() == "true"
                elif part.startswith("prune_layers_"):
                    layers_str = part.split("prune_layers_")[1]
                    prune_layers = layers_str.split('_')  # Split into a list of layer names
                elif part.startswith("prune_weights_"):
                    prune_weights_value = part.split("prune_weights_")[1]
                    prune_weights = prune_weights_value.lower() == "true"

            # Walk through each fold
            for fold_dir in config_dir.glob('fold_*'):
                if not fold_dir.is_dir():
                    continue

                # Parse fold number
                fold_part = fold_dir.name.split('_')[1]
                if fold_part == 'all':
                    fold_num = 'all'
                else:
                    fold_num = int(fold_part)

                # Check both final_model and best_model
                model_types = ["final_model", "best_model"]
                found_models = []

                for model_type in model_types:
                    model_dir = fold_dir / model_type
                    if model_dir.exists():
                        found_models.append((model_type, model_dir))

                if not found_models:
                    print(f"No final_model or best_model directory found in {fold_dir}")
                    continue

                # Process each found model
                for model_type, model_dir in found_models:
                    # Get performance from summary.json
                    performance = {}
                    summary_path = model_dir / 'summary.json'
                    if summary_path.exists():
                        try:
                            with open(summary_path, 'r') as f:
                                data = json.load(f)
                                performance = data.get('foreground_mean', {})
                            print(f"Extracted performance metrics from {summary_path}")
                        except Exception as e:
                            print(f"Error reading {summary_path}: {e}")
                    else:
                        print(f"Warning: {summary_path} does not exist")

                    # Get pruning stats from zero_parameter_analysis.txt
                    pruning_stats = {}
                    zero_path = model_dir / 'zero_parameter_analysis.txt'

                    if zero_path.exists():
                        try:
                            with open(zero_path, 'r') as f:
                                lines = f.readlines()

                                # Find the summary section
                                summary_index = -1
                                for i, line in enumerate(lines):
                                    if line.strip() == "SUMMARY:":
                                        summary_index = i
                                        break

                                if summary_index != -1 and summary_index + 3 < len(lines):
                                    weights_line = lines[summary_index + 1].strip()
                                    biases_line = lines[summary_index + 2].strip()
                                    total_line = lines[summary_index + 3].strip()

                                    # Extract weights percentage
                                    weights_match = re.search(r"Weights:.*\(([\d.]+)%\)", weights_line)
                                    if weights_match:
                                        pruning_stats['weights_percentage'] = float(weights_match.group(1))

                                        # Also extract the fraction
                                        fraction_match = re.search(r"Weights:\s+([\d,]+)/([\d,]+)", weights_line)
                                        if fraction_match:
                                            zeros = int(fraction_match.group(1).replace(',', ''))
                                            total = int(fraction_match.group(2).replace(',', ''))
                                            pruning_stats['weights_zeros'] = zeros
                                            pruning_stats['weights_total'] = total

                                    # Extract biases percentage
                                    biases_match = re.search(r"Biases:.*\(([\d.]+)%\)", biases_line)
                                    if biases_match:
                                        pruning_stats['biases_percentage'] = float(biases_match.group(1))

                                        # Also extract the fraction
                                        fraction_match = re.search(r"Biases:\s+([\d,]+)/([\d,]+)", biases_line)
                                        if fraction_match:
                                            zeros = int(fraction_match.group(1).replace(',', ''))
                                            total = int(fraction_match.group(2).replace(',', ''))
                                            pruning_stats['biases_zeros'] = zeros
                                            pruning_stats['biases_total'] = total

                                    # Extract total percentage
                                    total_match = re.search(r"Total:.*\(([\d.]+)%\)", total_line)
                                    if total_match:
                                        pruning_stats['total_percentage'] = float(total_match.group(1))

                                        # Also extract the fraction
                                        fraction_match = re.search(r"Total:\s+([\d,]+)/([\d,]+)", total_line)
                                        if fraction_match:
                                            zeros = int(fraction_match.group(1).replace(',', ''))
                                            total = int(fraction_match.group(2).replace(',', ''))
                                            pruning_stats['total_zeros'] = zeros
                                            pruning_stats['total_total'] = total

                            print(f"Extracted pruning stats from {zero_path}")
                        except Exception as e:
                            print(f"Error reading {zero_path}: {e}")
                    else:
                        print(f"Warning: {zero_path} does not exist")

                    # Save results for this model
                    result = {
                        'fold': fold_num,
                        'prune_method': pruning_method,
                        'eliminate_data_flow': eliminate_data_flow,
                        'prune_bias': prune_bias,
                        'prune_layers': prune_layers,
                        'prune_weights': prune_weights,
                        'model_type': model_type,  # Distinguish between final_model and best_model
                        'performance': performance,
                        'pruning_stats': pruning_stats
                    }

                    results.append(result)
                    print(f"Processed {model_type} for fold {fold_num} for {config_dir.name}")

    return results
