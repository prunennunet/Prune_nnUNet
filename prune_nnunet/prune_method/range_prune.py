import json
import re
from pathlib import Path

from torch.nn.utils import prune
import torch.nn as nn


class RangePruningMethod(prune.BasePruningMethod):
    """Prune weights that fall within a specified range of values.
    """
    PRUNING_TYPE = 'unstructured'  # Acts on individual weights

    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def compute_mask(self, t, default_mask):
        """Compute a mask that zeros weights in the specified range"""
        mask = default_mask.clone()
        # Create mask where True means "prune this weight"
        range_mask = ((t >= self.min_val) & (t <= self.max_val))
        # Apply the mask (setting pruned weights to 0 in the mask)
        mask[range_mask] = 0
        return mask


def prune_values_in_range(module, name, min_val, max_val):
    """Prunes weights in the specified parameter that fall within [min_val, max_val]"""
    RangePruningMethod.apply(module, name, min_val, max_val)
    return module


def apply_range_pruning_to_model(model, min_val: float, max_val: float, prune_weights: bool = True,
                               prune_bias: bool = False,
                               prune_layers: list = None):
    """
    Apply range-based pruning to specified layers in a model.

    Args:
        model: The PyTorch model to prune
        min_val: Minimum value of the range to prune
        max_val: Maximum value of the range to prune
        prune_weights: Whether to prune the weight parameters (default: True)
        prune_bias: Whether to also prune the bias parameters (default: False)
        prune_layers: List of layer specifications to prune. Can include:
                     - Exact layer names as strings (e.g., 'encoder.layer1')
                     - Layer types as strings (e.g., 'Conv2d', 'Linear')
                     - Generic categories as strings (e.g., 'conv', 'norm', 'linear')
                     If None, all layers with weights will be pruned (default: None)

    Returns:
        The pruned model
    """
    # Define layer type mappings for generic categories
    layer_type_mappings = {
        'conv': [nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d],
        'norm': [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.InstanceNorm1d,
                 nn.InstanceNorm2d, nn.InstanceNorm3d, nn.GroupNorm],
        'linear': [nn.Linear]
    }

    # Convert layer types from strings to actual classes if needed
    expanded_layer_types = []
    if prune_layers is not None:
        for layer_spec in prune_layers:
            # If it's a generic category, add all corresponding layer types
            if isinstance(layer_spec, str) and layer_spec.lower() in layer_type_mappings:
                expanded_layer_types.extend(layer_type_mappings[layer_spec.lower()])
            else:
                expanded_layer_types.append(layer_spec)

    pruned_count = 0
    total_count = 0

    for name, module in model.named_modules():
        # Check if this module should be pruned based on the prune_layers list
        should_prune = prune_layers is None  # Prune all if no specific layers provided

        if prune_layers is not None:
            # Check if module name matches any of the specified layer names
            for layer_spec in prune_layers:
                if isinstance(layer_spec, str):
                    # Check for exact name match or name contains the specification
                    # if layer_spec == name or (layer_spec in name and '.' not in layer_spec):
                    #     should_prune = True
                    #     print("=" * 60)
                    #     print(f"Pruning {name} based on layer name")
                    #     print("=" * 60)
                    #     break
                    if layer_spec == name:
                        should_prune = True
                        print("=" * 60)
                        print(f"Pruning {name} based on layer name")
                        print("=" * 60)
                        break

            # Check if module type matches any of the expanded layer types
            if not should_prune:
                for layer_type in expanded_layer_types:
                    if isinstance(layer_type, type) and isinstance(module, layer_type):
                        should_prune = True
                        break

        # Only process the module if it should be pruned based on layer criteria
        if should_prune:
            # Prune weights if specified and if the module has weights
            if prune_weights and hasattr(module, 'weight') and module.weight is not None:
                # Count weights before pruning
                original_weight = module.weight.data.clone()
                total_layer = original_weight.numel()
                total_count += total_layer

                # Apply pruning to weights
                prune_values_in_range(module, 'weight', min_val, max_val)

                # Count pruned weights
                if hasattr(module, 'weight_mask'):
                    pruned_layer = (module.weight_mask == 0).sum().item()
                    pruned_count += pruned_layer
                    print(f"{name}.weight: pruned {pruned_layer}/{total_layer} weights ({pruned_layer / total_layer:.2%})")

            # Prune bias if specified and if the module has bias
            if prune_bias and hasattr(module, 'bias') and module.bias is not None:
                original_bias = module.bias.data.clone()
                total_bias = original_bias.numel()
                total_count += total_bias

                # Apply pruning to bias
                prune_values_in_range(module, 'bias', min_val, max_val)

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


def process_range_pruning_results(root_dir):
    """Process all directories and extract required information."""
    results = []
    root_dir_path = Path(root_dir)

    # Walk through pruning methods first (e.g., RangePruning)
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
            min_val = ""
            max_val = ""
            prune_bias = None
            prune_layers = ""
            prune_weights = None

            # Extract parameters from each part
            for part in config_parts:
                if part.startswith("min_val_"):
                    min_val = part.split("min_val_")[1]
                elif part.startswith("max_val_"):
                    max_val = part.split("max_val_")[1]
                elif part.startswith("prune_bias_"):
                    prune_bias_value = part.split("prune_bias_")[1]
                    prune_bias = prune_bias_value.lower() == "true"
                elif part.startswith("prune_layers_"):
                    layers_str = part.split("prune_layers_")[1]
                    prune_layers = layers_str.split('_')  # Split into a list of layer types
                elif part.startswith("prune_weights_"):
                    prune_weights_value = part.split("prune_weights_")[1]
                    prune_weights = prune_weights_value.lower() == "true"

            # Walk through each fold
            for fold_dir in config_dir.glob('fold_*'):
                if not fold_dir.is_dir():
                    continue

                # fold_num = int(fold_dir.name.split('_')[1])
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
                        'min_val': min_val,
                        'max_val': max_val,
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


def save_range_pruning_results(results, json_file, csv_file):
    """Save results to JSON and CSV files."""
    # Save as JSON
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved JSON results to {json_file}")

    # Save as CSV
    with open(csv_file, 'w') as f:
        # Write header
        header = ['fold', 'prune_method', 'min_val', 'max_val', 'prune_bias', 'prune_layers', 'prune_weights', 'model_type']

        # Get all performance keys
        performance_keys = set()
        for result in results:
            performance_keys.update(result.get('performance', {}).keys())

        for key in sorted(performance_keys):
            header.append(f'performance_{key}')

        # Add pruning stats
        header.extend([
            'weights_zeros', 'weights_total', 'weights_percentage',
            'biases_zeros', 'biases_total', 'biases_percentage',
            'total_zeros', 'total_total', 'total_percentage'
        ])

        f.write(','.join(header) + '\n')

        # Write data rows
        for result in results:
            # Convert prune_layers list to a string if it's a list
            prune_layers_value = result.get('prune_layers', '')
            if isinstance(prune_layers_value, list):
                prune_layers_value = '_'.join(prune_layers_value)

            row = [
                str(result['fold']),
                result['prune_method'],
                result['min_val'],
                result['max_val'],
                str(result['prune_bias']),
                str(prune_layers_value),
                str(result.get('prune_weights', '')),
                result.get('model_type', '')  # New field to distinguish between final_model and best_model
            ]

            # Add performance values
            for key in sorted(performance_keys):
                row.append(str(result.get('performance', {}).get(key, '')))

            # Add pruning stats
            pruning_stats = result.get('pruning_stats', {})
            row.extend([
                str(pruning_stats.get('weights_zeros', '')),
                str(pruning_stats.get('weights_total', '')),
                str(pruning_stats.get('weights_percentage', '')),
                str(pruning_stats.get('biases_zeros', '')),
                str(pruning_stats.get('biases_total', '')),
                str(pruning_stats.get('biases_percentage', '')),
                str(pruning_stats.get('total_zeros', '')),
                str(pruning_stats.get('total_total', '')),
                str(pruning_stats.get('total_percentage', ''))
            ])

            f.write(','.join(row) + '\n')

    print(f"Saved CSV results to {csv_file}")