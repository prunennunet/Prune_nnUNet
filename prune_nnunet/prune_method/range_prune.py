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
                    if layer_spec == name or (layer_spec in name and '.' not in layer_spec):
                        should_prune = True
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
            if prune_weights and hasattr(module, 'weight'):
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