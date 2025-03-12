import torch
from torch.nn.utils import prune


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


def apply_range_pruning_to_model(model, min_val: float, max_val: float, prune_bias: bool = False, layers: list = None):
    """
    Apply range-based pruning to all layers with weights in a model.

    Args:
        model: The PyTorch model to prune
        min_val: Minimum value of the range to prune
        max_val: Maximum value of the range to prune
        prune_bias: Whether to also prune the bias parameters (default: False)

    Returns:
        The pruned model
    """
    pruned_count = 0
    total_count = 0

    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
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

        # If prune_bias is True and the module has a bias parameter, prune it too
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