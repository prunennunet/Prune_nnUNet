import os
import sys

import torch


def analyze_pruning_masks_model(model, output_path):
    """
    Analyze pruning masks from a PyTorch model, counting unique masks to avoid redundancy.
    Saves analysis to a text file at the provided output path.

    Args:
        model: PyTorch model containing pruning masks as buffers
        output_path: Directory path to save pruning mask analysis results

    Returns:
        A tuple containing:
            weight_stats: (weight_pruned, weight_total, weight_proportion)
            bias_stats: (bias_pruned, bias_total, bias_proportion)
            total_stats: (total_pruned, total_params, overall_proportion)
    """
    output_path = os.path.join(output_path, "pruning_mask_analysis.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    seen_buffer = set()
    seen_params = set()
    weight_total, weight_pruned = 0, 0
    bias_total, bias_pruned = 0, 0
    output_lines = []

    def log_line(line):
        print(line)
        output_lines.append(line + "\n")

    # First count the pruning masks
    for name, buffer in model.named_buffers():
        if 'weight_mask' in name or 'bias_mask' in name:
            storage_id = id(buffer.untyped_storage())
            if storage_id not in seen_buffer:
                seen_buffer.add(storage_id)

                num_pruned = (buffer == 0).sum().item()
                num_params = buffer.numel()

                if "weight_mask" in name:
                    # weight_total += num_params
                    weight_pruned += num_pruned
                    param_type = "weight"
                elif "bias_mask" in name:
                    # bias_total += num_params
                    bias_pruned += num_pruned
                    param_type = "bias"
                else:
                    sys.exit(f"There are no buffer for {name}")
                    # param_type = "unknown"

                sparsity = num_pruned / num_params
                log_line(f"{name} ({param_type}): {num_pruned}/{num_params} pruned ({sparsity:.2%})")
            else:
                log_line(f"{name}: Skipped redundant storage.")

    # Then count the parameters in the model
    for name, tensor in model.named_parameters():
        storage_id = id(tensor.untyped_storage())
        if storage_id not in seen_params:
            seen_params.add(storage_id)

            num_params = tensor.numel()

            if "weight" in name:
                weight_total += num_params
            elif "bias" in name:
                bias_total += num_params
            else:
                sys.exit(f"There are no tensor for {name}")
        else:
            log_line(f"{name}: Skipped redundant storage.")

    total_pruned = weight_pruned + bias_pruned
    total_params = weight_total + bias_total

    weight_proportion = weight_pruned / weight_total if weight_total else 0
    bias_proportion = bias_pruned / bias_total if bias_total else 0
    overall_sparsity = total_pruned / total_params if total_params else 0

    log_line("\nSUMMARY:")
    log_line(f"Weights: {weight_pruned:,}/{weight_total:,} pruned ({weight_proportion:.2%})")
    log_line(f"Biases: {bias_pruned:,}/{bias_total:,} pruned ({bias_proportion:.2%})")
    log_line(f"Total: {total_pruned:,}/{total_params:,} pruned ({overall_sparsity:.2%})")

    try:
        with open(output_path, 'w') as f:
            f.writelines(output_lines)
        print(f"\nPruning mask analysis saved to: {output_path}")
    except Exception as e:
        print(f"\nError saving pruning mask analysis to file: {e}")

    weight_stats = (weight_pruned, weight_total, weight_proportion)
    bias_stats = (bias_pruned, bias_total, bias_proportion)
    total_stats = (total_pruned, total_params, overall_sparsity)

    return weight_stats, bias_stats, total_stats


# def analyze_pruning_masks(model, output_path):
#     """
#     Analyze pruning masks in a model by counting parameters marked for pruning.
#     Save the results to a text file at output_path.
#
#     Args:
#         model: PyTorch model with pruning masks
#         output_path: Path to save the output text file
#
#     Returns:
#         A tuple of (weight_stats, bias_stats, total_stats) each containing:
#         (num_pruned, total_params, proportion_pruned)
#     """
#     output_path = os.path.join(output_path, "pruning_mask_analysis.txt")
#
#     # Count for pruned parameters (via masks)
#     weight_pruned = 0
#     bias_pruned = 0
#
#     # Count for all parameters in the model
#     weight_total = 0
#     bias_total = 0
#
#     # Prepare the output file
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#
#     # Create a list to store all output lines (for both console and file)
#     output_lines = []
#
#     # Function to both print and store a line
#     def log_line(line):
#         print(line)
#         output_lines.append(line + "\n")
#
#     # First, count all parameters in the model
#     for name, module in model.named_modules():
#         # Count all weights
#         if hasattr(module, 'weight') and module.weight is not None:
#             w_total = module.weight.numel()
#             weight_total += w_total
#
#         # Count all biases
#         if hasattr(module, 'bias') and module.bias is not None:
#             b_total = module.bias.numel()
#             bias_total += b_total
#
#     # Now, analyze pruning masks
#     for name, module in model.named_modules():
#         # Handle weight masks
#         if hasattr(module, 'weight_mask') and module.weight_mask is not None:
#             weight_mask = module.weight_mask.data
#             # Count zeros in weight masks (pruned weights)
#             w_pruned = (weight_mask == 0).sum().item()
#             w_mask_total = weight_mask.numel()
#             weight_pruned += w_pruned
#             if w_pruned > 0:
#                 log_line(f"{name}.weight_mask: {w_pruned}/{w_mask_total} pruned ({w_pruned / w_mask_total:.2%})")
#
#         # Handle bias masks
#         if hasattr(module, 'bias_mask') and module.bias_mask is not None:
#             bias_mask = module.bias_mask.data
#             # Count zeros in bias masks (pruned biases)
#             b_pruned = (bias_mask == 0).sum().item()
#             b_mask_total = bias_mask.numel()
#             bias_pruned += b_pruned
#             if b_pruned > 0:
#                 log_line(f"{name}.bias_mask: {b_pruned}/{b_mask_total} pruned ({b_pruned / b_mask_total:.2%})")
#
#     # Calculate totals and proportions based on all parameters
#     total_pruned = weight_pruned + bias_pruned
#     total_params = weight_total + bias_total
#     weight_proportion = weight_pruned / weight_total if weight_total > 0 else 0
#     bias_proportion = bias_pruned / bias_total if bias_total > 0 else 0
#     total_proportion = total_pruned / total_params if total_params > 0 else 0
#
#     # Print and log summary
#     log_line("\nSUMMARY:")
#     log_line(f"Weights: {weight_pruned:,}/{weight_total:,} pruned ({weight_proportion:.2%})")
#     log_line(f"Biases:  {bias_pruned:,}/{bias_total:,} pruned ({bias_proportion:.2%})")
#     log_line(f"Total:   {total_pruned:,}/{total_params:,} pruned ({total_proportion:.2%})")
#
#     # Save all output to the file
#     try:
#         with open(output_path, 'w') as f:
#             f.writelines(output_lines)
#         print(f"\nPruning mask analysis saved to: {output_path}")
#     except Exception as e:
#         print(f"\nError saving pruning mask analysis to file: {e}")
#
#     weight_stats = (weight_pruned, weight_total, weight_proportion)
#     bias_stats = (bias_pruned, bias_total, bias_proportion)
#     total_stats = (total_pruned, total_params, total_proportion)
#
#     return weight_stats, bias_stats, total_stats


def count_zero_parameters_model(model, output_path):
    """
    Count zero-valued parameters in a state_dict (OrderedDict), ensuring unique parameter counting.
    Saves the analysis to a text file at the provided output path.

    Args:
        state_dict: OrderedDict containing model parameters
        output_path: Directory path to save the zero-parameter analysis results

    Returns:
        A tuple containing:
            weight_stats: (weight_zeros, weight_total, weight_proportion)
            bias_stats: (bias_zeros, bias_total, bias_proportion)
            total_stats: (total_zeros, total_params, total_proportion)
    """
    output_path = os.path.join(output_path, "zero_parameter_analysis.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    seen_storages = set()
    weight_total, weight_zeros = 0, 0
    bias_total, bias_zeros = 0, 0
    output_lines = []

    def log_line(line):
        print(line)
        output_lines.append(line + "\n")

    print(type(model))

    for name, tensor in model.named_parameters():
        storage_id = id(tensor.untyped_storage())

        if storage_id not in seen_storages:
            seen_storages.add(storage_id)

            zeros = (tensor == 0).sum().item()
            total = tensor.numel()

            # Handle weights
            if "weight" in name:
                weight_zeros += zeros
                weight_total += total
                param_type = "weight"
            # Handle biases
            if "bias" in name:
                bias_zeros += zeros
                bias_total += total
                param_type = "bias"
            if zeros > 0:
                log_line(f"{name} ({param_type}): {zeros}/{total} zeros ({zeros / total:.2%})")

        else:
            log_line(f"{name}: Skipped redundant storage.")

    total_zeros = weight_zeros + bias_zeros
    total_params = weight_total + bias_total

    weight_proportion = weight_zeros / weight_total if weight_total else 0
    bias_proportion = bias_zeros / bias_total if bias_total else 0
    total_proportion = total_zeros / total_params if total_params else 0

    log_line("\nSUMMARY:")
    log_line(f"Weights: {weight_zeros:,}/{weight_total:,} zeros ({weight_proportion:.2%})")
    log_line(f"Biases: {bias_zeros:,}/{bias_total:,} zeros ({bias_proportion:.2%})")
    log_line(f"Total: {total_zeros:,}/{total_params:,} zeros ({total_proportion:.2%})")

    try:
        with open(output_path, 'w') as f:
            f.writelines(output_lines)
        print(f"\nZero parameter analysis saved to: {output_path}")
    except Exception as e:
        print(f"\nError saving zero parameter analysis to file: {e}")

    weight_stats = (weight_zeros, weight_total, weight_proportion)
    bias_stats = (bias_zeros, bias_total, bias_proportion)
    total_stats = (total_zeros, total_params, total_proportion)

    return weight_stats, bias_stats, total_stats


# def count_zero_parameters(model, output_path):
#     """
#     Count the number of zero-valued weights and biases in a model and their proportions.
#     Save the results to a text file at output_path.
#
#     Args:
#         model: PyTorch model
#         output_path: Path to save the output text file
#
#     Returns:
#         A tuple of (weight_stats, bias_stats, total_stats) each containing:
#         (num_zeros, total_params, proportion_zeros)
#     """
#     output_path = os.path.join(output_path, "zero_parameter_analysis.txt")
#
#     weight_zeros = 0
#     weight_total = 0
#     bias_zeros = 0
#     bias_total = 0
#
#     # Prepare the output file
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#
#     # Create a list to store all output lines (for both console and file)
#     output_lines = []
#
#     # Function to both print and store a line
#     def log_line(line):
#         print(line)
#         output_lines.append(line + "\n")
#
#     # Iterate through all modules that have parameters
#     for name, module in model.named_modules():
#         # Handle weights
#         if hasattr(module, 'weight') and module.weight is not None:
#             weight = module.weight.data
#             # Count zeros in weights
#             w_zeros = (weight == 0).sum().item()
#             w_total = weight.numel()
#
#             weight_zeros += w_zeros
#             weight_total += w_total
#
#             if w_zeros > 0:
#                 log_line(f"{name}.weight: {w_zeros}/{w_total} zeros ({w_zeros / w_total:.2%})")
#
#         # Handle biases
#         if hasattr(module, 'bias') and module.bias is not None:
#             bias = module.bias.data
#             # Count zeros in biases
#             b_zeros = (bias == 0).sum().item()
#             b_total = bias.numel()
#
#             bias_zeros += b_zeros
#             bias_total += b_total
#
#             if b_zeros > 0:
#                 log_line(f"{name}.bias: {b_zeros}/{b_total} zeros ({b_zeros / b_total:.2%})")
#
#     # Calculate totals
#     total_zeros = weight_zeros + bias_zeros
#     total_params = weight_total + bias_total
#     weight_proportion = weight_zeros / weight_total if weight_total > 0 else 0
#     bias_proportion = bias_zeros / bias_total if bias_total > 0 else 0
#     total_proportion = total_zeros / total_params if total_params > 0 else 0
#
#     # Print and log summary
#     log_line("\nSUMMARY:")
#     log_line(f"Weights: {weight_zeros:,}/{weight_total:,} zeros ({weight_proportion:.2%})")
#     log_line(f"Biases:  {bias_zeros:,}/{bias_total:,} zeros ({bias_proportion:.2%})")
#     log_line(f"Total:   {total_zeros:,}/{total_params:,} zeros ({total_proportion:.2%})")
#
#     # Save all output to the file
#     try:
#         with open(output_path, 'w') as f:
#             f.writelines(output_lines)
#         print(f"\nZero parameter analysis saved to: {output_path}")
#     except Exception as e:
#         print(f"\nError saving zero parameter analysis to file: {e}")
#
#     weight_stats = (weight_zeros, weight_total, weight_proportion)
#     bias_stats = (bias_zeros, bias_total, bias_proportion)
#     total_stats = (total_zeros, total_params, total_proportion)
#
#     return weight_stats, bias_stats, total_stats


def verify_pruning_model(model):
    """
    Verify pruning statistics directly from a state_dict (OrderedDict).
    Counts unique parameters without redundant counting.
    """
    seen_storages = set()
    total_params = 0
    total_zeros = 0

    print(type(model))

    for name, tensor in model.named_parameters():
        storage_id = id(tensor.untyped_storage())

        if storage_id not in seen_storages:
            seen_storages.add(storage_id)

            num_zeros = torch.sum(tensor == 0).item()
            num_params = tensor.numel()

            sparsity = 100.0 * num_zeros / num_params
            print(f"{name}: {num_zeros}/{num_params} pruned ({sparsity:.2f}% sparsity)")

            total_params += num_params
            total_zeros += num_zeros
        else:
            print(f"{name}: Skipped redundant storage.")

    print("\n----- Overall Pruning Statistics (Unique Params) -----")
    if total_params > 0:
        overall_sparsity = 100.0 * total_zeros / total_params
        print(f"Combined: {total_zeros}/{total_params} parameters pruned ({overall_sparsity:.2f}% sparsity)")
        return True
    else:
        print("No parameters found in the provided state_dict.")
        return False


# def verify_pruning(model):
#     """
#     Verify that pruning was successfully applied to the model.
#     Returns statistics about pruning for all layer types with weights or biases.
#     """
#     pruned_layers = 0
#     total_zeros_weights = 0
#     total_weights = 0
#     total_zeros_biases = 0
#     total_biases = 0
#
#     # Check each module for pruning masks
#     for name, module in model.named_modules():
#         has_pruned_params = False
#
#         # Check weights
#         if hasattr(module, 'weight') and hasattr(module, 'weight_mask'):
#             has_pruned_params = True
#             effective_weights = module.weight_orig * module.weight_mask
#             zeros_weights = (effective_weights == 0).sum().item()
#             total_weights_layer = effective_weights.numel()
#             total_zeros_weights += zeros_weights
#             total_weights += total_weights_layer
#             weight_sparsity = 100.0 * zeros_weights / total_weights_layer
#             print(
#                 f"Layer {name} weights: {zeros_weights}/{total_weights_layer} pruned ({weight_sparsity:.2f}% sparsity)")
#
#         # Check biases
#         if hasattr(module, 'bias') and module.bias is not None and hasattr(module, 'bias_mask'):
#             has_pruned_params = True
#             effective_biases = module.bias_orig * module.bias_mask
#             zeros_biases = (effective_biases == 0).sum().item()
#             total_biases_layer = effective_biases.numel()
#             total_zeros_biases += zeros_biases
#             total_biases += total_biases_layer
#             bias_sparsity = 100.0 * zeros_biases / total_biases_layer
#             print(f"Layer {name} biases: {zeros_biases}/{total_biases_layer} pruned ({bias_sparsity:.2f}% sparsity)")
#
#         if has_pruned_params:
#             pruned_layers += 1
#
#     # Print overall statistics
#     print("\n----- Overall Pruning Statistics -----")
#
#     if total_weights > 0:
#         overall_weight_sparsity = 100.0 * total_zeros_weights / total_weights
#         print(f"Weights: {total_zeros_weights}/{total_weights} pruned ({overall_weight_sparsity:.2f}% sparsity)")
#     else:
#         print("No pruned weights found.")
#
#     if total_biases > 0:
#         overall_bias_sparsity = 100.0 * total_zeros_biases / total_biases
#         print(f"Biases: {total_zeros_biases}/{total_biases} pruned ({overall_bias_sparsity:.2f}% sparsity)")
#     else:
#         print("No pruned biases found.")
#
#     total_params = total_weights + total_biases
#     total_zeros = total_zeros_weights + total_zeros_biases
#
#     if total_params > 0:
#         overall_sparsity = 100.0 * total_zeros / total_params
#         print(f"Combined: {total_zeros}/{total_params} parameters pruned ({overall_sparsity:.2f}% sparsity)")
#         print(f"Number of pruned layers: {pruned_layers}")
#
#         if pruned_layers == 0:
#             print("WARNING: No pruned layers found! Pruning may not have been applied.")
#             return False
#         return True
#     else:
#         print("No pruned parameters found in the model.")
#         return False
