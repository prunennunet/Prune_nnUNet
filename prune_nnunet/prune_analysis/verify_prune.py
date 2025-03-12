import os



def count_zero_parameters(model, output_path):
    """
    Count the number of zero-valued weights and biases in a model and their proportions.
    Save the results to a text file at output_path.

    Args:
        model: PyTorch model
        output_path: Path to save the output text file

    Returns:
        A tuple of (weight_stats, bias_stats, total_stats) each containing:
        (num_zeros, total_params, proportion_zeros)
    """
    output_path = os.path.join(output_path, "zero_parameter_analysis.txt")

    weight_zeros = 0
    weight_total = 0
    bias_zeros = 0
    bias_total = 0

    # Prepare the output file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create a list to store all output lines (for both console and file)
    output_lines = []

    # Function to both print and store a line
    def log_line(line):
        print(line)
        output_lines.append(line + "\n")

    # Iterate through all modules that have parameters
    for name, module in model.named_modules():
        # Handle weights
        if hasattr(module, 'weight') and module.weight is not None:
            weight = module.weight.data
            # Count zeros in weights
            w_zeros = (weight == 0).sum().item()
            w_total = weight.numel()

            weight_zeros += w_zeros
            weight_total += w_total

            if w_zeros > 0:
                log_line(f"{name}.weight: {w_zeros}/{w_total} zeros ({w_zeros / w_total:.2%})")

        # Handle biases
        if hasattr(module, 'bias') and module.bias is not None:
            bias = module.bias.data
            # Count zeros in biases
            b_zeros = (bias == 0).sum().item()
            b_total = bias.numel()

            bias_zeros += b_zeros
            bias_total += b_total

            if b_zeros > 0:
                log_line(f"{name}.bias: {b_zeros}/{b_total} zeros ({b_zeros / b_total:.2%})")

    # Calculate totals
    total_zeros = weight_zeros + bias_zeros
    total_params = weight_total + bias_total
    weight_proportion = weight_zeros / weight_total if weight_total > 0 else 0
    bias_proportion = bias_zeros / bias_total if bias_total > 0 else 0
    total_proportion = total_zeros / total_params if total_params > 0 else 0

    # Print and log summary
    log_line("\nSUMMARY:")
    log_line(f"Weights: {weight_zeros:,}/{weight_total:,} zeros ({weight_proportion:.2%})")
    log_line(f"Biases:  {bias_zeros:,}/{bias_total:,} zeros ({bias_proportion:.2%})")
    log_line(f"Total:   {total_zeros:,}/{total_params:,} zeros ({total_proportion:.2%})")

    # Save all output to the file
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


def verify_pruning(model):
    """
    Verify that pruning was successfully applied to the model.
    Returns statistics about pruning for all layer types with weights or biases.
    """
    pruned_layers = 0
    total_zeros_weights = 0
    total_weights = 0
    total_zeros_biases = 0
    total_biases = 0

    # Check each module for pruning masks
    for name, module in model.named_modules():
        has_pruned_params = False

        # Check weights
        if hasattr(module, 'weight') and hasattr(module, 'weight_mask'):
            has_pruned_params = True
            effective_weights = module.weight_orig * module.weight_mask
            zeros_weights = (effective_weights == 0).sum().item()
            total_weights_layer = effective_weights.numel()
            total_zeros_weights += zeros_weights
            total_weights += total_weights_layer
            weight_sparsity = 100.0 * zeros_weights / total_weights_layer
            print(
                f"Layer {name} weights: {zeros_weights}/{total_weights_layer} pruned ({weight_sparsity:.2f}% sparsity)")

        # Check biases
        if hasattr(module, 'bias') and module.bias is not None and hasattr(module, 'bias_mask'):
            has_pruned_params = True
            effective_biases = module.bias_orig * module.bias_mask
            zeros_biases = (effective_biases == 0).sum().item()
            total_biases_layer = effective_biases.numel()
            total_zeros_biases += zeros_biases
            total_biases += total_biases_layer
            bias_sparsity = 100.0 * zeros_biases / total_biases_layer
            print(f"Layer {name} biases: {zeros_biases}/{total_biases_layer} pruned ({bias_sparsity:.2f}% sparsity)")

        if has_pruned_params:
            pruned_layers += 1

    # Print overall statistics
    print("\n----- Overall Pruning Statistics -----")

    if total_weights > 0:
        overall_weight_sparsity = 100.0 * total_zeros_weights / total_weights
        print(f"Weights: {total_zeros_weights}/{total_weights} pruned ({overall_weight_sparsity:.2f}% sparsity)")
    else:
        print("No pruned weights found.")

    if total_biases > 0:
        overall_bias_sparsity = 100.0 * total_zeros_biases / total_biases
        print(f"Biases: {total_zeros_biases}/{total_biases} pruned ({overall_bias_sparsity:.2f}% sparsity)")
    else:
        print("No pruned biases found.")

    total_params = total_weights + total_biases
    total_zeros = total_zeros_weights + total_zeros_biases

    if total_params > 0:
        overall_sparsity = 100.0 * total_zeros / total_params
        print(f"Combined: {total_zeros}/{total_params} parameters pruned ({overall_sparsity:.2f}% sparsity)")
        print(f"Number of pruned layers: {pruned_layers}")

        if pruned_layers == 0:
            print("WARNING: No pruned layers found! Pruning may not have been applied.")
            return False
        return True
    else:
        print("No pruned parameters found in the model.")
        return False