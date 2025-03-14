import sys
from pprint import pprint

import os

from helper.config_manager import ConfigManager

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_performance_vs_pruning(csv_file, output_dir="plots"):
    """
    Generate plots showing performance_Dice vs total pruning percentage grouped by
    fold, prune_bias, prune_weights, and prune_layers.

    Args:
        csv_file: Path to the CSV file containing the analysis results
        output_dir: Directory to save the plots
    """
    # Create base output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the CSV file
    print(f"Reading data from {csv_file}")
    df = pd.read_csv(csv_file)

    # Print the first few rows to debug
    print("First few rows of the data:")
    print(df.head())

    # Print column names to confirm they exist
    print("Columns in the dataframe:", df.columns.tolist())

    # Check for missing data
    print("Missing values in key columns:")
    print(df[['fold', 'prune_method', 'prune_bias', 'prune_weights', 'prune_layers', 'performance_Dice',
              'total_percentage']].isna().sum())

    # Filter out data where prune_bias and prune_weights are both False
    df = df[~((df['prune_bias'] == False) & (df['prune_weights'] == False))]
    print(f"After filtering, {len(df)} data points remain")

    # Convert total_percentage to numeric, handling errors
    df['total_percentage'] = pd.to_numeric(df['total_percentage'], errors='coerce')

    # Check the range of total_percentage to determine if it needs scaling
    max_percentage = df['total_percentage'].max()
    print(f"Maximum total_percentage value: {max_percentage}")

    # If the percentage is already in 0-100 range, divide by 100
    if max_percentage > 1.0:
        print("Scaling total_percentage from 0-100 to 0-1 range")
        df['total_percentage'] = df['total_percentage'] / 100.0

    # Convert performance_Dice to numeric
    df['performance_Dice'] = pd.to_numeric(df['performance_Dice'], errors='coerce')

    # Get unique folds
    folds = df['fold'].unique()
    print(f"Found {len(folds)} unique folds: {sorted(folds)}")

    # Process each fold
    for fold in sorted(folds):
        print(f"Processing fold {fold}")

        # Filter data for this fold
        fold_data = df[df['fold'] == fold].copy()
        print(f"Found {len(fold_data)} data points for fold {fold}")

        if len(fold_data) == 0:
            print(f"No data for fold {fold}, skipping")
            continue

        # Create fold directory
        fold_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        # Group by prune_bias, prune_weights, and prune_layers
        groups = fold_data.groupby(['prune_bias', 'prune_weights', 'prune_layers'])

        # Process each group
        for (bias, weights, layers), group_data in groups:
            print(f"Processing group: prune_bias={bias}, prune_weights={weights}, prune_layers={layers}")
            print(f"Found {len(group_data)} data points in this group")

            # Create group directory with more readable names
            bias_str = str(bias).lower()
            weights_str = str(weights).lower()
            # Use the actual value for prune_layers
            layers_str = str(layers).replace("-", "_")  # Replace hyphens with underscores for path safety
            group_dir_name = f"bias_{bias_str}_weights_{weights_str}_layers_{layers_str}"
            group_dir = os.path.join(fold_dir, group_dir_name)
            os.makedirs(group_dir, exist_ok=True)

            # Create a new figure
            plt.figure(figsize=(10, 8))

            # Define colors and markers for different prune methods
            prune_methods = group_data['prune_method'].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(prune_methods)))
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']

            # For each prune method, create a line
            legend_handles = []
            legend_labels = []

            for i, method in enumerate(prune_methods):
                method_data = group_data[group_data['prune_method'] == method]

                # Extract numeric values from min_val for sorting
                def extract_number(val):
                    """Safely extract numeric value from threshold strings"""
                    if pd.isna(val):
                        return 0

                    # If it's already a number, return its absolute value
                    if isinstance(val, (int, float)):
                        return abs(float(val))

                    # Convert to string and handle various formats
                    val_str = str(val)

                    # Extract digits using regex
                    import re
                    numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', val_str)
                    if numbers:
                        try:
                            # Return the absolute value of the first number found
                            return abs(float(numbers[0]))
                        except:
                            return 0
                    return 0

                # Create a sorting key
                method_data['threshold_value'] = method_data['min_val'].apply(extract_number)

                # Sort first by total_percentage, then by threshold_value for points with same percentage
                method_data = method_data.sort_values(by=['total_percentage', 'threshold_value'])

                if len(method_data) == 0:
                    continue

                print(f"  - Plotting {len(method_data)} points for method={method}")

                # Plot the line
                color = colors[i % len(colors)]
                marker = markers[i % len(markers)]

                line, = plt.plot(
                    method_data['total_percentage'],
                    method_data['performance_Dice'],
                    marker=marker,
                    linestyle='-',
                    color=color,
                    linewidth=2,
                    markersize=8,
                    label=method
                )

                legend_handles.append(line)
                legend_labels.append(method)

                # Add annotations for min_val and max_val
                for _, row in method_data.iterrows():
                    min_val = row['min_val']
                    max_val = row['max_val']

                    # Format values for display
                    if isinstance(min_val, str) and 'e' in min_val:
                        min_val = min_val.replace('e-', 'e-')
                    if isinstance(max_val, str) and 'e' in max_val:
                        max_val = max_val.replace('e-', 'e-')

                    label = f"{min_val},{max_val}"
                    plt.annotate(
                        label,
                        (row['total_percentage'], row['performance_Dice']),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center',
                        fontsize=8
                    )

            # Set plot title and labels
            title = f'Fold {fold}: Dice Performance vs Pruning Percentage\n'
            title += f'Bias Pruning: {bias}, Weight Pruning: {weights}, Layers: {layers}'
            plt.title(title, fontsize=14)
            plt.xlabel('Total Pruning Percentage', fontsize=12)
            plt.ylabel('Dice Performance', fontsize=12)

            # Dynamically set x-axis limits based on data
            x_min = 0.0
            x_max = group_data['total_percentage'].max()

            # If all points are clustered near zero, zoom in
            if x_max < 0.1:
                # Add some padding (20% more than the max value)
                x_max = min(x_max * 1.2, 0.1)
                plt.xlim([x_min, x_max])
                # Adjust tick spacing based on the range
                x_ticks = np.linspace(x_min, x_max, 6)
                plt.xticks(x_ticks)
            else:
                # If points are spread out, use the default [0, 1] range
                plt.xlim([0.0, 1.0])
                plt.xticks(np.arange(0, 1.1, 0.1))

            # Set y-axis limits
            y_min = max(0.0, group_data['performance_Dice'].min() - 0.05)
            y_max = min(1.0, group_data['performance_Dice'].max() + 0.05)

            # If y range is very small, expand it to show differences better
            if y_max - y_min < 0.2:
                y_center = (y_min + y_max) / 2
                y_min = max(0.0, y_center - 0.1)
                y_max = min(1.0, y_center + 0.1)

            plt.ylim([y_min, y_max])
            plt.yticks(np.linspace(y_min, y_max, 6))

            # Add grid
            plt.grid(True, linestyle='--', alpha=0.7)

            # Add legend
            if legend_handles:
                plt.legend(legend_handles, legend_labels, loc='lower left')

            # Save the plot
            plot_filename = f"performance_vs_pruning.png"
            output_file = os.path.join(group_dir, plot_filename)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {output_file}")

            # Close the figure
            plt.close()

    print(f"All plots saved to {output_dir}")


def create_summary_plots(csv_file, output_dir="plots"):
    """
    Create summary plots for each unique combination of prune_bias, prune_weights, and prune_layers
    showing the average performance across all folds.

    Args:
        csv_file: Path to the CSV file containing the analysis results
        output_dir: Directory to save the plots
    """
    # Create summary directory
    summary_dir = os.path.join(output_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Filter out data where prune_bias and prune_weights are both False
    df = df[~((df['prune_bias'] == False) & (df['prune_weights'] == False))]

    # Convert total_percentage to numeric, handling errors
    df['total_percentage'] = pd.to_numeric(df['total_percentage'], errors='coerce')
    if df['total_percentage'].max() > 1.0:
        df['total_percentage'] = df['total_percentage'] / 100.0

    # Convert performance_Dice to numeric
    df['performance_Dice'] = pd.to_numeric(df['performance_Dice'], errors='coerce')

    # Group by prune_bias, prune_weights, and prune_layers
    groups = df.groupby(['prune_bias', 'prune_weights', 'prune_layers'])

    # Process each group
    for (bias, weights, layers), group_data in groups:
        print(f"Creating summary for: prune_bias={bias}, prune_weights={weights}, prune_layers={layers}")

        # Group by prune_method, min_val, max_val to get averages across folds
        summary = group_data.groupby(['prune_method', 'min_val', 'max_val']).agg({
            'total_percentage': 'mean',
            'performance_Dice': 'mean'
        }).reset_index()

        # Create a new figure
        plt.figure(figsize=(12, 10))

        # Define colors for different prune methods
        prune_methods = summary['prune_method'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(prune_methods)))
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']

        # For each prune method, create a line
        legend_handles = []
        legend_labels = []

        for i, method in enumerate(prune_methods):
            method_data = summary[summary['prune_method'] == method]

            # Extract numeric values from min_val for sorting
            def extract_number(val):
                """Safely extract numeric value from threshold strings"""
                if pd.isna(val):
                    return 0

                # If it's already a number, return its absolute value
                if isinstance(val, (int, float)):
                    return abs(float(val))

                # Convert to string and handle various formats
                val_str = str(val)

                # Extract digits using regex
                import re
                numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', val_str)
                if numbers:
                    try:
                        # Return the absolute value of the first number found
                        return abs(float(numbers[0]))
                    except:
                        return 0
                return 0

            # Create a sorting key
            method_data['threshold_value'] = method_data['min_val'].apply(extract_number)

            # Sort first by total_percentage, then by threshold_value for points with same percentage
            method_data = method_data.sort_values(by=['total_percentage', 'threshold_value'])

            if len(method_data) == 0:
                continue

            # Plot the line
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]

            line, = plt.plot(
                method_data['total_percentage'],
                method_data['performance_Dice'],
                marker=marker,
                linestyle='-',
                color=color,
                linewidth=2,
                markersize=8,
                label=method
            )

            legend_handles.append(line)
            legend_labels.append(method)

            # Add annotations for min_val and max_val
            for _, row in method_data.iterrows():
                min_val = row['min_val']
                max_val = row['max_val']

                # Format values for display
                if isinstance(min_val, str) and 'e' in min_val:
                    min_val = min_val.replace('e-', 'e-')
                if isinstance(max_val, str) and 'e' in max_val:
                    max_val = max_val.replace('e-', 'e-')

                label = f"{min_val},{max_val}"
                plt.annotate(
                    label,
                    (row['total_percentage'], row['performance_Dice']),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center',
                    fontsize=8
                )

        # Set plot title and labels
        title = f'Average Performance vs Pruning Percentage Across All Folds\n'
        title += f'Bias Pruning: {bias}, Weight Pruning: {weights}, Layers: {layers}'
        plt.title(title, fontsize=14)
        plt.xlabel('Total Pruning Percentage', fontsize=12)
        plt.ylabel('Dice Performance', fontsize=12)

        # Dynamically set x-axis limits based on data
        x_min = 0.0
        x_max = summary['total_percentage'].max()

        # If all points are clustered near zero, zoom in
        if x_max < 0.1:
            # Add some padding (20% more than the max value)
            x_max = min(x_max * 1.2, 0.1)
            plt.xlim([x_min, x_max])
            # Adjust tick spacing based on the range
            x_ticks = np.linspace(x_min, x_max, 6)
            plt.xticks(x_ticks)
        else:
            # If points are spread out, use the default [0, 1] range
            plt.xlim([0.0, 1.0])
            plt.xticks(np.arange(0, 1.1, 0.1))

        # Set y-axis limits
        y_min = max(0.0, summary['performance_Dice'].min() - 0.05)
        y_max = min(1.0, summary['performance_Dice'].max() + 0.05)

        # If y range is very small, expand it to show differences better
        if y_max - y_min < 0.2:
            y_center = (y_min + y_max) / 2
            y_min = max(0.0, y_center - 0.1)
            y_max = min(1.0, y_center + 0.1)

        plt.ylim([y_min, y_max])
        plt.yticks(np.linspace(y_min, y_max, 6))

        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)

        # Add legend
        if legend_handles:
            plt.legend(legend_handles, legend_labels, loc='lower left')

        # Save the plot
        bias_str = str(bias).lower()
        weights_str = str(weights).lower()
        layers_str = str(layers).replace("-", "_")
        group_name = f"bias_{bias_str}_weights_{weights_str}_layers_{layers_str}"
        output_file = os.path.join(summary_dir, f"summary_{group_name}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved summary plot to {output_file}")

        # Close the figure
        plt.close()

    print(f"All summary plots saved to {summary_dir}")


if __name__ == "__main__":

    if len(sys.argv) <= 1:
        print("Usage: python3 -m prune_nnunet.prune_analysis.plot_prune_results_temp prune_nnunet/config.yaml")
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

    csv_file = os.path.join(base_dir, prune_config.get('prune_method', ''), "pruning_analysis_results.csv")
    if not os.path.exists(csv_file):
        print(f"Error: pruning_analysis_results.csv file not found in {base_dir}")
        sys.exit(1)

    plot_output_dir = os.path.join(base_dir, prune_config.get('prune_method', ''), "analysis_plots")

    # Create individual fold plots
    plot_performance_vs_pruning(csv_file, plot_output_dir)

    # Create summary plots
    create_summary_plots(csv_file, plot_output_dir)

    print("Visualization complete!")