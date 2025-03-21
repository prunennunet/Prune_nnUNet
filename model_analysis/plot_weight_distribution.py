import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from batchgenerators.utilities.file_and_folder_operations import *
import seaborn as sns
from collections import defaultdict
from pprint import pprint
import matplotlib as mpl
from scipy.stats import norm, kurtosis
import argparse
import yaml

from helper.read_config import read_config


def setup_output_dirs(base_output_dir, fold, components):
    """Create directory structure for saving plots with encoder/decoder separation"""
    base_dir = Path(base_output_dir)
    fold_dir = base_dir / f'fold_{fold}'
    analysis_types = ['layer_wise', 'stage_wise', 'component_wise']

    for component in components:
        for analysis_type in analysis_types:
            dir_path = fold_dir / component / analysis_type
            dir_path.mkdir(parents=True, exist_ok=True)

    return fold_dir


def get_layer_shape(layer_name, state_dict):
    """Extract the actual shape of a layer from the state dictionary"""
    if layer_name + '.weight' in state_dict:
        return tuple(state_dict[layer_name + '.weight'].shape)
    return None


def collect_parameters(state_dict, filtered_layers):
    """Organize parameters by encoder/decoder and different categories"""
    params = {
        'encoder': {
            'layer_wise': {
                'conv': {
                    'weights': defaultdict(list),
                    'biases': defaultdict(list),
                    'shapes': {}
                },
                'norm': {
                    'weights': defaultdict(list),
                    'biases': defaultdict(list),
                    'shapes': {}
                }
            },
            'stage_wise': {
                'conv': {
                    'weights': defaultdict(list),
                    'biases': defaultdict(list)
                },
                'norm': {
                    'weights': defaultdict(list),
                    'biases': defaultdict(list)
                }
            },
            'component_wise': {
                'conv': {
                    'weights': [],
                    'biases': []
                },
                'norm': {
                    'weights': [],
                    'biases': []
                }
            }
        },
        'decoder': {
            'layer_wise': {
                'conv': {
                    'weights': defaultdict(list),
                    'biases': defaultdict(list),
                    'shapes': {}
                },
                'norm': {
                    'weights': defaultdict(list),
                    'biases': defaultdict(list),
                    'shapes': {}
                },
                'transpconv': {
                    'weights': defaultdict(list),
                    'biases': defaultdict(list),
                    'shapes': {}
                },
                'seg': {
                    'weights': defaultdict(list),
                    'biases': defaultdict(list),
                    'shapes': {}
                }
            },
            'stage_wise': {
                'conv': {
                    'weights': defaultdict(list),
                    'biases': defaultdict(list)
                },
                'norm': {
                    'weights': defaultdict(list),
                    'biases': defaultdict(list)
                },
                'transpconv': {
                    'weights': defaultdict(list),
                    'biases': defaultdict(list)
                },
                'seg': {
                    'weights': defaultdict(list),
                    'biases': defaultdict(list)
                }
            },
            'component_wise': {
                'conv': {
                    'weights': [],
                    'biases': []
                },
                'norm': {
                    'weights': [],
                    'biases': []
                },
                'transpconv': {
                    'weights': [],
                    'biases': []
                },
                'seg': {
                    'weights': [],
                    'biases': []
                }
            }
        }
    }

    for component in ['encoder', 'decoder']:
        for stage, layer_names in filtered_layers[component].items():
            for layer_name in layer_names:
                if layer_name not in state_dict:
                    continue

                param = state_dict[layer_name]
                param_data = param.detach().cpu().numpy().flatten()

                # Parse layer name to extract components
                parts = layer_name.split('.')
                stage_num = f"stage{stage}"

                # Determine the actual layer type and number
                if component == 'encoder':
                    layer_num = parts[5]
                    layer_type = parts[6]
                    base_layer_name = '.'.join(parts[:-1])
                else:
                    if 'transpconvs' in layer_name:
                        layer_type = 'transpconv'
                        layer_num = next(p for p in parts if p.isdigit())
                        base_layer_name = '.'.join(parts[:-1])
                    elif 'seg_layers' in layer_name:
                        layer_type = 'seg'
                        layer_num = next(p for p in parts if p.isdigit())
                        base_layer_name = '.'.join(parts[:-1])
                    else:
                        layer_num = parts[4]
                        layer_type = parts[5]
                        base_layer_name = '.'.join(parts[:-1])

                param_type = 'weights' if 'weight' in layer_name else 'biases'
                formatted_name = f"{stage_num}.{layer_num}.{param_type}"

                # Store layer shape
                if param_type == 'weights':
                    layer_shape = get_layer_shape(base_layer_name, state_dict)
                    if layer_shape is not None:
                        params[component]['layer_wise'][layer_type]['shapes'][formatted_name] = layer_shape

                # Add to appropriate collections
                params[component]['layer_wise'][layer_type][param_type][formatted_name].extend(param_data)
                params[component]['stage_wise'][layer_type][param_type][stage_num].extend(param_data)
                params[component]['component_wise'][layer_type][param_type].extend(param_data)

    return params


def get_initialization_distribution(shape, layer_type, neg_slope=1e-2):
    """Generate initialization distribution based on Kaiming initialization"""
    print(f"Generating initialization distribution for {layer_type} layer with shape {shape}")
    if layer_type in ['conv', 'transpconv', 'seg']:
        if len(shape) >= 3:  # Ensure we have enough dimensions
            fan_in = shape[1] * np.prod(shape[2:])  # Input channels * kernel dimensions
            std = np.sqrt(2.0 / ((1 + neg_slope ** 2) * fan_in))
            init_weights = np.random.normal(0, std, size=np.prod(shape))
            init_bias = np.zeros(shape[0]) if len(shape) > 0 else np.array([])
            return {'weights': init_weights, 'biases': init_bias}
    return None


def get_theoretical_init_distribution(shape, layer_type, neg_slope=1e-2):
    """Calculate theoretical initialization distribution parameters"""
    if layer_type in ['conv', 'transpconv', 'seg'] and len(shape) >= 3:
        fan_in = shape[1] * np.prod(shape[2:])  # Input channels * kernel dimensions
        std = np.sqrt(2.0 / ((1 + neg_slope ** 2) * fan_in))
        return 0.0, std  # mean, std for the normal distribution
    return None


def plot_layer_distributions(params, output_dir, component, plot_config):
    """Plot parameter distributions with normal approximations and theoretical distributions"""
    if not plot_config['analysis']['layer_wise']:
        print(f"Skipping layer-wise analysis for {component} as per config")
        return

    # Extract config parameters
    figsize = (plot_config['figsize']['width'], plot_config['figsize']['height'])
    dpi = plot_config['dpi']
    bins = plot_config['bins']
    alpha = plot_config['alpha']
    textbox_alpha = plot_config['textbox_alpha']
    max_cols = plot_config['max_cols']
    show_kurtosis = plot_config['show_kurtosis']
    show_theoretical = plot_config['show_theoretical']
    colors = plot_config['colors']

    for layer_type, param_types in params[component]['layer_wise'].items():
        for param_type, layers in param_types.items():
            if not layers or param_type == 'shapes':
                continue

            n_layers = len(layers)
            if n_layers == 0:
                continue

            n_cols = min(max_cols, n_layers)
            n_rows = (n_layers + n_cols - 1) // n_cols
            current_figsize = (figsize[0], figsize[1] * n_rows)

            # Create and save original plots with normal approximation
            fig_orig, axes_orig = plt.subplots(n_rows, n_cols, figsize=current_figsize)
            fig_orig.suptitle(f'{component.title()} {layer_type.title()} Layer-wise {param_type.title()} Distribution')

            if n_rows == 1:
                axes_orig = [axes_orig] if n_cols == 1 else axes_orig
            if n_cols == 1:
                axes_orig = [[ax] for ax in axes_orig]

            # Plot original distributions with fitted normal for specific layer types
            for idx, (name, values) in enumerate(sorted(layers.items())):
                row = idx // n_cols
                col = idx % n_cols
                ax = axes_orig[row][col]

                # Plot histogram
                counts, bins_hist, _ = ax.hist(values, bins=bins, density=True, color=colors['trained'], alpha=alpha,
                                               label='Distribution')

                # Fit normal distribution for specific layer types and weights
                if layer_type in ['conv', 'transpconv', 'seg'] and param_type == 'weights':
                    # Calculate kurtosis before fitting normal distribution
                    if show_kurtosis:
                        k = kurtosis(values, fisher=False)

                    mu, std = norm.fit(values)
                    x = np.linspace(min(values), max(values), 100)
                    p = norm.pdf(x, mu, std)
                    ax.plot(x, p, color=colors['fit'], lw=2, label='Normal fit')

                    # Add parameter text box
                    if show_kurtosis:
                        stats_text = f'kurtosis = {k:.2f}\nμ = {mu:.2e}\nσ = {std:.2e}'
                    else:
                        stats_text = f'μ = {mu:.2e}\nσ = {std:.2e}'

                    props = dict(boxstyle='round', facecolor='wheat', alpha=textbox_alpha)
                    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                            verticalalignment='top', horizontalalignment='right',
                            bbox=props)

                ax.set_title(name)
                ax.set_xlabel('Value')
                ax.set_ylabel('Density')
                ax.legend(loc='upper left')

            # Remove empty subplots in original plot
            for idx in range(len(layers), n_rows * n_cols):
                row = idx // n_cols
                col = idx % n_cols
                fig_orig.delaxes(axes_orig[row][col])

            plt.tight_layout()
            save_path_orig = output_dir / component / 'layer_wise' / f'{layer_type}_{param_type}_original.png'
            plt.savefig(save_path_orig, dpi=dpi, bbox_inches='tight')
            plt.close()

            # Create and save comparison plots with dual y-axes and theoretical distribution
            if show_theoretical and layer_type in ['conv', 'transpconv', 'seg'] and param_type == 'weights':
                fig_comp, axes_comp = plt.subplots(n_rows, n_cols, figsize=current_figsize)
                fig_comp.suptitle(
                    f'{component.title()} {layer_type.title()} Layer-wise {param_type.title()} Distribution (with Initialization)',
                    y=1.02)  # Moved title up slightly

                if n_rows == 1:
                    axes_comp = [axes_comp] if n_cols == 1 else axes_comp
                if n_cols == 1:
                    axes_comp = [[ax] for ax in axes_comp]

                # Plot comparison distributions
                for idx, (name, values) in enumerate(sorted(layers.items())):
                    row = idx // n_cols
                    col = idx % n_cols
                    ax1 = axes_comp[row][col]
                    ax2 = ax1.twinx()

                    # Plot trained weights histogram on first y-axis
                    counts_trained, bins_trained, _ = ax1.hist(values, bins=bins, color=colors['trained'], alpha=alpha,
                                                               label='Trained', density=False)
                    ax1.tick_params(axis='y', labelcolor=colors['trained'])

                    # Get the actual shape and theoretical distribution
                    actual_shape = params[component]['layer_wise'][layer_type]['shapes'].get(name)

                    if actual_shape:
                        theory_params = get_theoretical_init_distribution(actual_shape, layer_type)

                        if theory_params:
                            theory_mean, theory_std = theory_params

                            # Generate initialization samples and plot histogram
                            init_dist = get_initialization_distribution(actual_shape, layer_type)
                            if init_dist and param_type in init_dist:
                                init_weights = init_dist[param_type]
                                counts_init, bins_init, _ = ax2.hist(init_weights, bins=bins, color=colors['init'],
                                                                     alpha=alpha / 2,
                                                                     label='Init', density=False)

                                # Plot theoretical distribution line
                                x = np.linspace(min(init_weights), max(init_weights), 100)
                                theoretical_pdf = norm.pdf(x, theory_mean, theory_std)
                                scale_factor = np.max(counts_init) / np.max(theoretical_pdf)
                                theoretical_pdf_scaled = theoretical_pdf * scale_factor
                                ax2.plot(x, theoretical_pdf_scaled, color=colors['fit'], lw=2, label='Init Appx.')

                                # Add theoretical parameters text box
                                # Position it in the upper right but below where trained params would be
                                theory_text = f'μ = {theory_mean:.2e}\nσ = {theory_std:.2e}'
                                theory_props = dict(boxstyle='round', facecolor='mistyrose', alpha=textbox_alpha)
                                ax1.text(0.95, 0.75, theory_text, transform=ax1.transAxes, fontsize=8,
                                         verticalalignment='top', horizontalalignment='right',
                                         bbox=theory_props)

                            ax2.tick_params(axis='y', labelcolor=colors['init'])

                    ax1.set_title(f"{name}")
                    ax1.set_xlabel('Value')

                    # Add smaller, more compact legend in the upper right
                    lines1, labels1 = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    legend = ax1.legend(lines1 + lines2, labels1 + labels2,
                                        loc='upper right',
                                        bbox_to_anchor=(0.98, 0.98),
                                        prop={'size': 8},
                                        framealpha=0.8)
                    legend.get_frame().set_linewidth(0.5)

                # Remove empty subplots in comparison plot
                for idx in range(len(layers), n_rows * n_cols):
                    row = idx // n_cols
                    col = idx % n_cols
                    fig_comp.delaxes(axes_comp[row][col])

                # Adjust layout to prevent subplot overlap
                plt.tight_layout()
                fig_comp.subplots_adjust(top=0.95, right=0.95)  # Adjusted margins

                save_path_comp = output_dir / component / 'layer_wise' / f'{layer_type}_{param_type}_comparison.png'
                plt.savefig(save_path_comp, dpi=dpi, bbox_inches='tight')
                plt.close()


def plot_stage_distributions(params, output_dir, component, plot_config):
    """Plot stage-wise distributions for each layer type separately"""
    if not plot_config['analysis']['stage_wise']:
        print(f"Skipping stage-wise analysis for {component} as per config")
        return

    # Extract config parameters
    figsize = (plot_config['figsize']['width'], plot_config['figsize']['height'])
    dpi = plot_config['dpi']

    for layer_type, param_types in params[component]['stage_wise'].items():
        for param_type, stages in param_types.items():
            if not stages:
                continue

            plt.figure(figsize=figsize)

            data = [values for stage, values in sorted(stages.items())]
            labels = [stage for stage in sorted(stages.keys())]

            # Create violin plot
            plt.violinplot(data, showmeans=True)

            plt.xticks(range(1, len(labels) + 1), labels)
            plt.title(f'{component.title()} {layer_type.title()} Stage-wise {param_type.title()} Distribution')
            plt.xlabel('Stage')
            plt.ylabel('Value')

            save_path = output_dir / component / 'stage_wise' / f'{layer_type}_{param_type}.png'
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            plt.close()


def plot_component_distributions(params, output_dir, component, plot_config):
    """Plot component-wise distributions for each layer type separately"""
    if not plot_config['analysis']['component_wise']:
        print(f"Skipping component-wise analysis for {component} as per config")
        return

    # Extract config parameters
    figsize = (plot_config['figsize']['width'], plot_config['figsize']['height'])
    dpi = plot_config['dpi']

    plt.figure(figsize=figsize)

    # Collect all data for violin plots
    all_data = []
    labels = []

    for layer_type, param_types in params[component]['component_wise'].items():
        for param_type, values in param_types.items():
            if values:
                all_data.append(values)
                labels.append(f"{layer_type.title()} {param_type.title()}")

    if not all_data:
        return

    plt.violinplot(all_data, showmeans=True)
    plt.xticks(range(1, len(labels) + 1), labels, rotation=45)
    plt.title(f'{component.title()} Component-wise Parameter Distribution')
    plt.ylabel('Value')

    plt.tight_layout()
    save_path = output_dir / component / 'component_wise' / 'distributions.png'
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def analyze_parameter_distributions(state_dict, filtered_layers, output_dir, fold, plot_config):
    # TODO: This function is kinda hard coded
    """Main function to analyze and plot parameter distributions"""
    base_dir = setup_output_dirs(output_dir, fold, plot_config['components'])
    params = collect_parameters(state_dict, filtered_layers)

    for component in plot_config['components']:
        plot_layer_distributions(params, base_dir, component, plot_config)
        plot_stage_distributions(params, base_dir, component, plot_config)
        plot_component_distributions(params, base_dir, component, plot_config)

    return base_dir


def filter_unique_layers(state_dict):
    # TODO: This function is kinda hard coded
    """
    Filter and organize unique layers from the network state dict by stage.

    Args:
        state_dict (OrderedDict): Network state dictionary containing layer weights

    Returns:
        dict: Dictionary organized by component and stage containing unique layers
    """
    # Create dictionaries to store organized layers
    encoder_layers = defaultdict(list)
    decoder_layers = defaultdict(list)

    # Helper function to extract stage number
    def get_stage_num(key):
        parts = key.split('.')
        for part in parts:
            if part.isdigit():
                return int(part)
        return None

    # Process each layer key
    for key in state_dict.keys():
        # Skip duplicate all_modules entries as they're redundant
        if 'all_modules' in key:
            continue

        # Categorize layers by component and stage
        if key.startswith('encoder.stages'):
            stage = get_stage_num(key)
            if stage is not None:
                encoder_layers[stage].append(key)
        elif key.startswith('decoder.stages') or key.startswith('decoder.transpconvs') or key.startswith(
                'decoder.seg_layers'):
            stage = get_stage_num(key)
            if stage is not None:
                decoder_layers[stage].append(key)

    # Sort layers within each stage
    for stage in encoder_layers:
        encoder_layers[stage].sort()
    for stage in decoder_layers:
        decoder_layers[stage].sort()

    return {
        'encoder': dict(encoder_layers),
        'decoder': dict(decoder_layers)
    }


def print_layer_summary(filtered_layers):
    """
    Print a summary of the filtered layers by component and stage.

    Args:
        filtered_layers (dict): Dictionary containing filtered layers by component and stage
    """
    for component, stages in filtered_layers.items():
        print(f"\n{component.upper()} STAGES:")
        print("-" * 50)

        for stage, layers in sorted(stages.items()):
            print(f"\nStage {stage}:")
            for layer in layers:
                print(f"  {layer}")


def analyze_network_layers(state_dict):
    """
    Analyze network layers from the state dictionary.

    Args:
        state_dict (OrderedDict): Network state dictionary containing layer weights

    Returns:
        dict: Filtered layers organized by component and stage
    """
    filtered_layers = filter_unique_layers(state_dict)
    print_layer_summary(filtered_layers)

    # Count layers per stage
    stage_counts = {
        'encoder': len(filtered_layers['encoder']),
        'decoder': len(filtered_layers['decoder'])
    }
    print("\nLayer counts per component:")
    for component, count in stage_counts.items():
        print(f"{component}: {count} stages")

    return filtered_layers


def load_model_weights(base_dir, dataset_id, fold, model_version, checkpoint_name='checkpoint_final.pth'):
    """
    Load model weights from checkpoint.

    Args:
        base_dir (str or Path): Base directory for model results
        dataset_id (int): Dataset ID
        fold (int): Fold number
        model_version (str): Model version/trainer name
        checkpoint_name (str): Name of the checkpoint file

    Returns:
        OrderedDict: Model weights
    """
    base_path = Path(base_dir)
    dataset_dir = base_path / f'Dataset{dataset_id:03d}_ACDC'
    model_path = dataset_dir / model_version / f'fold_{fold}' / checkpoint_name

    print(f"\nLoading checkpoint from: {model_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {model_path}")

    checkpoint = torch.load(model_path, weights_only=False, map_location=torch.device('cpu'))

    print("\nCheckpoint contents:")
    print("Keys in checkpoint:", checkpoint.keys())

    # Print detailed information for each key
    for key, value in checkpoint.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            print("  Nested dictionary with keys:", value.keys())
        elif isinstance(value, torch.Tensor):
            print(f"\n{key}:")
            print(f"  Tensor shape: {value.shape}")
            print(f"  Tensor dtype: {value.dtype}")
        elif isinstance(value, (int, float, str, bool)):
            print(f"\n{key}:")
            print(f"  Value: {value}")
        else:
            print(f"\n{key}:")
            print(f"  Type: {type(value)}")

    # Return the network weights
    return checkpoint['network_weights']


def load_model_weights_with_path(model_path: str):
    """
    Load model weights from checkpoint.

    Args:
        base_dir (str or Path): Base directory for model results
        dataset_id (int): Dataset ID
        fold (int): Fold number
        model_version (str): Model version/trainer name
        checkpoint_name (str): Name of the checkpoint file

    Returns:
        OrderedDict: Model weights
    """
    model_path = Path(model_path)

    print(f"\nLoading checkpoint from: {model_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {model_path}")

    checkpoint = torch.load(model_path, weights_only=False, map_location=torch.device('cpu'))

    print("\nCheckpoint contents:")
    print("Keys in checkpoint:", checkpoint.keys())

    # Print detailed information for each key
    for key, value in checkpoint.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            print("  Nested dictionary with keys:", value.keys())
        elif isinstance(value, torch.Tensor):
            print(f"\n{key}:")
            print(f"  Tensor shape: {value.shape}")
            print(f"  Tensor dtype: {value.dtype}")
        elif isinstance(value, (int, float, str, bool)):
            print(f"\n{key}:")
            print(f"  Value: {value}")
        else:
            print(f"\n{key}:")
            print(f"  Type: {type(value)}")

    # Return the network weights
    return checkpoint['network_weights']


def main(config_path):
    """
    Main function to analyze parameter distributions using a YAML config file.

    Args:
        config_path (str): Path to the YAML configuration file
    """
    # Load configuration
    config = read_config(config_path)
    plot_config = config.get('plot', {})

    # Extract parameters from config
    dataset_id = plot_config.get('dataset_id', 27)
    base_dir = plot_config.get('base_dir', 'nnUNet_results')
    output_dir = plot_config.get('output_dir', 'parameter_distributions')
    model_version = plot_config.get('model_version', 'nnUNetTrainer__nnUNetPlans__2d')
    checkpoint_name = plot_config.get('checkpoint_name', 'checkpoint_final.pth')
    folds = plot_config.get('folds', list(range(5)))

    # Print configuration
    print(f"Parameter Analysis Configuration:")
    print(f"Dataset ID: {dataset_id}")
    print(f"Base Directory: {base_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Model Version: {model_version}")
    print(f"Checkpoint: {checkpoint_name}")
    print(f"Folds: {folds}")
    print(f"Components: {plot_config.get('components', ['encoder', 'decoder'])}")
    print(f"Analysis Types:")
    for analysis_type, enabled in plot_config.get('analysis', {}).items():
        print(f"  - {analysis_type}: {'Enabled' if enabled else 'Disabled'}")

    # Print current working directory
    print(f"\nCurrent working directory: {Path.cwd()}")

    # Print existence of base directory
    base_path = Path(base_dir)
    print(f"\n{base_dir} directory exists: {base_path.exists()}")

    if base_path.exists():
        print(f"\nContents of {base_dir}:")
        for item in base_path.iterdir():
            print(f"  {item}")

    # Analyze weights for each fold
    for fold in folds:
        print(f"\nAnalyzing fold {fold}")

        # Load weights
        try:
            state_dict = load_model_weights(base_dir, dataset_id, fold, model_version, checkpoint_name)
            print("\nSuccessfully loaded model weights")

            filtered_layers = analyze_network_layers(state_dict)
            print("\nLayer counts per component:")
            for component, stages in filtered_layers.items():
                print(f"{component}: {len(stages)} stages")

            import pprint
            print("\nDetailed layer structure:")
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(filtered_layers)

            fold_output_dir = analyze_parameter_distributions(
                state_dict, filtered_layers, output_dir, fold, plot_config)

            print(f"\nParameter distribution analysis complete")
            print(f"Plots have been saved to: {fold_output_dir}")

            # Print the types of plots generated
            print("\nGenerated plots:")
            if plot_config.get('analysis', {}).get('layer_wise', True):
                print("1. Layer-wise distributions:")
                print("   - Individual distributions for conv, norm, transpconv, and seg layers")
            if plot_config.get('analysis', {}).get('stage_wise', True):
                print("2. Stage-wise distributions:")
                print("   - Parameter distributions across different stages")
            if plot_config.get('analysis', {}).get('component_wise', True):
                print("3. Component-wise distributions:")
                print("   - Comparison between different layer types")

        except FileNotFoundError as e:
            print(f"Could not find weights for fold {fold}")
            expected_path = f"{base_dir}/Dataset{dataset_id:03d}_ACDC/{model_version}/fold_{fold}/{checkpoint_name}"
            print(f"Expected path: {expected_path}")
        except Exception as e:
            print(f"Error analyzing fold {fold}: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    config_path = "scripts/config/config.yaml"
    main(config_path)