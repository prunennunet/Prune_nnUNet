# Link to customized packages

nnUNetv2: https://github.com/tonyyunyang/nnUNet

dynamic network: https://github.com/tonyyunyang/dynamic-network-architectures


# nnUNetv2 Configuration Parameters Guide

This document explains the configuration parameters, a powerful framework for medical image segmentation. The configuration file is organized into four main sections: `train`, `predict`, `evaluate`, and `prune`.

## Training Parameters

| Parameter | Description | Default | Usage |
|-----------|-------------|---------|-------|
| `dataset_name_or_id` | The ID or name of the dataset to train on | **Required** | Integer ID (e.g., `27`) |
| `configuration` | Network architecture configuration | **Required** | Options: `2d`, `3d_fullres`, `3d_lowres`, `3d_cascade_fullres` |
| `fold` | Cross-validation folds to use | **Required** | Array of integers [0-4] or `all` |
| `tr` | Trainer class to use | `FlexibleTrainerV1` | String (e.g., `FlexibleTrainerV1`) |
| `p` | Plans configuration | `nnUNetPlans` | String (e.g., `nnUNetPlans`) |
| `pretrained_weights` | Path to pretrained weights | `null` | String path or `null` |
| `num_gpus` | Number of GPUs to use for training | `1` | Integer |
| `device` | Computing device | `cuda` | Options: `cuda`, `cpu`, `mps` |
| `npz` | Save softmax predictions as NPZ files | `false` | Boolean |
| `c` | Continue training from latest checkpoint | `false` | Boolean |
| `val` | Only run validation (requires training to have finished) | `false` | Boolean |
| `val_best` | Use the best checkpoint for validation | `false` | Boolean |
| `disable_checkpointing` | Disable saving model checkpoints | `false` | Boolean |

### Notes on Training Parameters

- The `fold` parameter accepts either individual integers (0-4), the string "all", or a list of integers.
- When using `val` or `val_best`, training must have been completed previously.
- The combination of `tr`, `p`, and `configuration` determines the specific network architecture and training strategy.

## Prediction Parameters

| Parameter | Description | Default | Usage |
|-----------|-------------|---------|-------|
| `model_folder` | Path to the trained model folder | Optional | String path |
| `input_folder` | Input folder with test images | **Required** | String path |
| `output_folder` | Output folder to save predictions | **Required** | String path |
| `fold` | Fold selection for prediction | **Required** | Nested array, e.g., `[[0], [1]]` |
| `step_size` | Step size for sliding window prediction | `0.5` | Float |
| `checkpoint_name` | Name of the checkpoint to use | `checkpoint_final.pth` | String |
| `device` | Device to run on | `cuda` | Options: `cuda`, `cpu`, `mps` |
| `num_processes_preprocessing` | Number of processes for preprocessing | `3` | Integer |
| `num_processes_segmentation_export` | Number of processes for segmentation export | `3` | Integer |
| `prev_stage_predictions` | Path to previous stage predictions | `null` | String path |
| `num_parts` | Total number of parallel prediction jobs | `1` | Integer |
| `part_id` | ID of this specific job (0-based) | `0` | Integer |
| `disable_tta` | Disable test-time augmentation | `false` | Boolean |
| `save_probabilities` | Save probability maps | `false` | Boolean |
| `continue_prediction` | Continue from a previous prediction | `false` | Boolean |
| `disable_progress_bar` | Disable progress bar | `false` | Boolean |
| `verbose` | Enable verbose output | `true` | Boolean |

### Notes on Prediction Parameters

- You must provide either `model_folder` OR both `dataset_name_or_id` and `configuration` from the train section.
- The `fold` parameter is formatted as a nested array. For example:
  - `[[0]]` - Use only fold 0 
  - `[[0, 1, 2]]` - Ensemble predictions from folds 0, 1, and 2
  - `[[0], [1], [2]]` - Generate separate predictions for each fold
- The system automatically creates output subdirectories based on:
  - The fold or ensemble of folds being used
  - The checkpoint type (best or final model)
- For cascade models, `prev_stage_predictions` must be specified to provide the results from the previous stage.
- The `num_parts` and `part_id` parameters allow parallel processing across multiple jobs.

## Evaluation Parameters

| Parameter | Description | Default | Usage |
|-----------|-------------|---------|-------|
| `gt_folder` | Ground truth segmentation folder | **Required** | String path |
| `output_file` | Output file for results | `null` | String path or `null` |
| `num_processes` | Number of processes for evaluation | `8` | Integer |
| `chill` | Less aggressive/verbose computation | `false` | Boolean |
| `pruned` | Whether to use pruned evaluations | `true` | Boolean |
| `checkpoint_name` | Name of checkpoint to use | `null` | String |
| `pred_folder` | Prediction folder to evaluate | `null` | String path |
| `result_base_dir` | Base directory for results | `null` | String path |

### Notes on Evaluation Parameters

- If `output_file` is `null`, results will be saved as `summary.json` in the prediction folder.
- The evaluation system will automatically find the appropriate prediction folder based on:
  - The fold structure used for prediction
  - The checkpoint type (best or final model)
- If `pred_folder` is not specified, it tries to use the `output_folder` from the predict configuration.
- The evaluation requires `dataset.json` and `plans.json` files from the model directory to correctly interpret the results.

## Code-Specific Details

### Fold Configuration Processing

The system processes fold configurations in a sophisticated way:
- Single values are converted to nested lists (e.g., `0` becomes `[[0]]`)
- Lists are wrapped if not already nested (e.g., `[0, 1]` becomes `[[0, 1]]`)
- Mixed cases handle both list and non-list elements
- Each fold value is validated to be between 0-4 or the string "all"

### Prediction Output Structure

The prediction system creates a structured output directory:
1. For single fold predictions: `output_folder/fold_X/[best|final]_model/`
2. For ensemble predictions: `output_folder/ensemble_X_Y_Z/[best|final]_model/`

### Automatic Model Folder Detection

When evaluating results, the system can automatically find the appropriate model folder based on:
1. The dataset ID
2. The trainer, plans, and configuration settings
3. The specific checkpoint being used

This automatic detection helps maintain consistency across the training, prediction, and evaluation pipelines.

## Pruning Parameters

| Parameter | Description | Default | Usage |
|-----------|-------------|---------|-------|
| `checkpoint_name` | Name of the checkpoint to use for pruning | `checkpoint_final.pth` | String |
| `fold` | Folds to perform pruning on | **Required** | Nested array, e.g., `[[0], [1], [2]]` |
| `gt_folder` | Ground truth segmentation folder | **Required** | String path |
| `input_folder` | Input folder with test images | **Required** | String path |
| `model_folder` | Path to the trained model folder | **Required** | String path |
| `output_folder` | Output folder to save pruned model predictions | **Required** | String path |
| `prune_method` | Method to use for pruning | **Required** | String (e.g., `RangePruning`) |
| `prune_parameters` | Configuration for the pruning process | Optional | Nested dictionary |

### Pruning Parameters Dictionary

The `prune_parameters` field contains specific settings for the pruning method:

| Parameter | Description | Default | Usage |
|-----------|-------------|---------|-------|
| `max_val` | Maximum threshold value for pruning | Depends on method | Float (e.g., `0.01`) |
| `min_val` | Minimum threshold value for pruning | Depends on method | Float (e.g., `-0.01`) |
| `prune_bias` | Whether to prune bias terms | `false` | Boolean |
| `prune_weights` | Whether to prune weights | `false` | Boolean |
| `prune_layers` | Types of layers to prune | Required for most methods | Array of strings (e.g., `["conv"]`) |

### Notes on Pruning

- The pruning system creates a structured output directory that includes:
  - The pruning method name (e.g., `RangePruning`)
  - The pruning parameters formatted as `param1_value1__param2_value2`
  - The fold or ensemble structure
  - The checkpoint type (best or final model)
- The system saves two versions of the pruned model:
  - `pruned_model_with_masks.pth`: Model with pruning masks still applied
  - `pruned_model_standard.pth`: Model with pruning parameterization removed
- Numerical pruning parameters (like thresholds) are formatted in scientific notation in the output directory structure
- Currently, ensemble prediction with multiple folds is not fully supported for pruning
- After pruning, statistics on zero-valued parameters (weights, biases, and total) are collected and saved

### Pruning Process

The pruning process follows these steps:
1. Load the trained model from the specified checkpoint
2. Apply the selected pruning method with the provided parameters
3. Verify that pruning was applied correctly
4. Save the pruned model in both masked and standard formats
5. Perform prediction using the pruned model
6. Evaluate the pruned model's performance against ground truth

This enables evaluation of how model compression through pruning affects segmentation accuracy.