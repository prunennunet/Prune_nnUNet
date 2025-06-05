import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import nibabel as nib

MAX_FEATURES = 32
MAX_IMAGES_PER_LAYER = 100

def save_featuremap_nifti(data_4d, out_path, affine=None):
    """
    Save a (C, D, H, W) or (1, C, D, H, W) feature map to NIfTI.
    We transpose to (D,H,W,C) so viewers see it as (D, H, W, C).
    """
    if affine is None:
        affine = np.eye(4, dtype=np.float32)

    if data_4d.ndim == 5 and data_4d.shape[0] == 1:
        data_4d = data_4d[0]  # => (C, D, H, W)
    data_4d = np.asarray(data_4d, dtype=np.float32)

    if data_4d.ndim == 4:
        data_4d = np.transpose(data_4d, (1, 2, 3, 0))

    img = nib.Nifti1Image(data_4d, affine)
    nib.save(img, out_path)
    print(f"Saved feature map NIfTI to {out_path}")


def get_robust_limits(arr, lower_percentile=1, upper_percentile=99, fallback=(0, 1)):
    arr = np.asarray(arr, dtype=np.float32).flatten()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return fallback
    vmin = np.percentile(arr, lower_percentile)
    vmax = np.percentile(arr, upper_percentile)
    if vmin >= vmax:
        vmin -= 1e-6
        vmax += 1e-6
    return vmin, vmax


def get_center_slice_indices(total_depth, num_to_pick):
    """
    Return `num_to_pick` slice indices centered around total_depth//2.
    If num_to_pick >= total_depth, return all slices.
    """
    if num_to_pick >= total_depth:
        return list(range(total_depth))

    center = total_depth // 2
    half = num_to_pick // 2
    start = center - half
    end = start + num_to_pick

    if start < 0:
        start = 0
        end = num_to_pick
    if end > total_depth:
        end = total_depth
        start = end - num_to_pick

    return list(range(start, end))


def plot_3d_mri_volume(data: np.ndarray, save_dir: str, dataset_name: str):
    """
    Plots up to 100 total (channel×slice) images for a 3D volume (C, D, H, W).
    If data is 2D (C, H, W), handle accordingly.
    """
    dataset_dir = os.path.join(save_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # Remove batch dimension if present
    if data.ndim == 5:  # (B, C, D, H, W)
        data = data[0]

    if data.ndim == 3:  # (C, H, W) => treat as 2D
        num_channels = min(data.shape[0], MAX_FEATURES)
        total_images = min(num_channels, MAX_IMAGES_PER_LAYER)
        vmin, vmax = get_robust_limits(data, 0.5, 99.5)

        for c in range(total_images):
            plt.figure(figsize=(8, 8))
            plt.imshow(data[c], cmap='gray', vmin=vmin, vmax=vmax)
            plt.axis('off')
            out_path = os.path.join(dataset_dir, f"{dataset_name}_channel_{c}.png")
            plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
            plt.close()
        return

    # data is (C, D, H, W)
    c_dim, d_dim, h_dim, w_dim = data.shape
    num_channels = min(c_dim, MAX_FEATURES)
    total_possible_images = num_channels * d_dim

    if total_possible_images <= MAX_IMAGES_PER_LAYER:
        for c in range(num_channels):
            channel_data = data[c]
            vmin, vmax = get_robust_limits(channel_data, 0.5, 99.5)
            for d in range(d_dim):
                plt.figure(figsize=(8, 8))
                plt.imshow(channel_data[d], cmap='gray', vmin=vmin, vmax=vmax)
                plt.axis('off')
                out_name = f"{dataset_name}_channel_{c}_slice_{d:03d}.png"
                plt.savefig(os.path.join(dataset_dir, out_name),
                            bbox_inches='tight', pad_inches=0)
                plt.close()
    else:
        # Sample from the center
        slices_per_channel = max(1, MAX_IMAGES_PER_LAYER // num_channels)
        for c in range(num_channels):
            channel_data = data[c]
            vmin, vmax = get_robust_limits(channel_data, 0.5, 99.5)
            slice_indices = get_center_slice_indices(d_dim, slices_per_channel)
            for d in slice_indices:
                plt.figure(figsize=(8, 8))
                plt.imshow(channel_data[d], cmap='gray', vmin=vmin, vmax=vmax)
                plt.axis('off')
                out_name = f"{dataset_name}_channel_{c}_slice_{d:03d}.png"
                plt.savefig(os.path.join(dataset_dir, out_name),
                            bbox_inches='tight', pad_inches=0)
                plt.close()


def plot_2d_feature_maps(data: np.ndarray, save_dir: str, dataset_name: str,
                         cmap='viridis', max_features=32):
    dataset_dir = os.path.join(save_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    if data.ndim == 4:  # (B, C, H, W)
        data = data[0]
    elif data.ndim == 2:  # (H, W)
        data = np.expand_dims(data, axis=0)

    num_channels = min(data.shape[0], max_features)
    num_to_plot = min(num_channels, MAX_IMAGES_PER_LAYER)

    for c in range(num_to_plot):
        channel_data = data[c]
        vmin, vmax = get_robust_limits(channel_data, 1, 99)
        plt.figure(figsize=(8, 8))
        plt.imshow(channel_data, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.axis('off')
        out_name = f"{dataset_name}_feature_{c}.png"
        out_path = os.path.join(dataset_dir, out_name)
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close()


def plot_3d_feature_maps(data: np.ndarray, save_dir: str,
                         dataset_name: str, cmap='viridis', max_features=3200):
    dataset_dir = os.path.join(save_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    if data.ndim == 5:  # (B, C, D, H, W)
        data = data[0]
    if data.ndim == 3:  # (C, H, W) => treat as 2D
        plot_2d_feature_maps(data, save_dir, dataset_name, cmap, max_features)
        return

    num_channels = min(data.shape[0], max_features)
    d_slices = data.shape[1]
    total_possible_images = num_channels * d_slices

    if total_possible_images <= MAX_IMAGES_PER_LAYER:
        for c in range(num_channels):
            channel_data = data[c]
            vmin, vmax = get_robust_limits(channel_data, 1, 99)
            norm = Normalize(vmin=vmin, vmax=vmax)
            for d in range(d_slices):
                plt.figure(figsize=(8, 8))
                plt.imshow(channel_data[d], cmap=cmap, norm=norm)
                plt.axis('off')
                out_name = f"{dataset_name}_feature_{c}_slice_{d:03d}.png"
                plt.savefig(os.path.join(dataset_dir, out_name),
                            bbox_inches='tight', pad_inches=0)
                plt.close()
    else:
        slices_per_channel = max(1, MAX_IMAGES_PER_LAYER // num_channels)
        for c in range(num_channels):
            channel_data = data[c]
            vmin, vmax = get_robust_limits(channel_data, 1, 99)
            norm = Normalize(vmin=vmin, vmax=vmax)
            slice_indices = get_center_slice_indices(d_slices, slices_per_channel)
            for d in slice_indices:
                plt.figure(figsize=(8, 8))
                plt.imshow(channel_data[d], cmap=cmap, norm=norm)
                plt.axis('off')
                out_name = f"{dataset_name}_feature_{c}_slice_{d:03d}.png"
                plt.savefig(os.path.join(dataset_dir, out_name),
                            bbox_inches='tight', pad_inches=0)
                plt.close()


def process_h5_group(h5_group, save_dir, group_path=""):
    """
    Recursively traverse the HDF5 group. Process datasets by visualizing slices
    and export to NIfTI if in a "norm" directory.
    """
    os.makedirs(save_dir, exist_ok=True)

    for key in h5_group.keys():
        item = h5_group[key]
        new_path = f"{group_path}/{key}" if group_path else key

        if isinstance(item, h5py.Group):
            sub_group_dir = os.path.join(save_dir, key)
            process_h5_group(item, sub_group_dir, new_path)
        else:
            data = item[()]  # load into memory
            print(f"Processing dataset {new_path} with shape {data.shape}")

            # If we are in a 'norm' subfolder => export entire volume as NIfTI
            if '/norm' in new_path:
                export_name = new_path.replace('/', '_') + ".nii.gz"
                nifti_out_path = os.path.join(save_dir, export_name)
                save_featuremap_nifti(data, nifti_out_path)

            try:
                if key == 'input_x':
                    plot_3d_mri_volume(data, save_dir, key)
                else:
                    plot_3d_feature_maps(
                        data, save_dir, key, cmap='viridis',
                        max_features=MAX_FEATURES
                    )
            except Exception as e:
                print(f"Error processing {key} with shape {data.shape}: {e}")
                error_file = os.path.join(save_dir, f"{key}_error.txt")
                with open(error_file, 'w') as f:
                    f.write(f"Error processing {key} with shape {data.shape}:\n{str(e)}")


# def process_all_h5_in_intermediate_dir(intermediate_h5_dir, max_files=1):
#     """
#     Given the path to an 'intermediate_features' directory,
#     find and process all .h5 files, but only up to `max_files`.
#     """
#     processed_count = 0
#     for fname in os.listdir(intermediate_h5_dir):
#         if fname.endswith(".h5"):
#             full_h5_path = os.path.join(intermediate_h5_dir, fname)
#             h5_dir_name = os.path.splitext(fname)[0]
#             output_dir_for_this_h5 = os.path.join(intermediate_h5_dir, h5_dir_name)
#             os.makedirs(output_dir_for_this_h5, exist_ok=True)
#
#             print(f"\n--- Processing {fname} in {intermediate_h5_dir} ---")
#             try:
#                 with h5py.File(full_h5_path, 'r') as f:
#                     process_h5_group(f, save_dir=output_dir_for_this_h5)
#             except Exception as e:
#                 print(f"Error processing {full_h5_path}: {e}")
#                 with open(os.path.join(output_dir_for_this_h5, "processing_error.txt"), 'w') as f_err:
#                     f_err.write(f"Error processing {fname}:\n{str(e)}")
#
#             processed_count += 1
#             if processed_count >= max_files:
#                 break

def process_all_h5_in_intermediate_dir(intermediate_h5_dir):
    """
    Given the path to an 'intermediate_features' directory,
    only process 'BRATS_487.h5' if it exists.
    """
    target_file = "prostate_03.h5"
    for fname in os.listdir(intermediate_h5_dir):
        if fname == target_file:
            full_h5_path = os.path.join(intermediate_h5_dir, fname)
            h5_dir_name = os.path.splitext(fname)[0]  # e.g., "BRATS_487"
            output_dir_for_this_h5 = os.path.join(intermediate_h5_dir, h5_dir_name)
            os.makedirs(output_dir_for_this_h5, exist_ok=True)

            print(f"\n--- Processing {fname} in {intermediate_h5_dir} ---")
            try:
                with h5py.File(full_h5_path, 'r') as f:
                    process_h5_group(f, save_dir=output_dir_for_this_h5)
            except Exception as e:
                print(f"Error processing {full_h5_path}: {e}")
                error_path = os.path.join(output_dir_for_this_h5, "processing_error.txt")
                with open(error_path, 'w') as f_err:
                    f_err.write(f"Error processing {fname}:\n{str(e)}")

            # Since we're only interested in this one file, break after processing
            # break
    else:
        # Optional: if we exit the for-loop without a break, it means BRATS_487.h5 wasn't found
        print(f"Did not find {target_file} in {intermediate_h5_dir}")


# def main():
#     range_pruning_dir = (
#         "/media/tonguyunyang/tony_data/data/pruning_nnunet_experiment_storage/nnUNet_pruning/Dataset001_BrainTumour/FlexibleTrainerV1__nnUNetPlans__3d_fullres/RangePruning"
#     )
#
#     # Walk the RangePruning dir; whenever we find an 'intermediate_features' folder,
#     # process it (optionally with a limit of 1 .h5 file per folder).
#     for root, dirs, files in os.walk(range_pruning_dir):
#         if os.path.basename(root) == "intermediate_features":
#             print(f"Found intermediate_features folder: {root}")
#             # process_all_h5_in_intermediate_dir(root, max_files=1)
#             process_all_h5_in_intermediate_dir(root)

def main():
    range_pruning_dir = (
        "/media/tonguyunyang/tony_data/data/pruning_nnunet_experiment_storage/nnUNet_pruning/Dataset005_Prostate/FlexibleTrainerV1__nnUNetPlans__3d_fullres/RangePruning"
    )

    # Only process intermediate_features folders whose parent directory
    # is one of these three names:
    allowed_parents = {
        "max_val_0e+__min_val_0e+__prune_bias_True__prune_layers_conv__prune_weights_True"
    }

    # Walk the RangePruning dir as before
    for root, dirs, files in os.walk(range_pruning_dir):
        if os.path.basename(root) == "intermediate_features":
            # Grab the name of the parent directory
            parent_dir_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(root))))
            print(f"{parent_dir_name}")
            if parent_dir_name in allowed_parents:
                print(f"Found intermediate_features folder under {parent_dir_name}: {root}")
                # Now process that folder
                process_all_h5_in_intermediate_dir(root)


if __name__ == "__main__":
    main()
