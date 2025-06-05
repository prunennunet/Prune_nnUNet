from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from batchgenerators.utilities.file_and_folder_operations import join, subdirs, maybe_mkdir_p
from nnunetv2.paths import nnUNet_raw

if __name__ == '__main__':
    """
    This snippet creates a separate dataset JSON for your validation/test images
    which do not have labels. Note that this, too, links directly to your data
    rather than copying it.
    """

    # Path to your validation/test images (no labels)
    extracted_BraTS2024_GLI_test_dir = '/mnt/data/downloads/BraTS2024-BraTS-GLI-ValidationData/validation_data'

    # Give this a new dataset name and ID to avoid overwriting your training dataset
    nnunet_dataset_name_test = 'BraTS2024-BraTS-GLI-Test'
    nnunet_dataset_id_test = 999  # Use a different ID from the training dataset
    dataset_name_test = f'Dataset{nnunet_dataset_id_test:03d}_{nnunet_dataset_name_test}'

    # This will be created under nnUNet_raw
    dataset_dir_test = join(nnUNet_raw, dataset_name_test)
    maybe_mkdir_p(dataset_dir_test)

    # Build the dictionary of cases. Note that there's no "label" entry here.
    test_dataset = {}
    casenames_test = subdirs(extracted_BraTS2024_GLI_test_dir, join=False)
    for c in casenames_test:
        test_dataset[c] = {
            'images': [
                join(extracted_BraTS2024_GLI_test_dir, c, c + '-t1n.nii.gz'),
                join(extracted_BraTS2024_GLI_test_dir, c, c + '-t1c.nii.gz'),
                join(extracted_BraTS2024_GLI_test_dir, c, c + '-t2w.nii.gz'),
                join(extracted_BraTS2024_GLI_test_dir, c, c + '-t2f.nii.gz')
            ]
        }

    # For the test dataset, you can still define the modalities and label values
    # so that nnU-Net knows the structure. Typically, label 0 is always "background."
    # Since these are unlabeled images, 'num_training_cases' is 0, and no 'label' file
    # is actually present in test_dataset.
    labels = {
        'background': 0,
        'NETC': 1,
        'SNFH': 2,
        'ET': 3,
        'RC': 4,
    }

    generate_dataset_json(
        dataset_dir_test,
        {
            0: 'T1',
            1: 'T1C',
            2: 'T2W',
            3: 'T2F'
        },
        labels,
        num_training_cases=0,   # 0 because this set is only for inference
        file_ending='.nii.gz',
        regions_class_order=None,
        dataset_name=dataset_name_test,
        reference='https://www.synapse.org/Synapse:syn53708249/wiki/627500',
        license='see https://www.synapse.org/Synapse:syn53708249/wiki/627508',
        dataset=test_dataset,
        description='This dataset JSON references unlabeled validation/test images for inference.'
    )
