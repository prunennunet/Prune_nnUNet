def predict_after_prune(config: dict):
    # Check whether all required parameters are provided correctly
    prune_config = config['prune']
    if 'model_folder' not in prune_config:
        print("Error: 'model_folder' is required in the config file for pruning.")
        sys.exit(1)
    if 'checkpoint_name' not in prune_config:
        print("Error: 'checkpoint_name' is required in the config file for pruning.")
        sys.exit(1)
    folds = process_fold_config(prune_config.get('fold'))
    print(f"Pruning for folds: {folds}")

    # Prune & Predict with the model trained in each fold
    pred_dirs = []
    for fold in folds:
        # Load the predictor and get the model weights
        fold_tuple = tuple(fold)
        predictor = load_predictor_from_folder(prune_config['model_folder'], fold_tuple, prune_config['checkpoint_name'])
        model = predictor.network

        # Find the function to the corresponding pruning method
        prune_func = get_pruning_function(prune_config['prune_method'])
        prune_params = prune_config.get('prune_parameters', {})
        prune_func(model, **prune_params)

        # ================== Start the prediction here =====================
        output_folder = create_output_path(prune_config, fold)
        pred_dirs.append(output_folder)

        torch.save(model.state_dict(), os.path.join(output_folder, "pruned_model_with_masks.pth"))

        weight_stats, bias_stats, total_stats = analyze_pruning_masks_model(model, output_folder)

        for name, module in model.named_modules():
            if hasattr(module, 'weight_orig'):
                print(f"Removing pruning parameterization for {module}")
                prune.remove(module, 'weight')
            if hasattr(module, 'bias_orig'):
                prune.remove(module, 'bias')
                print(f"Removing pruning parameterization for {module}")
        _ = verify_pruning_model(model)

        torch.save(model.state_dict(), os.path.join(output_folder, "pruned_model_standard.pth"))

        # Update the list_of_parameters with the pruned weights
        if len(fold) == 1:
            for i in range(len(fold)):
                predictor.list_of_parameters[i] = model.state_dict()
                print(f"Updated predictor.list_of_parameters with pruned weights for fold {fold[i]}")
        else:
            '''
            TODO: When multiple folds are used, weights for different folds are stored in model.list_of_parameters,
            this is not yet adapted, to be done.
            '''
            print("Error: Multiple folds are not yet supported for prediction after pruning.")
            sys.exit(1)

        # Perform prediction with the modified model
        predictor.predict_from_files(
            prune_config['input_folder'],
            output_folder,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=3,
            num_processes_segmentation_export=3,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0,
        )

        weight_stats, bias_stats, total_stats = count_zero_parameters_model(model, output_folder)

    return pred_dirs