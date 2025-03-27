import torch
import os
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def choose_device(device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    return device


def _test_force_modify_model(model):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'bias' in name:
                param.zero_()
    return


def print_model_layer_distribution(model):
    if model is not None:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Network has {num_params} parameters")

        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name}: mean={param.data.mean().item()}, std={param.data.std().item()}")
                # If all weights were zeros or ones, std would be 0 or very close to 0
    return

def load_predictor_from_folder(model_folder, fold, checkpoint_name):
    device = choose_device()
    predictor = nnUNetPredictor(tile_step_size=0.5,
                                use_gaussian=True,
                                use_mirroring=True,
                                perform_everything_on_device=True,
                                device=device,
                                verbose=False,
                                verbose_preprocessing=False,
                                allow_tqdm=False,
                                )
    predictor.initialize_from_trained_model_folder(model_folder, fold, checkpoint_name)
    print(f"There are a total of {len(predictor.list_of_parameters)} sets of parameters")
    return predictor


def load_state_dict_or_model_from_path(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    try:
        state_dict_model = torch.load(path, map_location=device, weights_only=False)
    except RuntimeError as e:
        print(f"Runtime error when loading model: {e}")
        state_dict_model = torch.load(path, map_location='cpu', weights_only=False)
        print("Loaded on CPU instead.")
    return state_dict_model


def create_model_load_from_state_dict(model_class, model_path, device=None, **model_params):
    try:
        device = choose_device(device)
        print(f"Loading model to {device}")
        # Init model, load state dict
        model = model_class(**model_params)
        state_dict = load_state_dict_or_model_from_path(model_path, device)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            if "size mismatch" in str(e) or "key doesn't exist" in str(e):
                print(f"Warning: Model architecture mismatch. Error: {e}")
                model.load_state_dict(state_dict, strict=False)
                print("Loaded with strict=False. Some weights may not be used.")
            else:
                raise Exception(f"Error loading model: {str(e)}")

        model = model.to(device)
        return model

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None


def load_model_from_pth(model_path, device=None):
    try:
        device = choose_device(device)
        print(f"Loading model to {device}")
        model = load_state_dict_or_model_from_path(model_path, device)
        return model

    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")
