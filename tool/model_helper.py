import warnings
from pickle import HIGHEST_PROTOCOL

import torch

from tool.log_helper import logger
from tool.path_helper import ROOT_DIR, mkdir_parent

MODEL_PATH = "out/checkpoints/%s.pt"


def save_model(model: torch.nn.Module, model_save_name: str):
    path = ROOT_DIR.joinpath(MODEL_PATH % model_save_name)
    mkdir_parent(path)
    torch.save(model, path, pickle_protocol=HIGHEST_PROTOCOL)
    logger.info(f"model saved: {path}")


def load_model(model_save_name: str, device: str = "cpu"):
    path = ROOT_DIR.joinpath(MODEL_PATH % model_save_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = torch.load(path, map_location=torch.device(device))
    model.config.main_device = device
    return model
