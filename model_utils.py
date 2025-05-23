import torch
import random
import numpy as np
from config import SEED, DEVICE

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_model(model_class=None, weights_path=None, device=DEVICE):
    if model_class and weights_path:
        model = model_class().to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT").eval().to(device)
    return model
