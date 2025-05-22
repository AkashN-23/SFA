import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn

feature_maps = {}

def register_feature_hook(model):
    def hook_fn(module, input, output):
        output.retain_grad()
        feature_maps['feat'] = output
    handle = model.backbone.body.layer4.register_forward_hook(hook_fn)
    return handle

def load_model(device='cpu'):
    model = fasterrcnn_resnet50_fpn(pretrained=True).eval().to(device)
    return model
