import torch
import torch.nn.functional as F
from torchvision import transforms
import lpips
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from config import DEVICE

lpips_model = lpips.LPIPS(net='alex').to(DEVICE)

def evaluate_all_metrics(orig_tensor, adv_tensor, model):
    model.eval()

    with torch.no_grad():
        f_orig = model.backbone.body.layer4(model.backbone.body(model.transform(orig_tensor)[0])).detach()
        f_adv = model.backbone.body.layer4(model.backbone.body(model.transform(adv_tensor)[0])).detach()

    orig_lpips = (orig_tensor * 2) - 1
    adv_lpips = (adv_tensor * 2) - 1

    def cosine_similarity(f1, f2):
        return F.cosine_similarity(f1.view(-1), f2.view(-1), dim=0).item()

    def l2_distance(f1, f2):
        return torch.norm(f1 - f2).item()

    def semantic_drift(delta):
        return delta.abs().mean().item()

    def lpips_score(img1, img2):
        return lpips_model(img1, img2).item()

    def psnr_score(img1, img2):
        img1_np = img1.squeeze().permute(1,2,0).cpu().numpy()
        img2_np = img2.squeeze().permute(1,2,0).cpu().numpy()
        return peak_signal_noise_ratio(img1_np, img2_np, data_range=1.0)

    def ssim_score(img1, img2):
        img1_np = img1.squeeze().permute(1,2,0).cpu().numpy()
        img2_np = img2.squeeze().permute(1,2,0).cpu().numpy()
        return structural_similarity(img1_np, img2_np, channel_axis=-1, data_range=1.0)

    def mse_score(img1, img2):
        img1_np = img1.squeeze().permute(1,2,0).cpu().numpy()
        img2_np = img2.squeeze().permute(1,2,0).cpu().numpy()
        return mean_squared_error(img1_np, img2_np)

    return {
        "Cosine Similarity": cosine_similarity(f_orig, f_adv),
        "L2 Distance": l2_distance(f_orig, f_adv),
        "Semantic Drift": semantic_drift(f_orig - f_adv),
        "LPIPS": lpips_score(orig_lpips, adv_lpips),
        "PSNR": psnr_score(orig_tensor, adv_tensor),
        "SSIM": ssim_score(orig_tensor, adv_tensor),
        "MSE": mse_score(orig_tensor, adv_tensor)
    }
