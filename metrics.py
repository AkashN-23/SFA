import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import lpips
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

# ---------------- CONFIG ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_model = lpips.LPIPS(net='alex').to(device)

# ---------------- LOAD MODEL ----------------
model = fasterrcnn_resnet50_fpn(weights="DEFAULT").eval().to(device)

# ---------------- LOAD IMAGES ----------------
original_path = "dog.jpg"
attacked_path = "test_images/attacked.jpg"

original_img = Image.open(original_path).convert('RGB')
attacked_img = Image.open(attacked_path).convert('RGB')

transform = transforms.ToTensor()
orig_tensor = transform(original_img).unsqueeze(0).to(device)
adv_tensor = transform(attacked_img).unsqueeze(0).to(device)

# For LPIPS (needs [-1, 1] range)
orig_lpips = (orig_tensor * 2) - 1
adv_lpips = (adv_tensor * 2) - 1

# ---------------- HOOK FOR FEATURES ----------------
feature_maps = {}
def hook_fn(module, input, output):
    feature_maps['features'] = output

hook = model.backbone.body.layer4.register_forward_hook(hook_fn)

# ---------------- FORWARD PASS ----------------
with torch.no_grad():
    _ = model(orig_tensor)
    f_orig = feature_maps['features'].detach()
    _ = model(adv_tensor)
    f_adv = feature_maps['features'].detach()

hook.remove()

# ---------------- METRICS ----------------
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
    return structural_similarity(img1_np, img2_np, multichannel=True, data_range=1.0)

def mse_score(img1, img2):
    img1_np = img1.squeeze().permute(1,2,0).cpu().numpy()
    img2_np = img2.squeeze().permute(1,2,0).cpu().numpy()
    return mean_squared_error(img1_np, img2_np)

# ---------------- FINAL REPORT ----------------
metrics = {
    "Cosine Similarity": cosine_similarity(f_orig, f_adv),
    "L2 Distance": l2_distance(f_orig, f_adv),
    "Semantic Drift": semantic_drift(f_orig - f_adv),
    "LPIPS": lpips_score(orig_lpips, adv_lpips),
    "PSNR": psnr_score(orig_tensor, adv_tensor),
    "SSIM": ssim_score(orig_tensor, adv_tensor),
    "MSE": mse_score(orig_tensor, adv_tensor)
}

print("\n--- Evaluation Metrics Report ---")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
