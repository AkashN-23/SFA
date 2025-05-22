import torch
import torch.nn.functional as Fnn
import numpy as np
from torchvision.transforms.functional import to_pil_image
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy as kl_divergence
import lpips  # Learned perceptual metric
import cv2

# ------------------------------
# 1. Feature-Level Metrics
# ------------------------------

def l2_feature_distance(f1, f2):
    return torch.norm(f1 - f2).item()

def cosine_feature_similarity(f1, f2):
    return Fnn.cosine_similarity(f1.view(-1), f2.view(-1), dim=0).item()

def semantic_drift(delta):
    return delta.abs().mean().item()


# ------------------------------
# 2. Image-Level (Perceptual)
# ------------------------------

def compute_ssim(img1, img2):
    img1_np = img1.squeeze().permute(1, 2, 0).cpu().numpy()
    img2_np = img2.squeeze().permute(1, 2, 0).cpu().numpy()
    return ssim(img1_np, img2_np, multichannel=True, data_range=1.0)

def pixel_l2_dist(img1, img2):
    return torch.norm(img1 - img2).item()

def psnr(img1, img2):
    mse = Fnn.mse_loss(img1, img2).item()
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))

def lpips_distance(lpips_fn, img1, img2):
    return lpips_fn(img1, img2).item()


# ------------------------------
# 3. Prediction-Level
# ------------------------------

def attack_success_rate(outputs_orig, outputs_adv, threshold=0.5):
    def extract_labels(outputs):
        return set([label.item() for label, score in zip(outputs[0]['labels'], outputs[0]['scores']) if score > threshold])
    orig_labels = extract_labels(outputs_orig)
    adv_labels = extract_labels(outputs_adv)
    suppressed = orig_labels - adv_labels
    return len(suppressed) / len(orig_labels) if orig_labels else 0.0


# ------------------------------
# 4. Statistical Drift
# ------------------------------

def histogram_kl_divergence(img1, img2, bins=256):
    img1_np = img1.squeeze().cpu().numpy().flatten()
    img2_np = img2.squeeze().cpu().numpy().flatten()
    hist1, _ = np.histogram(img1_np, bins=bins, range=(0, 1), density=True)
    hist2, _ = np.histogram(img2_np, bins=bins, range=(0, 1), density=True)
    hist1 += 1e-10
    hist2 += 1e-10
    return kl_divergence(hist1, hist2)


# ------------------------------
# 5. Multi-layer Feature Analysis
# ------------------------------

def multi_layer_feature_metrics(feat_dict_orig, feat_dict_adv):
    metrics = {}
    for layer in feat_dict_orig:
        f1, f2 = feat_dict_orig[layer], feat_dict_adv[layer]
        delta = f1 - f2
        metrics[f'{layer}_L2'] = l2_feature_distance(f1, f2)
        metrics[f'{layer}_Cosine'] = cosine_feature_similarity(f1, f2)
        metrics[f'{layer}_Drift'] = semantic_drift(delta)
    return metrics


# ------------------------------
# Normalize Metrics (for plots)
# ------------------------------

def normalize_metrics(metric_dict):
    # Min-max normalization assuming typical ranges
    normalized = {}
    bounds = {
        "L2 Feature Dist": (0, 10),
        "Cosine Similarity": (0.5, 1.0),
        "Semantic Drift": (0, 0.1),
        "Pixel L2 Dist": (0, 0.05),
        "SSIM": (0.5, 1.0),
        "PSNR": (10, 50),
        "KL Divergence": (0, 1),
        "LPIPS": (0, 0.5),
        "Suppression Rate": (0, 1)
    }

    for k, v in metric_dict.items():
        low, high = bounds.get(k, (0, 1))
        v_clamped = min(max(v, low), high)
        norm = (v_clamped - low) / (high - low) if high != low else 0
        normalized[k] = norm
    return normalized


# ------------------------------
# Unified Evaluation Function
# ------------------------------

def evaluate_attack_all(
    original_tensor,
    adv_tensor,
    feat_dict_orig,
    feat_dict_adv,
    outputs_orig=None,
    outputs_adv=None,
    lpips_fn=None
):
    results = {}

    # Feature-level
    results.update(multi_layer_feature_metrics(feat_dict_orig, feat_dict_adv))

    # Pixel & perceptual
    results['Pixel L2 Dist'] = pixel_l2_dist(original_tensor, adv_tensor)
    results['SSIM'] = compute_ssim(original_tensor, adv_tensor)
    results['PSNR'] = psnr(original_tensor, adv_tensor)

    if lpips_fn:
        results['LPIPS'] = lpips_distance(lpips_fn, original_tensor, adv_tensor)

    # Statistical
    results['KL Divergence'] = histogram_kl_divergence(original_tensor, adv_tensor)

    # Suppression Rate
    if outputs_orig and outputs_adv:
        results['Suppression Rate'] = attack_success_rate(outputs_orig, outputs_adv)

    return results, normalize_metrics(results)
