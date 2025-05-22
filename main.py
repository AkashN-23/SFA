# ================================
# main.py — Semantic Flow Attack
# ================================

import torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as TF
from metrics import evaluate_attack_all
import lpips
import os
import matplotlib.pyplot as plt
from utils import draw_boxes, save_image_comparison, save_metrics_csv

# ---------- Config ----------
IMAGE_PATH = 'dog.jpg'
SAVE_PATH = 'results/'
os.makedirs(SAVE_PATH, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_ITERS = 10
EPSILON = 0.01
SCORE_THRESH = 0.5


# ---------- Load Image ----------
img = Image.open(IMAGE_PATH).convert('RGB')
img_tensor = TF.to_tensor(img).unsqueeze(0).to(DEVICE)
img_tensor_adv = img_tensor.clone().detach()

# ---------- Load Model ----------
model = fasterrcnn_resnet50_fpn(weights="DEFAULT").to(DEVICE).eval()

# ---------- Feature Hook Setup ----------
feat_orig = {}
feat_adv = {}

def register_hooks(model, fmap_dict, layers=['layer2', 'layer3', 'layer4']):
    for layer in layers:
        module = model.backbone.body._modules[layer]
        module.register_forward_hook(lambda m, i, o, l=layer: fmap_dict.setdefault(l, o.detach()))

register_hooks(model, feat_orig)
_ = model(img_tensor)  # Trigger forward to fill feat_orig

# ---------- Adversarial Attack ----------
img_tensor_adv.requires_grad_(True)

for i in range(NUM_ITERS):
    print(f"[Iteration {i+1}/{NUM_ITERS}]")

    outputs = model(img_tensor_adv)
    scores = outputs[0]['scores']
    adv_loss = -torch.sum(scores) if len(scores) else torch.tensor(0.0, requires_grad=True)

    print(f"Loss: {adv_loss.item():.4f}")
    model.zero_grad()
    adv_loss.backward()

    grad = img_tensor_adv.grad
    img_tensor_adv = img_tensor_adv + EPSILON * torch.sign(grad)
    img_tensor_adv = torch.clamp(img_tensor_adv, 0, 1).detach()
    img_tensor_adv.requires_grad_(True)

# Capture features after attack
register_hooks(model, feat_adv)
_ = model(img_tensor_adv)

# ---------- Run Detections ----------
outputs_orig = model(img_tensor)
outputs_adv = model(img_tensor_adv)

# ---------- Visualizations ----------
img_orig_pil = TF.to_pil_image(img_tensor.squeeze(0).cpu())
img_adv_pil = TF.to_pil_image(img_tensor_adv.squeeze(0).cpu())

img_orig_annot = draw_boxes(img_orig_pil, outputs_orig, threshold=SCORE_THRESH)
img_adv_annot = draw_boxes(img_adv_pil, outputs_adv, threshold=SCORE_THRESH)

comparison_path = os.path.join(SAVE_PATH, 'comparison.png')
save_image_comparison(img_orig_annot, img_adv_annot, comparison_path)
print(f"✅ Side-by-side saved at: {comparison_path}")

# ---------- Metric Evaluation ----------
lpips_fn = lpips.LPIPS(net='alex').to(DEVICE)
results, normalized = evaluate_attack_all(
    original_tensor=img_tensor,
    adv_tensor=img_tensor_adv,
    feat_dict_orig=feat_orig,
    feat_dict_adv=feat_adv,
    outputs_orig=outputs_orig,
    outputs_adv=outputs_adv,
    lpips_fn=lpips_fn
)

print("\n📊 Final Metrics:")
for k, v in results.items():
    print(f"{k:<25}: {v:.4f}")

# ---------- Save Results ----------
save_metrics_csv(results, os.path.join(SAVE_PATH, "metrics_raw.csv"))
print("📁 Raw metrics saved.")

