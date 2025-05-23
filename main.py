import torch
from torchvision import transforms
from PIL import Image
from model_utils import load_model
from attacks import semantic_flow_attack
from metrics_evaluation import evaluate_all_metrics
from config import DEVICE
import os

image_path = "dog.jpg"
save_path = "attacked.jpg"
save_dir = os.path.dirname(save_path)
if save_dir:
    os.makedirs(save_dir, exist_ok=True)


# Load & preprocess image
transform = transforms.ToTensor()
image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(DEVICE)
label = torch.tensor([0]).to(DEVICE)  # placeholder; use correct class

# Load model
model = load_model().to(DEVICE)
model.eval()

# Run attack
adv_tensor = semantic_flow_attack(model, image_tensor.clone(), label, epsilon=0.03, alpha=1.0, steps=30)
adv_image = transforms.ToPILImage()(adv_tensor.squeeze().cpu())
adv_image.save(save_path)
print(f"[✓] Adversarial image saved at: {save_path}")

# Evaluate
metrics = evaluate_all_metrics(image_tensor, adv_tensor, model)

print("\n--- Enhanced Evaluation Metrics Report ---")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
