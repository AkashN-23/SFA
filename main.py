import torch
from torchvision import transforms
from PIL import Image
from model_utils import load_model
from attacks import semantic_flow_attack
from metrics_evaluation import evaluate_all_metrics
import os

# -------- CONFIG --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_path = "dog.jpg"
save_path = "test_images/attacked.jpg"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# -------- LOAD IMAGE --------
transform = transforms.ToTensor()
image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(device)

# -------- LOAD MODEL --------
model = load_model().to(device)
model.eval()

# -------- GENERATE ADVERSARIAL IMAGE --------
adv_tensor = semantic_flow_attack(model, image_tensor.clone(), epsilon=0.03, alpha=1.0, steps=30)
adv_image = transforms.ToPILImage()(adv_tensor.squeeze().cpu())
adv_image.save(save_path)
print(f"[✓] Adversarial image saved at: {save_path}")

# -------- EVALUATE METRICS --------
metrics = evaluate_all_metrics(image_tensor, adv_tensor, model)

print("\n--- Enhanced Evaluation Metrics Report ---")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
