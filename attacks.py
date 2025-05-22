import torch
import torch.nn.functional as F
from config import DEVICE

def semantic_flow_attack(model, image, label, epsilon=0.03, steps=10, alpha=0.005):
    perturbed = image.clone().detach().to(DEVICE)
    perturbed.requires_grad = True
    
    for _ in range(steps):
        output = model(perturbed)
        loss = F.cross_entropy(output, label)
        model.zero_grad()
        loss.backward()

        grad_sign = perturbed.grad.sign()
        perturbed = perturbed + alpha * grad_sign

        # Clip and project to epsilon ball
        perturbed = torch.clamp(perturbed, image - epsilon, image + epsilon)
        perturbed = torch.clamp(perturbed, 0, 1).detach()
        perturbed.requires_grad = True

    return perturbed
