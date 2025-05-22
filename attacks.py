import torch

def iterative_semantic_attack(model, img_tensor, num_iters=10, epsilon=0.005, feature_maps=None):
    adv_img = img_tensor.clone().detach().requires_grad_(True)

    for i in range(num_iters):
        feature_maps.clear()
        outputs = model(adv_img)

        scores = outputs[0]['scores']
        adv_loss = -torch.sum(scores) if len(scores) > 0 else torch.tensor(0.0, requires_grad=True)

        model.zero_grad()
        adv_loss.backward()

        grad = adv_img.grad
        adv_img = adv_img + epsilon * torch.sign(grad)
        adv_img = torch.clamp(adv_img, 0, 1).detach().requires_grad_(True)

    return adv_img.detach()
