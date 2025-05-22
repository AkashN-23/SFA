import matplotlib.pyplot as plt
import torch

def show_images(original, adversarial):
    o = original.squeeze().permute(1, 2, 0).cpu().numpy()
    a = adversarial.squeeze().permute(1, 2, 0).detach().cpu().numpy()

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(o)
    axs[0].set_title("Original")
    axs[1].imshow(a)
    axs[1].set_title("Adversarial")
    plt.show()
