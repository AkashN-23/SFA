import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
EPSILON = 0.03  # Perturbation limit
