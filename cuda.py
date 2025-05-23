'''import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))
    print("CUDA device count:", torch.cuda.device_count())
else:
    print("CUDA is not available.")'''

'''import cuda
import torch

print("CUDA Available:", torch.cuda.is_available())
print("CUDA Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
print("Torch Version:", torch.__version__)
print("Running on:", "GPU" if torch.cuda.is_available() else "CPU")'''

''''import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
inputs = inputs.to(device)'''

import torch
import torchvision.models as models

# Step 1: Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)

# Step 2: Load a pretrained model (example: ResNet18)
model = models.resnet18(pretrained=True)
model = model.to(device)  # MOVE model to GPU or CPU as per device
model.eval()  # set model to evaluation mode

# Step 3: Create a dummy input tensor to simulate your input batch
# Normally, you will load your real dataset here instead
dummy_input = torch.randn(1, 3, 224, 224)  # batch size 1, 3 channels, 224x224 image
dummy_input = dummy_input.to(device)  # MOVE input tensor to same device as model

# Step 4: Forward pass (example)
with torch.no_grad():
    output = model(dummy_input)

print("Output shape:", output.shape)





