import torch

# Check if an AMD GPU is available and accessible by PyTorch (via ROCm)
print("GPU Avaliable for PyTorch:", torch.cuda.is_available())

# If available, get the name of the detected AMD GPU
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
