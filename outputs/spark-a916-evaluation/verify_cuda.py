import torch


tensor = torch.ones(1, device="cuda:0")
print(f"torch={torch.__version__}")
print(f"cuda_build={torch.version.cuda}")
print(f"available={torch.cuda.is_available()}")
print(f"gpu={torch.cuda.get_device_name(0)}")
print(f"capability={torch.cuda.get_device_capability(0)}")
print(f"tensor_device={tensor.device}")
print(f"tensor_value={tensor.item()}")
