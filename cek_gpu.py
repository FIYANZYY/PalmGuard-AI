import torch
print(f"Apakah CUDA tersedia? {torch.cuda.is_available()}")
print(f"Nama GPU: {torch.cuda.get_device_name(0)}")