import torch
print("PyTorch version: " + torch.__version__.split("+")[0])
print("Cuda version: " + torch.__version__.split("+")[1])
if torch.cuda.is_available():
    print("Number of GPUs: %d" % (torch.cuda.device_count()))
    cpu = torch.device("cpu")
    gpu = torch.device("cuda")
    device = gpu
else:
    raise Exception("No GPU available.")

print("\nTraining of a Transformer Model using PyTorch!")
