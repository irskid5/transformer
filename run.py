import torch

def setup():
    print("PyTorch version: " + torch.__version__.split("+")[0])
    print("Cuda version: " + torch.__version__.split("+")[1])
    if torch.cuda.is_available():
        print("Number of GPUs: %d" % (torch.cuda.device_count()))
        gpu = torch.device("cuda")
        device = gpu
    else:
        print("Using CPU.")
        device = torch.device("cpu")
    return device

print("\nTraining of a Transformer Model using PyTorch.")

if __name__ == "__main__":
    device = setup()
    