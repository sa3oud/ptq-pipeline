import torch
import torchvision.models as models
import os

SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(SAVE_DIR, exist_ok=True)

def load_resnet18():
    print("Loading pretrained ResNet18 (FP32)...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    return model

def save_torchscript(model, path):
    scripted = torch.jit.script(model)
    torch.jit.save(scripted, path)
    size_mb = os.path.getsize(path) / (1024 ** 2)
    print(f"Saved TorchScript → {path}  ({size_mb:.2f} MB)")
    return size_mb

def save_state_dict(model, path):
    torch.save(model.state_dict(), path)
    size_mb = os.path.getsize(path) / (1024 ** 2)
    print(f"Saved state dict  → {path}  ({size_mb:.2f} MB)")
    return size_mb

if __name__ == "__main__":
    model = load_resnet18()
    fp32_path  = os.path.join(SAVE_DIR, "resnet18_fp32.pt")
    state_path = os.path.join(SAVE_DIR, "resnet18_state.pth")
    fp32_size = save_torchscript(model, fp32_path)
    save_state_dict(model, state_path)
    print(f"\n✅ FP32 model ready — size: {fp32_size:.2f} MB")
    print("   Next step: python quantize/ptq.py")
