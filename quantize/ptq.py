import torch
import torchvision.models as models
import os

torch.backends.quantized.engine = 'qnnpack'

SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(SAVE_DIR, exist_ok=True)

if __name__ == "__main__":
    print("Loading ResNet18 (FP32)...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()

    print("Applying dynamic INT8 quantization (qnnpack)...")
    int8_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    out_path = os.path.join(SAVE_DIR, "resnet18_int8.pth")
    torch.save(int8_model.state_dict(), out_path)
    size_mb = os.path.getsize(out_path) / (1024**2)
    print(f"✅ INT8 model saved → {out_path}  ({size_mb:.2f} MB)")
    print("   Next step: python3 benchmark/compare.py")
