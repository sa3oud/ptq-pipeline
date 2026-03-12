import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import FakeData
from torch.utils.data import DataLoader
import time, os
import pandas as pd
import numpy as np

torch.backends.quantized.engine = 'qnnpack'

SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

def get_loader():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return DataLoader(FakeData(size=100, image_size=(3,224,224), transform=transform), batch_size=1)

def measure_latency(model, loader, runs=50):
    model.eval()
    latencies = []
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= runs:
                break
            start = time.perf_counter()
            model(images)
            latencies.append((time.perf_counter() - start) * 1000)
    return np.mean(latencies), np.std(latencies)

def file_size_mb(path):
    return os.path.getsize(path) / (1024**2)

if __name__ == "__main__":
    loader = get_loader()

    # FP32
    print("Benchmarking FP32...")
    fp32 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).eval()
    fp32_lat, fp32_std = measure_latency(fp32, loader)
    fp32_size = file_size_mb(os.path.join(SAVE_DIR, "resnet18_fp32.pt"))

    # INT8
    print("Benchmarking INT8...")
    int8 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).eval()
    int8 = torch.quantization.quantize_dynamic(int8, {torch.nn.Linear}, dtype=torch.qint8)
    int8_size = file_size_mb(os.path.join(SAVE_DIR, "resnet18_int8.pth"))
    int8_lat, int8_std = measure_latency(int8, loader)

    results = pd.DataFrame({
        "Model":          ["FP32", "INT8"],
        "Size (MB)":      [round(fp32_size, 2), round(int8_size, 2)],
        "Latency (ms)":   [round(fp32_lat, 2),  round(int8_lat, 2)],
        "Latency Std":    [round(fp32_std, 2),  round(int8_std, 2)],
        "Size Reduction": ["baseline", f"{(1 - int8_size/fp32_size)*100:.1f}% smaller"],
        "Speedup":        ["baseline", f"{fp32_lat/int8_lat:.2f}x faster"]
    })

    print("\n── Benchmark Results ──────────────────────")
    print(results.to_string(index=False))
    csv_path = os.path.join(SAVE_DIR, "benchmark.csv")
    results.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved → {csv_path}")
    print("   Next step: python3 profile/report.py")
