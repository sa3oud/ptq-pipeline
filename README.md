# PTQ Pipeline — Post-Training Quantization on Apple Silicon

> How do you make a giant model small enough to fit on a chip?
> This project answers that question with real benchmarks, weight analysis,
> and a conceptual bridge between neural networks and human cognition.

---

## What This Project Does

Takes a pretrained ResNet18 (11 million weights, 44MB) and compresses it
to INT8 precision using Post-Training Quantization — then benchmarks the
real-world trade-off between size, speed, and accuracy on Apple Silicon CPU.

---

## Results

| Model | Size (MB) | Latency (ms) | Reduction |
|-------|-----------|--------------|-----------|
| FP32  | 44.72     | 18.85        | baseline  |
| INT8  | 43.20     | 17.22        | 1.09x faster |

> Dynamic quantization targets Linear layers only.
> Full static quantization of Conv layers would yield ~4x size reduction.

---

## Key Concepts

### What is a Weight?
A neural network is 11 million numbers (weights) chained through
multiplication. Each weight encodes "how much this neuron cares about
this input." Training is the process of finding the right values.

### What is Quantization?
Swapping 32-bit floats (FP32) for 8-bit integers (INT8):
```
FP32: 0.38472910  →  4 bytes per weight  →  44 MB total
INT8: 61           →  1 byte per weight   →  11 MB total (theoretical)

formula: INT8 = round(FP32 / scale) + zero_point
```

### The Human Analogy
Weights in a neural network work exactly like attention in a human brain:

| Human Brain         | Neural Network         | What It Means                     |
|---------------------|------------------------|-----------------------------------|
| Synapse strength    | Weight value           | How strongly a connection fires   |
| Learning from life  | Gradient descent       | Adjusting based on mistakes       |
| Loving a subject    | High learning rate     | Weights update faster             |
| Expert intuition    | High-magnitude weights | Key features strongly activated   |
| Forgetting          | Weight decay           | Unused weights shrink to zero     |
| Paying attention    | Attention mechanism    | Transformer models (GPT, Claude)  |
| Natural talent      | Weight initialization  | Starting point before training    |
| Bad at something    | Near-zero weights      | That domain was never trained     |

A concert pianist and a mathematician have the same brain architecture.
What differs is which weights got trained — and how much.

---

## Project Structure
```
ptq-pipeline/
├── model/
│   └── prepare.py          # Download ResNet18, save FP32 baseline
├── quantize/
│   └── ptq.py              # Apply dynamic INT8 quantization
├── benchmark/
│   └── compare.py          # Measure size + latency FP32 vs INT8
├── profile/
│   ├── report.py           # Generate benchmark dashboard
│   └── attention_explainer.py  # Weight distributions + human analogy
└── results/
    ├── resnet18_fp32.pt         # FP32 model
    ├── resnet18_int8.pth        # INT8 model
    ├── benchmark.csv            # Raw benchmark numbers
    ├── ptq_dashboard.png        # Size + latency chart
    └── attention_explainer.png  # Weight analysis + human analogy
```

---

## How to Run

### 1. Install dependencies
```bash
pip3 install torch==2.3.0 torchvision==0.18.0 \
  --index-url https://download.pytorch.org/whl/cpu
pip3 install numpy pandas matplotlib seaborn tqdm Pillow
```

### 2. Run the full pipeline
```bash
python3 model/prepare.py          # Download + save FP32 model
python3 quantize/ptq.py           # Quantize to INT8
python3 benchmark/compare.py      # Benchmark both models
python3 profile/report.py         # Generate dashboard
python3 profile/attention_explainer.py  # Generate explainer
```

---

## What I Learned (Interview Notes)

### On Apple Silicon compatibility
PyTorch's default quantization engine is `fbgemm` (x86 optimized).
On ARM (Apple Silicon) you must explicitly set:
```python
torch.backends.quantized.engine = 'qnnpack'
```
This is directly analogous to embedded targets like NXP i.MX —
the quantization kernel must match the hardware instruction set.

### On quantize_dynamic vs static quantization
- **Dynamic**: quantizes weights at load time, activations at runtime.
  Works on Linear layers, no calibration needed. Small gains on ResNet18.
- **Static**: quantizes both weights AND activations using calibration data.
  Requires a representative dataset. Targets Conv layers. Gives 4x gains.
- **QAT**: Quantization-Aware Training — simulates INT8 during training.
  Best accuracy, most expensive to run.

### On the size vs accuracy trade-off
Every bit you remove from a weight is precision you're trading for speed.
The art of edge AI deployment is finding the minimum precision that keeps
accuracy above your application threshold — identical to signal quantization
in embedded DSP systems.

---

## Hardware Context
- **Tested on**: MacBook Air M-series (Apple Silicon, CPU-only)
- **Target deployment**: ARM Cortex-A / NXP i.MX / Raspberry Pi class devices
- **Quantization engine**: qnnpack (ARM-optimized)

---

## Tech Stack
- Python 3.9
- PyTorch 2.3.0
- torchvision 0.18.0
- NumPy / Pandas / Matplotlib / Seaborn

---

## Next Steps
- [ ] Static quantization on Conv layers (target: 4x size reduction)
- [ ] Add FP16 as third comparison point
- [ ] Run on DistilBERT (Linear-heavy — dramatic INT8 gains)
- [ ] Deploy to Raspberry Pi and measure real edge latency
- [ ] Implement custom INT8 quantizer in NumPy from scratch
