#!/bin/bash
echo "Setting up PTQ Pipeline environment..."

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas matplotlib seaborn tqdm Pillow

echo ""
echo "✅ Setup complete. Run the pipeline with:"
echo "   python model/prepare.py"
echo "   python quantize/ptq.py"
echo "   python benchmark/compare.py"
echo "   python profile/report.py"
