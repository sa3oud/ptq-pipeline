import torch
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
from torchvision.datasets import FakeData
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load models ──────────────────────────────────────────────────
print("Loading models...")
trained   = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).eval()
untrained = models.resnet18(weights=None).eval()  # random init = "no experience"

# ── 2. Extract weights from key layers ──────────────────────────────
def get_weights(model, layer_name):
    for name, module in model.named_modules():
        if name == layer_name:
            return module.weight.data.numpy().flatten()
    return None

layers = {
    'layer1.0.conv1': 'Early Layer\n(Basic Edges)',
    'layer2.0.conv1': 'Middle Layer\n(Textures)',
    'layer4.0.conv1': 'Deep Layer\n(Complex Shapes)',
    'fc':             'Final Layer\n(Decision)'
}

# ── 3. Build the full figure ─────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#0f0f1a')

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.4)

# ── Title ────────────────────────────────────────────────────────────
fig.text(0.5, 0.97,
         'How Neural Networks Learn to Pay Attention',
         ha='center', va='top', fontsize=18, fontweight='bold',
         color='white')
fig.text(0.5, 0.935,
         'Trained weights concentrate around meaningful values  ·  '
         'Untrained weights are random noise  ·  '
         'Just like expertise vs ignorance in humans',
         ha='center', va='top', fontsize=10, color='#aaaacc', style='italic')

# ── Row 0: Weight distributions ─────────────────────────────────────
colors_trained   = '#4fc3f7'
colors_untrained = '#ef9a9a'

for col, (layer_name, label) in enumerate(layers.items()):
    ax = fig.add_subplot(gs[0, col])
    ax.set_facecolor('#1a1a2e')

    w_trained   = get_weights(trained,   layer_name)
    w_untrained = get_weights(untrained, layer_name)

    if w_trained is not None:
        ax.hist(w_trained,   bins=60, alpha=0.75,
                color=colors_trained,   label='Trained',   density=True)
        ax.hist(w_untrained, bins=60, alpha=0.55,
                color=colors_untrained, label='Untrained', density=True)

    ax.set_title(label, color='white', fontsize=9, pad=8)
    ax.set_xlabel('Weight value', color='#aaaacc', fontsize=7)
    ax.set_ylabel('Density',      color='#aaaacc', fontsize=7)
    ax.tick_params(colors='#aaaacc', labelsize=6)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')
    if col == 0:
        ax.legend(fontsize=7, facecolor='#1a1a2e',
                  labelcolor='white', framealpha=0.8)

# ── Row 1: Human analogy bars ────────────────────────────────────────
ax_musician = fig.add_subplot(gs[1, 0:2])
ax_musician.set_facecolor('#1a1a2e')

skills      = ['Rhythm', 'Melody', 'Harmony', 'Math', 'Geography', 'Cooking']
musician_w  = [0.95, 0.91, 0.87, 0.04, 0.02, 0.11]
random_w    = [0.48, 0.51, 0.46, 0.52, 0.49, 0.47]

x = np.arange(len(skills))
w = 0.35
bars1 = ax_musician.bar(x - w/2, musician_w, w,
                         label='Expert Musician',  color='#4fc3f7', alpha=0.85)
bars2 = ax_musician.bar(x + w/2, random_w,  w,
                         label='No Experience',    color='#ef9a9a', alpha=0.65)

ax_musician.set_title('Human Analogy — Attention Weights by Domain',
                       color='white', fontsize=10, pad=8)
ax_musician.set_xticks(x)
ax_musician.set_xticklabels(skills, color='#aaaacc', fontsize=8)
ax_musician.set_ylabel('Attention Weight (0=ignore, 1=focus)',
                        color='#aaaacc', fontsize=8)
ax_musician.set_ylim(0, 1.1)
ax_musician.tick_params(colors='#aaaacc')
ax_musician.legend(fontsize=8, facecolor='#1a1a2e',
                   labelcolor='white', framealpha=0.8)
for spine in ax_musician.spines.values():
    spine.set_edgecolor('#333355')
for bar in bars1:
    h = bar.get_height()
    ax_musician.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                     f'{h:.2f}', ha='center', va='bottom',
                     color='#4fc3f7', fontsize=6)

# ── Row 1 right: learning rate analogy ──────────────────────────────
ax_learn = fig.add_subplot(gs[1, 2:4])
ax_learn.set_facecolor('#1a1a2e')

epochs       = np.arange(0, 50)
love_curve   = 1 - np.exp(-epochs / 5)   + np.random.normal(0, 0.01, 50)
hate_curve   = 1 - np.exp(-epochs / 40)  + np.random.normal(0, 0.01, 50)
random_curve = np.ones(50) * 0.1         + np.random.normal(0, 0.02, 50)

ax_learn.plot(epochs, np.clip(love_curve,  0, 1), color='#4fc3f7',
              linewidth=2.5, label='Subject you love (fast weight updates)')
ax_learn.plot(epochs, np.clip(hate_curve,  0, 1), color='#ffcc80',
              linewidth=2.5, label='Subject you dislike (slow weight updates)')
ax_learn.plot(epochs, np.clip(random_curve,0, 1), color='#ef9a9a',
              linewidth=2.5, label='No exposure (weights stay random)',
              linestyle='--')

ax_learn.set_title('Learning Speed = Emotional Engagement\n'
                   '(Humans & Neural Networks)',
                   color='white', fontsize=10, pad=8)
ax_learn.set_xlabel('Experience / Training Epochs', color='#aaaacc', fontsize=8)
ax_learn.set_ylabel('Skill Level / Weight Quality', color='#aaaacc', fontsize=8)
ax_learn.set_ylim(0, 1.1)
ax_learn.tick_params(colors='#aaaacc')
ax_learn.legend(fontsize=7.5, facecolor='#1a1a2e',
                labelcolor='white', framealpha=0.8)
for spine in ax_learn.spines.values():
    spine.set_edgecolor('#333355')

# ── Row 2: Comparison table ──────────────────────────────────────────
ax_table = fig.add_subplot(gs[2, :])
ax_table.set_facecolor('#1a1a2e')
ax_table.axis('off')

table_data = [
    ['Synapse strength',      'Weight value',         'How strongly a connection fires'],
    ['Learning from life',    'Gradient descent',     'Adjusting based on mistakes'],
    ['Loving a subject',      'High learning rate',   'Weights update faster'],
    ['Expert intuition',      'High-magnitude weights','Key features strongly activated'],
    ['Forgetting',            'Weight decay',         'Unused weights shrink to zero'],
    ['Paying attention',      'Attention mechanism',  'Transformer models (GPT, Claude)'],
    ['Natural talent',        'Weight initialization','Starting point before training'],
    ['Being bad at something','Near-zero weights',    'That domain was never trained'],
]

col_labels = ['🧠  Human Brain', '🤖  Neural Network', '💡  What It Means']
colors_row  = ['#4fc3f7', '#81c784', '#ffcc80']

table = ax_table.table(
    cellText=table_data,
    colLabels=col_labels,
    loc='center',
    cellLoc='left'
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.7)

for (row, col), cell in table.get_celld().items():
    cell.set_facecolor('#1a1a2e' if row > 0 else '#16213e')
    cell.set_edgecolor('#333355')
    cell.set_text_props(color=colors_row[col] if row == 0 else 'white')

ax_table.set_title('The Deep Connection — Human Cognition vs Neural Networks',
                   color='white', fontsize=11, pad=12, fontweight='bold')

# ── Save ─────────────────────────────────────────────────────────────
out = 'results/attention_explainer.png'
plt.savefig(out, dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print(f"✅ Saved → {out}")
plt.show()
