import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

# Define the encoder architecture blocks based on model.py file
# For illustration purposes, this is a simplified layout

layers = [
    ("Input (1x256x256)", "input"),
    ("ConvBlock 1 (Conv+ResBlock)", "conv1"),
    ("ConvBlock 2", "conv2"),
    ("ConvBlock 3", "conv3"),
    ("ConvBlock 4", "conv4"),
    ("ConvBlock 5", "conv5"),
    ("ConvBlock 6", "conv6"),
    ("ConvBlock 7", "conv7"),
    ("ConvBlock 8", "conv8"),
    ("Minibatch Std + Flatten", "stddev_flatten"),
    ("Linear (512 → w)", "linear")
]

# Layout config
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 12)
ax.set_ylim(0, len(layers) + 1)
ax.axis('off')

# Plot blocks and arrows
for i, (label, name) in enumerate(reversed(layers)):
    y = i + 1
    rect = Rectangle((3, y), 6, 0.8, edgecolor='black', facecolor='lightblue', lw=2)
    ax.add_patch(rect)
    ax.text(6, y + 0.4, label, ha='center', va='center', fontsize=10)
    
    if i < len(layers) - 1:
        arrow = FancyArrow(6, y + 0.8, 0, 0.4, width=0.1, head_width=0.3, head_length=0.2, length_includes_head=True)
        ax.add_patch(arrow)

plt.title("StyleGAN2 Encoder Architecture (Grayscale Input)", fontsize=14)
plt.tight_layout()
plt.show()

