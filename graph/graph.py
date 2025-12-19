import os

from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import seaborn as sns

# Set style for a clean, professional look
sns.set_style("white")
plt.rcParams["font.size"] = 16
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 24
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 18

# Data
models = ["Gemini 2.5", "GPT5", "Grok4", "Latxa 70B 4bit"]
categories = ["Ciencia", "Política", "Igualdad"]

# Percentages for each model and category
data = {
    "Ciencia": [92.7, 83.8, 93.2, 94.0],
    "Política": [82.2, 62.5, 66.7, 78.1],
    "Igualdad": [95.9, 62.3, 84.3, 80.6],
}

# Color palette - using distinct, professional colors
colors = ["#4285F4", "#EA4335", "#FBBC04"]  # Blue, Red, Yellow

# Create figure and axis with extra space at bottom for logos
fig, ax = plt.subplots(figsize=(14, 9), facecolor="none")

# Set positions for bars
x = np.arange(len(models))
width = 0.25  # Width of each bar

# Create bars for each category
bars = []
for i, (category, values) in enumerate(data.items()):
    offset = (i - 1) * width
    bar = ax.bar(
        x + offset,
        values,
        width,
        label=category,
        color=colors[i],
        alpha=0.85,
        edgecolor="black",
        linewidth=1.2,
    )
    bars.append(bar)

    # Add value labels on top of bars
    for j, bar_rect in enumerate(bar):
        height = bar_rect.get_height()
        ax.text(
            bar_rect.get_x() + bar_rect.get_width() / 2.0,
            height + 1,
            f"{values[j]:.1f}%",
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
        )

# Customize the plot
# ax.set_title('Comparación de Rendimiento por Modelo y Categoría',
#             fontweight='bold', fontsize=24, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=18, fontweight="bold")
ax.set_ylim(0, 110)
ax.legend(
    loc="upper right",
    framealpha=0.9,
    fontsize=18,
    edgecolor="black",
    fancybox=True,
    shadow=True,
    facecolor="white",
)

# Remove grid and axes spines for a cleaner look
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)

# Remove y-axis ticks and labels for minimal look
ax.set_yticks([])
ax.set_ylabel("")

# Set transparent background for axes
ax.patch.set_alpha(0)


# Function to add logos under the x-axis labels
def add_logo(logo_path, x_position, zoom=0.10):
    """Add a logo image to the plot"""
    try:
        img = Image.open(logo_path)
        # Convert to RGBA to handle different image formats
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA")
        imagebox = OffsetImage(img, zoom=zoom)
        # Position logos below the x-axis using axes coordinates for y
        ab = AnnotationBbox(
            imagebox,
            (x_position, -0.18),
            frameon=False,
            box_alignment=(0.5, 0.5),
            xycoords=("data", "axes fraction"),
            pad=0,
        )
        ax.add_artist(ab)
        print(f"Added logo: {os.path.basename(logo_path)}")
    except Exception as e:
        print(f"Could not load logo {logo_path}: {e}")


# Logo paths and positions
script_dir = os.path.dirname(os.path.abspath(__file__))
logo_info = [
    (os.path.join(script_dir, "gemini-app-icon-hd_processed.png"), 0, 0.20),
    (os.path.join(script_dir, "ChatGPT-Logo_processed.png"), 1, 0.20),
    (os.path.join(script_dir, "grok-seeklogo_processed.png"), 2, 0.20),
    (os.path.join(script_dir, "latxa_processed.png"), 3, 0.20),
]

# Adjust layout to make room for logos BEFORE adding them
plt.tight_layout()
plt.subplots_adjust(bottom=0.20)

# Add logos below the x-axis
for logo_file, x_pos, zoom_level in logo_info:
    add_logo(logo_file, x_pos, zoom=zoom_level)

# Save the figure in high resolution for PowerPoint
output_path = os.path.join(script_dir, "model_comparison.png")
# Use pad_inches to ensure logos aren't cut off, transparent=True for no background
plt.savefig(output_path, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.5)
print(f"Graph saved as '{output_path}'")

# Display the plot
plt.show()
