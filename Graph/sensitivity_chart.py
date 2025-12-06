#!/usr/bin/env python3
"""
Bar chart showing Sensitivity at 80% Specificity threshold for all 12 models.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load the models results data
base_dir = Path(__file__).parent.parent
results_dir = base_dir / "results"

# Define all 12 models with their paths
MODELS = {
    "Densenet121": results_dir / "PAD_UFES20" / "Densenet121",
    "EfficientNetB3": results_dir / "PAD_UFES20" / "EfficientNetB3",
    "InceptionV3": results_dir / "PAD_UFES20" / "InceptionV3",
    "ResNet50": results_dir / "PAD_UFES20" / "ResNet50",
    "CoAtNet0": results_dir / "PAD_UFES20_Hybrid" / "CoAtNet0",
    "MaxViTTiny": results_dir / "PAD_UFES20_Hybrid" / "MaxViTTiny",
    "CrossViTSmall": results_dir / "PAD_UFES20_ViT" / "CrossViTSmall",
    "DeiTSmall": results_dir / "PAD_UFES20_ViT" / "DeiTSmall",
    "SwinTiny": results_dir / "PAD_UFES20_ViT" / "SwinTiny",
    "ViTSmall": results_dir / "PAD_UFES20_ViT" / "ViTSmall",
    "CLIPViTB16": results_dir / "PAD_UFES20_VLM" / "CLIPViTB16",
    "SigLIPBase384": results_dir / "PAD_UFES20_VLM" / "SigLIPBase384",
}

# Extract sensitivity at 80% specificity from overall_test_results.csv files
models_list = []
sensitivity_80spec = []

for model_name, model_path in MODELS.items():
    test_results_file = model_path / "overall_test_results.csv"
    
    if test_results_file.exists():
        try:
            df_test = pd.read_csv(test_results_file)
            test_dict = dict(zip(df_test.iloc[:, 0], df_test.iloc[:, 1]))
            # Get sensitivity at 80% specificity
            sens = float(test_dict.get("posthoc_sensitivity_at_80pct_specificity_test", 0)) * 100  # Convert to percentage
            models_list.append(model_name)
            sensitivity_80spec.append(sens)
        except Exception as e:
            print(f"⚠️  Warning: Could not load sensitivity for {model_name}: {e}")
    else:
        print(f"⚠️  Warning: {test_results_file} not found for {model_name}")

# Convert to numpy arrays
models_array = np.array(models_list)
sensitivity_array = np.array(sensitivity_80spec)

# Sort by sensitivity in ascending order (lowest to highest)
sort_idx = np.argsort(sensitivity_array)
models_sorted = models_array[sort_idx]
sensitivity_sorted = sensitivity_array[sort_idx]

# Set up the plot
fig, ax = plt.subplots(figsize=(14, 8))

# Set up bar positions
x = np.arange(len(models_sorted))
width = 0.6  # Width of the bars

# Create bars
bars = ax.bar(x, sensitivity_sorted, width, label='Sensitivity at 80% Specificity', 
              color='#43A047', alpha=1.0, edgecolor='#1B5E20', linewidth=1.2)

# Customize the plot
ax.set_xlabel('Model', fontsize=14, fontweight='bold')
ax.set_ylabel('Sensitivity (%)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models_sorted, rotation=45, ha='right', fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

# Set y-axis range to better show differences
min_sens = sensitivity_sorted.min()
max_sens = sensitivity_sorted.max()
y_margin = (max_sens - min_sens) * 0.1
ax.set_ylim([max(0, min_sens - y_margin), min(100, max_sens + y_margin)])

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add value labels on bars
def add_value_labels(bars):
    """Add value labels on top of bars."""
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

add_value_labels(bars)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the figure
output_dir = base_dir / "Graph"
output_dir.mkdir(exist_ok=True)
output_file = output_dir / "sensitivity_comparison.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✅ Chart saved to: {output_file}")

# Also save as PDF for high quality
output_file_pdf = output_dir / "sensitivity_comparison.pdf"
plt.savefig(output_file_pdf, bbox_inches='tight')
print(f"✅ Chart saved to: {output_file_pdf}")

# Show the plot
plt.show()

