#!/usr/bin/env python3
"""
Bar chart showing AUC scores for all models.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load the models results data
base_dir = Path(__file__).parent.parent
data_file = base_dir / "all_models_results.csv"

# Read the data
df = pd.read_csv(data_file)

# Convert AUC from 0-1 range to percentage (0-100)
df['AUC (%)'] = df['AUC'] * 100

# Sort by AUC in ascending order (lowest to highest)
df_sorted = df.sort_values(by="AUC (%)", ascending=True).reset_index(drop=True)

# Extract data from sorted dataframe
models = df_sorted["模型"].values  # 使用中文列名
auc_scores = df_sorted["AUC (%)"].values

# Set up the plot
fig, ax = plt.subplots(figsize=(14, 8))

# Set up bar positions
x = np.arange(len(models))
width = 0.6  # Width of the bars

# Create bars
bars = ax.bar(x, auc_scores, width, label='AUC Score', 
              color='#43A047', alpha=1.0, edgecolor='#1B5E20', linewidth=1.2)

# Customize the plot
ax.set_xlabel('Model', fontsize=14, fontweight='bold')
ax.set_ylabel('AUC (%)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right', fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

# Set y-axis range to better show differences
min_auc = auc_scores.min()
max_auc = auc_scores.max()
y_margin = (max_auc - min_auc) * 0.1
ax.set_ylim([max(0, min_auc - y_margin), min(100, max_auc + y_margin)])

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
output_file = output_dir / "model_auc_comparison.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✅ Chart saved to: {output_file}")

# Also save as PDF for high quality
output_file_pdf = output_dir / "model_auc_comparison.pdf"
plt.savefig(output_file_pdf, bbox_inches='tight')
print(f"✅ Chart saved to: {output_file_pdf}")

# Show the plot
plt.show()

