# src/visualize_data.py
# Visualization script for RFMiD DenseNet121 results
# Reads final_test_results.csv and overall_test_results.csv and creates bar charts

import os
os.environ.setdefault("MPLBACKEND", "Agg")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style for better-looking plots
plt.style.use('default')

# ----------------------
# Paths
# ----------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
FINAL_RESULTS_CSV = RESULTS_DIR / "final_test_results.csv"
OVERALL_RESULTS_CSV = RESULTS_DIR / "overall_test_results.csv"

# ----------------------
# Plotting Functions
# ----------------------
def plot_overall_metrics():
    """Plot overall test metrics as a bar chart"""
    try:
        # Read overall results
        df = pd.read_csv(OVERALL_RESULTS_CSV)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Main Performance Metrics
        main_metrics = ['test_balanced_accuracy', 'test_sensitivity', 'test_specificity', 'test_auc']
        main_values = [df[df['metric'] == m]['value'].iloc[0] for m in main_metrics]
        main_labels = ['Balanced\nAccuracy', 'Sensitivity', 'Specificity', 'AUC']
        
        bars1 = ax1.bar(main_labels, main_values, color=['#2E8B57', '#4169E1', '#DC143C', '#FF8C00'], alpha=0.8)
        ax1.set_title('Overall Test Performance Metrics', fontsize=14, fontweight='bold', pad=20)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars1, main_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Loss and Validation AUC
        loss_metrics = ['test_loss', 'best_validation_auc']
        loss_values = [df[df['metric'] == m]['value'].iloc[0] for m in loss_metrics]
        loss_labels = ['Test Loss', 'Best Validation\nAUC']
        
        bars2 = ax2.bar(loss_labels, loss_values, color=['#8B0000', '#4B0082'], alpha=0.8)
        ax2.set_title('Loss and Validation Performance', fontsize=14, fontweight='bold', pad=20)
        ax2.set_ylabel('Score', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars2, loss_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "overall_metrics_bar_chart.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("✅ Overall metrics bar chart saved to: overall_metrics_bar_chart.png")
        
    except Exception as e:
        print(f"❌ Error plotting overall metrics: {e}")

def plot_per_class_sensitivity():
    """Plot per-class sensitivity as bar chart sorted in ascending order"""
    try:
        # Read final test results
        df = pd.read_csv(FINAL_RESULTS_CSV)
        
        # Remove empty rows
        df = df.dropna()
        
        # Sort by sensitivity in ascending order
        df = df.sort_values('test_sensitivity', ascending=True)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(20, 12))
        
        # Prepare data
        diseases = df['class_name'].tolist()
        sensitivities = df['test_sensitivity'].tolist()
        
        # Create horizontal bar chart
        bars = ax.barh(range(len(diseases)), sensitivities, color='#4169E1', alpha=0.8)
        
        # Customize plot
        ax.set_xlabel('Sensitivity Score', fontsize=12)
        ax.set_ylabel('Disease Classes', fontsize=12)
        ax.set_title('Per-Class Test Sensitivity (Sorted Ascending)', fontsize=16, fontweight='bold', pad=20)
        ax.set_yticks(range(len(diseases)))
        ax.set_yticklabels(diseases, fontsize=10)
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim(0, 1)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, sensitivities)):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.3f}', ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "per_class_sensitivity_bar_chart.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("✅ Per-class sensitivity bar chart saved to: per_class_sensitivity_bar_chart.png")
        
    except Exception as e:
        print(f"❌ Error plotting per-class sensitivity: {e}")

def plot_per_class_specificity():
    """Plot per-class specificity as bar chart sorted in ascending order"""
    try:
        # Read final test results
        df = pd.read_csv(FINAL_RESULTS_CSV)
        
        # Remove empty rows
        df = df.dropna()
        
        # Sort by specificity in ascending order
        df = df.sort_values('test_specificity', ascending=True)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(20, 12))
        
        # Prepare data
        diseases = df['class_name'].tolist()
        specificities = df['test_specificity'].tolist()
        
        # Create horizontal bar chart
        bars = ax.barh(range(len(diseases)), specificities, color='#DC143C', alpha=0.8)
        
        # Customize plot
        ax.set_xlabel('Specificity Score', fontsize=12)
        ax.set_ylabel('Disease Classes', fontsize=12)
        ax.set_title('Per-Class Test Specificity (Sorted Ascending)', fontsize=16, fontweight='bold', pad=20)
        ax.set_yticks(range(len(diseases)))
        ax.set_yticklabels(diseases, fontsize=10)
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim(0, 1)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, specificities)):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.3f}', ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "per_class_specificity_bar_chart.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("✅ Per-class specificity bar chart saved to: per_class_specificity_bar_chart.png")
        
    except Exception as e:
        print(f"❌ Error plotting per-class specificity: {e}")

def plot_top_performing_diseases():
    """Plot top 10 performing diseases by balanced accuracy"""
    try:
        # Read final test results
        df = pd.read_csv(FINAL_RESULTS_CSV)
        df = df.dropna()
        
        # Calculate balanced accuracy
        df['balanced_accuracy'] = (df['test_sensitivity'] + df['test_specificity']) / 2
        
        # Get top 10 diseases
        top_10 = df.nlargest(10, 'balanced_accuracy')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create horizontal bar chart
        bars = ax.barh(range(len(top_10)), top_10['balanced_accuracy'], 
                      color='#2E8B57', alpha=0.8)
        
        # Customize plot
        ax.set_xlabel('Balanced Accuracy', fontsize=12)
        ax.set_ylabel('Disease Classes', fontsize=12)
        ax.set_title('Top 10 Performing Diseases (by Balanced Accuracy)', fontsize=14, fontweight='bold', pad=20)
        ax.set_yticks(range(len(top_10)))
        ax.set_yticklabels(top_10['class_name'], fontsize=10)
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim(0, 1)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, top_10['balanced_accuracy'])):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.3f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "top_10_diseases_bar_chart.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("✅ Top 10 diseases bar chart saved to: top_10_diseases_bar_chart.png")
        
    except Exception as e:
        print(f"❌ Error plotting top diseases: {e}")

def plot_specificity_distribution():
    """Plot distribution of specificity scores"""
    try:
        # Read final test results
        df = pd.read_csv(FINAL_RESULTS_CSV)
        df = df.dropna()
        
        # Create figure with single plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram of specificity
        ax.hist(df['test_specificity'], bins=15, color='#DC143C', alpha=0.7, edgecolor='black')
        ax.axvline(df['test_specificity'].mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {df["test_specificity"].mean():.3f}')
        ax.axvline(0.9, color='green', linestyle='--', linewidth=2, 
                   label='90% Target')
        ax.set_xlabel('Specificity', fontsize=12)
        ax.set_ylabel('Number of Diseases', fontsize=12)
        ax.set_title('Distribution of Specificity Scores', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "specificity_distribution_chart.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("✅ Specificity distribution chart saved to: specificity_distribution_chart.png")
        
    except Exception as e:
        print(f"❌ Error plotting specificity distribution: {e}")


def main():
    """Main function to generate all visualizations"""
    print("🎨 Starting visualization of RFMiD DenseNet121 results...")
    
    # Check if files exist
    if not FINAL_RESULTS_CSV.exists():
        print(f"❌ File not found: {FINAL_RESULTS_CSV}")
        return
    
    if not OVERALL_RESULTS_CSV.exists():
        print(f"❌ File not found: {OVERALL_RESULTS_CSV}")
        return
    
    print(f"📊 Reading data from:")
    print(f"   - {FINAL_RESULTS_CSV}")
    print(f"   - {OVERALL_RESULTS_CSV}")
    
    # Generate all visualizations
    print("\n📈 Generating visualizations...")
    
    plot_overall_metrics()
    plot_per_class_sensitivity()
    plot_per_class_specificity()
    plot_top_performing_diseases()
    plot_specificity_distribution()
    
    print(f"\n🎉 All visualizations completed!")
    print(f"📁 Charts saved to: {RESULTS_DIR}")
    print("\nGenerated files:")
    print("   - overall_metrics_bar_chart.png")
    print("   - per_class_sensitivity_bar_chart.png")
    print("   - per_class_specificity_bar_chart.png")
    print("   - top_10_diseases_bar_chart.png")
    print("   - specificity_distribution_chart.png")

if __name__ == "__main__":
    main()
