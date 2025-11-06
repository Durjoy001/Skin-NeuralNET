#!/usr/bin/env python3
"""
Statistical Tests for CNN Architecture Comparison
- Delong Test: Compare AUC differences between architectures
- McNemar Test: Compare classification accuracy differences
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import itertools

def delong_test(y_true, y_scores_dict, alpha=0.05):
    """
    Perform Delong test to compare AUCs between multiple models
    """
    print("=" * 60)
    print("DELONG TEST RESULTS")
    print("=" * 60)
    print("Testing for statistically significant differences in AUC between architectures")
    print(f"Significance level: α = {alpha}")
    print()
    
    # Calculate AUCs
    aucs = {}
    for model_name, y_score in y_scores_dict.items():
        auc = roc_auc_score(y_true, y_score)
        aucs[model_name] = auc
        print(f"{model_name:15s}: AUC = {auc:.4f}")
    
    print()
    
    # Perform pairwise Delong tests
    models = list(y_scores_dict.keys())
    results = []
    
    for i, j in itertools.combinations(range(len(models)), 2):
        model1, model2 = models[i], models[j]
        
        # Calculate AUC difference
        auc_diff = aucs[model1] - aucs[model2]
        
        # Simplified Delong test (using bootstrap approximation)
        # In practice, you'd use the full Delong implementation
        n = len(y_true)
        
        # Bootstrap sampling for confidence interval
        n_bootstrap = 1000
        auc_diffs = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(n, n, replace=True)
            auc1_boot = roc_auc_score(y_true[indices], y_scores_dict[model1][indices])
            auc2_boot = roc_auc_score(y_true[indices], y_scores_dict[model2][indices])
            auc_diffs.append(auc1_boot - auc2_boot)
        
        auc_diffs = np.array(auc_diffs)
        se = np.std(auc_diffs)
        z_score = auc_diff / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # Confidence interval
        ci_lower = np.percentile(auc_diffs, 2.5)
        ci_upper = np.percentile(auc_diffs, 97.5)
        
        results.append({
            'Model1': model1,
            'Model2': model2,
            'AUC_Diff': auc_diff,
            'SE': se,
            'Z_Score': z_score,
            'P_Value': p_value,
            'CI_Lower': ci_lower,
            'CI_Upper': ci_upper,
            'Significant': p_value < alpha
        })
        
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        print(f"{model1} vs {model2}:")
        print(f"  AUC Difference: {auc_diff:+.4f}")
        print(f"  Standard Error: {se:.4f}")
        print(f"  Z-Score: {z_score:+.3f}")
        print(f"  P-Value: {p_value:.4f} {significance}")
        print(f"  95% CI: [{ci_lower:+.4f}, {ci_upper:+.4f}]")
        print(f"  Significant: {'Yes' if p_value < alpha else 'No'}")
        print()
    
    return pd.DataFrame(results)

def mcnemar_test(y_true, y_preds_dict, alpha=0.05):
    """
    Perform McNemar test to compare classification accuracies between models
    """
    print("=" * 60)
    print("MCNEMAR TEST RESULTS")
    print("=" * 60)
    print("Testing for statistically significant differences in classification accuracy")
    print(f"Significance level: α = {alpha}")
    print()
    
    # Calculate accuracies
    accuracies = {}
    for model_name, y_pred in y_preds_dict.items():
        acc = accuracy_score(y_true, y_pred)
        accuracies[model_name] = acc
        print(f"{model_name:15s}: Accuracy = {acc:.4f}")
    
    print()
    
    # Perform pairwise McNemar tests
    models = list(y_preds_dict.keys())
    results = []
    
    for i, j in itertools.combinations(range(len(models)), 2):
        model1, model2 = models[i], models[j]
        
        y_pred1 = y_preds_dict[model1]
        y_pred2 = y_preds_dict[model2]
        
        # Create contingency table
        # b: model1 correct, model2 incorrect
        # c: model1 incorrect, model2 correct
        b = np.sum((y_pred1 == y_true) & (y_pred2 != y_true))
        c = np.sum((y_pred1 != y_true) & (y_pred2 == y_true))
        
        # McNemar test statistic
        if b + c == 0:
            chi2_stat = 0
            p_value = 1.0
        else:
            chi2_stat = (b - c)**2 / (b + c)
            p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)
        
        # Accuracy difference
        acc_diff = accuracies[model1] - accuracies[model2]
        
        results.append({
            'Model1': model1,
            'Model2': model2,
            'Acc_Diff': acc_diff,
            'b': b,
            'c': c,
            'Chi2_Stat': chi2_stat,
            'P_Value': p_value,
            'Significant': p_value < alpha
        })
        
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        print(f"{model1} vs {model2}:")
        print(f"  Accuracy Difference: {acc_diff:+.4f}")
        print(f"  Model1 correct, Model2 incorrect (b): {b}")
        print(f"  Model1 incorrect, Model2 correct (c): {c}")
        print(f"  Chi-square statistic: {chi2_stat:.3f}")
        print(f"  P-Value: {p_value:.4f} {significance}")
        print(f"  Significant: {'Yes' if p_value < alpha else 'No'}")
        print()
    
    return pd.DataFrame(results)

def main():
    # Load data for all architectures
    architectures = ['Densenet121', 'EfficientNetB3', 'InceptionV3', 'ResNet50']
    
    y_scores = {}
    y_preds = {}
    y_true = None
    
    print("Loading test data for all architectures...")
    
    for arch in architectures:
        try:
            data = np.load(f'results/CNN/{arch}/cnn_anyabnormal_test_outputs.npz')
            
            if y_true is None:
                y_true = data['y_true']
            
            y_scores[arch] = data['y_score']
            y_preds[arch] = data['y_pred_at_spec80']
            
            print(f"✓ Loaded {arch}: {len(data['y_score'])} samples")
            
        except Exception as e:
            print(f"✗ Error loading {arch}: {e}")
    
    if len(y_scores) < 2:
        print("Error: Need at least 2 architectures for comparison")
        return
    
    print(f"\nTotal test samples: {len(y_true)}")
    print(f"Architectures loaded: {list(y_scores.keys())}")
    print()
    
    # Run Delong test
    delong_results = delong_test(y_true, y_scores)
    
    # Run McNemar test  
    mcnemar_results = mcnemar_test(y_true, y_preds)
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("\nDelong Test Summary:")
    significant_delong = delong_results[delong_results['Significant'] == True]
    if len(significant_delong) > 0:
        print("Significant AUC differences found:")
        for _, row in significant_delong.iterrows():
            print(f"  {row['Model1']} vs {row['Model2']}: p = {row['P_Value']:.4f}")
    else:
        print("No significant AUC differences found between architectures")
    
    print("\nMcNemar Test Summary:")
    significant_mcnemar = mcnemar_results[mcnemar_results['Significant'] == True]
    if len(significant_mcnemar) > 0:
        print("Significant accuracy differences found:")
        for _, row in significant_mcnemar.iterrows():
            print(f"  {row['Model1']} vs {row['Model2']}: p = {row['P_Value']:.4f}")
    else:
        print("No significant accuracy differences found between architectures")
    
    # Save results
    delong_results.to_csv('delong_test_results.csv', index=False)
    mcnemar_results.to_csv('mcnemar_test_results.csv', index=False)
    print(f"\nResults saved to:")
    print(f"  - delong_test_results.csv")
    print(f"  - mcnemar_test_results.csv")

if __name__ == "__main__":
    main()
