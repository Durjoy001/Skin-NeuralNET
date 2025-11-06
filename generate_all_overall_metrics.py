#!/usr/bin/env python3
"""
Generate complete overall test results for all 4 CNN models (ResNet50, DenseNet121, EfficientNetB3, InceptionV3).
Loads existing trained models and generates comprehensive overall test metrics.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm  # kept in case you want progress bars later

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))


def _remap_state_dict_keys(state_dict: dict) -> dict:
    """
    Make checkpoints compatible across older/newer model wrappers/heads.
    - Strip 'module.' from DataParallel
    - Strip 'backbone.' from older wrapper
    - EfficientNet old head: classifier.1.1 -> classifier.1, classifier.1.4 -> classifier.4
    """
    new_sd = {}
    for k, v in state_dict.items():
        k2 = k

        # 1) DataParallel prefix
        if k2.startswith("module."):
            k2 = k2[len("module.") :]

        # 2) Old wrapper prefix
        if k2.startswith("backbone."):
            k2 = k2[len("backbone.") :]

        # 3) EfficientNet old nested head -> flat head
        #    Map 'classifier.1.1.*' -> 'classifier.1.*'
        #         'classifier.1.4.*' -> 'classifier.4.*'
        if k2.startswith("classifier.1.1."):
            k2 = "classifier.1." + k2.split("classifier.1.1.", 1)[1]
        if k2.startswith("classifier.1.4."):
            k2 = "classifier.4." + k2.split("classifier.1.4.", 1)[1]

        new_sd[k2] = v
    return new_sd


def generate_all_overall_metrics():
    """Generate overall metrics for all 4 CNN models"""
    print("🚀 Generating Overall Test Results for All CNN Models")
    print("=" * 70)

    try:
        from train_cnn import (
            build_model,
            RFMiDDataset,
            get_transforms,
            evaluate_model,
            overall_confusion_from_batches,
            _compute_overall_f1max,
        )

        # Set up paths
        ROOT_DIR = Path(__file__).resolve().parent
        RESULTS_DIR = ROOT_DIR / "results" / "CNN"
        DATA_DIR = ROOT_DIR / "data" / "RFMiD_Challenge_Dataset"

        # Model configurations
        model_cfgs = {
            "ResNet50": {
                "model_name": "resnet50",
                "checkpoint_name": "resnet50_rfmid_best.pth",
                "metrics_name": "resnet50_metrics.csv",
            },
            "Densenet121": {
                "model_name": "densenet121",
                "checkpoint_name": "densenet121_rfmid_best.pth",
                "metrics_name": "densenet121_metrics.csv",
            },
            "EfficientNetB3": {
                "model_name": "efficientnet_b3",
                "checkpoint_name": "efficientnet_b3_rfmid_best.pth",
                "metrics_name": "efficientnet_b3_metrics.csv",
            },
            "InceptionV3": {
                "model_name": "inception_v3",
                "checkpoint_name": "inception_v3_rfmid_best.pth",
                "metrics_name": "inception_v3_metrics.csv",
            },
        }

        # Load data once
        print("\n📊 Loading test data...")
        test_labels = pd.read_csv(
            DATA_DIR / "2. Groundtruths" / "c. RFMiD_Testing_Labels.csv"
        )
        train_labels = pd.read_csv(
            DATA_DIR / "2. Groundtruths" / "a. RFMiD_Training_Labels.csv"
        )

        label_columns = [c for c in train_labels.columns if c != "ID"]
        test_labels = test_labels.reindex(columns=["ID"] + label_columns, fill_value=0)
        class_names = label_columns

        print(f"   Test samples: {len(test_labels)}")
        print(f"   Number of classes: {len(class_names)}")

        # Set up device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   Device: {device}")

        # Process each model
        results_summary = []

        for model_display_name, config in model_cfgs.items():
            print(f"\n{'='*20} {model_display_name} {'='*20}")

            try:
                # Set up paths for this model
                model_results_dir = RESULTS_DIR / model_display_name
                checkpoint_path = model_results_dir / config["checkpoint_name"]
                thresholds_path = model_results_dir / "optimal_thresholds.npy"
                metrics_path = model_results_dir / config["metrics_name"]
                overall_results_path = model_results_dir / "overall_test_results.csv"

                # Check if files exist
                if not checkpoint_path.exists():
                    print(f"❌ Checkpoint not found: {checkpoint_path}")
                    continue

                if not thresholds_path.exists():
                    print(f"❌ Thresholds not found: {thresholds_path}")
                    continue

                print(f"✅ Processing {model_display_name}...")
                print(f"   Checkpoint: {checkpoint_path.name}")
                print(f"   Thresholds: {thresholds_path.name}")

                # Build model
                model = build_model(config["model_name"], len(class_names)).to(device)

                # Load trained weights (with compatibility remap)
                ckpt = torch.load(checkpoint_path, map_location=device)
                raw_sd = ckpt["model_state_dict"]
                remapped_sd = _remap_state_dict_keys(raw_sd)
                missing, unexpected = model.load_state_dict(remapped_sd, strict=False)
                if missing:
                    print(f"   ⚠️ Missing keys after remap: {missing[:5]}{' ...' if len(missing)>5 else ''}")
                if unexpected:
                    print(f"   ⚠️ Unexpected keys after remap: {unexpected[:5]}{' ...' if len(unexpected)>5 else ''}")

                # Set model to evaluation mode
                model.eval()

                # Get best validation AUC from metrics file
                best_val_auc = 0.0
                if metrics_path.exists():
                    try:
                        metrics_df = pd.read_csv(metrics_path)
                        if "val_auc" in metrics_df.columns:
                            best_val_auc = float(metrics_df["val_auc"].max())
                        elif "best_val_auc" in metrics_df.columns:
                            best_val_auc = float(metrics_df["best_val_auc"].iloc[-1])
                    except Exception as e:
                        print(f"   ⚠️ Could not load validation AUC: {e}")

                # Load calibrated thresholds
                calibrated_thresholds = np.load(thresholds_path)
                if calibrated_thresholds.shape[0] != len(class_names):
                    raise ValueError(
                        f"Thresholds length ({calibrated_thresholds.shape[0]}) "
                        f"!= #classes ({len(class_names)})."
                    )

                # Set up data loader
                test_transform = get_transforms(config["model_name"], train=False)
                test_dataset = RFMiDDataset(
                    DATA_DIR / "1. Original Images" / "c. Testing Set",
                    test_labels,
                    label_columns,
                    transform=test_transform,
                )
                test_loader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=16, shuffle=False, num_workers=0
                )

                # Set up criterion
                criterion = torch.nn.BCEWithLogitsLoss()

                # Run evaluation with no-grad context
                print(f"   🧪 Running evaluation...")
                with torch.inference_mode():
                    (
                        test_loss,
                        test_bal_acc,
                        test_sens,
                        test_spec,
                        test_auc,
                        test_all_labels,
                        test_all_preds,
                        test_sens_per_class,
                        test_spec_per_class,
                    ) = evaluate_model(
                        model, test_loader, criterion, device, calibrated_thresholds
                    )

                # Calculate additional metrics
                test_tp, test_tn, test_fp, test_fn = overall_confusion_from_batches(
                    test_all_labels, test_all_preds, calibrated_thresholds
                )
                precision_overall = test_tp / (test_tp + test_fp + 1e-8)
                recall_overall = test_tp / (test_tp + test_fn + 1e-8)

                # Calculate F1max (micro over all elements)
                (
                    f1max_overall,
                    thr_f1_overall,
                    prec_f1_overall,
                    rec_f1_overall,
                ) = _compute_overall_f1max(test_all_labels, test_all_preds)

                # Display results
                print(f"   📊 Results:")
                print(f"      Loss: {test_loss:.6f}")
                print(f"      Balanced Accuracy: {test_bal_acc:.6f}")
                print(f"      Sensitivity: {test_sens:.6f}")
                print(f"      Specificity: {test_spec:.6f}")
                print(f"      AUC: {test_auc:.6f}")
                print(f"      Precision: {precision_overall:.6f}")
                print(f"      Recall: {recall_overall:.6f}")
                print(f"      F1max: {f1max_overall:.6f}")
                print(f"      TP: {test_tp}, TN: {test_tn}, FP: {test_fp}, FN: {test_fn}")

                # Write updated overall test results
                print(f"   💾 Writing overall_test_results.csv...")
                with open(overall_results_path, "w") as f:
                    f.write("metric,value\n")
                    f.write(f"test_loss,{test_loss:.6f}\n")
                    f.write(f"test_balanced_accuracy,{test_bal_acc:.6f}\n")
                    f.write(f"test_sensitivity,{test_sens:.6f}\n")
                    f.write(f"test_specificity,{test_spec:.6f}\n")
                    f.write(f"test_auc,{test_auc:.6f}\n")
                    f.write(f"best_validation_auc,{best_val_auc:.6f}\n")
                    f.write(f"test_precision,{precision_overall:.6f}\n")
                    f.write(f"test_recall,{recall_overall:.6f}\n")
                    f.write(f"test_tp,{int(test_tp)}\n")
                    f.write(f"test_tn,{int(test_tn)}\n")
                    f.write(f"test_fp,{int(test_fp)}\n")
                    f.write(f"test_fn,{int(test_fn)}\n")
                    f.write(f"test_f1max,{f1max_overall:.6f}\n")
                    f.write(f"test_f1max_threshold,{thr_f1_overall:.6f}\n")
                    f.write(f"test_f1max_precision,{prec_f1_overall:.6f}\n")
                    f.write(f"test_f1max_recall,{rec_f1_overall:.6f}\n")

                print(f"   ✅ Updated: {overall_results_path.name}")

                # Store results for summary
                results_summary.append(
                    {
                        "Model": model_display_name,
                        "Loss": test_loss,
                        "Balanced_Accuracy": test_bal_acc,
                        "Sensitivity": test_sens,
                        "Specificity": test_spec,
                        "AUC": test_auc,
                        "Precision": precision_overall,
                        "Recall": recall_overall,
                        "F1max": f1max_overall,
                        "Best_Val_AUC": best_val_auc,
                        "TP": test_tp,
                        "TN": test_tn,
                        "FP": test_fp,
                        "FN": test_fn,
                    }
                )

            except Exception as e:
                print(f"❌ Error processing {model_display_name}: {e}")
                import traceback

                traceback.print_exc()
                continue

        # Create summary comparison
        if results_summary:
            print(f"\n{'='*70}")
            print("📊 SUMMARY COMPARISON - ALL MODELS")
            print(f"{'='*70}")

            summary_df = pd.DataFrame(results_summary)

            # Display key metrics
            print("\n🎯 Key Performance Metrics:")
            key_metrics = [
                "Model",
                "AUC",
                "Sensitivity",
                "Specificity",
                "Precision",
                "Recall",
                "F1max",
            ]
            print(summary_df[key_metrics].to_string(index=False, float_format="%.4f"))

            # Find best performing model for each metric
            print(f"\n🏆 Best Performing Models:")
            for metric in ["AUC", "Sensitivity", "Specificity", "Precision", "Recall", "F1max"]:
                if metric in summary_df.columns and not summary_df.empty:
                    best_idx = summary_df[metric].idxmax()
                    best_model = summary_df.loc[best_idx, "Model"]
                    best_value = summary_df.loc[best_idx, metric]
                    print(f"   {metric}: {best_model} ({best_value:.4f})")

            # Save summary to CSV
            summary_path = RESULTS_DIR / "all_models_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f"\n💾 Summary saved to: {summary_path}")

            # Compare with any_abnormal metrics
            print(f"\n📈 Comparison with Any_Abnormal Metrics:")
            print(f"{'Model':<15} {'Any_Abnormal AUC':<15} {'Overall AUC':<12} {'Difference':<10}")
            print("-" * 60)

            for result in results_summary:
                model_name = result["Model"]
                any_abnormal_path = RESULTS_DIR / model_name / "overall_any_abnormal_metrics.csv"

                if any_abnormal_path.exists():
                    try:
                        any_abnormal_df = pd.read_csv(any_abnormal_path)
                        any_abnormal_auc = (
                            any_abnormal_df[any_abnormal_df["Metric"] == "AUC (%)"]["Value"].iloc[0]
                            / 100.0
                        )
                        overall_auc = result["AUC"]
                        difference = any_abnormal_auc - overall_auc

                        print(
                            f"{model_name:<15} {any_abnormal_auc:<15.4f} {overall_auc:<12.4f} {difference:<10.4f}"
                        )
                    except Exception:
                        print(
                            f"{model_name:<15} {'Error':<15} {result['AUC']:<12.4f} {'N/A':<10}"
                        )
                else:
                    print(
                        f"{model_name:<15} {'Not found':<15} {result['AUC']:<12.4f} {'N/A':<10}"
                    )

        print(f"\n🎉 Overall metrics generation completed for all models!")
        return True

    except Exception as e:
        print(f"❌ Error generating overall metrics: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = generate_all_overall_metrics()

    if success:
        print("\n✅ SUCCESS!")
        print("All models now have complete overall test metrics.")
        print("No model retraining was required!")
    else:
        print("\n❌ FAILED!")
        print("Please check the error messages above.")
