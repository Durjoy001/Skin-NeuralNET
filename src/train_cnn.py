# -*- coding: utf-8 -*-
"""
Binary CNN training for PAD-UFES-20:
- Task: skin cancer vs non cancer lesion (image-only)
- Cancer classes: BCC, MEL, SCC
- Non cancer classes: ACK, NEV, SEK
- Architectures: ResNet50, EfficientNet-B3, InceptionV3, DenseNet121 (optional)
- ImageNet pretrained transforms per model
- Sensitivity / Specificity tracking, AUC checkpointing
- Threshold calibration to get ~0.80 specificity on validation set
- Saves best-AUC checkpoint and final test metrics

Expected data layout:

project_root/
  src/
    train_cnn.py
  data/
    metadata.csv
    imgs_part_1/
      imgs_part_1/
        <img_id>.png
    imgs_part_2/
      imgs_part_2/
        <img_id>.png
    imgs_part_3/
      imgs_part_3/
        <img_id>.png
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path
import random, warnings, numpy as np, pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import (
    DenseNet121_Weights,
    ResNet50_Weights,
    EfficientNet_B3_Weights,
    Inception_V3_Weights
)
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
from sklearn.exceptions import UndefinedMetricWarning
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------
# Hard-coded parameters
# ----------------------
SEED, BATCH_SIZE, EPOCHS, LR, NUM_WORKERS = 42, 16, 20, 1e-4, 0  # NUM_WORKERS=0 is macOS-safe
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ----------------------
# Paths (root)
# ----------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

RESULTS_DIR = None
SAVE_PATH = None
METRICS_CSV = None
THRESHOLDS_PATH = None

PATIENCE = 5        # stop if val loss doesn't improve for 5 epochs
MIN_DELTA = 1e-4    # minimum improvement to be considered "better"

# ----------------------
# Reproducibility
# ----------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ----------------------
# Dataset
# ----------------------
class PADUFESDataset(Dataset):
    """
    Dataset wrapper for PAD-UFES-20.

    labels_df must contain:
      - "ID": image identifier (from img_id column)
      - "is_cancer": binary label (1 = BCC/MEL/SCC, 0 = ACK/NEV/SEK)
    """
    def __init__(self, img_dirs, labels_df, label_columns, transform=None):
        # img_dirs can be a single Path or a list of Paths
        if isinstance(img_dirs, (list, tuple)):
            self.img_dirs = [Path(d) for d in img_dirs]
        else:
            self.img_dirs = [Path(img_dirs)]
        self.labels_df = labels_df.reset_index(drop=True)
        self.transform = transform
        self.label_columns = label_columns
        self.num_classes = len(self.label_columns)

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        img_id = str(row["ID"])

        # Try several common extensions across all image directories
        img_path = None
        for img_dir in self.img_dirs:
            for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
                candidate = img_dir / f"{img_id}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break
            if img_path is not None:
                break
        
        # Fallback: assume img_id already includes extension
        if img_path is None:
            for img_dir in self.img_dirs:
                candidate = img_dir / img_id
                if candidate.exists():
                    img_path = candidate
                    break
        
        if img_path is None or not img_path.exists():
            raise FileNotFoundError(f"Image not found for ID={img_id} in any image directory")

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Convert to numpy array first, then to float32, then to tensor
        label_values = row[self.label_columns].values.astype(np.float32)
        labels = torch.tensor(label_values, dtype=torch.float32)
        return image, labels

# ----------------------
# Per-model pretrained transforms
# ----------------------
def get_transforms(model_name: str, train: bool):
    model_name = model_name.lower()
    W = {
        "resnet50": ResNet50_Weights.IMAGENET1K_V1,
        "efficientnet_b3": EfficientNet_B3_Weights.IMAGENET1K_V1,
        "inception_v3": Inception_V3_Weights.IMAGENET1K_V1,
        "densenet121": DenseNet121_Weights.IMAGENET1K_V1,
    }[model_name]
    base = W.transforms(antialias=True)
    if train:
        return transforms.Compose([
            base,
            transforms.RandomHorizontalFlip(0.5),
        ])
    return base

# ----------------------
# Multi-CNN model builder
# ----------------------
def build_model(model_name, num_classes):
    """Return CNN backbone for PAD-UFES-20 binary classification."""
    model_name = model_name.lower()

    if model_name == "densenet121":
        model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        in_f = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_f, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    elif model_name == "resnet50":
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        in_f = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_f, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    elif model_name == "efficientnet_b3":
        model = models.efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        in_f = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_f, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    elif model_name == "inception_v3":
        model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        in_f = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_f, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        if model.AuxLogits is not None:
            aux_in = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Linear(aux_in, num_classes)

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model

# ----------------------
# Metric Helpers
# ----------------------
def compute_sens_spec(preds: torch.Tensor, labels: torch.Tensor, average="macro"):
    tp = (preds * labels).sum(dim=0).float()
    tn = ((1 - preds) * (1 - labels)).sum(dim=0).float()
    fp = (preds * (1 - labels)).sum(dim=0).float()
    fn = ((1 - preds) * labels).sum(dim=0).float()

    if average == "micro":
        TP, TN, FP, FN = tp.sum(), tn.sum(), fp.sum(), fn.sum()
        sens = (TP / (TP + FN + 1e-6)).item()
        spec = (TN / (TN + FP + 1e-6)).item()
        return sens, spec

    pos_mask = (tp + fn) > 0
    neg_mask = (tn + fp) > 0
    sens = (tp[pos_mask] / (tp[pos_mask] + fn[pos_mask])).mean().item() if pos_mask.any() else float("nan")
    spec = (tn[neg_mask] / (tn[neg_mask] + fp[neg_mask])).mean().item() if neg_mask.any() else float("nan")
    return sens, spec

def compute_per_class_sens_spec(preds: torch.Tensor, labels: torch.Tensor):
    tp = (preds * labels).sum(dim=0).float()
    tn = ((1 - preds) * (1 - labels)).sum(dim=0).float()
    fp = (preds * (1 - labels)).sum(dim=0).float()
    fn = ((1 - preds) * labels).sum(dim=0).float()
    sens_per_class = tp / (tp + fn + 1e-6)
    spec_per_class = tn / (tn + fp + 1e-6)
    return sens_per_class.cpu().numpy(), spec_per_class.cpu().numpy()

# ----------------------
# Threshold Calibration
# ----------------------
def compute_optimal_thresholds(y_true, y_pred, target_spec=0.8):
    """
    For each column, choose threshold whose specificity is closest to target_spec.
    Here we only have one class, but keep general form.
    """
    thresholds = []
    for i in range(y_true.shape[1]):
        try:
            fpr, tpr, thr = roc_curve(y_true[:, i], y_pred[:, i])
            spec = 1 - fpr
            idx = np.argmin(np.abs(spec - target_spec))
            thresholds.append(thr[idx])
        except Exception:
            thresholds.append(0.5)
    return np.array(thresholds)

# ----------------------
# Train/Eval routines
# ----------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch; handle Inception aux loss when present."""
    model.train()
    running_loss = 0.0

    TP = TN = FP = FN = None
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        if isinstance(outputs, tuple):  # Inception train mode: (main, aux)
            main_out, aux_out = outputs
            loss = criterion(main_out, labels) + 0.4 * criterion(aux_out, labels)
            logits_for_metrics = main_out
        else:
            loss = criterion(outputs, labels)
            logits_for_metrics = outputs

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        preds = (torch.sigmoid(logits_for_metrics) > 0.5).float()
        tp = (preds * labels).sum(dim=0)
        tn = ((1 - preds) * (1 - labels)).sum(dim=0)
        fp = (preds * (1 - labels)).sum(dim=0)
        fn = ((1 - preds) * labels).sum(dim=0)

        if TP is None:
            TP, TN, FP, FN = tp, tn, fp, fn
        else:
            TP += tp; TN += tn; FP += fp; FN += fn

    sens = (TP.sum() / (TP.sum() + FN.sum() + 1e-6)).item()
    spec = (TN.sum() / (TN.sum() + FP.sum() + 1e-6)).item()
    bal_acc = 0.5 * (sens + spec)
    avg_loss = running_loss / len(loader)
    return avg_loss, bal_acc, sens, spec

@torch.no_grad()
def evaluate_model(model, loader, criterion, device, thresholds=None):
    """Evaluate with epoch-level sensitivity/specificity and per-class metrics."""
    model.eval()
    running_loss = 0.0
    TP = TN = FP = FN = None
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="Evaluating", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        if isinstance(outputs, tuple):  # just in case (Inception eval)
            outputs = outputs[0]
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        probs = torch.sigmoid(outputs).cpu().numpy()
        labels_np = labels.cpu().numpy()

        if thresholds is None:
            preds = (probs > 0.5).astype(float)
        else:
            preds = (probs > thresholds).astype(float)

        preds_t = torch.tensor(preds)
        labels_t = torch.tensor(labels_np)

        tp = (preds_t * labels_t).sum(dim=0)
        tn = ((1 - preds_t) * (1 - labels_t)).sum(dim=0)
        fp = (preds_t * (1 - labels_t)).sum(dim=0)
        fn = ((1 - preds_t) * labels_t).sum(dim=0)

        if TP is None:
            TP, TN, FP, FN = tp, tn, fp, fn
        else:
            TP += tp; TN += tn; FP += fp; FN += fn

        all_preds.append(probs)
        all_labels.append(labels_np)

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    sens = (TP.sum() / (TP.sum() + FN.sum() + 1e-6)).item()
    spec = (TN.sum() / (TN.sum() + FP.sum() + 1e-6)).item()
    bal_acc = 0.5 * (sens + spec)

    sens_per_class = (TP / (TP + FN + 1e-6)).cpu().numpy()
    spec_per_class = (TN / (TN + FP + 1e-6)).cpu().numpy()

    try:
        valid_cols = (np.sum(all_labels, axis=0) > 0) & (np.sum(all_labels == 0, axis=0) > 0)
        if np.any(valid_cols):
            auc_score = roc_auc_score(all_labels[:, valid_cols], all_preds[:, valid_cols], average='macro')
        else:
            auc_score = 0.0
    except Exception:
        auc_score = 0.0

    avg_loss = running_loss / len(loader)
    return avg_loss, bal_acc, sens, spec, auc_score, all_labels, all_preds, sens_per_class, spec_per_class

def overall_confusion_from_batches(all_labels, all_preds, thresholds=None):
    """Calculate overall TP, TN, FP, FN from batched predictions and labels"""
    thr = 0.5 if thresholds is None else thresholds
    preds = (all_preds > thr).astype(float)
    TP = (preds * all_labels).sum()
    TN = ((1 - preds) * (1 - all_labels)).sum()
    FP = (preds * (1 - all_labels)).sum()
    FN = ((1 - preds) * all_labels).sum()
    return int(TP), int(TN), int(FP), int(FN)

# ----------------------
# Plotting
# ----------------------
def plot_training_curves(train_losses, train_accs, val_losses, val_accs, out_path):
    try:
        epochs = np.arange(0, len(train_losses))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(epochs, train_losses, label="Train Loss", color='blue')
        ax1.plot(epochs, val_losses, label="Val Loss", color='red')
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)
        ax1.set_title("Training and Validation Loss")
        ax2.plot(epochs, train_accs, label="Train Balanced Acc", color='blue')
        ax2.plot(epochs, val_accs, label="Val Balanced Acc", color='red')
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Balanced Accuracy"); ax2.legend(); ax2.grid(True, alpha=0.3)
        ax2.set_title("Training and Validation Balanced Accuracy")
        plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()
    except Exception as e:
        print(f"[WARN] Failed to plot training curves: {e}")

def plot_loss_curves(train_losses, val_losses, out_path):
    try:
        epochs = np.arange(0, len(train_losses))
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, label="Training Loss", linewidth=2, marker='o', markersize=4)
        plt.plot(epochs, val_losses, label="Validation Loss", linewidth=2, marker='s', markersize=4)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("Training and Validation Loss Over Time", fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ Loss curves saved to: {out_path}")
    except Exception as e:
        print(f"[WARN] Failed to plot loss curves: {e}")

def plot_sensitivity_specificity_curves(train_sens, train_spec, val_sens, val_spec, out_path):
    try:
        epochs = np.arange(0, len(train_sens))
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_sens, label="Training Sensitivity", linewidth=2, marker='o', markersize=4)
        plt.plot(epochs, train_spec, label="Training Specificity", linewidth=2, marker='^', markersize=4)
        plt.plot(epochs, val_sens, label="Validation Sensitivity", linewidth=2, marker='s', markersize=4)
        plt.plot(epochs, val_spec, label="Validation Specificity", linewidth=2, marker='d', markersize=4)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Sensitivity / Specificity", fontsize=12)
        plt.title("Training and Validation Sensitivity & Specificity Over Time", fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ Sensitivity/Specificity curves saved to: {out_path}")
    except Exception as e:
        print(f"[WARN] Failed to plot sensitivity/specificity curves: {e}")

def plot_roc_curve(y_true, y_score, auc_score, out_path):
    """Plot ROC curve for test set evaluation."""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier (AUC = 0.5000)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        plt.title('ROC Curve - Test Set', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ ROC curve saved to: {out_path}")
    except Exception as e:
        print(f"[WARN] Failed to plot ROC curve: {e}")

def plot_precision_recall_curve(y_true, y_score, f1max, out_path):
    """Plot Precision-Recall curve for test set evaluation."""
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        plt.figure(figsize=(8, 8))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (F1-max = {f1max:.4f})')
        plt.xlabel('Recall (Sensitivity)', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve - Test Set', fontsize=14, fontweight='bold')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.legend(loc="lower left", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ Precision-Recall curve saved to: {out_path}")
    except Exception as e:
        print(f"[WARN] Failed to plot Precision-Recall curve: {e}")

def plot_confusion_matrix(tn, fp, fn, tp, out_path):
    """Plot confusion matrix visualization."""
    try:
        cm = np.array([[tn, fp], [fn, tp]])
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(2):
            for j in range(2):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=14, fontweight='bold')
        
        # Labels
        ax.set(xticks=np.arange(2), yticks=np.arange(2),
               xticklabels=['Non-Cancer', 'Cancer'],
               yticklabels=['Non-Cancer', 'Cancer'],
               title='Confusion Matrix - Test Set',
               ylabel='True Label', xlabel='Predicted Label')
        
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ Confusion matrix saved to: {out_path}")
    except Exception as e:
        print(f"[WARN] Failed to plot confusion matrix: {e}")

# ----------------------
# Helpers for threshold selection and F1max
# ----------------------
def _pick_threshold_for_specificity(y_true_binary, y_score, target_spec=0.8):
    fpr, tpr, thr = roc_curve(y_true_binary, y_score)
    spec = 1 - fpr
    idx = np.argmin(np.abs(spec - target_spec))
    return float(thr[idx]), float(spec[idx]), float(tpr[idx])

def _compute_f1max(y_true_binary, y_score):
    precision, recall, thr = precision_recall_curve(y_true_binary, y_score)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    f1_use = f1[:-1]
    best_idx = int(np.nanargmax(f1_use))
    return float(f1_use[best_idx]), float(thr[best_idx]), float(precision[best_idx]), float(recall[best_idx])

# ----------------------
# Core training pipeline — runs for one model
# ----------------------
def run_for_model(model_name: str):
    global RESULTS_DIR, SAVE_PATH, METRICS_CSV, THRESHOLDS_PATH

    pretty = {
        "densenet121": "Densenet121",
        "resnet50": "ResNet50",
        "efficientnet_b3": "EfficientNetB3",
        "inception_v3": "InceptionV3",
    }[model_name.lower()]

    RESULTS_DIR = ROOT_DIR / "results" / "PAD_UFES20" / pretty
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_PATH = RESULTS_DIR / f"{model_name.lower()}_padufes20_best.pth"
    METRICS_CSV = RESULTS_DIR / f"{model_name.lower()}_metrics.csv"
    THRESHOLDS_PATH = RESULTS_DIR / "optimal_thresholds.npy"

    print(f"🚀 Starting {pretty} training on PAD-UFES-20 (cancer vs non cancer)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------
    # Load metadata and build binary label
    # ----------------------
    meta = pd.read_csv(DATA_DIR / "metadata.csv")
    # Standardise names
    if "img_id" in meta.columns:
        meta = meta.rename(columns={"img_id": "ID"})
    else:
        raise ValueError("metadata.csv must contain an 'img_id' column")

    cancer_labels = {"BCC", "MEL", "SCC"}
    non_cancer_labels = {"ACK", "NEV", "SEK"}

    meta["diagnostic"] = meta["diagnostic"].str.upper().str.strip()
    valid_diags = cancer_labels.union(non_cancer_labels)
    meta = meta[meta["diagnostic"].isin(valid_diags)].copy()

    meta["is_cancer"] = meta["diagnostic"].isin(cancer_labels).astype(np.float32)

    # ----------------------
    # Patient-level split: 70% train, 15% val, 15% test
    # ----------------------
    unique_patients = meta["patient_id"].unique()
    rng = np.random.RandomState(SEED)
    rng.shuffle(unique_patients)
    n = len(unique_patients)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    train_patients = set(unique_patients[:n_train])
    val_patients = set(unique_patients[n_train:n_train + n_val])
    test_patients = set(unique_patients[n_train + n_val:])

    train_labels = meta[meta["patient_id"].isin(train_patients)].reset_index(drop=True)
    val_labels   = meta[meta["patient_id"].isin(val_patients)].reset_index(drop=True)
    test_labels  = meta[meta["patient_id"].isin(test_patients)].reset_index(drop=True)

    label_columns = ["is_cancer"]

    # Pos_weight from train only
    y = train_labels[label_columns].values
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    pos_weight = torch.tensor((neg / (pos + 1e-6)).astype(np.float32)).to(device)

    print(f"Training samples: {len(train_labels)}, Validation: {len(val_labels)}, Test: {len(test_labels)}")

    # ----------------------
    # Datasets and loaders
    # ----------------------
    # Images are split across three directories
    img_dirs = [
        DATA_DIR / "imgs_part_1" / "imgs_part_1",
        DATA_DIR / "imgs_part_2" / "imgs_part_2",
        DATA_DIR / "imgs_part_3" / "imgs_part_3"
    ]

    train_transform    = get_transforms(model_name, train=True)
    val_test_transform = get_transforms(model_name, train=False)

    train_dataset = PADUFESDataset(img_dirs, train_labels, label_columns, train_transform)
    val_dataset   = PADUFESDataset(img_dirs, val_labels,   label_columns, val_test_transform)
    test_dataset  = PADUFESDataset(img_dirs, test_labels,  label_columns, val_test_transform)

    train_loader = DataLoader(train_dataset, BATCH_SIZE, True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_dataset,   BATCH_SIZE, False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_dataset,  BATCH_SIZE, False, num_workers=NUM_WORKERS)

    # ----------------------
    # Model, loss, optimizer
    # ----------------------
    model = build_model(model_name, num_classes=len(label_columns)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Different LR for backbone vs classifier
    if hasattr(model, "features"):  # DenseNet
        backbone_params = list(model.features.parameters())
        classifier_params = list(model.classifier.parameters())
    elif hasattr(model, "fc"):  # ResNet/Inception
        backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
        classifier_params = list(model.fc.parameters())
    elif hasattr(model, "classifier"):  # EfficientNet
        backbone_params = [p for n, p in model.named_parameters() if not n.startswith("classifier.")]
        classifier_params = list(model.classifier.parameters())
    else:
        backbone_params = list(model.parameters())
        classifier_params = []

    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': LR * 0.1},
        {'params': classifier_params, 'lr': LR}
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # ----------------------
    # CSV header
    # ----------------------
    class_names = label_columns
    header = "epoch,train_loss,train_bal_acc,train_sens,train_spec,val_loss,val_bal_acc,val_sens,val_spec,val_auc"
    for class_name in class_names:
        header += f",val_sens_{class_name},val_spec_{class_name}"
    with open(METRICS_CSV, "w") as f:
        f.write(header + "\n")

    best_val_auc = 0.0
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    train_sens_list, train_spec_list, val_sens_list, val_spec_list = [], [], [], []

    # Initial evaluation (epoch 0)
    print("\n📊 Epoch 0: Evaluating initial model performance...")
    train_loss_0, train_bal_acc_0, train_sens_0, train_spec_0, _, _, _, _, _ = evaluate_model(model, train_loader, criterion, device)
    val_loss_0, val_bal_acc_0, val_sens_0, val_spec_0, val_auc_0, _, _, val_sens_per_class_0, val_spec_per_class_0 = evaluate_model(model, val_loader, criterion, device)

    train_losses.append(train_loss_0); val_losses.append(val_loss_0)
    train_accs.append(train_bal_acc_0); val_accs.append(val_bal_acc_0)
    train_sens_list.append(train_sens_0); train_spec_list.append(train_spec_0)
    val_sens_list.append(val_sens_0); val_spec_list.append(val_spec_0)

    print(f"Initial Train Balanced Acc: {train_bal_acc_0:.4f} | Sens: {train_sens_0:.4f} | Spec: {train_spec_0:.4f}")
    print(f"Initial Val Balanced Acc: {val_bal_acc_0:.4f} | Sens: {val_sens_0:.4f} | Spec: {val_spec_0:.4f} | AUC: {val_auc_0:.4f}")

    csv_line = f"0,{train_loss_0:.6f},{train_bal_acc_0:.6f},{train_sens_0:.6f},{train_spec_0:.6f},"
    csv_line += f"{val_loss_0:.6f},{val_bal_acc_0:.6f},{val_sens_0:.6f},{val_spec_0:.6f},{val_auc_0:.6f}"
    for i in range(len(class_names)):
        csv_line += f",{val_sens_per_class_0[i]:.6f},{val_spec_per_class_0[i]:.6f}"
    with open(METRICS_CSV, "a") as f:
        f.write(csv_line + "\n")

    # Early stopping on validation AUC
    best_val_auc_es = -1.0
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss, train_bal_acc, train_sens, train_spec = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_bal_acc, val_sens, val_spec, val_auc, y_true_val_all, y_pred_val_all, val_sens_per_class, val_spec_per_class = evaluate_model(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        train_losses.append(train_loss); val_losses.append(val_loss)
        train_accs.append(train_bal_acc); val_accs.append(val_bal_acc)
        train_sens_list.append(train_sens); train_spec_list.append(train_spec)
        val_sens_list.append(val_sens); val_spec_list.append(val_spec)

        print(f"Train Balanced Acc: {train_bal_acc:.4f} | Sens: {train_sens:.4f} | Spec: {train_spec:.4f}")
        print(f"Val Balanced Acc: {val_bal_acc:.4f} | Sens: {val_sens:.4f} | Spec: {val_spec:.4f} | AUC: {val_auc:.4f}")

        # Write metrics to CSV
        csv_line = f"{epoch},{train_loss:.6f},{train_bal_acc:.6f},{train_sens:.6f},{train_spec:.6f},"
        csv_line += f"{val_loss:.6f},{val_bal_acc:.6f},{val_sens:.6f},{val_spec:.6f},{val_auc:.6f}"
        for i in range(len(class_names)):
            csv_line += f",{val_sens_per_class[i]:.6f},{val_spec_per_class[i]:.6f}"
        with open(METRICS_CSV, "a") as f:
            f.write(csv_line + "\n")

        # Early stopping
        if val_auc > best_val_auc_es + MIN_DELTA:
            best_val_auc_es = val_auc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"[ES] No val-AUC improvement for {epochs_no_improve}/{PATIENCE} epoch(s).")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({'model_state_dict': model.state_dict()}, SAVE_PATH)
            print(f"💾 Best model saved! (AUC={val_auc:.4f})")

        # Update plots each epoch
        plot_training_curves(train_losses, train_accs, val_losses, val_accs, RESULTS_DIR / "training_curves.png")
        plot_loss_curves(train_losses, val_losses, RESULTS_DIR / "loss_curves.png")
        plot_sensitivity_specificity_curves(train_sens_list, train_spec_list, val_sens_list, val_spec_list, RESULTS_DIR / "sensitivity_specificity_curves.png")

        if epochs_no_improve >= PATIENCE:
            print(f"[ES] Early stopping triggered (patience={PATIENCE}).")
            break

    # ----------------------
    # Threshold calibration at ~0.80 specificity (validation)
    # ----------------------
    print("\n📊 Calibrating threshold (target specificity=0.8)...")
    if os.path.exists(SAVE_PATH):
        checkpoint = torch.load(SAVE_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Loaded best saved model for calibration.")
    else:
        print("⚠️ No best model saved; using last trained model for calibration.")

    _, _, _, _, val_auc, y_true_val, y_pred_val, _, _ = evaluate_model(model, val_loader, criterion, device)
    y_true_bin_val = y_true_val[:, 0].astype(np.int32)
    y_score_bin_val = y_pred_val[:, 0]
    thr_spec80, spec80_val, sens80_val = _pick_threshold_for_specificity(y_true_bin_val, y_score_bin_val, target_spec=0.8)
    thresholds = np.array([thr_spec80], dtype=np.float32)
    np.save(THRESHOLDS_PATH, thresholds)
    print(f"🔧 Validation operating point @~0.80 specificity: thr={thr_spec80:.4f}, spec={spec80_val:.4f}, sens={sens80_val:.4f}")
    print(f"Optimal threshold saved to: {THRESHOLDS_PATH}")

    # ----------------------
    # Final evaluation on test set using calibrated threshold
    # ----------------------
    print("\n🧪 Final evaluation on test set (using calibrated threshold)...")
    test_loss, test_bal_acc, test_sens, test_spec, test_auc_raw, y_true_test_all, y_pred_test_all, test_sens_per_class, test_spec_per_class = evaluate_model(
        model, test_loader, criterion, device, thresholds=thresholds
    )

    y_true_bin_test = y_true_test_all[:, 0].astype(np.int32)
    y_score_bin_test = y_pred_test_all[:, 0]

    # ROC AUC (continuous scores)
    auc_test = roc_auc_score(y_true_bin_test, y_score_bin_test)

    # Confusion matrix at thr_spec80
    y_pred_bin_test = (y_score_bin_test >= thr_spec80).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y_true_bin_test, y_pred_bin_test, labels=[0, 1]).ravel()

    precision_at_thr = tp / (tp + fp + 1e-8)
    recall_at_thr = tp / (tp + fn + 1e-8)

    # F1max on test (continuous threshold sweep)
    f1max, thr_f1, prec_f1, rec_f1 = _compute_f1max(y_true_bin_test, y_score_bin_test)

    # Confusion matrix at F1max threshold
    y_pred_bin_test_f1max = (y_score_bin_test >= thr_f1).astype(np.int32)
    tn_f1max, fp_f1max, fn_f1max, tp_f1max = confusion_matrix(y_true_bin_test, y_pred_bin_test_f1max, labels=[0, 1]).ravel()

    # Post-hoc analysis: Calculate sensitivity at 80% specificity on TEST set
    # NOTE: This is a post-hoc analysis. The official operating point uses the validation-chosen threshold (thr_spec80)
    _, spec80_test_posthoc, sens80_test_posthoc = _pick_threshold_for_specificity(y_true_bin_test, y_score_bin_test, target_spec=0.8)

    # Generate evaluation plots
    print("\n📊 Generating evaluation plots...")
    plot_roc_curve(y_true_bin_test, y_score_bin_test, auc_test, RESULTS_DIR / "roc_curve.png")
    plot_precision_recall_curve(y_true_bin_test, y_score_bin_test, f1max, RESULTS_DIR / "precision_recall_curve.png")
    plot_confusion_matrix(tn, fp, fn, tp, RESULTS_DIR / "confusion_matrix.png")

    # Save per-class (only one class) sensitivity/specificity
    test_results_csv = RESULTS_DIR / "final_test_results.csv"
    with open(test_results_csv, "w") as f:
        f.write("class_name,test_sensitivity,test_specificity\n")
        f.write(f"{class_names[0]},{test_sens_per_class[0]:.6f},{test_spec_per_class[0]:.6f}\n")

    # Overall test metrics
    overall_results_csv = RESULTS_DIR / "overall_test_results.csv"
    with open(overall_results_csv, "w") as f:
        f.write("metric,value\n")
        f.write(f"test_loss,{test_loss:.6f}\n")
        f.write(f"test_balanced_accuracy,{test_bal_acc:.6f}\n")
        f.write(f"test_sensitivity,{test_sens:.6f}\n")
        f.write(f"test_specificity,{test_spec:.6f}\n")
        f.write(f"test_auc_continuous,{auc_test:.6f}\n")
        f.write(f"best_validation_auc,{best_val_auc:.6f}\n")
        f.write(f"threshold_spec80_validation,{thr_spec80:.6f}\n")
        f.write(f"precision_at_validation_threshold,{precision_at_thr:.6f}\n")
        f.write(f"recall_at_validation_threshold,{recall_at_thr:.6f}\n")
        f.write(f"test_sensitivity_at_validation_threshold,{test_sens:.6f}\n")
        f.write(f"test_specificity_at_validation_threshold,{test_spec:.6f}\n")
        f.write(f"posthoc_sensitivity_at_80pct_specificity_test,{sens80_test_posthoc:.6f}\n")
        f.write(f"posthoc_specificity_at_80pct_target_test,{spec80_test_posthoc:.6f}\n")
        f.write(f"tp_at_validation_threshold,{int(tp)}\n")
        f.write(f"tn_at_validation_threshold,{int(tn)}\n")
        f.write(f"fp_at_validation_threshold,{int(fp)}\n")
        f.write(f"fn_at_validation_threshold,{int(fn)}\n")
        f.write(f"f1max,{f1max:.6f}\n")
        f.write(f"f1max_threshold,{thr_f1:.6f}\n")
        f.write(f"f1max_precision,{prec_f1:.6f}\n")
        f.write(f"f1max_recall,{rec_f1:.6f}\n")
        f.write(f"tp_at_f1max_threshold,{int(tp_f1max)}\n")
        f.write(f"tn_at_f1max_threshold,{int(tn_f1max)}\n")
        f.write(f"fp_at_f1max_threshold,{int(fp_f1max)}\n")
        f.write(f"fn_at_f1max_threshold,{int(fn_f1max)}\n")

    print(f"\n🎉 Training completed for {pretty}!")
    print(f"Best validation AUC: {best_val_auc:.4f}")
    print(f"Test AUC (continuous): {auc_test:.4f}")
    print(f"\n📊 Official Operating Point (validation-chosen threshold):")
    print(f"  - Threshold: {thr_spec80:.4f}")
    print(f"  - Test Sensitivity: {test_sens:.4f}")
    print(f"  - Test Specificity: {test_spec:.4f}")
    print(f"\n📁 All results saved to: {RESULTS_DIR}")
    print(f"  - Model: {SAVE_PATH.name}")
    print(f"  - Thresholds: {THRESHOLDS_PATH.name}")
    print(f"  - Training metrics: {METRICS_CSV.name}")
    print(f"  - Test results: {test_results_csv.name}")
    print(f"  - Overall metrics: {overall_results_csv.name}")
    print(f"  - Visualizations: roc_curve.png, precision_recall_curve.png, confusion_matrix.png")

# ----------------------
# Entry point
# ----------------------
if __name__ == "__main__":
    # Train all 4 CNN models
    model_names = ["resnet50", "efficientnet_b3", "inception_v3", "densenet121"]
    for m in model_names:
        print(f"\n==================== {m.upper()} ====================")
        run_for_model(m)
