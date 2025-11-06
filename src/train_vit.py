# src/train_vit.py
# Multi-ViT training script for RFMiD retinal fundus multi-disease classification
# - Swin-Tiny, ViT-Small/16, DeiT-Small/16, CrossViT-Small
# - ImageNet pretrained transforms per model
# - Sens/Spec tracking, AUC checkpointing, 0.80-spec thresholding
# - Any-abnormal metrics + per-image NPZ for DeLong & McNemar

import os
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path
import random, warnings, numpy as np, pandas as pd, math
from PIL import Image

# Suppress PyTorch deprecation warnings for mixed precision
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
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
DATA_DIR = ROOT_DIR / "data" / "RFMiD_Challenge_Dataset"

# These will be set per-model inside run_for_model(...)
RESULTS_DIR = None
SAVE_PATH = None
METRICS_CSV = None
THRESHOLDS_PATH = None
SAVE_PATH_ANY = None

PATIENCE = 10        # stop if val loss doesn't improve for 10 epochs (was 5)
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
class RFMiDDataset(Dataset):
    def __init__(self, img_dir, labels_df, label_columns, transform=None):
        self.img_dir, self.labels_df, self.transform = img_dir, labels_df, transform
        self.label_columns = label_columns
        self.num_classes = len(self.label_columns)

    def __len__(self): return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        img_id = row['ID']
        img_path = self.img_dir / f"{img_id}.png"
        # fail loudly if image missing/unreadable to avoid poisoning the dataset
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        labels = torch.tensor(row[self.label_columns].values, dtype=torch.float32)
        return image, labels

# ----------------------
# Cosine with warmup scheduler for ViTs
# ----------------------
def cosine_with_warmup(optimizer, warmup_epochs=5, total_epochs=20):
    """Cosine decay with warmup scheduler for ViT training"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ----------------------
# Per-model pretrained transforms using timm configs
# ----------------------
def vit_transforms(model_name: str, train: bool):
    """Get transforms for ViT models using timm's pretrained configurations"""
    model_name = model_name.lower()
    
    # Map model names to timm model identifiers
    model_map = {
        "swin_tiny": "swin_tiny_patch4_window7_224",
        "vit_small": "vit_small_patch16_224", 
        "deit_small": "deit_small_patch16_224",
        "crossvit_small": "crossvit_15_240",
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown ViT model name: {model_name}")
    
    # Create model to get its data config
    m = timm.create_model(model_map[model_name], pretrained=True, num_classes=1)
    cfg = resolve_data_config({}, model=m)
    
    # Create transforms using timm's factory
    return create_transform(**cfg, is_training=train)

# ----------------------
# Multi-ViT model builder
# ----------------------
def build_model(model_name, num_classes):
    """Return ViT backbone for RFMiD classification (same head style for all)."""
    model_name = model_name.lower()

    if model_name == "swin_tiny":
        # Swin-Tiny: swin_tiny_patch4_window7_224 (~28M params)
        model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0, drop_path_rate=0.2)
        in_f = model.num_features  # Get feature dimension
        
    elif model_name == "vit_small":
        # ViT-Small/16: vit_small_patch16_224 (~22M params)
        model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=0, drop_path_rate=0.2)
        in_f = model.num_features
        
    elif model_name == "deit_small":
        # DeiT-Small/16: deit_small_patch16_224 (~22M params)
        model = timm.create_model('deit_small_patch16_224', pretrained=True, num_classes=0, drop_path_rate=0.2)
        in_f = model.num_features
        
    elif model_name == "crossvit_small":
        # CrossViT-Small: crossvit_15_240 (~27M params)
        model = timm.create_model('crossvit_15_240', pretrained=True, num_classes=0, drop_path_rate=0.2)
        in_f = model.num_features
        
    else:
        raise ValueError(f"Unknown ViT model name: {model_name}")

    # Add consistent classifier head for all ViT models
    classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_f, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    
    # Create wrapper model
    class ViTWrapper(nn.Module):
        def __init__(self, backbone, classifier):
            super().__init__()
            self.backbone = backbone
            self.classifier = classifier
            
        def forward(self, x):
            features = self.backbone(x)
            return self.classifier(features)
    
    return ViTWrapper(model, classifier)

# ---------- Temperature scaling ----------
class _TempScaler(nn.Module):
    def __init__(self, init_T=1.0):
        super().__init__()
        self.logT = nn.Parameter(torch.tensor(math.log(init_T), dtype=torch.float32))

    def forward(self, logits):
        T = torch.exp(self.logT)
        return logits / T

def _gather_val_logits_labels(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            all_logits.append(logits.cpu())
            all_labels.append(y)
    return torch.cat(all_logits, 0), torch.cat(all_labels, 0)

def fit_temperature(model, val_loader, device):
    logits, labels = _gather_val_logits_labels(model, val_loader, device)
    scaler = _TempScaler(init_T=1.0).to(device)
    optimizer = torch.optim.LBFGS(scaler.parameters(), lr=0.1, max_iter=50, line_search_fn="strong_wolfe")
    criterion = nn.BCEWithLogitsLoss(reduction="mean")

    logits, labels = logits.to(device), labels.to(device)

    def closure():
        optimizer.zero_grad()
        scaled = scaler(logits)
        loss = criterion(scaled, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        T = torch.exp(scaler.logT).item()
    return T

class _WithTemp(nn.Module):
    def __init__(self, base, T):
        super().__init__()
        self.base = base
        self.T = float(T)
    def forward(self, x):
        return self.base(x) / self.T

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
# F1 Metrics at Thresholds
# ----------------------
def compute_f1_at_thresholds(all_labels, all_preds, thresholds=None):
    """Compute Macro and Micro F1 scores at chosen per-class thresholds"""
    from sklearn.metrics import f1_score
    
    if thresholds is None:
        thresholds = np.full(all_preds.shape[1], 0.5)
    
    # Apply thresholds to get binary predictions
    preds_binary = (all_preds > thresholds).astype(int)
    
    try:
        # Macro F1: average of per-class F1 scores
        macro_f1 = f1_score(all_labels, preds_binary, average='macro', zero_division=0)
        
        # Micro F1: treats all classes as one big binary problem
        micro_f1 = f1_score(all_labels, preds_binary, average='micro', zero_division=0)
        
        return macro_f1, micro_f1
    except Exception:
        return 0.0, 0.0

# ----------------------
# Train/Eval routines
# ----------------------
def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    """Train for one epoch with mixed precision; handle Inception aux loss when present."""
    model.train()
    running_loss = 0.0

    TP = TN = FP = FN = None
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            outputs = model(images)
            if isinstance(outputs, tuple):  # Inception train mode: (main, aux)
                main_out, aux_out = outputs
                loss = criterion(main_out, labels) + 0.4 * criterion(aux_out, labels)
                logits_for_metrics = main_out
            else:
                loss = criterion(outputs, labels)
                logits_for_metrics = outputs

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
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
    """Evaluate with global epoch-level sensitivity/specificity and per-class metrics."""
    model.eval()
    running_loss = 0.0
    TP = TN = FP = FN = None
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="Evaluating", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        if isinstance(outputs, tuple):  # just in case (should not happen in eval)
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
            # Macro AUROC (existing)
            auc_score = roc_auc_score(all_labels[:, valid_cols], all_preds[:, valid_cols], average='macro')
            # Micro AUROC (handles prevalence/imbalance by pooling)
            auc_micro_score = roc_auc_score(all_labels[:, valid_cols], all_preds[:, valid_cols], average='micro')
        else:
            auc_score = 0.0
            auc_micro_score = 0.0
    except Exception:
        auc_score = 0.0
        auc_micro_score = 0.0

    # Calculate Macro and Micro F1 at chosen per-class thresholds
    macro_f1, micro_f1 = compute_f1_at_thresholds(all_labels, all_preds, thresholds)

    avg_loss = running_loss / len(loader)
    return avg_loss, bal_acc, sens, spec, auc_score, auc_micro_score, macro_f1, micro_f1, all_labels, all_preds, sens_per_class, spec_per_class

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
        plt.plot(epochs, train_losses, label="Training Loss", color='blue', linewidth=2, marker='o', markersize=4)
        plt.plot(epochs, val_losses, label="Validation Loss", color='red', linewidth=2, marker='s', markersize=4)
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
        plt.plot(epochs, train_sens, label="Training Sensitivity", color='blue', linewidth=2, marker='o', markersize=4)
        plt.plot(epochs, train_spec, label="Training Specificity", color='lightblue', linewidth=2, marker='^', markersize=4)
        plt.plot(epochs, val_sens, label="Validation Sensitivity", color='red', linewidth=2, marker='s', markersize=4)
        plt.plot(epochs, val_spec, label="Validation Specificity", color='lightcoral', linewidth=2, marker='d', markersize=4)
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

# ----------------------
# Core training pipeline — runs for one model
# ----------------------
def run_for_model(model_name: str):
    global RESULTS_DIR, SAVE_PATH, METRICS_CSV, THRESHOLDS_PATH, SAVE_PATH_ANY

    pretty = {
        "swin_tiny": "SwinTiny",
        "vit_small": "ViTSmall",
        "deit_small": "DeiTSmall",
        "crossvit_small": "CrossViTSmall",
    }[model_name.lower()]

    RESULTS_DIR = ROOT_DIR / "results" / "ViT" / pretty
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_PATH = RESULTS_DIR / f"{model_name.lower()}_rfmid_best.pth"
    METRICS_CSV = RESULTS_DIR / f"{model_name.lower()}_metrics.csv"
    THRESHOLDS_PATH = RESULTS_DIR / "optimal_thresholds.npy"
    SAVE_PATH_ANY = RESULTS_DIR / f"{model_name.lower()}_rfmid_best_any_abnormal.pth"

    print(f"🚀 Starting {pretty} training with Sens/Spec tracking + AUC threshold calibration")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_labels = pd.read_csv(DATA_DIR / "2. Groundtruths" / "a. RFMiD_Training_Labels.csv")
    val_labels = pd.read_csv(DATA_DIR / "2. Groundtruths" / "b. RFMiD_Validation_Labels.csv")
    test_labels = pd.read_csv(DATA_DIR / "2. Groundtruths" / "c. RFMiD_Testing_Labels.csv")

    # Freeze label schema from TRAIN and reindex VAL/TEST to match
    label_columns = [c for c in train_labels.columns if c != "ID"]
    val_labels  = val_labels.reindex(columns=["ID"] + label_columns, fill_value=0)
    test_labels = test_labels.reindex(columns=["ID"] + label_columns, fill_value=0)

    # Compute per-class pos_weight from TRAIN ONLY (for BCEWithLogitsLoss)
    y = train_labels[label_columns].values
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    pos_weight = torch.tensor((neg / (pos + 1e-6)).astype(np.float32)).to(device)

    print(f"Training samples: {len(train_labels)}, Validation: {len(val_labels)}, Test: {len(test_labels)}")

    # Per-model transforms from pretrained weights
    train_transform    = vit_transforms(model_name, train=True)
    val_test_transform = vit_transforms(model_name, train=False)

    train_dataset = RFMiDDataset(DATA_DIR / "1. Original Images" / "a. Training Set", train_labels, label_columns, train_transform)
    val_dataset   = RFMiDDataset(DATA_DIR / "1. Original Images" / "b. Validation Set", val_labels, label_columns, val_test_transform)
    test_dataset  = RFMiDDataset(DATA_DIR / "1. Original Images" / "c. Testing Set", test_labels, label_columns, val_test_transform)

    train_loader = DataLoader(train_dataset, BATCH_SIZE, True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_dataset,   BATCH_SIZE, False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_dataset,  BATCH_SIZE, False, num_workers=NUM_WORKERS)

    # Build model and criterion
    model = build_model(model_name, num_classes=len(label_columns)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer/scheduler for ViT models (AdamW + cosine with warmup)
    backbone_params = list(model.backbone.parameters())
    classifier_params = list(model.classifier.parameters())

    optimizer = torch.optim.AdamW([
        {'params': backbone_params,   'lr': 5e-5, 'weight_decay': 0.05},  # Gentle LR for backbone
        {'params': classifier_params, 'lr': 5e-4, 'weight_decay': 0.00},  # Zero WD on head
    ])
    scheduler = cosine_with_warmup(optimizer, warmup_epochs=5, total_epochs=EPOCHS)

    # CSV header with per-class columns
    class_names = label_columns
    header = "epoch,train_loss,train_bal_acc,train_sens,train_spec,val_loss,val_bal_acc,val_sens,val_spec,val_auc"
    for class_name in class_names:
        header += f",val_sens_{class_name},val_spec_{class_name}"
    with open(METRICS_CSV, "w") as f:
        f.write(header + "\n")

    best_val_auc = 0.0
    best_any_auc = 0.0
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    train_sens_list, train_spec_list, val_sens_list, val_spec_list = [], [], [], []

    # Epoch 0: initial evaluation
    print("\n📊 Epoch 0: Evaluating initial model performance...")
    train_loss_0, train_bal_acc_0, train_sens_0, train_spec_0, *_, = evaluate_model(model, train_loader, criterion, device)
    val_loss_0, val_bal_acc_0, val_sens_0, val_spec_0, val_auc_0, _, _, _, _, _, val_sens_per_class_0, val_spec_per_class_0 = evaluate_model(model, val_loader, criterion, device)

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

    # Early stopping state (on validation AUC)
    best_val_auc_es = -1.0
    epochs_no_improve = 0
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss, train_bal_acc, train_sens, train_spec = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_bal_acc, val_sens, val_spec, val_auc, val_auc_micro, val_macro_f1, val_micro_f1, y_true_val_all, y_pred_val_all, val_sens_per_class, val_spec_per_class = evaluate_model(model, val_loader, criterion, device)

        scheduler.step()
        train_losses.append(train_loss); val_losses.append(val_loss)
        train_accs.append(train_bal_acc); val_accs.append(val_bal_acc)
        train_sens_list.append(train_sens); train_spec_list.append(train_spec)
        val_sens_list.append(val_sens); val_spec_list.append(val_spec)

        print(f"Train Balanced Acc: {train_bal_acc:.4f} | Sens: {train_sens:.4f} | Spec: {train_spec:.4f}")
        print(f"Val Balanced Acc: {val_bal_acc:.4f} | Sens: {val_sens:.4f} | Spec: {val_spec:.4f} | AUC: {val_auc:.4f} | Micro AUC: {val_auc_micro:.4f} | Macro F1: {val_macro_f1:.4f} | Micro F1: {val_micro_f1:.4f}")

        # Any-abnormal validation AUC and checkpoint
        y_true_any_val = (np.sum(y_true_val_all, axis=1) > 0).astype(np.int32)
        y_score_any_val = np.max(y_pred_val_all, axis=1)
        val_auc_any = roc_auc_score(y_true_any_val, y_score_any_val)
        if val_auc_any > best_any_auc:
            best_any_auc = val_auc_any
            torch.save({'model_state_dict': model.state_dict()}, SAVE_PATH_ANY)
            print(f"💾 Best ANY-ABNORMAL model saved! (val AUC_any={val_auc_any:.4f})")

        # Write metrics to CSV
        csv_line = f"{epoch},{train_loss:.6f},{train_bal_acc:.6f},{train_sens:.6f},{train_spec:.6f},"
        csv_line += f"{val_loss:.6f},{val_bal_acc:.6f},{val_sens:.6f},{val_spec:.6f},{val_auc:.6f}"
        for i in range(len(class_names)):
            csv_line += f",{val_sens_per_class[i]:.6f},{val_spec_per_class[i]:.6f}"
        with open(METRICS_CSV, "a") as f:
            f.write(csv_line + "\n")

        # Early stopping check (val AUC)
        if val_auc > best_val_auc_es + MIN_DELTA:
            best_val_auc_es = val_auc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"[ES] No val-AUC improvement for {epochs_no_improve}/{PATIENCE} epoch(s).")

        if epochs_no_improve >= PATIENCE:
            print(f"[ES] Early stopping triggered (patience={PATIENCE}).")
            plot_training_curves(train_losses, train_accs, val_losses, val_accs, RESULTS_DIR / "training_curves.png")
            plot_loss_curves(train_losses, val_losses, RESULTS_DIR / "loss_curves.png")
            plot_sensitivity_specificity_curves(train_sens_list, train_spec_list, val_sens_list, val_spec_list, RESULTS_DIR / "sensitivity_specificity_curves.png")
            break

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({'model_state_dict': model.state_dict()}, SAVE_PATH)
            print(f"💾 Best model saved! (AUC={val_auc:.4f})")

        # Update plots
        plot_training_curves(train_losses, train_accs, val_losses, val_accs, RESULTS_DIR / "training_curves.png")
        plot_loss_curves(train_losses, val_losses, RESULTS_DIR / "loss_curves.png")
        plot_sensitivity_specificity_curves(train_sens_list, train_spec_list, val_sens_list, val_spec_list, RESULTS_DIR / "sensitivity_specificity_curves.png")

    # ----------------------
    # Threshold calibration
    # ----------------------
    print("\n📊 Calibrating thresholds (target specificity=0.8)...")
    if os.path.exists(SAVE_PATH):
        checkpoint = torch.load(SAVE_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Loaded best saved model for calibration.")
    else:
        print("⚠️ No best model saved yet (AUC=nan or interrupted training). Using last trained model instead.")

    _, _, _, _, val_auc, _, _, _, y_true_val, y_pred_val, _, _ = evaluate_model(model, val_loader, criterion, device)
    thresholds = compute_optimal_thresholds(np.array(y_true_val), np.array(y_pred_val), target_spec=0.8)
    np.save(THRESHOLDS_PATH, thresholds)
    print(f"Optimal thresholds saved to: {THRESHOLDS_PATH}")

    # =============== Overall "Any Abnormal vs Normal" metrics (single-row) ===============
    y_true_any_val = (np.sum(y_true_val, axis=1) > 0).astype(np.int32)
    y_score_any_val = np.max(y_pred_val, axis=1)
    thr_any, spec_any_val, sens_any_val = _pick_threshold_for_specificity(y_true_any_val, y_score_any_val, target_spec=0.8)
    print(f"🔧 Any-abnormal validation operating point @~0.80 specificity: thr={thr_any:.4f}, spec={spec_any_val:.4f}, sens={sens_any_val:.4f}")

    if os.path.exists(SAVE_PATH_ANY):
        checkpoint_any = torch.load(SAVE_PATH_ANY, map_location=device)
        model.load_state_dict(checkpoint_any['model_state_dict'])
        print("✅ Loaded best any-abnormal model for overall metrics.")
        _, _, _, _, _, _, _, _, y_true_val_anyCkpt, y_pred_val_anyCkpt, _, _ = evaluate_model(model, val_loader, criterion, device)
        y_true_any_val_ckpt  = (np.sum(y_true_val_anyCkpt, axis=1) > 0).astype(np.int32)
        y_score_any_val_ckpt = np.max(y_pred_val_anyCkpt, axis=1)
        thr_any, spec_any_val, sens_any_val = _pick_threshold_for_specificity(y_true_any_val_ckpt, y_score_any_val_ckpt, target_spec=0.8)
        print(f"🔧 Recomputed any-abnormal val operating point for any-ckpt: thr={thr_any:.4f}, spec={spec_any_val:.4f}, sens={sens_any_val:.4f}")
    else:
        print("⚠️ No any-abnormal checkpoint found; using current model for overall metrics.")

    _, _, _, _, _, _, _, _, y_true_test_all, y_pred_test_all, _, _ = evaluate_model(model, test_loader, criterion, device)
    y_true_any_test = (np.sum(y_true_test_all, axis=1) > 0).astype(np.int32)
    y_score_any_test = np.max(y_pred_test_all, axis=1)

    auc_any = roc_auc_score(y_true_any_test, y_score_any_test)
    y_pred_any_test = (y_score_any_test >= thr_any).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y_true_any_test, y_pred_any_test, labels=[0,1]).ravel()
    precision_at_thr = tp / (tp + fp + 1e-8)
    recall_at_thr = tp / (tp + fn + 1e-8)

    f1max, thr_f1, prec_f1, rec_f1 = _compute_f1max(y_true_any_test, y_score_any_test)

    overall_csv = RESULTS_DIR / "overall_any_abnormal_metrics.csv"
    with open(overall_csv, "w") as f:
        f.write("Metric,Value\n")
        f.write(f"AUC (%),{auc_any*100:.4f}\n")
        f.write(f"Threshold@0.80spec,{thr_any:.6f}\n")
        f.write(f"Precision@Thr,{precision_at_thr*100:.4f}\n")
        f.write(f"Recall@Thr (%),{recall_at_thr*100:.4f}\n")
        f.write(f"TP,{int(tp)}\n")
        f.write(f"TN,{int(tn)}\n")
        f.write(f"FP,{int(fp)}\n")
        f.write(f"FN,{int(fn)}\n")
        f.write(f"F1max,{f1max:.6f}\n")
        f.write(f"F1max_Threshold,{thr_f1:.6f}\n")
        f.write(f"F1max_Precision,{prec_f1*100:.4f}\n")
        f.write(f"F1max_Recall (%),{rec_f1*100:.4f}\n")
    print(f"🧾 Wrote overall any-abnormal metrics to: {overall_csv}")

    # ================== SAVE PER-IMAGE OUTPUTS FOR LATER STATS (DeLong & McNemar) ==================
    val_stats_npz = RESULTS_DIR / "vit_anyabnormal_val_outputs.npz"
    np.savez(val_stats_npz,
             ids=val_labels["ID"].values,
             y_true=(np.sum(y_true_val, axis=1) > 0).astype(np.int8),
             y_score=np.max(y_pred_val, axis=1).astype(np.float32))
    print(f"💾 Saved validation per-image any-abnormal outputs to: {val_stats_npz}")

    test_stats_npz = RESULTS_DIR / "vit_anyabnormal_test_outputs.npz"
    np.savez(test_stats_npz,
             ids=test_labels["ID"].values,
             y_true=y_true_any_test.astype(np.int8),
             y_score=y_score_any_test.astype(np.float32),
             y_pred_at_spec80=(y_score_any_test >= thr_any).astype(np.int8),
             thr_spec80=float(thr_any))
    print(f"💾 Saved test per-image any-abnormal outputs to: {test_stats_npz}")

    # 🔁 Restore best-AUC checkpoint for per-class final evaluation
    if os.path.exists(SAVE_PATH):
        checkpoint = torch.load(SAVE_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Restored best-AUC checkpoint for per-class thresholded evaluation.")
    else:
        print("⚠️ Expected best-AUC checkpoint not found; proceeding with current weights.")

    print("\n🧪 Final evaluation on test set (using calibrated thresholds)...")
    test_loss, test_bal_acc, test_sens, test_spec, test_auc, test_auc_micro, test_macro_f1, test_micro_f1, test_all_labels, test_all_preds, test_sens_per_class, test_spec_per_class = evaluate_model(model, test_loader, criterion, device, thresholds)
    print(f"Test Balanced Acc: {test_bal_acc:.4f} | Sens: {test_sens:.4f} | Spec: {test_spec:.4f} | AUC: {test_auc:.4f} | Micro AUC: {test_auc_micro:.4f} | Macro F1: {test_macro_f1:.4f} | Micro F1: {test_micro_f1:.4f}")

    test_results_csv = RESULTS_DIR / "final_test_results.csv"
    with open(test_results_csv, "w") as f:
        f.write("class_name,test_sensitivity,test_specificity\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name},{test_sens_per_class[i]:.6f},{test_spec_per_class[i]:.6f}\n")

    # Calculate additional metrics for overall test results
    test_tp, test_tn, test_fp, test_fn = overall_confusion_from_batches(test_all_labels, test_all_preds, thresholds)
    precision_overall = test_tp / (test_tp + test_fp + 1e-8)
    recall_overall = test_tp / (test_tp + test_fn + 1e-8)
    
    # Calculate F1max for overall results
    f1max_overall, thr_f1_overall, prec_f1_overall, rec_f1_overall = _compute_overall_f1max(test_all_labels, test_all_preds)
    
    overall_results_csv = RESULTS_DIR / "overall_test_results.csv"
    with open(overall_results_csv, "w") as f:
        f.write("metric,value\n")
        f.write(f"test_loss,{test_loss:.6f}\n")
        f.write(f"test_balanced_accuracy,{test_bal_acc:.6f}\n")
        f.write(f"test_sensitivity,{test_sens:.6f}\n")
        f.write(f"test_specificity,{test_spec:.6f}\n")
        f.write(f"test_auc,{test_auc:.6f}\n")
        f.write(f"test_auc_micro,{test_auc_micro:.6f}\n")
        f.write(f"test_macro_f1,{test_macro_f1:.6f}\n")
        f.write(f"test_micro_f1,{test_micro_f1:.6f}\n")
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

    print(f"\n🎉 Training completed for {pretty}!")
    print(f"Best validation AUC: {best_val_auc:.4f}")
    print(f"Model saved to: {SAVE_PATH}")
    print(f"Thresholds saved to: {THRESHOLDS_PATH}")
    print(f"Training metrics saved to: {METRICS_CSV}")
    print(f"Final test per-class results saved to: {test_results_csv}")
    print(f"Overall test results saved to: {overall_results_csv}")

# ----------------------
# Helpers for any-abnormal operating point
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

def _compute_overall_f1max(all_labels, all_preds):
    """Compute F1max for overall multi-class classification using micro-averaging"""
    # For multi-class, we need to compute F1max using micro-averaging
    # This means treating all classes as one big binary classification problem
    
    # Flatten all predictions and labels for micro-averaging
    y_true_flat = all_labels.flatten()
    y_score_flat = all_preds.flatten()
    
    # Only consider valid predictions (where there are both positive and negative samples)
    valid_mask = np.isfinite(y_score_flat)
    y_true_valid = y_true_flat[valid_mask]
    y_score_valid = y_score_flat[valid_mask]
    
    if len(y_true_valid) == 0 or len(np.unique(y_true_valid)) < 2:
        return 0.0, 0.5, 0.0, 0.0
    
    # Use precision_recall_curve for micro-averaged F1max
    precision, recall, thr = precision_recall_curve(y_true_valid, y_score_valid)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    f1_use = f1[:-1]  # Remove last element as it's always 1.0
    best_idx = int(np.nanargmax(f1_use))
    return float(f1_use[best_idx]), float(thr[best_idx]), float(precision[best_idx]), float(recall[best_idx])

# ----------------------
# Generate overall metrics for all ViT models
# ----------------------
def generate_all_vit_overall_metrics():
    """Generate overall metrics for all 4 ViT models after training"""
    print("🚀 Generating Overall Test Results for All ViT Models")
    print("=" * 70)
    
    try:
        # Set up paths
        ROOT_DIR = Path(__file__).resolve().parent.parent
        RESULTS_DIR = ROOT_DIR / "results" / "ViT"
        DATA_DIR = ROOT_DIR / "data" / "RFMiD_Challenge_Dataset"
        
        # Model configurations
        model_cfgs = {
            "SwinTiny": {
                "model_name": "swin_tiny",
                "checkpoint_name": "swin_tiny_rfmid_best.pth",
                "metrics_name": "swin_tiny_metrics.csv",
            },
            "ViTSmall": {
                "model_name": "vit_small", 
                "checkpoint_name": "vit_small_rfmid_best.pth",
                "metrics_name": "vit_small_metrics.csv",
            },
            "DeiTSmall": {
                "model_name": "deit_small",
                "checkpoint_name": "deit_small_rfmid_best.pth", 
                "metrics_name": "deit_small_metrics.csv",
            },
            "CrossViTSmall": {
                "model_name": "crossvit_small",
                "checkpoint_name": "crossvit_small_rfmid_best.pth",
                "metrics_name": "crossvit_small_metrics.csv",
            },
        }
        
        # Load data once
        print("\n📊 Loading test data...")
        test_labels = pd.read_csv(DATA_DIR / "2. Groundtruths" / "c. RFMiD_Testing_Labels.csv")
        train_labels = pd.read_csv(DATA_DIR / "2. Groundtruths" / "a. RFMiD_Training_Labels.csv")
        
        # Freeze label schema from TRAIN and reindex TEST to match
        label_columns = [c for c in train_labels.columns if c != "ID"]
        test_labels = test_labels.reindex(columns=["ID"] + label_columns, fill_value=0)
        class_names = label_columns
        
        # Setup device and criterion
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        y = train_labels[label_columns].values
        pos = y.sum(axis=0)
        neg = y.shape[0] - pos
        pos_weight = torch.tensor((neg / (pos + 1e-6)).astype(np.float32)).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        summary_data = []
        
        for model_display_name, config in model_cfgs.items():
            print(f"\n🔍 Processing {model_display_name}...")
            
            model_path = RESULTS_DIR / model_display_name
            checkpoint_path = model_path / config["checkpoint_name"]
            metrics_path = model_path / config["metrics_name"]
            thresholds_path = model_path / "optimal_thresholds.npy"
            
            if not checkpoint_path.exists():
                print(f"   ⚠️ Checkpoint not found: {checkpoint_path}")
                continue
                
            if not thresholds_path.exists():
                print(f"   ⚠️ Thresholds not found: {thresholds_path}")
                continue
                
            # Load best validation AUC from metrics
            try:
                metrics_df = pd.read_csv(metrics_path)
                best_val_auc = metrics_df['val_auc'].max()
            except:
                print(f"   ⚠️ Could not load validation AUC from {metrics_path}")
                best_val_auc = 0.0
            
            # Load calibrated thresholds
            calibrated_thresholds = np.load(thresholds_path)
            
            # Create test dataset and loader with model-specific transforms
            test_transform = vit_transforms(config["model_name"], train=False)
            test_dataset = RFMiDDataset(DATA_DIR / "1. Original Images" / "c. Testing Set", test_labels, label_columns, test_transform)
            test_loader = DataLoader(test_dataset, 16, False, num_workers=0)
            
            # Build model
            model = build_model(config["model_name"], len(class_names)).to(device)
            
            # Load trained weights
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            
            # Set model to evaluation mode
            model.eval()
            
            # Run evaluation with no-grad context
            print(f"   🧪 Running evaluation...")
            with torch.inference_mode():
                (
                    test_loss,
                    test_bal_acc,
                    test_sens,
                    test_spec,
                    test_auc,
                    test_auc_micro,
                    test_macro_f1,
                    test_micro_f1,
                    test_all_labels,
                    test_all_preds,
                    test_sens_per_class,
                    test_spec_per_class,
                ) = evaluate_model(model, test_loader, criterion, device, calibrated_thresholds)
            
            # Calculate additional metrics
            test_tp, test_tn, test_fp, test_fn = overall_confusion_from_batches(test_all_labels, test_all_preds, calibrated_thresholds)
            precision_overall = test_tp / (test_tp + test_fp + 1e-8)
            recall_overall = test_tp / (test_tp + test_fn + 1e-8)
            
            # Calculate F1max for overall results
            f1max_overall, thr_f1_overall, prec_f1_overall, rec_f1_overall = _compute_overall_f1max(test_all_labels, test_all_preds)
            
            # Display results
            print(f"   📊 Results:")
            print(f"      Loss: {test_loss:.6f}")
            print(f"      Balanced Accuracy: {test_bal_acc:.6f}")
            print(f"      Sensitivity: {test_sens:.6f}")
            print(f"      Specificity: {test_spec:.6f}")
            print(f"      AUC: {test_auc:.6f}")
            print(f"      Micro AUC: {test_auc_micro:.6f}")
            print(f"      Macro F1: {test_macro_f1:.6f}")
            print(f"      Micro F1: {test_micro_f1:.6f}")
            print(f"      Precision: {precision_overall:.6f}")
            print(f"      Recall: {recall_overall:.6f}")
            print(f"      F1max: {f1max_overall:.6f}")
            print(f"      TP: {test_tp}, TN: {test_tn}, FP: {test_fp}, FN: {test_fn}")
            
            # Write to overall_test_results.csv
            overall_results_csv = model_path / "overall_test_results.csv"
            with open(overall_results_csv, "w") as f:
                f.write("metric,value\n")
                f.write(f"test_loss,{test_loss:.6f}\n")
                f.write(f"test_balanced_accuracy,{test_bal_acc:.6f}\n")
                f.write(f"test_sensitivity,{test_sens:.6f}\n")
                f.write(f"test_specificity,{test_spec:.6f}\n")
                f.write(f"test_auc,{test_auc:.6f}\n")
                f.write(f"test_auc_micro,{test_auc_micro:.6f}\n")
                f.write(f"test_macro_f1,{test_macro_f1:.6f}\n")
                f.write(f"test_micro_f1,{test_micro_f1:.6f}\n")
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
            
            print(f"   ✅ Updated: {overall_results_csv}")
            
            # Store for summary
            summary_data.append({
                'Model': model_display_name,
                'Loss': test_loss,
                'Balanced_Accuracy': test_bal_acc,
                'Sensitivity': test_sens,
                'Specificity': test_spec,
                'AUC': test_auc,
                'Micro_AUC': test_auc_micro,
                'Macro_F1': test_macro_f1,
                'Micro_F1': test_micro_f1,
                'Precision': precision_overall,
                'Recall': recall_overall,
                'F1max': f1max_overall,
                'Best_Val_AUC': best_val_auc,
                'TP': test_tp,
                'TN': test_tn,
                'FP': test_fp,
                'FN': test_fn
            })
        
        # Create summary comparison
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_csv = RESULTS_DIR / "all_models_summary.csv"
            summary_df.to_csv(summary_csv, index=False)
            print(f"\n📊 Summary saved to: {summary_csv}")
            
            # Display summary
            print(f"\n🏆 MODEL COMPARISON SUMMARY:")
            print("=" * 80)
            print(f"{'Model':<15} {'AUC':<8} {'Sens':<8} {'Spec':<8} {'Prec':<8} {'F1max':<8} {'Loss':<8}")
            print("-" * 80)
            for _, row in summary_df.iterrows():
                print(f"{row['Model']:<15} {row['AUC']:<8.4f} {row['Sensitivity']:<8.4f} {row['Specificity']:<8.4f} {row['Precision']:<8.4f} {row['F1max']:<8.4f} {row['Loss']:<8.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating overall metrics: {e}")
        import traceback
        traceback.print_exc()
        return False

# ----------------------
# Entry point
# ----------------------
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--generate-metrics":
        # Generate overall metrics for all trained ViT models
        success = generate_all_vit_overall_metrics()
        if success:
            print("\n✅ SUCCESS!")
            print("All ViT models now have complete overall test metrics.")
        else:
            print("\n❌ FAILED!")
            print("Please check the error messages above.")
    else:
        # Train all 4 ViT models
        model_names = ["swin_tiny", "vit_small", "deit_small", "crossvit_small"]
        for m in model_names:
            print(f"\n==================== {m.upper()} ====================")
            run_for_model(m)
