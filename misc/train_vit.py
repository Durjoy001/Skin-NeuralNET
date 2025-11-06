# src/train_composition_min.py
# Minimal ViT training on HDF5 frames -> TIRADS composition (3 classes)
# Now with training curves (loss & accuracy) and confusion matrix.
# Hard-coded params, single train/test split.

import os
os.environ.setdefault("MPLBACKEND", "Agg")  # headless rendering
from pathlib import Path
import random

import h5py
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from timm import create_model
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import matplotlib
try:
    matplotlib.use("Agg")  # belt + suspenders
except Exception:
    pass

import matplotlib.pyplot as plt  # safe to import now
# ----------------------
# Hard-coded parameters
# ----------------------
SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 25
LR = 3e-5
TEST_FRAC = 0.2
MODEL_NAME = "vit_tiny_patch16_224"  # use vit_base_patch16_224 on GPU

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DATABASE_PATH = DATA_DIR / "thyroid_dataset.h5"
METADATA_PATH = DATA_DIR / "metadata.csv"
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PATH = RESULTS_DIR / f"{MODEL_NAME}_composition_best.pth"
CURVES_PNG = RESULTS_DIR / "training_curves.png"
CM_PNG = RESULTS_DIR / "confusion_matrix.png"
METRICS_CSV = RESULTS_DIR / "metrics.csv"


# ----------------------
# Reproducibility
# ----------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 8))

# ----------------------
# Dataset
# ----------------------
class H5FrameDataset(Dataset):
    """Frames-as-samples dataset (reads directly from one HDF5)."""
    def __init__(self, h5_path, indices, id_to_label, transform):
        self.h5_path = str(h5_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.id_to_label = id_to_label
        self.transform = transform
        self._h5 = None
        self._annot = None
        self._images = None

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
            self._annot = self._h5["annot_id"]
            self._images = self._h5["image"]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        self._ensure_open()
        real_idx = int(self.indices[idx])
        a = self._annot[real_idx]
        annot_id = a.decode("utf-8") if isinstance(a, (bytes, np.bytes_)) else str(a)
        img_np = self._images[real_idx]               # (H, W) uint8
        img_pil = Image.fromarray(img_np).convert("RGB")  # to 3ch for ViT
        img = self.transform(img_pil)
        label = int(self.id_to_label[annot_id])
        return img, torch.tensor(label, dtype=torch.long)

    def __del__(self):
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass

# ----------------------
# Build splits (by nodule)
# ----------------------
def build_splits_by_nodule(h5_path, meta_df):
    with h5py.File(h5_path, "r") as h5f:
        annot_ids_raw = h5f["annot_id"][:]
    annot_ids = [a.decode("utf-8") if isinstance(a, (bytes, np.bytes_)) else str(a) for a in annot_ids_raw]

    id_to_indices = {}
    for i, a in enumerate(annot_ids):
        id_to_indices.setdefault(a, []).append(i)

    meta_df = meta_df[["annot_id", "ti-rads_composition"]].dropna()
    meta_df = meta_df[meta_df["annot_id"].isin(id_to_indices.keys())]

    id_to_label = {row["annot_id"]: int(row["ti-rads_composition"]) for _, row in meta_df.iterrows()}
    labeled_ids = [a for a in id_to_indices if a in id_to_label]

    random.shuffle(labeled_ids)
    n_test = max(1, int(len(labeled_ids) * TEST_FRAC))
    test_ids = set(labeled_ids[:n_test])
    train_ids = set(labeled_ids[n_test:])

    train_idx = sorted([i for a in train_ids for i in id_to_indices[a]])
    test_idx  = sorted([i for a in test_ids  for i in id_to_indices[a]])

    classes = sorted({id_to_label[a] for a in labeled_ids})
    return train_idx, test_idx, id_to_label, classes

# ----------------------
# Train / Eval
# ----------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, nseen = 0.0, 0
    correct = 0
    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # stats
        running_loss += loss.item() * imgs.size(0)
        nseen += imgs.size(0)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == labels).sum().item()

    epoch_loss = running_loss / max(1, nseen)
    epoch_acc = correct / max(1, nseen)
    return epoch_loss, epoch_acc

@torch.no_grad()
def evaluate_with_preds(model, loader, device):
    model.eval()
    preds, gts = [], []
    for imgs, labels in tqdm(loader, desc="Eval", leave=False):
        imgs = imgs.to(device)
        logits = model(imgs)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        preds.extend(pred.tolist())
        gts.extend(labels.numpy().tolist())
    acc = accuracy_score(gts, preds)
    return acc, np.array(gts), np.array(preds)

# ----------------------
# Plotting
# ----------------------
def plot_training_curves(losses, accs, out_path):
    """Safe, headless plotting of training curves (won't crash training)."""
    try:
        epochs = np.arange(1, len(losses) + 1)
        plt.figure()
        plt.plot(epochs, losses, label="Train Loss")
        plt.plot(epochs, accs, label="Train Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Training Curves")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
    except Exception as e:
        # don't let plotting kill the run
        print(f"[WARN] Failed to plot training curves to {out_path}: {e}")
    finally:
        try:
            plt.close()
        except Exception:
            pass

def plot_conf_mat(gts, preds, classes, out_path):
    cm = confusion_matrix(gts, preds, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    plt.figure()
    disp.plot(values_format="d", cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix (Test)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# ----------------------
# Main
# ----------------------
def main():
    assert DATABASE_PATH.exists(), f"Missing HDF5: {DATABASE_PATH}"
    assert METADATA_PATH.exists(), f"Missing metadata.csv: {METADATA_PATH}"

    # Load metadata & splits
    meta_df = pd.read_csv(METADATA_PATH)
    train_idx, test_idx, id_to_label, classes = build_splits_by_nodule(DATABASE_PATH, meta_df)
    num_classes = len(classes)
    print(f"Classes: {classes} (num={num_classes})")
    print(f"Train frames: {len(train_idx)} | Test frames: {len(test_idx)}")

    # Transforms
    tfm = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    # Datasets & loaders
    train_ds = H5FrameDataset(DATABASE_PATH, train_idx, id_to_label, transform=tfm)
    test_ds  = H5FrameDataset(DATABASE_PATH, test_idx,  id_to_label, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=False)

    # Model / Optim / Loss
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(MODEL_NAME, pretrained=True, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # Train
    best_acc = 0.0
    train_losses, train_accs = [], []

    # fresh metrics file header
    with open(METRICS_CSV, "w") as f:
        f.write("epoch,train_loss,train_acc,test_acc\n")

    for epoch in range(1, EPOCHS + 1):
        print(f"\n— Epoch {epoch}/{EPOCHS} —")
        epoch_loss, epoch_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_acc, gts, preds = evaluate_with_preds(model, test_loader, device)

        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        print(f"epoch_loss: {epoch_loss:.4f} | train_acc: {epoch_acc:.4f} | test_acc: {test_acc:.4f}")

        # 1) append metrics row immediately
        with open(METRICS_CSV, "a") as f:
            f.write(f"{epoch},{epoch_loss:.6f},{epoch_acc:.6f},{test_acc:.6f}\n")

        # 2) save/update curves after every epoch
        plot_training_curves(train_losses, train_accs, CURVES_PNG)

        # 3) save confusion matrix snapshot for this epoch
        epoch_cm_path = RESULTS_DIR / f"confusion_matrix_epoch_{epoch}.png"
        plot_conf_mat(gts, preds, classes, epoch_cm_path)

        # 4) checkpoint best model so far
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), SAVE_PATH)


    # Plots
    plot_training_curves(train_losses, train_accs, CURVES_PNG)
    # Final confusion matrix on the last eval preds
    plot_conf_mat(gts, preds, classes, CM_PNG)

    print(f"\nDone. Best test accuracy: {best_acc:.4f}")
    print(f"Saved best model to: {SAVE_PATH.resolve()}")
    print(f"Saved training curves to: {CURVES_PNG.resolve()}")
    print(f"Saved confusion matrix to: {CM_PNG.resolve()}")

if __name__ == "__main__":
    main()
