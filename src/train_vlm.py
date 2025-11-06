# src/train_vlm.py
# Vision-Language Model (CLIP/SigLIP) training script for RFMiD retinal fundus multi-disease classification
# - CLIP via OpenCLIP (ViT-L/14-336 or ViT-B/16)
# - SigLIP (base patch 16 384 or large)
# - Per-class logits from cosine similarity between image embeddings and cached class prompt embeddings
# - Temperature scaling and per-class thresholding

import os
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path
import random, warnings, numpy as np, pandas as pd, math
from PIL import Image

# Suppress PyTorch deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import open_clip
import timm  # For SigLIP
import timm.data
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
from sklearn.exceptions import UndefinedMetricWarning
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------
# Hard-coded parameters
# ----------------------
SEED, BATCH_SIZE, EPOCHS, NUM_WORKERS = 42, 4, 20, 0  # NUM_WORKERS=0 is macOS-safe
ACCUM_STEPS = 4
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

PATIENCE = 10
MIN_DELTA = 1e-4

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
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        labels = torch.tensor(row[self.label_columns].values, dtype=torch.float32)
        return image, labels

# ----------------------
# Cosine with warmup scheduler
# ----------------------
def cosine_with_warmup(optimizer, warmup_epochs=5, total_epochs=20):
    """Cosine decay with warmup scheduler"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ----------------------
# Transforms for CLIP/SigLIP
# ----------------------
# Cache for SigLIP data config to avoid rebuilding models
_SIGLIP_CFG_CACHE = {}

def vlm_transforms(model_name: str, train: bool):
    """Get transforms for CLIP/SigLIP models"""
    model_name = model_name.lower()
    
    if "clip" in model_name:
        # OpenCLIP-aligned transforms without instantiating a model (avoid double load)
        size = 336 if "336" in model_name else 224
        aug = [
            transforms.RandomResizedCrop(size, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                std=(0.26862954, 0.26130258, 0.27577711)),
        ]
        eval_tf = [
            transforms.Resize(size, interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                std=(0.26862954, 0.26130258, 0.27577711)),
        ]
        return transforms.Compose(aug if train else eval_tf)
    
    elif "siglip" in model_name:
        # Use timm's official config (with caching to avoid rebuilding)
        # Handle different naming across timm versions by trying common aliases
        if "large" in model_name:
            candidates = [
                "vit_large_patch16_siglip_384",  # common
                "siglip_large_patch16_384",      # older/alternate
                "vit_large_patch16_siglip_224",
                "siglip_large_patch16_224",
            ]
        else:
            candidates = [
                "vit_base_patch16_siglip_384",
                "siglip_base_patch16_384",
                "vit_base_patch16_siglip_224",
                "siglip_base_patch16_224",
            ]

        timm_name = None
        tmp = None
        for name in candidates:
            try:
                tmp = timm.create_model(name, pretrained=True, num_classes=0)
                timm_name = name
                break
            except Exception:
                continue
        if timm_name is None:
            available = [m for m in timm.list_models("*siglip*")]
            raise RuntimeError(f"No matching SigLIP model found among {candidates}. Available: {available}")

        if timm_name not in _SIGLIP_CFG_CACHE:
            _SIGLIP_CFG_CACHE[timm_name] = timm.data.resolve_data_config({}, model=tmp)
        cfg = _SIGLIP_CFG_CACHE[timm_name]
        return timm.data.create_transform(**cfg, is_training=train)
    
    else:
        raise ValueError(f"Unknown VLM model name: {model_name}")

# ----------------------
# Model builders
# ----------------------
def build_model(model_name, num_classes, class_names):
    """Return CLIP or SigLIP model for RFMiD classification"""
    model_name = model_name.lower()
    
    if "biomedclip" in model_name or "clip" in model_name:
        # CLIP via OpenCLIP
        if "biomedclip" in model_name:
            clip_name = "ViT-L-14"
            base, _, _ = open_clip.create_model_and_transforms(clip_name, pretrained="biomedclip")
        else:
            if "336" in model_name:
                clip_name = "ViT-L-14-336"
            elif "b16" in model_name or "vit-b/16" in model_name:
                clip_name = "ViT-B-16"
            else:
                clip_name = "ViT-B-16"  # safe default
            base, _, _ = open_clip.create_model_and_transforms(clip_name, pretrained="openai")
        print(f"✅ Loaded CLIP {clip_name}")
        
        # Safer freeze strategy: freeze all first, then unfreeze vision
        for p in base.parameters():
            p.requires_grad = False
        for p in base.visual.parameters():
            p.requires_grad = True
        
        # Wrapper (learn logit scale separately so optimizer sees it)
        wrapper = CLIPWrapper(base, class_names)
        return wrapper
        
    elif "siglip" in model_name:
        # SigLIP (handle naming differences across timm versions)
        if "large" in model_name:
            candidates = [
                "vit_large_patch16_siglip_384",
                "siglip_large_patch16_384",
                "vit_large_patch16_siglip_224",
                "siglip_large_patch16_224",
            ]
        else:
            candidates = [
                "vit_base_patch16_siglip_384",
                "siglip_base_patch16_384",
                "vit_base_patch16_siglip_224",
                "siglip_base_patch16_224",
            ]
        backbone = None
        last_err = None
        for name in candidates:
            try:
                backbone = timm.create_model(name, pretrained=True, num_classes=0)
                print(f"✅ Loaded SigLIP {name}")
                break
            except Exception as e:
                last_err = e
                continue
        if backbone is None:
            available = [m for m in timm.list_models("*siglip*")]
            raise RuntimeError(f"Could not create SigLIP model from {candidates}. Available: {available}. Last error: {last_err}")
        # Default: freeze backbone; we'll selectively unfreeze later at runtime
        for p in backbone.parameters():
            p.requires_grad = False
        return SigLIPWrapper(backbone, class_names)
    
    else:
        raise ValueError(f"Unknown VLM model name: {model_name}")

# ----------------------
# CLIP Wrapper with cached class embeddings
# ----------------------
class CLIPWrapper(nn.Module):
    def __init__(self, base_model, class_names):
        super().__init__()
        self.base_model = base_model
        # logit_scale as an explicit learnable param in the wrapper
        with torch.no_grad():
            init = base_model.logit_scale.exp().clamp(1.0, 100.0).log()
        self.logit_scale = nn.Parameter(init.clone())
        # Per-class bias to allow class-specific calibration when encoders are frozen
        self.class_bias = nn.Parameter(torch.zeros(len(class_names), dtype=torch.float32))
        
        self.class_names = class_names
        self.register_buffer("cached_text_embeds", torch.empty(0), persistent=True)  # placeholder - will be resized in cache_prompts
        self._prompts_cached = False

    def cache_prompts(self, device):
        if self._prompts_cached:
            return
        
        # Prompt set for RFMiD classes
        class_prompts = {
            "Disease_Risk": [
                "a retinal fundus photograph showing any retinal pathology or abnormal finding",
                "a retinal fundus photograph with signs of disease or abnormal retina"
            ],
            "DR": [
                "a retinal fundus photograph with diabetic retinopathy featuring microaneurysms dot blot hemorrhages and hard exudates near the macula",
                "a retinal fundus photograph with diabetic retinopathy showing scattered intraretinal hemorrhages cotton wool spots and possible neovascularization"
            ],
            "ARMD": [
                "a retinal fundus photograph with age related macular degeneration showing yellow drusen and macular pigmentary change",
                "a retinal fundus photograph with age related macular degeneration with subretinal deposits and geographic atrophy in the macula"
            ],
            "MH": [
                "a retinal fundus photograph with a full thickness macular hole showing a round foveal defect and surrounding cuff of fluid",
                "a retinal fundus photograph with a macular hole causing loss of foveal reflex and central retinal thinning"
            ],
            "DN": [
                "a retinal fundus photograph with drusen at the posterior pole with multiple yellow white deposits in the macula",
                "a retinal fundus photograph showing soft and hard drusen clustered in the macular region"
            ],
            "MYA": [
                "a retinal fundus photograph with myopic degeneration showing peripapillary atrophy and a tilted optic disc",
                "a retinal fundus photograph with pathologic myopia showing chorioretinal atrophy and visible choroidal vessels"
            ],
            "BRVO": [
                "a retinal fundus photograph with branch retinal vein occlusion showing sectoral flame and dot blot hemorrhages and venous dilation",
                "a retinal fundus photograph with branch retinal vein occlusion with cotton wool spots and tortuous veins in one quadrant"
            ],
            "TSLN": [
                "a retinal fundus photograph with tessellated fundus showing prominent choroidal pattern and generalized fundus tessellation",
                "a retinal fundus photograph with tessellated fundus and visible large choroidal vessels due to thinning"
            ],
            "ERM": [
                "a retinal fundus photograph with epiretinal membrane showing macular surface wrinkling and a glistening sheen",
                "a retinal fundus photograph with epiretinal membrane causing retinal folds and distortion at the macula"
            ],
            "LS": [
                "a retinal fundus photograph with laser photocoagulation scars showing multiple pale discrete burns in the peripheral retina",
                "a retinal fundus photograph showing panretinal photocoagulation scars with evenly spaced whitening spots"
            ],
            "MS": [
                "a retinal fundus photograph with a macular scar showing fibrotic change and mottled retinal pigment epithelium at the fovea",
                "a retinal fundus photograph with central chorioretinal scar and macular fibrosis"
            ],
            "CSR": [
                "a retinal fundus photograph with central serous chorioretinopathy showing a dome shaped serous detachment at the macula",
                "a retinal fundus photograph with central serous chorioretinopathy showing a well circumscribed area of subretinal fluid"
            ],
            "ODC": [
                "a retinal fundus photograph with optic disc cupping showing increased cup to disc ratio and rim thinning",
                "a retinal fundus photograph with glaucomatous cupping and vertical elongation of the optic cup"
            ],
            "CRVO": [
                "a retinal fundus photograph with central retinal vein occlusion showing diffuse blood and thunder hemorrhages and dilated tortuous veins",
                "a retinal fundus photograph with central retinal vein occlusion and disc edema with cotton wool spots"
            ],
            "TV": [
                "a retinal fundus photograph with tortuous retinal vessels showing generalized vessel tortuosity",
                "a retinal fundus photograph with markedly serpentine arterioles and venules across the posterior pole"
            ],
            "AH": [
                "a retinal fundus photograph with hypertensive arteriolar change showing arteriolar narrowing arteriovenous nicking and copper wiring",
                "a retinal fundus photograph with chronic hypertension signs including focal arteriolar narrowing and reflective vessel walls"
            ],
            "ODP": [
                "a retinal fundus photograph with an optic disc pit showing a small gray depression on the optic nerve head",
                "a retinal fundus photograph with congenital optic disc pit and adjacent serous maculopathy risk"
            ],
            "ODE": [
                "a retinal fundus photograph with optic disc edema showing blurred hyperemic disc margins and obscured vessels",
                "a retinal fundus photograph with swollen optic nerve head and peripapillary flame hemorrhages"
            ],
            "ST": [
                "a retinal fundus photograph with posterior staphyloma showing outpouching curvature and peripapillary atrophy in high myopia",
                "a retinal fundus photograph with posterior staphyloma and elongated axial contour at the posterior pole"
            ],
            "AION": [
                "a retinal fundus photograph with anterior ischemic optic neuropathy showing pallid swollen optic disc",
                "a retinal fundus photograph with anterior ischemic optic neuropathy and sectoral disc edema with altitudinal pattern"
            ],
            "PT": [
                "a retinal fundus photograph with papillitis showing hyperemic swollen optic disc and thickened nerve fiber layer",
                "a retinal fundus photograph with optic neuritis appearance including disc edema and blurred margins"
            ],
            "RT": [
                "a retinal fundus photograph with a retinal tear showing horseshoe shaped break with surrounding hemorrhage",
                "a retinal fundus photograph with peripheral retinal tear and adjacent lattice degeneration"
            ],
            "RS": [
                "a retinal fundus photograph with retinoschisis showing smooth dome shaped peripheral splitting with well demarcated borders",
                "a retinal fundus photograph with degenerative retinoschisis and immobile thin inner layer"
            ],
            "CRS": [
                "a retinal fundus photograph with chorioretinal scar showing well defined atrophic area with pigment clumping",
                "a retinal fundus photograph with old chorioretinal scar from prior inflammation or laser"
            ],
            "EDN": [
                "a retinal fundus photograph with macular edema showing thickening and circinate hard exudates",
                "a retinal fundus photograph with retinal edema and loss of foveal depression"
            ],
            "RPEC": [
                "a retinal fundus photograph with retinal pigment epithelium changes showing mottling and hyper or hypopigmentation",
                "a retinal fundus photograph with irregular retinal pigment epithelium at the macula and midperiphery"
            ],
            "MHL": [
                "a retinal fundus photograph with lamellar macular hole showing partial thickness foveal defect with irregular foveal contour",
                "a retinal fundus photograph with lamellar macular hole and epiretinal proliferation"
            ],
            "RP": [
                "a retinal fundus photograph with retinitis pigmentosa showing bone spicule pigmentation attenuated vessels and waxy disc pallor",
                "a retinal fundus photograph with peripheral bone spicule pigment and narrow arterioles typical of retinitis pigmentosa"
            ],
            "OTHER": [
                "a retinal fundus photograph with an uncommon retinal pathology not listed among other classes",
                "a retinal fundus photograph with miscellaneous retinal abnormality outside predefined categories"
            ],
        }
        
        # Encode all class prompts and average per class
        all_txt = []
        for cname in self.class_names:
            variants = class_prompts.get(cname, [f"a retinal fundus photograph with {cname.lower()}"])
            tokens = open_clip.tokenize(variants).to(device)
            self.base_model.eval()
            with torch.no_grad():
                emb = self.base_model.encode_text(tokens)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                emb = emb.mean(dim=0, keepdim=True)  # average over variants
            all_txt.append(emb)
        
        txt = torch.cat(all_txt, dim=0)  # [num_classes, D]
        
        # Resize and copy to keep it as a registered buffer
        if self.cached_text_embeds.numel() == 0:
            self.cached_text_embeds.resize_as_(txt.detach().cpu()).copy_(txt.detach().cpu())
        else:
            self.cached_text_embeds.copy_(txt.detach().cpu())
        self._prompts_cached = True

    def forward(self, x):
        device = x.device
        self.cache_prompts(device)
        # Clamp logit scale to a safe range to avoid runaway temperatures
        with torch.no_grad():
            self.logit_scale.data.clamp_(math.log(1.0), math.log(100.0))
        img = self.base_model.encode_image(x)  # pooled & projected
        img = img / img.norm(dim=-1, keepdim=True)
        txt = self.cached_text_embeds.to(device)
        logits = self.logit_scale.exp() * (img @ txt.T)
        logits = logits + self.class_bias  # broadcast per-class bias
        return logits

# ----------------------
# SigLIP Wrapper
# ----------------------
class SigLIPWrapper(nn.Module):
    def __init__(self, backbone, class_names):
        super().__init__()
        self.backbone = backbone
        
        # Add classifier head
        in_features = backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, len(class_names))
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

# ----------------------
# Stage-wise fine-tuning helpers
# ----------------------
def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False

def unfreeze_last_vit_blocks(model, n_blocks=2):
    # Works for both CLIP visual.transformer and timm ViT
    blocks = None
    if isinstance(model, CLIPWrapper):
        blocks = list(model.base_model.visual.transformer.resblocks)
    elif isinstance(model, SigLIPWrapper):
        # For timm ViT-like backbones
        if hasattr(model.backbone, "blocks"):
            blocks = list(model.backbone.blocks)
        elif hasattr(model.backbone, "stages"):
            # fallback if SigLIP uses stages
            blocks = []
            for s in model.backbone.stages:
                if hasattr(s, "blocks"): blocks.extend(list(s.blocks))
    if blocks:
        for b in blocks[-n_blocks:]:
            for p in b.parameters():
                p.requires_grad = True

# ----------------------
# Temperature scaling (same as before)
# ----------------------
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
    # Move optimization to CPU unless CUDA is available; L-BFGS is often more stable on CPU
    if device.type != "cuda":
        logits, labels = logits.cpu(), labels.cpu()
        dev = torch.device("cpu")
    else:
        dev = device
    scaler = _TempScaler(init_T=1.0).to(dev)
    optimizer = torch.optim.LBFGS(scaler.parameters(), lr=0.1, max_iter=50, line_search_fn="strong_wolfe")
    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    
    logits, labels = logits.to(dev), labels.to(dev)
    
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
# Metric helpers (same as train_hybrid.py)
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

def compute_f1_at_thresholds(all_labels, all_preds, thresholds=None):
    from sklearn.metrics import f1_score
    
    if thresholds is None:
        thresholds = np.full(all_preds.shape[1], 0.5)
    
    preds_binary = (all_preds > thresholds).astype(int)
    
    try:
        macro_f1 = f1_score(all_labels, preds_binary, average='macro', zero_division=0)
        micro_f1 = f1_score(all_labels, preds_binary, average='micro', zero_division=0)
        return macro_f1, micro_f1
    except Exception:
        return 0.0, 0.0

# ----------------------
# Train/Eval routines (same structure as train_hybrid.py)
# ----------------------
def train_one_epoch(model, loader, criterion, optimizer, device, scaler, accum_steps=4):
    model.train()
    running_loss = 0.0
    TP = TN = FP = FN = None
    optimizer.zero_grad(set_to_none=True)

    device_type = device.type  # 'cuda', 'mps', or 'cpu'
    for step, (images, labels) in enumerate(tqdm(loader, desc="Training", leave=False), 1):
        images, labels = images.to(device), labels.to(device)

        with torch.amp.autocast(device_type=device_type, enabled=(device_type in ["cuda", "mps"])):
            logits = model(images)
            loss = criterion(logits, labels) / accum_steps

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if step % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if step % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * accum_steps

        preds = (torch.sigmoid(logits) > 0.5).float()
        tp = (preds * labels).sum(dim=0); tn = ((1 - preds) * (1 - labels)).sum(dim=0)
        fp = (preds * (1 - labels)).sum(dim=0); fn = ((1 - preds) * labels).sum(dim=0)
        if TP is None:
            TP, TN, FP, FN = tp, tn, fp, fn
        else:
            TP += tp; TN += tn; FP += fp; FN += fn

    # Final optimizer step for remaining gradient accumulation tail
    if (step % accum_steps) != 0:
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    sens = (TP.sum() / (TP.sum() + FN.sum() + 1e-6)).item()
    spec = (TN.sum() / (TN.sum() + FP.sum() + 1e-6)).item()
    bal_acc = 0.5 * (sens + spec)
    avg_loss = running_loss / len(loader)
    return avg_loss, bal_acc, sens, spec

@torch.no_grad()
def evaluate_model(model, loader, criterion, device, thresholds=None):
    model.eval()
    running_loss = 0.0
    TP = TN = FP = FN = None
    all_preds, all_labels = [], []
    
    pbar = tqdm(loader, desc="Evaluating", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        running_loss += loss.item()
        
        probs = torch.sigmoid(logits).cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        if thresholds is None:
            preds = (probs > 0.5).astype(float)
        else:
            preds = (probs > thresholds).astype(float)
        
        # Compute confusion components in NumPy to avoid unnecessary CPU<->Torch churn
        tp = (preds * labels_np).sum(axis=0)
        tn = ((1 - preds) * (1 - labels_np)).sum(axis=0)
        fp = (preds * (1 - labels_np)).sum(axis=0)
        fn = ((1 - preds) * labels_np).sum(axis=0)
        
        if TP is None:
            TP = torch.tensor(tp)
            TN = torch.tensor(tn)
            FP = torch.tensor(fp)
            FN = torch.tensor(fn)
        else:
            TP += torch.tensor(tp); TN += torch.tensor(tn); FP += torch.tensor(fp); FN += torch.tensor(fn)
        
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
            auc_micro_score = roc_auc_score(all_labels[:, valid_cols], all_preds[:, valid_cols], average='micro')
        else:
            auc_score = 0.0
            auc_micro_score = 0.0
    except Exception:
        auc_score = 0.0
        auc_micro_score = 0.0
    
    macro_f1, micro_f1 = compute_f1_at_thresholds(all_labels, all_preds, thresholds)
    
    avg_loss = running_loss / len(loader)
    return avg_loss, bal_acc, sens, spec, auc_score, auc_micro_score, macro_f1, micro_f1, all_labels, all_preds, sens_per_class, spec_per_class

def overall_confusion_from_batches(all_labels, all_preds, thresholds=None):
    thr = 0.5 if thresholds is None else thresholds
    preds = (all_preds > thr).astype(float)
    TP = (preds * all_labels).sum()
    TN = ((1 - preds) * (1 - all_labels)).sum()
    FP = (preds * (1 - all_labels)).sum()
    FN = ((1 - preds) * all_labels).sum()
    return int(TP), int(TN), int(FP), int(FN)

# ----------------------
# Plotting functions (same as train_hybrid.py)
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
# Core training pipeline — runs for one VLM model
# ----------------------
def run_for_model(model_name: str):
    global RESULTS_DIR, SAVE_PATH, METRICS_CSV, THRESHOLDS_PATH, SAVE_PATH_ANY

    pretty = {
        "clip_vit_b16": "CLIPViTB16",
        "siglip_base_384": "SigLIPBase384",
    }[model_name.lower()]

    RESULTS_DIR = ROOT_DIR / "results" / "VLM" / pretty
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_PATH = RESULTS_DIR / f"{model_name.lower()}_rfmid_best.pth"
    METRICS_CSV = RESULTS_DIR / f"{model_name.lower()}_metrics.csv"
    THRESHOLDS_PATH = RESULTS_DIR / "optimal_thresholds.npy"
    SAVE_PATH_ANY = RESULTS_DIR / f"{model_name.lower()}_rfmid_best_any_abnormal.pth"

    print(f"🚀 Starting {pretty} training with Sens/Spec tracking + AUC threshold calibration")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

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

    # Per-model transforms
    train_transform    = vlm_transforms(model_name, train=True)
    val_test_transform = vlm_transforms(model_name, train=False)

    train_dataset = RFMiDDataset(DATA_DIR / "1. Original Images" / "a. Training Set", train_labels, label_columns, train_transform)
    val_dataset   = RFMiDDataset(DATA_DIR / "1. Original Images" / "b. Validation Set", val_labels, label_columns, val_test_transform)
    test_dataset  = RFMiDDataset(DATA_DIR / "1. Original Images" / "c. Testing Set", test_labels, label_columns, val_test_transform)

    pin = torch.cuda.is_available()
    # Toggle for using weighted sampler; default False to avoid double-weighting with pos_weight
    USE_WEIGHTED_SAMPLER = False
    if USE_WEIGHTED_SAMPLER:
        # Safer sampler: give all examples a base weight, boost positives by rarity
        cls_freq = train_labels[label_columns].sum(axis=0) + 1e-6
        per_cls_w = (len(train_labels) - cls_freq) / cls_freq
        ex_w = (train_labels[label_columns].values * per_cls_w.values).sum(axis=1)
        ex_w = np.asarray(ex_w, dtype=np.float64)
        ex_w = ex_w + 1.0  # base weight so all-negative samples still get sampled
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.tensor(ex_w, dtype=torch.double), len(ex_w), replacement=True
        )
        train_loader = DataLoader(train_dataset, BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS,
                                  pin_memory=pin, persistent_workers=NUM_WORKERS>0)
    else:
        # Deterministic shuffling with a seeded generator
        g = torch.Generator()
        g.manual_seed(SEED)
        train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                                  generator=g, pin_memory=pin, persistent_workers=NUM_WORKERS>0)
    val_loader   = DataLoader(val_dataset,   BATCH_SIZE, False, num_workers=NUM_WORKERS,
                              pin_memory=pin, persistent_workers=NUM_WORKERS>0)
    test_loader  = DataLoader(test_dataset,  BATCH_SIZE, False, num_workers=NUM_WORKERS,
                              pin_memory=pin, persistent_workers=NUM_WORKERS>0)

    # Build model and criterion
    model = build_model(model_name, num_classes=len(label_columns), class_names=label_columns).to(device)
    # Avoid double-weighting: if sampler is enabled, drop pos_weight
    criterion = nn.BCEWithLogitsLoss(pos_weight=None if USE_WEIGHTED_SAMPLER else pos_weight)

    # ---------------- Stage-wise LR: Stage 1 (freeze, train small head) ----------------
    freeze_all(model)
    if isinstance(model, CLIPWrapper):
        # Only train logit_scale in stage 1
        for p in model.base_model.visual.parameters():
            p.requires_grad = False
        # Re-enable grad for logit_scale after global freeze
        if hasattr(model, "logit_scale"):
            model.logit_scale.requires_grad = True
        if hasattr(model, "class_bias"):
            model.class_bias.requires_grad = True
        stage1_params = [
            {"params": [model.logit_scale, model.class_bias], "lr": 1e-4, "weight_decay": 0.0},
        ]
    elif isinstance(model, SigLIPWrapper):
        # Only train classifier head in stage 1
        for p in model.classifier.parameters():
            p.requires_grad = True
        stage1_params = [
            {"params": model.classifier.parameters(), "lr": 5e-4, "weight_decay": 0.0},
        ]
    else:
        stage1_params = [{"params": model.parameters(), "lr": 5e-5, "weight_decay": 0.05}]

    optimizer = torch.optim.AdamW(stage1_params)
    scheduler = cosine_with_warmup(optimizer, warmup_epochs=5, total_epochs=EPOCHS)

    # Epoch to switch to stage 2 (unfreeze last ViT blocks)
    UNFREEZE_EPOCH = 5
    UNFREEZE_BLOCKS = 4

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
    
    # Mixed precision scaler (CUDA only)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        # ---------------- Stage switch to Stage 2 ----------------
        if epoch == UNFREEZE_EPOCH:
            unfreeze_last_vit_blocks(model, n_blocks=UNFREEZE_BLOCKS)
            if isinstance(model, CLIPWrapper):
                params = [
                    {"params": [model.logit_scale, model.class_bias], "lr": 1e-4, "weight_decay": 0.0},
                ]
                last_blocks = list(model.base_model.visual.transformer.resblocks)[-UNFREEZE_BLOCKS:]
                block_params = []
                for b in last_blocks:
                    block_params += [p for p in b.parameters() if p.requires_grad]
                params.append({"params": block_params, "lr": 1e-5, "weight_decay": 0.05})
            else:  # SigLIPWrapper
                params = [
                    {"params": model.classifier.parameters(), "lr": 5e-4, "weight_decay": 0.0},
                ]
                if hasattr(model.backbone, "blocks"):
                    last_blocks = list(model.backbone.blocks)[-UNFREEZE_BLOCKS:]
                    block_params = []
                    for b in last_blocks:
                        block_params += [p for p in b.parameters() if p.requires_grad]
                    params.append({"params": block_params, "lr": 1e-5, "weight_decay": 0.05})
            optimizer = torch.optim.AdamW(params)
            scheduler = cosine_with_warmup(optimizer, warmup_epochs=2, total_epochs=max(3, EPOCHS - UNFREEZE_EPOCH))
            print(f"🔓 Unfroze last {UNFREEZE_BLOCKS} ViT blocks with LR=1e-5 (head/logit_scale kept higher LR)")
        train_loss, train_bal_acc, train_sens, train_spec = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, accum_steps=ACCUM_STEPS)
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
    # Temperature scaling & Threshold calibration
    # ----------------------
    print("\n🌡️ Calibrating temperature on validation logits...")
    if os.path.exists(SAVE_PATH):
        checkpoint = torch.load(SAVE_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Loaded best saved model for temperature calibration.")
    else:
        print("⚠️ No best model saved yet (AUC=nan or interrupted training). Using last trained model instead.")
    
    T = fit_temperature(model, val_loader, device)
    model = _WithTemp(model, T).to(device)
    
    # Persist temperature
    with open(RESULTS_DIR / "temperature.txt", "w") as f:
        f.write(f"{T:.6f}\n")
    print(f"✅ Learned temperature T = {T:.3f}")

    print("\n📊 Calibrating thresholds (target specificity=0.8)...")
    _, _, _, _, val_auc, _, _, _, y_true_val, y_pred_val, _, _ = evaluate_model(model, val_loader, criterion, device)
    thresholds = compute_optimal_thresholds(np.array(y_true_val), np.array(y_pred_val), target_spec=0.8)
    np.save(THRESHOLDS_PATH, thresholds)
    print(f"Optimal thresholds saved to: {THRESHOLDS_PATH}")

    # =============== Overall "Any Abnormal vs Normal" metrics ===============
    y_true_any_val = (np.sum(y_true_val, axis=1) > 0).astype(np.int32)
    y_score_any_val = np.max(y_pred_val, axis=1)
    thr_any, spec_any_val, sens_any_val = _pick_threshold_for_specificity(y_true_any_val, y_score_any_val, target_spec=0.8)
    print(f"🔧 Any-abnormal validation operating point @~0.80 specificity: thr={thr_any:.4f}, spec={spec_any_val:.4f}, sens={sens_any_val:.4f}")
    
    # Persist the any-abnormal operating point threshold
    with open(RESULTS_DIR / "any_abnormal_thr_at_spec80.txt", "w") as f:
        f.write(f"{thr_any:.6f}\n")
    print(f"💾 Saved any-abnormal threshold to: any_abnormal_thr_at_spec80.txt")

    if os.path.exists(SAVE_PATH_ANY):
        checkpoint_any = torch.load(SAVE_PATH_ANY, map_location=device)
        
        # Unwrap if temperature wrapper is active
        if isinstance(model, _WithTemp):
            base = model.base
            base.load_state_dict(checkpoint_any['model_state_dict'])
            # Re-wrap with the same temperature
            model = _WithTemp(base, T).to(device)
        else:
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

    # ================== SAVE PER-IMAGE OUTPUTS ==================
    val_stats_npz = RESULTS_DIR / "vlm_anyabnormal_val_outputs.npz"
    np.savez(val_stats_npz,
             ids=val_labels["ID"].values,
             y_true=(np.sum(y_true_val, axis=1) > 0).astype(np.int8),
             y_score=np.max(y_pred_val, axis=1).astype(np.float32))
    print(f"💾 Saved validation per-image any-abnormal outputs to: {val_stats_npz}")

    test_stats_npz = RESULTS_DIR / "vlm_anyabnormal_test_outputs.npz"
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
        # Remove _WithTemp wrapper to get original model
        if isinstance(model, _WithTemp):
            model = model.base
        # Load directly into the current (unwrapped) model
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Restored best-AUC checkpoint for per-class thresholded evaluation.")
    else:
        print("⚠️ Expected best-AUC checkpoint not found; proceeding with current weights.")
        # Remove _WithTemp wrapper if present
        if isinstance(model, _WithTemp):
            model = model.base
    
    # Load temperature if it was persisted
    temp_path = RESULTS_DIR / "temperature.txt"
    if temp_path.exists():
        with open(temp_path, "r") as f:
            T_eval = float(f.read().strip())
        model = _WithTemp(model, T_eval).to(device)
        print(f"✅ Loaded temperature T = {T_eval:.3f} for evaluation.")
    else:
        print("⚠️ No temperature file found; using unscaled logits.")

    print("\n🧪 Final evaluation on test set (using calibrated thresholds)...")
    test_loss, test_bal_acc, test_sens, test_spec, test_auc, test_auc_micro, test_macro_f1, test_micro_f1, test_all_labels, test_all_preds, test_sens_per_class, test_spec_per_class = evaluate_model(model, test_loader, criterion, device, thresholds)
    print(f"Test Balanced Acc: {test_bal_acc:.4f} | Sens: {test_sens:.4f} | Spec: {test_spec:.4f} | AUC: {test_auc:.4f} | Micro AUC: {test_auc_micro:.4f} | Macro F1: {test_macro_f1:.4f} | Micro F1: {test_micro_f1:.4f}")

    test_results_csv = RESULTS_DIR / "final_test_results.csv"
    with open(test_results_csv, "w") as f:
        f.write("class_name,test_sensitivity,test_specificity\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name},{test_sens_per_class[i]:.6f},{test_spec_per_class[i]:.6f}\n")

    # Calculate additional metrics
    test_tp, test_tn, test_fp, test_fn = overall_confusion_from_batches(test_all_labels, test_all_preds, thresholds)
    precision_overall = test_tp / (test_tp + test_fp + 1e-8)
    recall_overall = test_tp / (test_tp + test_fn + 1e-8)
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
# Helper functions (from train_hybrid.py)
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
    y_true_flat = all_labels.flatten()
    y_score_flat = all_preds.flatten()
    valid_mask = np.isfinite(y_score_flat)
    y_true_valid = y_true_flat[valid_mask]
    y_score_valid = y_score_flat[valid_mask]
    
    if len(y_true_valid) == 0 or len(np.unique(y_true_valid)) < 2:
        return 0.0, 0.5, 0.0, 0.0
    
    precision, recall, thr = precision_recall_curve(y_true_valid, y_score_valid)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    f1_use = f1[:-1]
    best_idx = int(np.nanargmax(f1_use))
    return float(f1_use[best_idx]), float(thr[best_idx]), float(precision[best_idx]), float(recall[best_idx])

# ----------------------
# Entry point
# ----------------------
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--generate-metrics":
        print("❌ Metrics generation not yet implemented for VLM models.")
        print("Train models first, then implement generate_all_vlm_overall_metrics().")
    else:
        # Train selected smaller VLM models
        model_names = ["clip_vit_b16", "siglip_base_384"]
        for m in model_names:
            print(f"\n==================== {m.upper()} ====================")
            run_for_model(m)
