# code_final_fixed_v3.py
# Thesis-friendly fixed version:
# - Frozen feature extraction (ResNet/DenseNet/ViT/etc.) with deterministic cache loading
# - Real validation split support: uses data_dir/valid if available; otherwise stratified split from train features
# - Test set is NEVER used for training, threshold tuning, scaler/PCA fitting, or ensemble-weight optimization
# - Safe normal/no-tumor class detection; no more dangerous assumption that class 0 is normal
# - Binary AND multiclass threshold tuning using validation data
# - Per-class Recall/FNR/Precision/F1 in output CSV
# - Optional Confidence-Based Hard Sample Weighting with tumor-class FN penalty
# - Optimized soft-voting ensemble weights on validation data
# - Artifact export for UI/EXE inference: scaler, PCA, classifier(s), thresholds, metadata
# - Can reuse existing cached features without re-extracting deep features
# - TTA for valid/test; train remains deterministic without TTA

import os
import gc
import re
import json
import time
import glob
import argparse
import hashlib
import pickle
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision.models as models

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import xgboost as xgb

try:
    import torchxrayvision as xrv
    HAS_XRV = True
except Exception:
    HAS_XRV = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False

try:
    import lightgbm as lgb
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False


# -----------------------------
# v4 changes vs v3:
#   - Patient-level aggregation (patient id parsed from filename prefix "PID__...")
#   - Reports metrics BOTH with tuned thresholds and with plain argmax
#   - Bootstrap 95% confidence intervals for accuracy / balanced accuracy
#   - Optional patient-wise re-split of train+valid via --valid_seed (no image-level leakage)
#   - Defaults set to the configuration that performed best on the patient-wise corpus
# -----------------------------

# -----------------------------
# Configuration
# -----------------------------

@dataclass
class Cfg:
    data_dir: str
    out_dir: str
    cache_dir: str
    artifact_dir: str
    models_to_run: List[str]
    use_hard_weighting: bool
    backbones: List[str]
    batch_size: int
    num_workers: int
    amp: bool
    show_plots: bool
    xgb_gpu: bool
    lgbm_try_gpu: bool
    xgb_rounds: int
    lgb_rounds: int
    cat_rounds: int
    svm_c: float
    pca_var: float
    tta: int
    threshold_metric: str
    min_tumor_recall: float
    valid_ratio: float
    random_state: int
    normal_class: str
    hard_weight_gamma: float
    hard_weight_fn_penalty: float
    prefer_cache: bool
    cache_only: bool
    optimize_ensemble: bool
    save_artifacts: bool
    class_names: List[str]
    patient_level: bool = True
    bootstrap_n: int = 1000
    valid_seed: int = 0
    valid_patient_frac: float = 0.2
    img_size: int = 224


# -----------------------------
# Basic utilities
# -----------------------------

def set_torch_fast():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s.replace("+", "PLUS"))


def list_images(folder: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(exts)]


def discover_classes(data_dir: str, split: str = "train") -> List[str]:
    split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Missing split folder: {split_dir}")
    classes = []
    for name in sorted(os.listdir(split_dir)):
        p = os.path.join(split_dir, name)
        if os.path.isdir(p) and len(list_images(p)) > 0:
            classes.append(name)
    if len(classes) < 2:
        raise RuntimeError(f"Need >=2 class folders inside: {split_dir}")
    return classes


def make_class_to_idx(class_names: List[str]) -> Dict[str, int]:
    return {c: i for i, c in enumerate(class_names)}


def normalize_class_name(s: str) -> str:
    return s.lower().replace("-", "_").replace(" ", "_").strip()


def find_normal_class_index(class_names: List[str], requested: str = "auto") -> Optional[int]:
    """Find no-tumor/normal class safely. Never assume class index 0 is normal."""
    if requested and requested.lower() not in ["auto", "none", ""]:
        req = normalize_class_name(requested)
        for i, c in enumerate(class_names):
            if normalize_class_name(c) == req:
                return i
        raise ValueError(f"Requested normal_class='{requested}' not found in class_names={class_names}")

    aliases = {
        "no_tumor", "notumor", "no_tumour", "notumour", "normal", "healthy",
        "no", "none", "negative", "non_tumor", "non_tumour"
    }
    for i, c in enumerate(class_names):
        if normalize_class_name(c) in aliases:
            return i
    return None


def split_file_paths(data_dir: str, split: str, class_names: List[str]) -> List[str]:
    """Ordered paths exactly matching FolderDataset item order (class order, then sorted files)."""
    paths: List[str] = []
    split_dir = os.path.join(data_dir, split)
    for cls in class_names:
        paths.extend(list_images(os.path.join(split_dir, cls)))
    return paths


def patient_id_from_path(p: str) -> str:
    """Patient id = filename prefix before '__'. Falls back to the stem."""
    base = str(p).replace("\\", "/").rstrip("/").split("/")[-1]
    stem = os.path.splitext(base)[0]
    return stem.split("__")[0] if "__" in stem else stem


def patient_ids_for_split(data_dir: str, split: str, class_names: List[str],
                          expected: Optional[int] = None) -> Optional[np.ndarray]:
    paths = split_file_paths(data_dir, split, class_names)
    if not paths:
        return None
    if expected is not None and len(paths) != expected:
        tqdm.write(f"[PATIENT][WARN] {split}: {len(paths)} files but {expected} feature rows -> "
                   f"patient-level metrics disabled for this split.")
        return None
    return np.array([patient_id_from_path(p) for p in paths])


def aggregate_by_patient(proba: np.ndarray, y: np.ndarray, pids: np.ndarray):
    """Mean probability per patient. Returns (proba_pat, y_pat, pid_order)."""
    order, seen = [], set()
    for q in pids:
        if q not in seen:
            seen.add(q)
            order.append(q)
    P = np.zeros((len(order), proba.shape[1]), dtype=np.float64)
    Y = np.zeros(len(order), dtype=np.int64)
    for k, q in enumerate(order):
        m = pids == q
        P[k] = proba[m].mean(axis=0)
        Y[k] = int(np.bincount(y[m]).argmax())
    ssum = P.sum(axis=1, keepdims=True)
    P = P / np.maximum(ssum, 1e-12)
    return P, Y, order


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 1000,
                 seed: int = 42) -> Dict[str, float]:
    """Percentile bootstrap CI for accuracy and balanced accuracy."""
    if len(y_true) == 0 or n_boot <= 0:
        return {}
    rng = np.random.default_rng(seed)
    n = len(y_true)
    accs, bals = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        accs[b] = float((yt == yp).mean())
        recs = [float((yp[yt == c] == c).mean()) for c in np.unique(yt)]
        bals[b] = float(np.mean(recs)) if recs else 0.0
    return {
        "acc_ci_low": float(np.percentile(accs, 2.5)),
        "acc_ci_high": float(np.percentile(accs, 97.5)),
        "bal_ci_low": float(np.percentile(bals, 2.5)),
        "bal_ci_high": float(np.percentile(bals, 97.5)),
    }


def dataset_fingerprint(split_dir: str, class_names: List[str]) -> Dict[str, Any]:
    fp = {"dir": os.path.abspath(split_dir), "counts": {}, "mtimes": {}}
    for cls in class_names:
        cls_dir = os.path.join(split_dir, cls)
        files = list_images(cls_dir)
        fp["counts"][cls] = len(files)
        latest = 0.0
        for p in files:
            try:
                latest = max(latest, os.path.getmtime(p))
            except FileNotFoundError:
                pass
        fp["mtimes"][cls] = latest
    return fp


def hash_fp(obj: Dict[str, Any]) -> str:
    s = json.dumps(obj, sort_keys=True).encode("utf-8")
    return hashlib.md5(s).hexdigest()


# -----------------------------
# Dataset and feature extraction
# -----------------------------

def collate_pil(batch):
    imgs, ys = zip(*batch)
    return list(imgs), torch.tensor(ys, dtype=torch.long)


class FolderDataset(Dataset):
    def __init__(self, split_dir: str, class_names: List[str], transform, class_to_idx: Dict[str, int]):
        self.items: List[Tuple[str, int]] = []
        self.transform = transform
        for cls in class_names:
            idx = class_to_idx[cls]
            cls_dir = os.path.join(split_dir, cls)
            for p in list_images(cls_dir):
                self.items.append((p, idx))
        if len(self.items) == 0:
            raise RuntimeError(f"No images found in: {split_dir}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int):
        path, y = self.items[i]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, y


def build_extractor(backbone: str, device: torch.device, img_size: int = 224) -> Tuple[nn.Module, int]:
    b = backbone.lower().strip()
    if b == "resnet50":
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        feat_dim = 2048
        extractor = nn.Sequential(*list(base.children())[:-1])
    elif b in ["densenet121", "densenet", "d121"]:
        base = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        feat_dim = 1024
        extractor = nn.Sequential(base.features, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)))
    elif b == "resnet18":
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feat_dim = 512
        extractor = nn.Sequential(*list(base.children())[:-1])
    elif b in ["efficientnet_b0", "efficientnet", "effb0"]:
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        feat_dim = 1280
        extractor = nn.Sequential(base.features, nn.AdaptiveAvgPool2d(1))
    elif b in ["vit_b_16", "vit", "vitb16"]:
        base = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        if int(img_size) != 224:
            # ViT has fixed positional embeddings for 224/16 = 14x14 patches.
            # torchvision provides bicubic interpolation of those embeddings.
            if img_size % 16 != 0:
                raise ValueError(f"img_size must be divisible by 16 for ViT-B/16, got {img_size}")
            from torchvision.models.vision_transformer import interpolate_embeddings
            new_state = interpolate_embeddings(int(img_size), 16, base.state_dict())
            base = models.vit_b_16(weights=None, image_size=int(img_size))
            base.load_state_dict(new_state)
            tqdm.write(f"[VIT] Positional embeddings interpolated 224 -> {img_size} "
                       f"({(img_size//16)**2} patches)")
        feat_dim = base.hidden_dim
        base.heads = nn.Identity()
        extractor = base
    elif b in ["xraytorchvision", "xrv", "xrv_densenet"]:
        if not HAS_XRV:
            raise ImportError("torchxrayvision is not installed. Run: pip install torchxrayvision")
        xrv_model = xrv.models.DenseNet(weights="densenet121-res224-all")
        feat_dim = 1024
        pool = nn.AdaptiveAvgPool2d((1, 1))

        class _XRVExtractor(nn.Module):
            def __init__(self, model, pool):
                super().__init__()
                self.model = model
                self.pool = pool

            def forward(self, x):
                gray = x.mean(dim=1, keepdim=True)
                gray = gray * 2048.0 - 1024.0
                feats = self.model.features(gray)
                feats = self.pool(feats)
                feats = feats.flatten(1)
                return feats

        extractor = _XRVExtractor(xrv_model, pool).to(device).eval()
        return extractor, feat_dim
    else:
        raise ValueError(
            "Unknown backbone. Use: resnet50, densenet121, resnet18, efficientnet_b0, vit_b_16, xraytorchvision"
        )

    extractor = extractor.to(device).eval()
    if device.type == "cuda" and b not in ["vit_b_16", "vit", "vitb16"]:
        extractor = extractor.to(memory_format=torch.channels_last)
    return extractor, feat_dim


def base_preprocess_transform(img_size: int = 224) -> T.Compose:
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return T.Compose([T.Resize((int(img_size), int(img_size))), T.ToTensor(), normalize])


def tta_variants(n: int):
    variants = [("id", lambda im: im)]
    if n >= 2:
        variants.append(("hflip", lambda im: TF.hflip(im)))
    if n >= 3:
        variants.append(("rot_p7", lambda im: TF.rotate(im, 7)))
    if n >= 4:
        variants.append(("rot_m7", lambda im: TF.rotate(im, -7)))
    if n >= 5:
        def zoom(im: Image.Image) -> Image.Image:
            w, h = im.size
            crop_w, crop_h = int(w * 0.92), int(h * 0.92)
            left = (w - crop_w) // 2
            top = (h - crop_h) // 2
            im2 = TF.crop(im, top, left, crop_h, crop_w)
            return im2.resize((w, h))
        variants.append(("zoom", zoom))
    return variants[:max(1, n)]


def latest_cached_pair(cache_dir: str, backbone: str, split: str,
                       img_size: int = 224) -> Optional[Tuple[str, str]]:
    """Latest cached X/y pair. v4: image size is part of the key so that changing
    --img_size can never silently reuse features extracted at another resolution."""
    tag = f"{backbone}_s{int(img_size)}"
    x_pattern = os.path.join(cache_dir, f"X_{tag}_{split}_*.npy")
    candidates = sorted(glob.glob(x_pattern), key=lambda p: os.path.getmtime(p), reverse=True)
    for x_path in candidates:
        y_path = x_path.replace(os.path.basename(x_path), os.path.basename(x_path).replace(f"X_{tag}_", f"y_{tag}_"))
        if os.path.exists(y_path):
            return x_path, y_path
    return None


@torch.inference_mode()
def extract_features_split_cached(cfg: Cfg, backbone: str, split: str, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    split_dir = os.path.join(cfg.data_dir, split)

    if cfg.prefer_cache:
        cached = latest_cached_pair(cfg.cache_dir, backbone, split, cfg.img_size)
        if cached is not None:
            x_path, y_path = cached
            tqdm.write(f"[CACHE LATEST] Loaded {split} features for {backbone}: {os.path.basename(x_path)}")
            return np.load(x_path), np.load(y_path)
        elif cfg.cache_only:
            raise FileNotFoundError(f"Cache-only mode: missing cached features for backbone={backbone}, split={split}")

    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Missing split folder and cache not found: {split_dir}")

    class_to_idx = make_class_to_idx(cfg.class_names)
    tta_n = cfg.tta if split in ["valid", "test"] else 1
    tta_ids = [name for name, _ in tta_variants(tta_n)]

    fp = dataset_fingerprint(split_dir, cfg.class_names)
    fp.update({"backbone": backbone, "split": split, "amp": cfg.amp, "tta": tta_ids,
               "img_size": int(cfg.img_size)})
    fp_hash = hash_fp(fp)

    ensure_dir(cfg.cache_dir)
    _tag = f"{backbone}_s{int(cfg.img_size)}"
    x_path = os.path.join(cfg.cache_dir, f"X_{_tag}_{split}_{fp_hash}.npy")
    y_path = os.path.join(cfg.cache_dir, f"y_{_tag}_{split}_{fp_hash}.npy")

    if os.path.exists(x_path) and os.path.exists(y_path):
        tqdm.write(f"[CACHE HIT] Loaded {split} features for {backbone} from deterministic cache.")
        return np.load(x_path), np.load(y_path)

    if cfg.cache_only:
        raise FileNotFoundError(f"Cache-only mode: deterministic cache missing for {backbone}:{split}")

    extractor, _ = build_extractor(backbone, device, cfg.img_size)
    preprocess = base_preprocess_transform(cfg.img_size)

    ds = FolderDataset(split_dir, cfg.class_names, transform=T.Lambda(lambda im: im), class_to_idx=class_to_idx)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=2 if cfg.num_workers > 0 else None,
        collate_fn=collate_pil,
    )

    tqdm.write(f"[EXTRACT] {split} | {backbone} | N={len(ds)} | batch={cfg.batch_size} | tta={tta_n} | amp={cfg.amp}")

    X_list, y_list = [], []
    variants = tta_variants(tta_n)
    is_vit = backbone.lower().strip() in ["vit_b_16", "vit", "vitb16"]

    for imgs_pil, yb in tqdm(dl, desc=f"Extracting {backbone}:{split}", ncols=110):
        feats_sum = None
        for _, vfunc in variants:
            xb = torch.stack([preprocess(vfunc(im)) for im in imgs_pil], dim=0)
            xb = xb.to(device, non_blocking=True)
            if device.type == "cuda" and not is_vit:
                xb = xb.to(memory_format=torch.channels_last)
            if cfg.amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    feats = extractor(xb)
            else:
                feats = extractor(xb)
            if feats.dim() == 4:
                feats = feats.squeeze(-1).squeeze(-1)
            feats = feats.detach().float()
            feats_sum = feats if feats_sum is None else feats_sum + feats

        feats_avg = feats_sum / float(len(variants))
        X_list.append(feats_avg.cpu().numpy().astype(np.float32))
        y_list.append(yb.numpy().astype(np.int64))

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    np.save(x_path, X)
    np.save(y_path, y)

    del extractor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return X, y


def load_features_for_backbone_spec(cfg: Cfg, backbone_spec: str, split: str, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    parts = [p.strip() for p in backbone_spec.split("+") if p.strip()]
    if not parts:
        raise ValueError("Empty backbone_spec")
    Xs = []
    y_ref = None
    for bb in parts:
        X, y = extract_features_split_cached(cfg, bb, split, device)
        if y_ref is None:
            y_ref = y
        else:
            if len(y) != len(y_ref) or not np.array_equal(y, y_ref):
                raise RuntimeError(f"Label mismatch for fusion in split={split}, backbone={bb}.")
        Xs.append(X)
    X_fused = Xs[0] if len(Xs) == 1 else np.concatenate(Xs, axis=1)
    return X_fused, y_ref


def load_train_valid_test_features(cfg: Cfg, backbone_spec: str, device: torch.device):
    """Guarantees test is isolated; valid uses data_dir/valid if available or stratified split from train."""
    X_train_full, y_train_full = load_features_for_backbone_spec(cfg, backbone_spec, "train", device)

    valid_dir = os.path.join(cfg.data_dir, "valid")
    has_valid_dir = os.path.isdir(valid_dir)
    has_valid_cache = True
    try:
        # If prefer_cache/cache_only, this will also allow valid from cached features even without valid_dir.
        if has_valid_dir or latest_cached_pair(cfg.cache_dir, backbone_spec.split("+")[0].strip(), "valid", cfg.img_size):
            X_valid, y_valid = load_features_for_backbone_spec(cfg, backbone_spec, "valid", device)
        else:
            has_valid_cache = False
    except Exception as e:
        tqdm.write(f"[WARN] Could not load valid split/cache: {e}")
        has_valid_cache = False

    if has_valid_dir or has_valid_cache:
        X_train, y_train = X_train_full, y_train_full
        valid_source = "valid_folder_or_cache"
    else:
        if cfg.valid_ratio <= 0:
            raise ValueError("No valid split found and --valid_ratio <= 0. Set valid_ratio > 0 or create data_dir/valid.")
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_full,
            y_train_full,
            test_size=cfg.valid_ratio,
            stratify=y_train_full,
            random_state=cfg.random_state,
        )
        valid_source = f"stratified_from_train_{cfg.valid_ratio:.2f}"
        tqdm.write(f"[VALID] Created validation split from train features: train={len(y_train)}, valid={len(y_valid)}")

    # v4: optional patient-wise re-split of train+valid (avoids image-level leakage
    # that train_test_split would introduce between train and validation).
    if cfg.valid_seed and cfg.valid_seed > 0:
        p_tr = patient_ids_for_split(cfg.data_dir, "train", cfg.class_names, len(y_train))
        p_va = patient_ids_for_split(cfg.data_dir, "valid", cfg.class_names, len(y_valid))
        if p_tr is not None and p_va is not None:
            Xa = np.vstack([X_train, X_valid])
            ya = np.concatenate([y_train, y_valid])
            pa = np.concatenate([p_tr, p_va])
            pid_major: Dict[str, int] = {}
            for q in np.unique(pa):
                pid_major[q] = int(np.bincount(ya[pa == q]).argmax())
            rng = np.random.default_rng(cfg.valid_seed)
            vp: set = set()
            for c in sorted(set(pid_major.values())):
                grp = sorted([q for q, m in pid_major.items() if m == c])
                rng.shuffle(grp)
                k = max(1, int(round(len(grp) * cfg.valid_patient_frac)))
                vp.update(grp[:k])
            mask_v = np.isin(pa, list(vp))
            X_train, y_train = Xa[~mask_v], ya[~mask_v]
            X_valid, y_valid = Xa[mask_v], ya[mask_v]
            valid_source = f"patientwise_reslit_seed{cfg.valid_seed}_frac{cfg.valid_patient_frac:.2f}"
            tqdm.write(f"[VALID] Patient-wise re-split (seed={cfg.valid_seed}): "
                       f"train={len(y_train)} ({len(set(pa[~mask_v]))} patients), "
                       f"valid={len(y_valid)} ({len(vp)} patients), overlap=0")
        else:
            tqdm.write("[VALID][WARN] Patient ids unavailable -> keeping original train/valid split.")

    X_test, y_test = load_features_for_backbone_spec(cfg, backbone_spec, "test", device)
    return X_train, y_train, X_valid, y_valid, X_test, y_test, valid_source


# -----------------------------
# Metrics, thresholds, decisions
# -----------------------------

def ensure_proba_matrix(p: np.ndarray, n_classes: int) -> np.ndarray:
    p = np.asarray(p, dtype=np.float32)
    if n_classes == 2 and p.ndim == 1:
        return np.column_stack([1.0 - p, p]).astype(np.float32)
    if p.ndim == 1:
        raise ValueError(f"Expected probability matrix for n_classes={n_classes}, got shape={p.shape}")
    if p.shape[1] != n_classes:
        raise ValueError(f"Probability class dimension mismatch: expected {n_classes}, got {p.shape}")
    return p.astype(np.float32)


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "bal_acc":
        return float(balanced_accuracy_score(y_true, y_pred))
    if metric == "f1_macro":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    if metric == "recall_macro":
        return float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    if metric == "f1_pos":
        # only meaningful for binary; fallback for multiclass handled elsewhere
        return float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    return float(accuracy_score(y_true, y_pred))


def predict_with_thresholds(proba: np.ndarray, thresholds: np.ndarray, normal_idx: Optional[int]) -> np.ndarray:
    """Binary and multiclass threshold-based decision.

    For multiclass, choose highest normalized confidence among classes passing their threshold.
    If no class passes threshold, fallback to argmax.
    """
    proba = np.asarray(proba, dtype=np.float32)
    n_classes = proba.shape[1]
    thresholds = np.asarray(thresholds, dtype=np.float32)
    thresholds = np.clip(thresholds, 1e-6, 0.999999)

    if n_classes == 2:
        if normal_idx is not None and normal_idx in [0, 1]:
            tumor_idx = 1 - normal_idx
        else:
            tumor_idx = 1
            normal_idx = 0
        pred = np.full(proba.shape[0], normal_idx, dtype=np.int64)
        pred[proba[:, tumor_idx] >= thresholds[tumor_idx]] = tumor_idx
        return pred

    pred = np.empty(proba.shape[0], dtype=np.int64)
    norm_score = proba / thresholds.reshape(1, -1)
    eligible = proba >= thresholds.reshape(1, -1)
    for i in range(proba.shape[0]):
        if eligible[i].any():
            pred[i] = int(np.argmax(np.where(eligible[i], norm_score[i], -np.inf)))
        else:
            pred[i] = int(np.argmax(proba[i]))
    return pred


def tune_thresholds(
    y_true: np.ndarray,
    proba: np.ndarray,
    metric: str,
    min_tumor_recall: float,
    normal_idx: Optional[int],
    n_classes: int,
) -> Tuple[np.ndarray, float]:
    """Tune class-specific thresholds on validation data only.

    Binary: tunes tumor-vs-normal threshold.
    Multiclass: one-vs-rest threshold per class, then evaluated using multiclass decision rule.
    """
    proba = ensure_proba_matrix(proba, n_classes)
    thresholds = np.full(n_classes, 0.5, dtype=np.float32)
    grid = np.linspace(0.05, 0.95, 37)

    if n_classes == 2:
        tumor_idx = 1
        if normal_idx is not None and normal_idx in [0, 1]:
            tumor_idx = 1 - normal_idx
        best_t, best_v = 0.5, -1.0
        for t in grid:
            thr = thresholds.copy()
            thr[tumor_idx] = float(t)
            pred = predict_with_thresholds(proba, thr, normal_idx)
            if min_tumor_recall and min_tumor_recall > 0:
                tr = recall_score((y_true == tumor_idx).astype(int), (pred == tumor_idx).astype(int), zero_division=0)
                if tr < min_tumor_recall:
                    continue
            local_metric = metric if metric != "f1_pos" else "f1_pos"
            if local_metric == "f1_pos":
                v = f1_score((y_true == tumor_idx).astype(int), (pred == tumor_idx).astype(int), zero_division=0)
            else:
                v = score_predictions(y_true, pred, local_metric)
            if v > best_v:
                best_v, best_t = float(v), float(t)
        if best_v < 0:
            best_t = 0.5
        thresholds[tumor_idx] = best_t
        pred = predict_with_thresholds(proba, thresholds, normal_idx)
        return thresholds, score_predictions(y_true, pred, "bal_acc" if metric == "f1_pos" else metric)

    # Multiclass: tune each class as one-vs-rest first.
    for c in range(n_classes):
        y_bin = (y_true == c).astype(np.int64)
        if y_bin.sum() == 0:
            thresholds[c] = 0.5
            continue
        best_t, best_v = 0.5, -1.0
        for t in grid:
            pred_bin = (proba[:, c] >= t).astype(np.int64)
            # For tumor classes, enforce minimum one-vs-rest recall if requested.
            if min_tumor_recall and min_tumor_recall > 0 and (normal_idx is None or c != normal_idx):
                rc = recall_score(y_bin, pred_bin, zero_division=0)
                if rc < min_tumor_recall:
                    continue
            if metric == "accuracy":
                v = accuracy_score(y_bin, pred_bin)
            elif metric == "bal_acc":
                v = balanced_accuracy_score(y_bin, pred_bin)
            elif metric in ["f1_pos", "f1_macro"]:
                v = f1_score(y_bin, pred_bin, zero_division=0)
            else:
                v = recall_score(y_bin, pred_bin, zero_division=0)
            if v > best_v:
                best_v, best_t = float(v), float(t)
        thresholds[c] = best_t if best_v >= 0 else 0.5

    pred = predict_with_thresholds(proba, thresholds, normal_idx)
    final_metric = "f1_macro" if metric == "f1_pos" else metric
    return thresholds, score_predictions(y_true, pred, final_metric)


def eval_metrics_general(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> Dict[str, float]:
    n_classes = len(class_names)
    labels = list(range(n_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    out: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    fnr_list = []
    for i, cname in enumerate(class_names):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - tp - fn - fp
        recall_i = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        precision_i = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        fnr_i = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1_i = float(2 * precision_i * recall_i / (precision_i + recall_i)) if (precision_i + recall_i) > 0 else 0.0
        key = safe_name(cname)
        out[f"recall_{key}"] = recall_i
        out[f"precision_{key}"] = precision_i
        out[f"f1_{key}"] = f1_i
        out[f"FNR_{key}"] = fnr_i
        out[f"support_{key}"] = float((y_true == i).sum())
        fnr_list.append(fnr_i)

    out["FNR_macro"] = float(np.mean(fnr_list))

    if n_classes == 2:
        out.update({
            "pos_f1_label1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "pos_precision_label1": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "pos_recall_label1": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        })
    return out


def cm_counts(cm: np.ndarray):
    if cm.shape == (2, 2):
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        return tn, fp, fn, tp
    return None


# -----------------------------
# Plots and reports
# -----------------------------

def save_confusion_matrix(cm: np.ndarray, title: str, out_path: str, class_names: List[str]):
    fig = plt.figure(figsize=(6.5, 5.8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=25, ha="right")
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0 if cm.size and cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_bar_plot(df: pd.DataFrame, metric: str, out_path: str, title: str):
    fig = plt.figure(figsize=(10, 4.8))
    plt.bar(df["name"].tolist(), df[metric].values)
    plt.title(title)
    plt.ylabel(metric)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Training algorithms
# -----------------------------

def compute_confidence_weights(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    normal_idx: Optional[int],
    gamma: float = 2.0,
    fn_penalty: float = 5.0,
) -> np.ndarray:
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42, n_jobs=-1)
    clf.fit(X, y)
    probs = clf.predict_proba(X)
    p_true = probs[np.arange(len(y)), y]
    weights = (1.0 - p_true) ** gamma

    if normal_idx is not None:
        tumor_mask = y != normal_idx
        weights[tumor_mask] *= fn_penalty
        tqdm.write(f"[WEIGHT] Tumor FN penalty applied to classes != '{normal_idx}'")
    else:
        # Safe fallback: no semantic tumor-vs-normal penalty if normal class cannot be identified.
        tqdm.write("[WEIGHT][WARN] Normal/no_tumor class not found. Applying difficulty weighting only, no tumor FN penalty.")

    weights = weights / max(np.mean(weights), 1e-8)
    return weights.astype(np.float32)


def train_xgb_proba(X_tr, y_tr, X_va, y_va, X_te, rounds: int, use_gpu: bool, n_classes: int, sample_weight=None):
    dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=sample_weight)
    dvalid = xgb.DMatrix(X_va, label=y_va)
    dtest = xgb.DMatrix(X_te)

    params = {
        "max_depth": 6,
        "eta": 0.02,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 2.0,
        "alpha": 1.0,
        "min_child_weight": 2,
        "seed": 42,
        "nthread": -1,
        "tree_method": "hist",
        "device": "cuda" if use_gpu else "cpu",
        "eval_metric": "logloss" if n_classes == 2 else "mlogloss",
    }

    if n_classes == 2:
        params["objective"] = "binary:logistic"
        neg = float((y_tr == 0).sum())
        pos = float((y_tr == 1).sum())
        if pos > 0 and neg > 0:
            params["scale_pos_weight"] = neg / pos
    else:
        params["objective"] = "multi:softprob"
        params["num_class"] = n_classes

    tqdm.write(f"[TRAIN] XGBoost ({'GPU' if use_gpu else 'CPU'}) rounds={rounds}")
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=rounds,
        evals=[(dvalid, "valid")],
        verbose_eval=False,
    )

    p_va = booster.predict(dvalid)
    p_te = booster.predict(dtest)
    if n_classes == 2:
        p_va = ensure_proba_matrix(p_va, 2)
        p_te = ensure_proba_matrix(p_te, 2)
    else:
        p_va = p_va.reshape(-1, n_classes).astype(np.float32)
        p_te = p_te.reshape(-1, n_classes).astype(np.float32)
    name = "XGBoost_GPU" if use_gpu else "XGBoost_CPU"
    return p_va, p_te, name, booster


def train_catboost_proba(X_tr, y_tr, X_va, y_va, X_te, rounds: int, n_classes: int, sample_weight=None):
    if not HAS_CATBOOST:
        return None, None, "CatBoost_MISSING", None

    cat = CatBoostClassifier(
        iterations=rounds,
        learning_rate=0.03,
        depth=6,
        task_type="GPU" if torch.cuda.is_available() else "CPU",
        verbose=False,
        random_state=42,
        auto_class_weights="Balanced" if sample_weight is None else None,
        loss_function="Logloss" if n_classes == 2 else "MultiClass",
    )

    tqdm.write(f"[TRAIN] CatBoost rounds={rounds}")
    cat.fit(X_tr, y_tr, sample_weight=sample_weight, eval_set=(X_va, y_va), use_best_model=False)

    p_va = cat.predict_proba(X_va).astype(np.float32)
    p_te = cat.predict_proba(X_te).astype(np.float32)
    return ensure_proba_matrix(p_va, n_classes), ensure_proba_matrix(p_te, n_classes), "CatBoost", cat


def train_svm_proba(X_tr, y_tr, X_va, X_te, c: float, n_classes: int, sample_weight=None):
    svm = SVC(kernel="rbf", C=c, probability=True, class_weight="balanced", random_state=42)
    tqdm.write(f"[TRAIN] SVM RBF -> fitting on {X_tr.shape[0]} samples")
    svm.fit(X_tr, y_tr, sample_weight=sample_weight)
    p_va = svm.predict_proba(X_va).astype(np.float32)
    p_te = svm.predict_proba(X_te).astype(np.float32)
    return ensure_proba_matrix(p_va, n_classes), ensure_proba_matrix(p_te, n_classes), "SVM_RBF", svm


def train_lgbm_proba(X_tr, y_tr, X_va, y_va, X_te, rounds: int, try_gpu: bool, n_classes: int, sample_weight=None):
    if not HAS_LGBM:
        return None, None, "LightGBM_MISSING", None

    params = dict(
        n_estimators=rounds,
        learning_rate=0.02,
        num_leaves=63,
        max_depth=6,
        min_data_in_leaf=20,
        min_gain_to_split=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    if n_classes == 2:
        model = lgb.LGBMClassifier(**params, objective="binary", class_weight="balanced")
    else:
        model = lgb.LGBMClassifier(**params, objective="multiclass", num_class=n_classes)

    mode = "CPU"
    if try_gpu:
        model.set_params(device_type="gpu", gpu_use_dp=False, max_bin=255, verbose=-1)
        mode = "GPU_TRY"

    tqdm.write(f"[TRAIN] LightGBM ({mode}) rounds={rounds}")
    try:
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], sample_weight=sample_weight)
    except Exception as e:
        if try_gpu:
            tqdm.write(f"[WARN] LightGBM GPU failed ({e}). Retrying on CPU.")
            model.set_params(device_type="cpu")
            mode = "CPU"
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], sample_weight=sample_weight)
        else:
            raise

    p_va = model.predict_proba(X_va).astype(np.float32)
    p_te = model.predict_proba(X_te).astype(np.float32)
    return ensure_proba_matrix(p_va, n_classes), ensure_proba_matrix(p_te, n_classes), f"LightGBM_{mode}", model


def make_predict_fn(model_name: str, model_obj: Any, n_classes: int):
    lname = model_name.lower()
    if "xgboost" in lname:
        def _predict(X_in):
            p = model_obj.predict(xgb.DMatrix(X_in))
            return ensure_proba_matrix(p, n_classes)
        return _predict
    else:
        def _predict(X_in):
            return ensure_proba_matrix(model_obj.predict_proba(X_in), n_classes)
        return _predict


# -----------------------------
# Ensemble
# -----------------------------

def default_ensemble_weights(names: List[str]) -> List[float]:
    w_map = {"xgboost": 0.40, "svm": 0.30, "lightgbm": 0.15, "catboost": 0.15}
    weights = []
    for n in names:
        ln = n.lower()
        w = 1.0
        for k, v in w_map.items():
            if k in ln:
                w = v
        weights.append(w)
    s = sum(weights)
    return [float(w / s) for w in weights]


def weighted_average_probs(probs: List[np.ndarray], weights: List[float]) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / max(float(weights.sum()), 1e-8)
    out = np.zeros_like(probs[0], dtype=np.float32)
    for w, p in zip(weights, probs):
        out += float(w) * p.astype(np.float32)
    return out


def generate_weight_grid(n: int, step: float = 0.1):
    units = int(round(1.0 / step))
    if n == 1:
        yield [1.0]
        return

    def rec(k, remaining, prefix):
        if k == n - 1:
            yield prefix + [remaining]
        else:
            for v in range(remaining + 1):
                yield from rec(k + 1, remaining - v, prefix + [v])

    for combo_units in rec(0, units, []):
        if sum(combo_units) == units and any(v > 0 for v in combo_units):
            yield [v / units for v in combo_units]


def optimize_ensemble_weights(
    probs_va: List[np.ndarray],
    names: List[str],
    y_va: np.ndarray,
    n_classes: int,
    normal_idx: Optional[int],
    metric: str,
    min_tumor_recall: float,
) -> Tuple[List[float], np.ndarray, float]:
    if len(probs_va) == 1:
        thresholds, score = tune_thresholds(y_va, probs_va[0], metric, min_tumor_recall, normal_idx, n_classes)
        return [1.0], thresholds, score

    best_w = default_ensemble_weights(names)
    best_p = weighted_average_probs(probs_va, best_w)
    best_thr, best_score = tune_thresholds(y_va, best_p, metric, min_tumor_recall, normal_idx, n_classes)

    for weights in generate_weight_grid(len(probs_va), step=0.1):
        p = weighted_average_probs(probs_va, weights)
        thr, _ = tune_thresholds(y_va, p, metric, min_tumor_recall, normal_idx, n_classes)
        pred = predict_with_thresholds(p, thr, normal_idx)
        score = score_predictions(y_va, pred, "f1_macro" if metric == "f1_pos" else metric)
        if score > best_score:
            best_score = float(score)
            best_w = [float(x) for x in weights]
            best_thr = thr

    tqdm.write(f"[ENSEMBLE] Optimized weights: {dict(zip(names, best_w))} | valid_score={best_score:.4f}")
    return best_w, best_thr, float(best_score)


# -----------------------------
# Artifact export
# -----------------------------

def save_artifact(
    cfg: Cfg,
    run_dir: str,
    backbone_spec: str,
    model_name: str,
    scaler: StandardScaler,
    pca: Optional[PCA],
    models_dict: Dict[str, Any],
    thresholds: np.ndarray,
    normal_idx: Optional[int],
    metrics_row: Dict[str, Any],
    ensemble_info: Optional[Dict[str, Any]] = None,
):
    if not cfg.save_artifacts:
        return None

    artifact_root = cfg.artifact_dir if cfg.artifact_dir else os.path.join(run_dir, "artifacts")
    artifact_name = safe_name(f"{backbone_spec}__{model_name}")
    artifact_path = os.path.join(artifact_root, artifact_name)
    ensure_dir(artifact_path)

    with open(os.path.join(artifact_path, "preprocess.pkl"), "wb") as f:
        pickle.dump({"scaler": scaler, "pca": pca}, f)

    with open(os.path.join(artifact_path, "models.pkl"), "wb") as f:
        pickle.dump(models_dict, f)

    metadata = {
        "version": "code_final_fixed_v3",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backbone_spec": backbone_spec,
        "model_name": model_name,
        "class_names": cfg.class_names,
        "normal_idx": normal_idx,
        "normal_class": cfg.class_names[normal_idx] if normal_idx is not None else None,
        "thresholds": [float(x) for x in thresholds],
        "pca_var": cfg.pca_var,
        "tta": cfg.tta,
        "threshold_metric": cfg.threshold_metric,
        "min_tumor_recall": cfg.min_tumor_recall,
        "metrics": {k: (float(v) if isinstance(v, (np.floating, float, np.integer, int)) else v) for k, v in metrics_row.items() if k not in ["cm_path"]},
        "ensemble_info": ensemble_info,
        "note": "Use train-fitted scaler/PCA and validation-tuned thresholds. Test metrics are for reporting only.",
    }
    with open(os.path.join(artifact_path, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return artifact_path


# -----------------------------
# Evaluation and main run
# -----------------------------

def evaluate_one(
    y_va: np.ndarray,
    y_te: np.ndarray,
    proba_va: np.ndarray,
    proba_te: np.ndarray,
    name: str,
    cfg: Cfg,
    n_classes: int,
    normal_idx: Optional[int],
    feat_time_sec: float,
    class_time_sec: float,
    thresholds: Optional[np.ndarray] = None,
    test_pids: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    if thresholds is None:
        thresholds, threshold_score = tune_thresholds(
            y_va, proba_va, cfg.threshold_metric, cfg.min_tumor_recall, normal_idx, n_classes
        )
    else:
        pred_va_tmp = predict_with_thresholds(proba_va, thresholds, normal_idx)
        threshold_score = score_predictions(y_va, pred_va_tmp, "f1_macro" if cfg.threshold_metric == "f1_pos" else cfg.threshold_metric)

    y_pred_va = predict_with_thresholds(proba_va, thresholds, normal_idx)
    y_pred_te = predict_with_thresholds(proba_te, thresholds, normal_idx)

    m_va = eval_metrics_general(y_va, y_pred_va, cfg.class_names)
    m_te = eval_metrics_general(y_te, y_pred_te, cfg.class_names)

    cm = confusion_matrix(y_te, y_pred_te, labels=range(n_classes))
    tn_fp_fn_tp = cm_counts(cm)

    row: Dict[str, Any] = {
        "backbone": "",
        "model": name,
        "accuracy": m_te["accuracy"],
        "balanced_accuracy": m_te["balanced_accuracy"],
        "f1_macro": m_te["f1_macro"],
        "precision_macro": m_te["precision_macro"],
        "recall_macro": m_te["recall_macro"],
        "FNR_macro": m_te["FNR_macro"],
        "feature_extraction_time_sec": float(feat_time_sec),
        "classification_time_sec": float(class_time_sec),
        "valid_accuracy": m_va["accuracy"],
        "valid_balanced_accuracy": m_va["balanced_accuracy"],
        "valid_f1_macro": m_va["f1_macro"],
        "thresholds_json": json.dumps([float(x) for x in thresholds]),
        "threshold_score": float(threshold_score),
    }

    # Add per-class metrics from test and validation in explicit columns.
    for k, v in m_te.items():
        if k not in row:
            row[k] = v
    for k, v in m_va.items():
        row[f"valid_{k}"] = v

    if tn_fp_fn_tp is not None:
        tn, fp, fn, tp = tn_fp_fn_tp
        row.update({"TN": tn, "FP": fp, "FN": fn, "TP": tp})

    # ---- v4: plain argmax (no thresholds) on the same probabilities ----
    y_pred_te_am = proba_te.argmax(axis=1)
    m_te_am = eval_metrics_general(y_te, y_pred_te_am, cfg.class_names)
    row["argmax_accuracy"] = m_te_am["accuracy"]
    row["argmax_balanced_accuracy"] = m_te_am["balanced_accuracy"]
    row["argmax_f1_macro"] = m_te_am["f1_macro"]
    row["argmax_FNR_macro"] = m_te_am["FNR_macro"]
    row["threshold_gain_acc"] = float(m_te["accuracy"] - m_te_am["accuracy"])

    # ---- v4: bootstrap confidence intervals (slice level) ----
    if cfg.bootstrap_n and cfg.bootstrap_n > 0:
        ci = bootstrap_ci(y_te, y_pred_te, cfg.bootstrap_n, cfg.random_state)
        for k, v in ci.items():
            row[f"slice_{k}"] = v

    # ---- v4: patient-level aggregation ----
    if cfg.patient_level and test_pids is not None and len(test_pids) == len(y_te):
        Pp, Yp, _ = aggregate_by_patient(proba_te, y_te, test_pids)
        pat_thr = predict_with_thresholds(Pp, thresholds, normal_idx)
        pat_am = Pp.argmax(axis=1)
        m_pt = eval_metrics_general(Yp, pat_thr, cfg.class_names)
        m_pa = eval_metrics_general(Yp, pat_am, cfg.class_names)
        row["n_patients"] = int(len(Yp))
        row["patient_accuracy"] = m_pt["accuracy"]
        row["patient_balanced_accuracy"] = m_pt["balanced_accuracy"]
        row["patient_f1_macro"] = m_pt["f1_macro"]
        row["patient_FNR_macro"] = m_pt["FNR_macro"]
        row["patient_argmax_accuracy"] = m_pa["accuracy"]
        row["patient_argmax_balanced_accuracy"] = m_pa["balanced_accuracy"]
        row["patient_argmax_f1_macro"] = m_pa["f1_macro"]
        row["patient_argmax_FNR_macro"] = m_pa["FNR_macro"]
        row["patient_threshold_gain_acc"] = float(m_pt["accuracy"] - m_pa["accuracy"])
        if cfg.bootstrap_n and cfg.bootstrap_n > 0:
            best_pred = pat_am if m_pa["accuracy"] >= m_pt["accuracy"] else pat_thr
            ci = bootstrap_ci(Yp, best_pred, cfg.bootstrap_n, cfg.random_state)
            for k, v in ci.items():
                row[f"patient_{k}"] = v

    return row, cm, thresholds


def run_for_spec(cfg: Cfg, backbone_spec: str, device: torch.device, run_dir: str) -> List[Dict[str, Any]]:
    n_classes = len(cfg.class_names)
    _test_pids_cache: Dict[str, Any] = {}
    if cfg.patient_level:
        _test_pids_cache["test"] = None  # filled after features are loaded
    normal_idx = find_normal_class_index(cfg.class_names, cfg.normal_class)
    if normal_idx is None:
        tqdm.write("[WARN] No normal/no_tumor class detected. Tumor-specific FN penalty/recall constraints will be disabled.")
    else:
        tqdm.write(f"[INFO] Normal/no_tumor class detected: index={normal_idx}, name='{cfg.class_names[normal_idx]}'")

    start_feat_time = time.time()
    X_tr, y_tr, X_va, y_va, X_te, y_te, valid_source = load_train_valid_test_features(cfg, backbone_spec, device)

    # v4: patient ids for the test split (used for patient-level aggregation)
    if cfg.patient_level:
        _pids = patient_ids_for_split(cfg.data_dir, "test", cfg.class_names, len(y_te))
        _test_pids_cache["test"] = _pids
        if _pids is not None:
            tqdm.write(f"[PATIENT] test: {len(y_te)} slices from {len(set(_pids))} patients")
    total_feature_extraction_time = time.time() - start_feat_time

    # Critical anti-leakage rule: scaler and PCA are fitted on TRAIN only.
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    pca = None
    if cfg.pca_var and cfg.pca_var > 0:
        pca = PCA(n_components=float(cfg.pca_var), random_state=cfg.random_state)
        X_tr_s = pca.fit_transform(X_tr_s)
        X_va_s = pca.transform(X_va_s)
        X_te_s = pca.transform(X_te_s)
        tqdm.write(f"[PCA] Applied PCA n_components={pca.n_components_}, explained={np.sum(pca.explained_variance_ratio_):.4f}")
    else:
        tqdm.write("[PCA] Disabled (pca_var=0.0).")

    sample_weights = None
    if cfg.use_hard_weighting:
        tqdm.write(f"[INFO] Computing Confidence-Based Sample Weights for {backbone_spec}...")
        sample_weights = compute_confidence_weights(
            X_tr_s, y_tr, n_classes, normal_idx,
            gamma=cfg.hard_weight_gamma,
            fn_penalty=cfg.hard_weight_fn_penalty,
        )

    rows: List[Dict[str, Any]] = []
    probs_dict: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    time_dict: Dict[str, float] = {}
    model_objects: Dict[str, Any] = {}
    predict_fns = {}

    requested = set([m.strip().lower() for m in cfg.models_to_run])

    if "xgb" in requested or "xgboost" in requested:
        start_class = time.time()
        pva_xgb, pte_xgb, name_xgb, xgb_model = train_xgb_proba(
            X_tr_s, y_tr, X_va_s, y_va, X_te_s,
            rounds=cfg.xgb_rounds,
            use_gpu=bool(cfg.xgb_gpu and device.type == "cuda"),
            n_classes=n_classes,
            sample_weight=sample_weights,
        )
        time_dict[name_xgb] = time.time() - start_class
        probs_dict[name_xgb] = (pva_xgb, pte_xgb)
        model_objects[name_xgb] = xgb_model
        predict_fns[name_xgb] = make_predict_fn(name_xgb, xgb_model, n_classes)

    if "svm" in requested:
        start_class = time.time()
        pva_svm, pte_svm, name_svm, svm_model = train_svm_proba(
            X_tr_s, y_tr, X_va_s, X_te_s, c=cfg.svm_c, n_classes=n_classes, sample_weight=sample_weights
        )
        time_dict[name_svm] = time.time() - start_class
        probs_dict[name_svm] = (pva_svm, pte_svm)
        model_objects[name_svm] = svm_model
        predict_fns[name_svm] = make_predict_fn(name_svm, svm_model, n_classes)

    if "lgb" in requested or "lightgbm" in requested:
        start_class = time.time()
        pva_lgb, pte_lgb, name_lgb, lgb_model = train_lgbm_proba(
            X_tr_s, y_tr, X_va_s, y_va, X_te_s,
            rounds=cfg.lgb_rounds,
            try_gpu=cfg.lgbm_try_gpu,
            n_classes=n_classes,
            sample_weight=sample_weights,
        )
        if name_lgb != "LightGBM_MISSING":
            time_dict[name_lgb] = time.time() - start_class
            probs_dict[name_lgb] = (pva_lgb, pte_lgb)
            model_objects[name_lgb] = lgb_model
            predict_fns[name_lgb] = make_predict_fn(name_lgb, lgb_model, n_classes)

    if "cat" in requested or "catboost" in requested:
        start_class = time.time()
        pva_cat, pte_cat, name_cat, cat_model = train_catboost_proba(
            X_tr_s, y_tr, X_va_s, y_va, X_te_s,
            rounds=cfg.cat_rounds,
            n_classes=n_classes,
            sample_weight=sample_weights,
        )
        if name_cat != "CatBoost_MISSING":
            time_dict[name_cat] = time.time() - start_class
            probs_dict[name_cat] = (pva_cat, pte_cat)
            model_objects[name_cat] = cat_model
            predict_fns[name_cat] = make_predict_fn(name_cat, cat_model, n_classes)

    if not probs_dict:
        raise RuntimeError("No model was trained. Check --models and installed packages.")

    # Individual model evaluation and artifact export.
    for name, (pva, pte) in probs_dict.items():
        r, cm, thresholds = evaluate_one(
            y_va, y_te, pva, pte, name, cfg, n_classes, normal_idx,
            total_feature_extraction_time, time_dict[name],
            test_pids=_test_pids_cache.get("test"),
        )
        r["backbone"] = backbone_spec
        r["valid_source"] = valid_source
        r["normal_class"] = cfg.class_names[normal_idx] if normal_idx is not None else "NOT_FOUND"
        cm_path = os.path.join(run_dir, f"cm_{safe_name(backbone_spec)}_{safe_name(name)}.png")
        save_confusion_matrix(cm, f"CM | {backbone_spec} | {name}", cm_path, cfg.class_names)
        r["cm_path"] = cm_path

        artifact_path = save_artifact(
            cfg, run_dir, backbone_spec, name, scaler, pca,
            {name: model_objects[name]}, thresholds, normal_idx, r,
            ensemble_info=None,
        )
        if artifact_path:
            r["artifact_path"] = artifact_path
        rows.append(r)

        # Fast, honest interpretability: top-k over a subsample. Label as approximate.

    # Ensemble evaluation and artifact export.
    if len(probs_dict) > 1:
        start_ens = time.time()
        ens_names = list(probs_dict.keys())
        ens_pva_list = [probs_dict[n][0] for n in ens_names]
        ens_pte_list = [probs_dict[n][1] for n in ens_names]

        if cfg.optimize_ensemble:
            ens_weights, ens_thresholds, ens_valid_score = optimize_ensemble_weights(
                ens_pva_list, ens_names, y_va, n_classes, normal_idx,
                cfg.threshold_metric, cfg.min_tumor_recall,
            )
        else:
            ens_weights = default_ensemble_weights(ens_names)
            ens_pva_tmp = weighted_average_probs(ens_pva_list, ens_weights)
            ens_thresholds, ens_valid_score = tune_thresholds(
                y_va, ens_pva_tmp, cfg.threshold_metric, cfg.min_tumor_recall, normal_idx, n_classes
            )

        ens_va = weighted_average_probs(ens_pva_list, ens_weights)
        ens_te = weighted_average_probs(ens_pte_list, ens_weights)
        total_ens_time = (time.time() - start_ens) + sum(time_dict.values())

        ens_name = "Ensemble(" + "+".join([n.split("_")[0] for n in ens_names]) + ")"
        r, cm, thresholds = evaluate_one(
            y_va, y_te, ens_va, ens_te, ens_name, cfg, n_classes, normal_idx,
            total_feature_extraction_time, total_ens_time, thresholds=ens_thresholds,
            test_pids=_test_pids_cache.get("test"),
        )
        r["backbone"] = backbone_spec
        r["valid_source"] = valid_source
        r["normal_class"] = cfg.class_names[normal_idx] if normal_idx is not None else "NOT_FOUND"
        r.update({
            "ensemble_members": " | ".join(ens_names),
            "ensemble_weights": json.dumps([float(w) for w in ens_weights]),
            "ensemble_valid_score": float(ens_valid_score),
            "ensemble_weight_optimization": bool(cfg.optimize_ensemble),
        })
        cm_path = os.path.join(run_dir, f"cm_{safe_name(backbone_spec)}_{safe_name(ens_name)}.png")
        save_confusion_matrix(cm, f"CM | {backbone_spec} | {ens_name}", cm_path, cfg.class_names)
        r["cm_path"] = cm_path

        artifact_path = save_artifact(
            cfg, run_dir, backbone_spec, ens_name, scaler, pca,
            {n: model_objects[n] for n in ens_names}, thresholds, normal_idx, r,
            ensemble_info={
                "members": ens_names,
                "weights": [float(w) for w in ens_weights],
                "optimized_on_validation": bool(cfg.optimize_ensemble),
            },
        )
        if artifact_path:
            r["artifact_path"] = artifact_path
        rows.append(r)

        # Error image logs for qualitative validation.
        try:
            test_dir = os.path.join(cfg.data_dir, "test")
            class_to_idx = make_class_to_idx(cfg.class_names)
            ds_test = FolderDataset(test_dir, cfg.class_names, transform=lambda x: x, class_to_idx=class_to_idx)
            test_paths = [p for p, _ in ds_test.items]
            y_pred_te_ens = predict_with_thresholds(ens_te, thresholds, normal_idx)
            error_log_path = os.path.join(run_dir, f"ERROR_IMAGES_{safe_name(backbone_spec)}.txt")
            with open(error_log_path, "w", encoding="utf-8") as f:
                f.write(f"Normal class index: {normal_idx}, name: {cfg.class_names[normal_idx] if normal_idx is not None else 'N/A'}\n")
                f.write("=== Misclassified Images ===\n")
                for i in range(len(y_te)):
                    if y_te[i] != y_pred_te_ens[i]:
                        f.write(f"TRUE={cfg.class_names[y_te[i]]}\tPRED={cfg.class_names[y_pred_te_ens[i]]}\t{test_paths[i]}\n")
                if normal_idx is not None:
                    f.write("\n=== Tumor False Negatives: tumor predicted as normal ===\n")
                    for i in range(len(y_te)):
                        if y_te[i] != normal_idx and y_pred_te_ens[i] == normal_idx:
                            f.write(f"TRUE={cfg.class_names[y_te[i]]}\tPRED={cfg.class_names[y_pred_te_ens[i]]}\t{test_paths[i]}\n")
        except Exception as e:
            tqdm.write(f"[WARN] Could not write error-image log: {e}")

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return rows


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Fixed thesis pipeline: frozen deep features + ML classifiers + valid-safe evaluation")
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset folder with train/test and optionally valid subfolders")
    parser.add_argument("--out_dir", type=str, default="outputs_fixed_v3")
    parser.add_argument("--cache_dir", type=str, default="cache_features_final_report")
    parser.add_argument("--artifact_dir", type=str, default="artifacts_fixed_v3")
    parser.add_argument("--models", type=str, default="xgb,svm,lgb,cat", help="Comma-separated: xgb,svm,lgb,cat")
    parser.add_argument("--use_hard_weighting", type=int, default=1)
    parser.add_argument("--backbones", type=str, default="resnet18,densenet121,vit_b_16,resnet18+densenet121,resnet18+vit_b_16,densenet121+vit_b_16,resnet18+densenet121+vit_b_16")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_false", dest="amp")
    parser.add_argument("--show_plots", action="store_true")
    parser.add_argument("--xgb_gpu", action="store_true", default=True)
    parser.add_argument("--no_xgb_gpu", action="store_false", dest="xgb_gpu")
    parser.add_argument("--lgbm_try_gpu", action="store_true", default=True)
    parser.add_argument("--no_lgbm_try_gpu", action="store_false", dest="lgbm_try_gpu")
    parser.add_argument("--xgb_rounds", type=int, default=1200)
    parser.add_argument("--lgb_rounds", type=int, default=1000)
    parser.add_argument("--cat_rounds", type=int, default=1000)
    parser.add_argument("--svm_c", type=float, default=1.5)
    parser.add_argument("--pca_var", type=float, default=0.95, help="0 disables PCA; e.g., 0.95 keeps 95% variance")
    parser.add_argument("--tta", type=int, default=3)
    parser.add_argument("--threshold_metric", type=str, default="bal_acc", choices=["accuracy", "bal_acc", "f1_pos", "f1_macro", "recall_macro"])
    parser.add_argument("--min_tumor_recall", type=float, default=0.0, help="Validation constraint for tumor classes when normal class is known")
    parser.add_argument("--valid_ratio", type=float, default=0.15, help="Used only when data_dir/valid is missing")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--img_size", type=int, default=224,
                        help="اندازه ورودی شبکه‌ها؛ برای ViT باید مضرب ۱۶ باشد (۲۲۴، ۳۲۰، ۳۸۴)")
    parser.add_argument("--patient_level", type=int, default=1,
                        help="گزارش معیارها در سطح بیمار (شناسه از پیشوند نام فایل تا __)")
    parser.add_argument("--bootstrap_n", type=int, default=1000,
                        help="تعداد بازنمونه‌گیری بوت‌استرپ برای فاصله اطمینان؛ ۰ = خاموش")
    parser.add_argument("--valid_seed", type=int, default=0,
                        help="بازتقسیم بیمارمحور train+valid با این seed؛ ۰ = بدون تغییر")
    parser.add_argument("--valid_patient_frac", type=float, default=0.2,
                        help="نسبت بیماران در اعتبارسنجی هنگام بازتقسیم")
    parser.add_argument("--normal_class", type=str, default="auto", help="auto or explicit folder/class name like notumor")
    parser.add_argument("--hard_weight_gamma", type=float, default=2.0)
    parser.add_argument("--hard_weight_fn_penalty", type=float, default=5.0)
    parser.add_argument("--prefer_cache", action="store_true", default=True, help="Load latest cached X/y files if available")
    parser.add_argument("--no_prefer_cache", action="store_false", dest="prefer_cache")
    parser.add_argument("--cache_only", action="store_true", help="Never extract features; fail if cache is missing")
    parser.add_argument("--optimize_ensemble", action="store_true", default=True)
    parser.add_argument("--no_optimize_ensemble", action="store_false", dest="optimize_ensemble")
    parser.add_argument("--save_artifacts", action="store_true", default=True)
    parser.add_argument("--no_save_artifacts", action="store_false", dest="save_artifacts")
    return parser.parse_args()


def main():
    args = parse_args()
    set_torch_fast()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM] device={device}")

    class_names = discover_classes(args.data_dir, split="train")
    print(f"[DATA] class_names={class_names}")

    models_to_run = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    backbones = [b.strip() for b in args.backbones.split(",") if b.strip()]

    cfg = Cfg(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        cache_dir=args.cache_dir,
        artifact_dir=args.artifact_dir,
        models_to_run=models_to_run,
        use_hard_weighting=bool(args.use_hard_weighting),
        backbones=backbones,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        amp=bool(args.amp),
        show_plots=bool(args.show_plots),
        xgb_gpu=bool(args.xgb_gpu),
        lgbm_try_gpu=bool(args.lgbm_try_gpu),
        xgb_rounds=int(args.xgb_rounds),
        lgb_rounds=int(args.lgb_rounds),
        cat_rounds=int(args.cat_rounds),
        svm_c=float(args.svm_c),
        pca_var=float(args.pca_var),
        tta=int(args.tta),
        threshold_metric=str(args.threshold_metric),
        min_tumor_recall=float(args.min_tumor_recall),
        valid_ratio=float(args.valid_ratio),
        random_state=int(args.random_state),
        normal_class=str(args.normal_class),
        hard_weight_gamma=float(args.hard_weight_gamma),
        hard_weight_fn_penalty=float(args.hard_weight_fn_penalty),
        prefer_cache=bool(args.prefer_cache),
        cache_only=bool(args.cache_only),
        optimize_ensemble=bool(args.optimize_ensemble),
        save_artifacts=bool(args.save_artifacts),
        class_names=class_names,
        patient_level=bool(args.patient_level),
        bootstrap_n=int(args.bootstrap_n),
        valid_seed=int(args.valid_seed),
        valid_patient_frac=float(args.valid_patient_frac),
        img_size=int(args.img_size),
    )

    ensure_dir(cfg.out_dir)
    ensure_dir(cfg.cache_dir)
    ensure_dir(cfg.artifact_dir)

    run_dir = os.path.join(cfg.out_dir, f"run_{time.strftime('%Y%m%d_%H%M%S')}")
    ensure_dir(run_dir)

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    all_rows: List[Dict[str, Any]] = []
    for spec in tqdm(cfg.backbones, desc="Backbone specs", ncols=110):
        try:
            rows = run_for_spec(cfg, spec, device, run_dir)
            for r in rows:
                r.update({"tta": cfg.tta, "pca_var": cfg.pca_var, "img_size": cfg.img_size})
            all_rows.extend(rows)
        except Exception as e:
            print(f"[ERROR] spec={spec} failed: {e}")

    if not all_rows:
        raise RuntimeError("No successful results were produced.")

    df = pd.DataFrame(all_rows)

    leftmost_cols = [
        "backbone", "model", "accuracy", "balanced_accuracy", "f1_macro", "precision_macro",
        "recall_macro", "FNR_macro", "feature_extraction_time_sec", "classification_time_sec",
        "argmax_accuracy", "argmax_balanced_accuracy", "threshold_gain_acc",
        "n_patients", "patient_accuracy", "patient_balanced_accuracy",
        "patient_argmax_accuracy", "patient_argmax_balanced_accuracy",
        "patient_threshold_gain_acc",
        "slice_acc_ci_low", "slice_acc_ci_high",
        "patient_acc_ci_low", "patient_acc_ci_high",
        "valid_source", "normal_class", "thresholds_json", "valid_accuracy", "valid_balanced_accuracy", "valid_f1_macro"
    ]
    remaining_cols = [c for c in df.columns if c not in leftmost_cols]
    df = df[leftmost_cols + remaining_cols]

    df = df.sort_values(by=["balanced_accuracy", "f1_macro", "accuracy"], ascending=False).reset_index(drop=True)
    csv_path = os.path.join(run_dir, "results_all_models_fixed_v3.csv")
    df.to_csv(csv_path, index=False)

    df_plot = df.copy()
    df_plot["name"] = df_plot["backbone"] + " | " + df_plot["model"]
    save_bar_plot(df_plot.head(10), "balanced_accuracy", os.path.join(run_dir, "top10_balanced_accuracy.png"), "Top-10 Balanced Accuracy")
    save_bar_plot(df_plot.head(10), "FNR_macro", os.path.join(run_dir, "top10_FNR_macro.png"), "Top-10 FNR Macro (lower is better)")

    print("\n[SAVED]")
    print(" -", csv_path)
    print(" -", os.path.join(run_dir, "top10_balanced_accuracy.png"))
    print(" -", os.path.join(run_dir, "top10_FNR_macro.png"))
    print(" - artifacts:", cfg.artifact_dir)

    print("\n[TOP RESULTS - VALID-SAFE]")
    show_cols = [c for c in leftmost_cols if c in df.columns]
    print(df[show_cols].head(20).to_string(index=False))

    if cfg.show_plots:
        img = plt.imread(os.path.join(run_dir, "top10_balanced_accuracy.png"))
        plt.figure()
        plt.imshow(img)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
