"""
train_static_v2.py
==================
Enhanced training for STATIC Reweighting - same setup as dynamic for fair comparison.

Same improvements as dynamic:
1. Larger CLIP model (ViT-L/14 instead of ViT-B/32)
2. Partial CLIP unfreezing (last 4 layers)
3. Focal Loss for hard examples
4. Optimal threshold search
5. Gradient accumulation for larger effective batch
6. Cosine annealing with warm restarts

Only difference: Uses fixed alpha=0.5 instead of learned gating network.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import CLIPModel, CLIPProcessor, get_cosine_schedule_with_warmup
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
from PIL import Image
import io

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
CONFIG = {
    "train_parquet"  : "../Data/train-00000-of-00001-6587b3a58d350036.parquet",
    "val_parquet"    : "../Data/validation-00000-of-00001-1508d9e5032c2c1f.parquet",
    "clip_model"     : "openai/clip-vit-large-patch14",  # Larger model
    "batch_size"     : 8,              # Smaller batch for larger model
    "grad_accum"     : 4,              # Effective batch = 32
    "num_epochs"     : 20,
    "unfreeze_layers": 4,              # Unfreeze last N CLIP layers
    "clip_lr"        : 2e-6,           # Lower LR for CLIP
    "fusion_lr"      : 1e-4,
    "weight_decay"   : 0.01,
    "warmup_ratio"   : 0.1,
    "focal_alpha"    : 0.75,           # Focal loss alpha (weight for positive class)
    "focal_gamma"    : 2.0,            # Focal loss gamma (focus on hard examples)
    "label_smoothing": 0.1,
    "static_alpha"   : 0.5,            # STATIC: Fixed fusion weight
    "fairness_lambda": 0.05,
    "use_balanced_sampler": True,
    "use_temperature_scaling": True,
    "temperature_grid_min": 0.70,
    "temperature_grid_max": 1.50,
    "temperature_grid_step": 0.05,
    "group_columns": ["group", "demographic_group", "protected_group", "identity", "race", "gender"],
    "checkpoint_dir" : "../checkpoints",
    "checkpoint_name": "best_model_static_v2.pt",
    "patience"       : 7
}


# ─────────────────────────────────────────────────────────────────
# FOCAL LOSS
# ─────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance and hard examples."""
    def __init__(self, alpha=0.75, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, logits, targets):
        # Apply label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weight for class balance
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        loss = alpha_t * focal_weight * ce_loss
        return loss.mean()


# ─────────────────────────────────────────────────────────────────
# ENHANCED MODEL
# ─────────────────────────────────────────────────────────────────
class CrossModalAttentionV2(nn.Module):
    """Enhanced cross-modal attention with deeper interaction."""
    def __init__(self, embed_dim=768, num_heads=12, ffn_dim=2048, dropout=0.1):
        super().__init__()
        self.img_to_text_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.text_to_img_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm_img_1 = nn.LayerNorm(embed_dim)
        self.norm_text_1 = nn.LayerNorm(embed_dim)
        
        self.ffn_img = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.ffn_text = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.norm_img_2 = nn.LayerNorm(embed_dim)
        self.norm_text_2 = nn.LayerNorm(embed_dim)
        
        # Second attention layer for deeper interaction
        self.img_to_text_attn_2 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.text_to_img_attn_2 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_img_3 = nn.LayerNorm(embed_dim)
        self.norm_text_3 = nn.LayerNorm(embed_dim)

    def forward(self, text_feat, img_feat):
        t = text_feat.unsqueeze(1)
        i = img_feat.unsqueeze(1)
        
        # First attention layer
        img_attn, _ = self.img_to_text_attn(query=i, key=t, value=t)
        text_attn, _ = self.text_to_img_attn(query=t, key=i, value=i)
        
        img_out = self.norm_img_1(i + img_attn)
        text_out = self.norm_text_1(t + text_attn)
        
        # FFN
        img_out = self.norm_img_2(img_out + self.ffn_img(img_out))
        text_out = self.norm_text_2(text_out + self.ffn_text(text_out))
        
        # Second attention layer
        img_attn_2, _ = self.img_to_text_attn_2(query=img_out, key=text_out, value=text_out)
        text_attn_2, _ = self.text_to_img_attn_2(query=text_out, key=img_out, value=img_out)
        
        img_out = self.norm_img_3(img_out + img_attn_2).squeeze(1)
        text_out = self.norm_text_3(text_out + text_attn_2).squeeze(1)
        
        return text_out, img_out


class DynamicGatingNetworkV2(nn.Module):
    """Enhanced gating network with attention-based feature comparison."""
    def __init__(self, embed_dim=768):
        super().__init__()
        self.query = nn.Linear(embed_dim, 128)
        self.key = nn.Linear(embed_dim, 128)
        
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 4 + 128, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, text_feat, img_feat):
        # Compute attention-based similarity
        q = self.query(img_feat)
        k = self.key(text_feat)
        attn_score = (q * k).sum(dim=-1, keepdim=True) / (128 ** 0.5)
        
        delta = torch.abs(img_feat - text_feat)
        prod = img_feat * text_feat
        combined = torch.cat([img_feat, text_feat, delta, prod, attn_score.expand(-1, 128)], dim=-1)
        
        logits = self.net(combined)
        alpha = torch.sigmoid(logits)
        return alpha


class StaticGatingNetworkV2(nn.Module):
    """Static gating - returns fixed alpha for all samples."""
    def __init__(self, static_alpha=0.5):
        super().__init__()
        self.static_alpha = static_alpha
    
    def forward(self, text_feat, img_feat):
        batch_size = text_feat.size(0)
        alpha = torch.full((batch_size, 1), self.static_alpha, 
                          device=text_feat.device, dtype=text_feat.dtype)
        return alpha


class StaticFusionModelV2(nn.Module):
    """Enhanced model for ViT-L/14 CLIP with STATIC gating."""
    def __init__(self, clip_model_name="openai/clip-vit-large-patch14", unfreeze_layers=4, static_alpha=0.5):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.embed_dim = self.clip.config.projection_dim  # 768 for ViT-L
        
        # Freeze CLIP except last N layers
        self._freeze_clip(unfreeze_layers)
        
        self.cross_attn = CrossModalAttentionV2(
            embed_dim=self.embed_dim,
            num_heads=12,
            ffn_dim=2048,
            dropout=0.1
        )
        
        # STATIC gating instead of dynamic
        self.gating = StaticGatingNetworkV2(static_alpha=static_alpha)
        
        # Deeper classifier (same as dynamic)
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim * 2 + 1, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
    
    def _freeze_clip(self, unfreeze_layers):
        """Freeze CLIP except last N layers of vision and text encoders."""
        # Freeze all first
        for param in self.clip.parameters():
            param.requires_grad = False
        
        if unfreeze_layers > 0:
            # Unfreeze last N vision layers
            vision_layers = self.clip.vision_model.encoder.layers
            for layer in vision_layers[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            
            # Unfreeze last N text layers
            text_layers = self.clip.text_model.encoder.layers
            for layer in text_layers[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            
            # Unfreeze projection layers
            if hasattr(self.clip, 'visual_projection'):
                for param in self.clip.visual_projection.parameters():
                    param.requires_grad = True
            if hasattr(self.clip, 'text_projection'):
                for param in self.clip.text_projection.parameters():
                    param.requires_grad = True

    def forward(self, input_ids, attention_mask, pixel_values):
        # Get CLIP embeddings (these return tensors directly)
        text_out = self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        img_out = self.clip.get_image_features(pixel_values=pixel_values)
        
        # Handle case where output might be a tuple or object
        if hasattr(text_out, 'pooler_output'):
            text_out = text_out.pooler_output
        if hasattr(img_out, 'pooler_output'):
            img_out = img_out.pooler_output
        
        # Normalize
        text_feat = F.normalize(text_out, dim=-1)
        img_feat = F.normalize(img_out, dim=-1)
        
        # Cross-modal attention
        text_feat, img_feat = self.cross_attn(text_feat, img_feat)
        text_feat = F.normalize(text_feat, dim=-1)
        img_feat = F.normalize(img_feat, dim=-1)
        
        # Static gating (fixed alpha)
        alpha = self.gating(text_feat, img_feat)
        
        # Fusion
        img_branch = alpha * img_feat
        text_branch = (1.0 - alpha) * text_feat
        fused = torch.cat([img_branch, text_branch, alpha], dim=-1)
        
        logit = self.classifier(fused).squeeze(-1)
        alpha_out = alpha.squeeze(-1)
        
        return logit, alpha_out


# ─────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────
class HatefulMemesDatasetV2:
    def __init__(self, parquet_path, processor, max_text_length=77):
        self.df = pd.read_parquet(parquet_path)
        self.processor = processor
        self.max_text_length = max_text_length
        self.group_col = self._detect_group_column()
        if self.group_col is not None:
            group_values = self.df[self.group_col].fillna("unknown").astype(str)
            self.group_to_id = {g: i for i, g in enumerate(sorted(group_values.unique()))}
        else:
            self.group_to_id = {"all": 0}
        print(f"Loaded {len(self.df)} samples from {parquet_path}")
        if self.group_col is not None:
            print(f"Detected subgroup column: {self.group_col} ({len(self.group_to_id)} groups)")

    def _detect_group_column(self):
        for col in CONFIG["group_columns"]:
            if col in self.df.columns:
                return col
        return None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        img_data = row['image']
        if isinstance(img_data, dict) and 'bytes' in img_data:
            image = Image.open(io.BytesIO(img_data['bytes'])).convert("RGB")
        elif isinstance(img_data, bytes):
            image = Image.open(io.BytesIO(img_data)).convert("RGB")
        else:
            image = Image.open(str(img_data)).convert("RGB")
        
        text = str(row['text']) if pd.notna(row['text']) else ""
        label = int(row['label']) if 'label' in row.index and pd.notna(row['label']) else -1
        if self.group_col is not None:
            group_name = str(row[self.group_col]) if pd.notna(row[self.group_col]) else "unknown"
        else:
            group_name = "all"
        group_id = self.group_to_id.get(group_name, 0)
        
        encoding = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'pixel_values': encoding['pixel_values'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float32),
            'group': torch.tensor(group_id, dtype=torch.long)
        }


# ─────────────────────────────────────────────────────────────────
# OPTIMAL THRESHOLD SEARCH
# ─────────────────────────────────────────────────────────────────
def find_optimal_threshold(labels, probs):
    """Find threshold that maximizes accuracy."""
    best_threshold = 0.5
    best_acc = 0
    
    for thresh in np.arange(0.3, 0.7, 0.01):
        preds = (probs >= thresh).astype(int)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_threshold = thresh
    
    return best_threshold, best_acc


def fit_temperature_by_brier(labels, probs):
    if not CONFIG["use_temperature_scaling"]:
        return 1.0
    eps = 1e-6
    labels = labels.astype(np.float32)
    probs = np.clip(probs.astype(np.float32), eps, 1 - eps)
    logits = np.log(probs / (1 - probs))
    temps = np.arange(
        CONFIG["temperature_grid_min"],
        CONFIG["temperature_grid_max"] + 1e-9,
        CONFIG["temperature_grid_step"]
    )
    best_t, best_brier = 1.0, float("inf")
    for t in temps:
        p = 1.0 / (1.0 + np.exp(-logits / t))
        brier = np.mean((p - labels) ** 2)
        if brier < best_brier:
            best_brier = brier
            best_t = float(t)
    return best_t


def compute_group_stats(labels, probs, preds, groups):
    groups = groups.astype(np.int64)
    unique_groups = np.unique(groups)
    if unique_groups.size <= 1:
        return None
    stats = {}
    tprs, fprs, fnrs, aucs = [], [], [], []
    for g in unique_groups:
        mask = groups == g
        y = labels[mask]
        p = probs[mask]
        yhat = preds[mask]
        pos = np.sum(y == 1)
        neg = np.sum(y == 0)
        tp = np.sum((yhat == 1) & (y == 1))
        fp = np.sum((yhat == 1) & (y == 0))
        fn = np.sum((yhat == 0) & (y == 1))
        tpr = tp / max(pos, 1)
        fpr = fp / max(neg, 1)
        fnr = fn / max(pos, 1)
        if np.unique(y).size > 1:
            auc = roc_auc_score(y, p)
            aucs.append(auc)
        else:
            auc = float("nan")
        tprs.append(tpr)
        fprs.append(fpr)
        fnrs.append(fnr)
        stats[int(g)] = {
            "size": int(mask.sum()),
            "tpr": float(tpr),
            "fpr": float(fpr),
            "fnr": float(fnr),
            "auroc": float(auc) if not np.isnan(auc) else float("nan")
        }
    gap = max(np.max(tprs) - np.min(tprs), np.max(fprs) - np.min(fprs))
    result = {
        "stats": stats,
        "equalized_odds_gap": float(gap),
        "fpr_gap": float(np.max(fprs) - np.min(fprs)),
        "fnr_gap": float(np.max(fnrs) - np.min(fnrs)),
        "auroc_gap": float(np.max(aucs) - np.min(aucs)) if len(aucs) >= 2 else float("nan")
    }
    return result


def find_fair_threshold(labels, probs, groups):
    best_threshold = 0.5
    best_score = -1e9
    best_acc = 0.0
    best_gap = 0.0
    for thresh in np.arange(0.3, 0.7, 0.01):
        preds = (probs >= thresh).astype(int)
        acc = accuracy_score(labels, preds)
        grp = compute_group_stats(labels, probs, preds, groups)
        gap = grp["equalized_odds_gap"] if grp is not None else 0.0
        score = acc - CONFIG["fairness_lambda"] * gap
        if score > best_score:
            best_score = score
            best_threshold = float(thresh)
            best_acc = float(acc)
            best_gap = float(gap)
    return best_threshold, best_acc, best_gap


def make_weighted_sampler(dataset):
    labels = dataset.df["label"].astype(int).tolist()
    if dataset.group_col is not None:
        groups = dataset.df[dataset.group_col].fillna("unknown").astype(str).tolist()
        keys = list(zip(labels, groups))
    else:
        keys = [(y, "all") for y in labels]
    counts = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    weights = [1.0 / counts[k] for k in keys]
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True
    )


# ─────────────────────────────────────────────────────────────────
# TRAINING FUNCTIONS
# ─────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, grad_accum, entropy_weight):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['label'].to(device)
        
        logits, alpha = model(input_ids, attention_mask, pixel_values)
        
        # Main loss
        loss = criterion(logits, labels)
        
        # Entropy regularization for diverse alpha
        alpha_clamped = torch.clamp(alpha, 0.01, 0.99)
        entropy = -(alpha_clamped * torch.log(alpha_clamped) + 
                   (1 - alpha_clamped) * torch.log(1 - alpha_clamped))
        entropy_loss = -entropy.mean()  # Maximize entropy
        
        loss = loss + entropy_weight * entropy_loss
        loss = loss / grad_accum
        loss.backward()
        
        if (batch_idx + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * grad_accum
        
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy())
        
        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item()*grad_accum:.4f}")
    
    # Handle remaining gradients
    if len(loader) % grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    auroc = roc_auc_score(all_labels, all_preds)
    return total_loss / len(loader), auroc


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_alphas, all_groups = [], [], [], []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)
            
            groups = batch['group'].to(device)
            logits, alpha = model(input_ids, attention_mask, pixel_values)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            all_alphas.extend(alpha.cpu().numpy())
            all_groups.extend(groups.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_alphas = np.array(all_alphas)
    all_groups = np.array(all_groups)

    temperature = fit_temperature_by_brier(all_labels, all_preds)
    logits_np = np.log(np.clip(all_preds, 1e-6, 1 - 1e-6) / np.clip(1 - all_preds, 1e-6, 1 - 1e-6))
    all_preds_cal = 1.0 / (1.0 + np.exp(-logits_np / temperature))

    # Find fairness-aware threshold
    opt_thresh, opt_acc, fairness_gap = find_fair_threshold(all_labels, all_preds_cal, all_groups)
    
    # Metrics with default threshold
    bin_preds = (all_preds_cal >= 0.5).astype(int)
    accuracy = accuracy_score(all_labels, bin_preds)
    
    # Metrics with optimal threshold
    bin_preds_opt = (all_preds_cal >= opt_thresh).astype(int)
    accuracy_opt = accuracy_score(all_labels, bin_preds_opt)
    
    auroc = roc_auc_score(all_labels, all_preds_cal)
    f1 = f1_score(all_labels, bin_preds, average='macro')
    f1_opt = f1_score(all_labels, bin_preds_opt, average='macro')
    group_metrics = compute_group_stats(all_labels, all_preds_cal, bin_preds_opt, all_groups)
    
    return {
        'loss': total_loss / len(loader),
        'auroc': auroc,
        'accuracy': accuracy,
        'accuracy_opt': accuracy_opt,
        'f1': f1,
        'f1_opt': f1_opt,
        'opt_thresh': opt_thresh,
        'temperature': temperature,
        'fairness_gap': fairness_gap,
        'group_metrics': group_metrics,
        'alpha_mean': all_alphas.mean(),
        'alpha_std': all_alphas.std(),
        'alpha_min': all_alphas.min(),
        'alpha_max': all_alphas.max()
    }


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    print(f"CLIP model: {CONFIG['clip_model']}")
    
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    
    # Load processor and create datasets
    processor = CLIPProcessor.from_pretrained(CONFIG["clip_model"])
    
    train_dataset = HatefulMemesDatasetV2(CONFIG["train_parquet"], processor)
    val_dataset = HatefulMemesDatasetV2(CONFIG["val_parquet"], processor)
    
    if CONFIG["use_balanced_sampler"]:
        train_sampler = make_weighted_sampler(train_dataset)
        train_loader = DataLoader(
            train_dataset, batch_size=CONFIG["batch_size"],
            sampler=train_sampler, num_workers=2, pin_memory=False
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=CONFIG["batch_size"],
            shuffle=True, num_workers=2, pin_memory=False
        )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"],
        shuffle=False, num_workers=2, pin_memory=False
    )
    
    # Create model - STATIC version
    model = StaticFusionModelV2(
        clip_model_name=CONFIG["clip_model"],
        unfreeze_layers=CONFIG["unfreeze_layers"],
        static_alpha=CONFIG["static_alpha"]
    ).to(device)
    
    print(f"Static Alpha: {CONFIG['static_alpha']}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    
    # Loss
    criterion = FocalLoss(
        alpha=CONFIG["focal_alpha"],
        gamma=CONFIG["focal_gamma"],
        label_smoothing=CONFIG["label_smoothing"]
    )
    
    # Optimizer with differential learning rates
    clip_params = []
    other_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'clip' in name:
                clip_params.append(param)
            else:
                other_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': clip_params, 'lr': CONFIG["clip_lr"]},
        {'params': other_params, 'lr': CONFIG["fusion_lr"]}
    ], weight_decay=CONFIG["weight_decay"])
    
    # Scheduler
    total_steps = CONFIG["num_epochs"] * len(train_loader) // CONFIG["grad_accum"]
    warmup_steps = int(total_steps * CONFIG["warmup_ratio"])
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_score = -1e9
    best_auroc = 0.0
    best_acc = 0.0
    patience_ctr = 0
    
    for epoch in range(CONFIG["num_epochs"]):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1}/{CONFIG['num_epochs']} -- STATIC Reweighting V2 (alpha={CONFIG['static_alpha']})")
        print(f"{'='*60}")
        
        train_loss, train_auroc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device,
            CONFIG["grad_accum"], 0.0  # No entropy weight for static
        )
        
        metrics = evaluate(model, val_loader, criterion, device)
        
        print(f"\nTrain --> Loss: {train_loss:.4f} | AUROC: {train_auroc:.4f}")
        print(f"Val   --> Loss: {metrics['loss']:.4f} | AUROC: {metrics['auroc']:.4f}")
        print(f"Val   --> Acc@0.5: {metrics['accuracy']:.4f} | Acc@{metrics['opt_thresh']:.2f}: {metrics['accuracy_opt']:.4f}")
        print(f"Val   --> F1@0.5: {metrics['f1']:.4f} | F1@opt: {metrics['f1_opt']:.4f}")
        print(f"Val   --> Temp: {metrics['temperature']:.2f} | FairGap: {metrics['fairness_gap']:.4f}")
        print(f"Alpha --> Mean: {metrics['alpha_mean']:.3f} | Std: {metrics['alpha_std']:.3f} | "
              f"Range: [{metrics['alpha_min']:.3f}, {metrics['alpha_max']:.3f}]")
        if metrics['group_metrics'] is not None:
            gm = metrics['group_metrics']
            print(
                f"Group --> EqOddsGap: {gm['equalized_odds_gap']:.4f} | "
                f"FPRGap: {gm['fpr_gap']:.4f} | FNRGap: {gm['fnr_gap']:.4f}"
            )

        # Save best model (fairness-aware objective)
        score = metrics['auroc'] - CONFIG["fairness_lambda"] * metrics['fairness_gap']
        if score > best_score:
            best_score = score
            best_auroc = metrics['auroc']
            best_acc = metrics['accuracy_opt']
            patience_ctr = 0
            
            path = os.path.join(CONFIG["checkpoint_dir"], CONFIG["checkpoint_name"])
            torch.save({
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_auroc': metrics['auroc'],
                'val_acc': metrics['accuracy'],
                'val_acc_opt': metrics['accuracy_opt'],
                'opt_threshold': metrics['opt_thresh'],
                'temperature': metrics['temperature'],
                'fairness_gap': metrics['fairness_gap'],
                'selection_score': score,
                'alpha_mean': metrics['alpha_mean'],
                'config': CONFIG
            }, path)
            print(
                f"✅ Saved best model --> Score: {best_score:.4f} | "
                f"AUROC: {best_auroc:.4f} | Acc: {best_acc:.4f}"
            )
        else:
            patience_ctr += 1
            print(f"No improvement. Patience: {patience_ctr}/{CONFIG['patience']}")
            if patience_ctr >= CONFIG["patience"]:
                print("Early stopping triggered.")
                break
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"Best Val AUROC: {best_auroc:.4f}")
    print(f"Best Val Accuracy (optimal threshold): {best_acc:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
