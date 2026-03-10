"""
baselines.py
============
All baseline model variants for ablation study.
Run via run_ablation.py — do not run directly.

Models:
    1. TextOnlyModel              — no fusion
    2. ImageOnlyModel             — no fusion
    3. ConcatFusionModel          — fixed concatenation, no attention, no gating
    4. CrossAttnNoGatingModel     — cross-attention but fixed 0.5/0.5 blend
    5. AdaptiveFusionModel        — cross-attn + static reweighting (in model.py)

NOTE: All models use the same CLIP feature extraction pattern:
    text_out  = self.clip.text_model(...)
    text_feat = self.clip.text_projection(text_out.pooler_output)

    img_out   = self.clip.vision_model(...)
    img_feat  = self.clip.visual_projection(img_out.pooler_output)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel


# ─────────────────────────────────────────────────────────────────
# BASELINE 1: Text Only
# ─────────────────────────────────────────────────────────────────
class TextOnlyModel(nn.Module):
    """
    CLIP text encoder → classifier only.
    No image, no fusion, no attention, no gating.
    """
    def __init__(self, embed_dim=512, dropout=0.1, freeze_clip=True):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        if freeze_clip:
            for p in self.clip.parameters():
                p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        # FIXED — extract tensor properly from CLIP output object
        text_out  = self.clip.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_feat = self.clip.text_projection(text_out.pooler_output)   # (B, 512)
        text_feat = F.normalize(text_feat, dim=-1)

        logit = self.classifier(text_feat).squeeze(1)

        # Dummy alpha — zeros (no image used)
        alpha = torch.zeros(text_feat.size(0)).to(text_feat.device)
        return logit, alpha


# ─────────────────────────────────────────────────────────────────
# BASELINE 2: Image Only
# ─────────────────────────────────────────────────────────────────
class ImageOnlyModel(nn.Module):
    """
    CLIP image encoder → classifier only.
    No text, no fusion, no attention, no gating.
    """
    def __init__(self, embed_dim=512, dropout=0.1, freeze_clip=True):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        if freeze_clip:
            for p in self.clip.parameters():
                p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        # FIXED — extract tensor properly from CLIP output object
        img_out   = self.clip.vision_model(pixel_values=pixel_values)
        img_feat  = self.clip.visual_projection(img_out.pooler_output)  # (B, 512)
        img_feat  = F.normalize(img_feat, dim=-1)

        logit = self.classifier(img_feat).squeeze(1)

        # Dummy alpha — ones (full image weight)
        alpha = torch.ones(img_feat.size(0)).to(img_feat.device)
        return logit, alpha


# ─────────────────────────────────────────────────────────────────
# BASELINE 3: Concat Fusion (no attention, no gating)
# ─────────────────────────────────────────────────────────────────
class ConcatFusionModel(nn.Module):
    """
    CLIP text + image → concatenate → classifier.
    No cross-attention. No dynamic gating. Just gluing features together.
    """
    def __init__(self, embed_dim=512, dropout=0.1, freeze_clip=True):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        if freeze_clip:
            for p in self.clip.parameters():
                p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),   # 1024 input
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        # FIXED — extract tensors properly from CLIP output objects
        text_out  = self.clip.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_feat = self.clip.text_projection(text_out.pooler_output)   # (B, 512)

        img_out   = self.clip.vision_model(pixel_values=pixel_values)
        img_feat  = self.clip.visual_projection(img_out.pooler_output)  # (B, 512)

        text_feat = F.normalize(text_feat, dim=-1)
        img_feat  = F.normalize(img_feat,  dim=-1)

        fused = torch.cat([text_feat, img_feat], dim=-1)   # (B, 1024)
        logit = self.classifier(fused).squeeze(1)

        # Fixed equal weight — no dynamic gating
        alpha = torch.full((text_feat.size(0),), 0.5).to(text_feat.device)
        return logit, alpha


# ─────────────────────────────────────────────────────────────────
# BASELINE 4: Cross-Attention WITHOUT Reweighting
# ─────────────────────────────────────────────────────────────────
class CrossAttnNoGatingModel(nn.Module):
    """
    Full cross-modal attention BUT fixed 0.5/0.5 blend after attention.
    No static or dynamic gating.

    This isolates the contribution of reweighting:
    AdaptiveFusionModel AUROC - CrossAttnNoGatingModel AUROC
    = the value added purely by static reweighting (Phase 1)
    """
    def __init__(self, embed_dim=512, num_heads=8, ffn_dim=1024,
                 dropout=0.1, freeze_clip=True):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        if freeze_clip:
            for p in self.clip.parameters():
                p.requires_grad = False

        # Cross-attention (same as full model)
        self.img_to_text_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.text_to_img_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_img_1  = nn.LayerNorm(embed_dim)
        self.norm_text_1 = nn.LayerNorm(embed_dim)

        self.ffn_img = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.ffn_text = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm_img_2  = nn.LayerNorm(embed_dim)
        self.norm_text_2 = nn.LayerNorm(embed_dim)

        # Same classifier as full model
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 3, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        # FIXED — extract tensors properly from CLIP output objects
        text_out  = self.clip.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_feat = self.clip.text_projection(text_out.pooler_output)   # (B, 512)

        img_out   = self.clip.vision_model(pixel_values=pixel_values)
        img_feat  = self.clip.visual_projection(img_out.pooler_output)  # (B, 512)

        text_feat = F.normalize(text_feat, dim=-1)
        img_feat  = F.normalize(img_feat,  dim=-1)

        # Cross-attention
        t = text_feat.unsqueeze(1)
        i = img_feat.unsqueeze(1)

        img_attn,  _ = self.img_to_text_attn(query=i, key=t, value=t)
        text_attn, _ = self.text_to_img_attn(query=t, key=i, value=i)

        img_out  = self.norm_img_1 (i + img_attn ).squeeze(1)
        text_out = self.norm_text_1(t + text_attn).squeeze(1)

        img_out  = self.norm_img_2 (img_out  + self.ffn_img (img_out ))
        text_out = self.norm_text_2(text_out + self.ffn_text(text_out))

        img_out  = F.normalize(img_out,  dim=-1)
        text_out = F.normalize(text_out, dim=-1)

        # FIXED 0.5/0.5 — no gating network at all
        fused = 0.5 * img_out + 0.5 * text_out

        combined = torch.cat([fused, text_out, img_out], dim=-1)  # (B, 1536)
        logit    = self.classifier(combined).squeeze(1)

        # Alpha fixed at 0.5
        alpha = torch.full((text_feat.size(0),), 0.5).to(text_feat.device)
        return logit, alpha
