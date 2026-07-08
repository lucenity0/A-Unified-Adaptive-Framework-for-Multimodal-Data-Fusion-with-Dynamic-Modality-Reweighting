"""
model.py
========
All model architectures.

V3 changes over V2:
  - CrossModalAttentionV2   : attention dropout raised to 0.2 (configurable)
  - DynamicGatingNetworkV3  : accepts per-modality uncertainty signals from aux heads
  - AdaptiveFusionModelV3   : adds img_aux_head / txt_aux_head; exposes
                               extract_features() + fuse_and_classify() so the
                               training loop can apply MixUp between the two steps
  - StaticFusionModelV2     : unchanged (baseline — keep identical for fair comparison)

The dataset class lives in dataset.py; loss functions live in losses.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

from clip_features import extract_clip_image_features, extract_clip_text_features


def freeze_clip_layers(clip, unfreeze_layers: int):
    """Freeze all CLIP parameters, then unfreeze the last `unfreeze_layers`
    transformer blocks of both encoders plus the projection layers.
    Shared by the dynamic and static models."""
    for p in clip.parameters():
        p.requires_grad = False

    if unfreeze_layers <= 0:
        return

    for layer in clip.vision_model.encoder.layers[-unfreeze_layers:]:
        for p in layer.parameters():
            p.requires_grad = True

    for layer in clip.text_model.encoder.layers[-unfreeze_layers:]:
        for p in layer.parameters():
            p.requires_grad = True

    for attr in ("visual_projection", "text_projection"):
        if hasattr(clip, attr):
            for p in getattr(clip, attr).parameters():
                p.requires_grad = True


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-MODAL ATTENTION  (unchanged architecture, dropout raised)
# ─────────────────────────────────────────────────────────────────────────────

class CrossModalAttentionV2(nn.Module):
    """
    Two-layer bidirectional cross-modal attention.
    dropout default raised to 0.2 vs 0.1 in V2 to reduce overfitting.
    """

    def __init__(self, embed_dim: int = 768, num_heads: int = 12,
                 ffn_dim: int = 2048, dropout: float = 0.2):
        super().__init__()

        # ── Layer 1 ──
        self.img_to_txt_1  = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.txt_to_img_1  = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_img_1    = nn.LayerNorm(embed_dim)
        self.norm_txt_1    = nn.LayerNorm(embed_dim)
        self.ffn_img       = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim), nn.Dropout(dropout),
        )
        self.ffn_txt       = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim), nn.Dropout(dropout),
        )
        self.norm_img_2    = nn.LayerNorm(embed_dim)
        self.norm_txt_2    = nn.LayerNorm(embed_dim)

        # ── Layer 2 ──
        self.img_to_txt_2  = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.txt_to_img_2  = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_img_3    = nn.LayerNorm(embed_dim)
        self.norm_txt_3    = nn.LayerNorm(embed_dim)

    def forward(self, txt_feat: torch.Tensor, img_feat: torch.Tensor):
        t = txt_feat.unsqueeze(1)   # (B, 1, D)
        i = img_feat.unsqueeze(1)

        # Layer 1
        i_attn, _ = self.img_to_txt_1(query=i, key=t, value=t)
        t_attn, _ = self.txt_to_img_1(query=t, key=i, value=i)
        i_out = self.norm_img_1(i + i_attn)
        t_out = self.norm_txt_1(t + t_attn)
        i_out = self.norm_img_2(i_out + self.ffn_img(i_out))
        t_out = self.norm_txt_2(t_out + self.ffn_txt(t_out))

        # Layer 2
        i_attn2, _ = self.img_to_txt_2(query=i_out, key=t_out, value=t_out)
        t_attn2, _ = self.txt_to_img_2(query=t_out, key=i_out, value=i_out)
        i_out = self.norm_img_3(i_out + i_attn2).squeeze(1)   # (B, D)
        t_out = self.norm_txt_3(t_out + t_attn2).squeeze(1)

        return t_out, i_out


# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC GATING NETWORK  (V3 — uncertainty-aware)
# ─────────────────────────────────────────────────────────────────────────────

class DynamicGatingNetworkV3(nn.Module):
    """
    Per-sample scalar gating network α ∈ (0,1).

    V3 adds two uncertainty scalars (img_entropy, txt_entropy) from the
    auxiliary per-modality classifier heads to the input descriptor.
    Input dim: 768*4 + 128 + 2 = 3202
    Dropout raised to 0.4 / 0.3 (anti-overfitting, resubmission).
    """

    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.query = nn.Linear(embed_dim, 128)
        self.key   = nn.Linear(embed_dim, 128)

        in_dim = embed_dim * 4 + 128 + 2   # 3202
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
        nn.init.zeros_(self.net[-1].bias)   # start at α = 0.5

    def forward(self, txt_feat: torch.Tensor, img_feat: torch.Tensor,
                img_entropy: torch.Tensor = None,
                txt_entropy: torch.Tensor = None) -> torch.Tensor:

        q          = self.query(img_feat)                              # (B, 128)
        k          = self.key(txt_feat)
        attn_score = (q * k).sum(dim=-1, keepdim=True) / (128 ** 0.5) # (B, 1)

        delta = (img_feat - txt_feat).abs()
        prod  = img_feat * txt_feat

        # Uncertainty signals — fall back to zeros if not provided
        if img_entropy is None:
            img_entropy = torch.zeros(img_feat.size(0), 1, device=img_feat.device)
        if txt_entropy is None:
            txt_entropy = torch.zeros(txt_feat.size(0), 1, device=txt_feat.device)

        combined = torch.cat([
            img_feat, txt_feat, delta, prod,
            attn_score.expand(-1, 128),
            img_entropy, txt_entropy,
        ], dim=-1)   # (B, 3202)

        return torch.sigmoid(self.net(combined))   # (B, 1)


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE FUSION MODEL  (V3 — with aux heads + split forward)
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveFusionModelV3(nn.Module):
    """
    V3 enhancements:
      • Per-modality auxiliary classifier heads (img_aux_head, txt_aux_head).
        Their prediction entropy feeds uncertainty into the gating network.
        Their logits contribute an auxiliary focal loss during training.
      • extract_features() / fuse_and_classify() split so the training loop
        can apply feature-level MixUp between the two calls.
      • forward() wraps both calls for normal inference.

    Classifier dropout raised: 0.4 / 0.3 / 0.2.
    """

    def __init__(self, clip_model_name: str = "openai/clip-vit-large-patch14",
                 unfreeze_layers: int = 4, attn_dropout: float = 0.2):
        super().__init__()
        self.clip      = CLIPModel.from_pretrained(clip_model_name)
        self.embed_dim = self.clip.config.projection_dim   # 768

        freeze_clip_layers(self.clip, unfreeze_layers)

        self.cross_attn = CrossModalAttentionV2(
            embed_dim=self.embed_dim, num_heads=12,
            ffn_dim=2048, dropout=attn_dropout,
        )
        self.gating = DynamicGatingNetworkV3(embed_dim=self.embed_dim)

        # Lightweight per-modality auxiliary heads
        self.img_aux_head = nn.Linear(self.embed_dim, 1)
        self.txt_aux_head = nn.Linear(self.embed_dim, 1)

        # Main classifier — deeper dropout to fight overfitting
        fused_dim = self.embed_dim * 2 + 1   # 1537
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    # ── Feature extraction (CLIP only — expensive, run once per batch) ────────
    def extract_features(self, input_ids, attention_mask, pixel_values):
        """Returns L2-normalised (txt_feat, img_feat), each (B, D)."""
        txt = extract_clip_text_features(
            self.clip, input_ids=input_ids, attention_mask=attention_mask
        )
        img = extract_clip_image_features(self.clip, pixel_values=pixel_values)
        return F.normalize(txt, dim=-1), F.normalize(img, dim=-1)

    # ── Fusion + classification (cheap — run twice for R-Drop) ───────────────
    def fuse_and_classify(self, txt_feat, img_feat):
        """
        Returns:
            logit        (B,)  — main classification logit
            alpha        (B,)  — per-sample fusion weight
            img_aux_logit (B,) — auxiliary image-only logit
            txt_aux_logit (B,) — auxiliary text-only logit
        """
        # Cross-modal attention
        txt_feat, img_feat = self.cross_attn(txt_feat, img_feat)
        txt_feat = F.normalize(txt_feat, dim=-1)
        img_feat = F.normalize(img_feat, dim=-1)

        # Auxiliary per-modality predictions
        img_aux_logit = self.img_aux_head(img_feat).squeeze(-1)   # (B,)
        txt_aux_logit = self.txt_aux_head(txt_feat).squeeze(-1)

        # Per-modality uncertainty (entropy of aux predictions)
        def _entropy(logit):
            p = torch.sigmoid(logit).unsqueeze(1)   # (B, 1)
            return -(p * (p + 1e-8).log() + (1 - p) * (1 - p + 1e-8).log())

        img_entropy = _entropy(img_aux_logit)   # (B, 1)
        txt_entropy = _entropy(txt_aux_logit)

        # Dynamic gating with uncertainty signals
        alpha = self.gating(txt_feat, img_feat, img_entropy, txt_entropy)  # (B, 1)

        # Weighted fusion
        img_branch = alpha * img_feat
        txt_branch = (1.0 - alpha) * txt_feat
        fused      = torch.cat([img_branch, txt_branch, alpha], dim=-1)   # (B, 1537)

        logit = self.classifier(fused).squeeze(-1)   # (B,)
        return logit, alpha.squeeze(-1), img_aux_logit, txt_aux_logit

    # ── Standard forward (used for inference / evaluate) ─────────────────────
    def forward(self, input_ids, attention_mask, pixel_values):
        txt, img = self.extract_features(input_ids, attention_mask, pixel_values)
        return self.fuse_and_classify(txt, img)


# ─────────────────────────────────────────────────────────────────────────────
# STATIC FUSION MODEL  (V2 — unchanged for fair baseline comparison)
# ─────────────────────────────────────────────────────────────────────────────

class _StaticGating(nn.Module):
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, txt_feat, img_feat):
        B = txt_feat.size(0)
        return torch.full((B, 1), self.alpha,
                          device=txt_feat.device, dtype=txt_feat.dtype)


class StaticFusionModelV2(nn.Module):
    """Identical to AdaptiveFusionModelV3 except gating is fixed at alpha=0.5."""

    def __init__(self, clip_model_name: str = "openai/clip-vit-large-patch14",
                 unfreeze_layers: int = 4, static_alpha: float = 0.5,
                 attn_dropout: float = 0.2):
        super().__init__()
        self.clip      = CLIPModel.from_pretrained(clip_model_name)
        self.embed_dim = self.clip.config.projection_dim

        freeze_clip_layers(self.clip, unfreeze_layers)

        self.cross_attn = CrossModalAttentionV2(
            embed_dim=self.embed_dim, num_heads=12,
            ffn_dim=2048, dropout=attn_dropout,
        )
        self.gating = _StaticGating(alpha=static_alpha)

        fused_dim = self.embed_dim * 2 + 1
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        txt = extract_clip_text_features(
            self.clip, input_ids=input_ids, attention_mask=attention_mask
        )
        img = extract_clip_image_features(self.clip, pixel_values=pixel_values)

        txt = F.normalize(txt, dim=-1)
        img = F.normalize(img, dim=-1)

        txt, img = self.cross_attn(txt, img)
        txt = F.normalize(txt, dim=-1)
        img = F.normalize(img, dim=-1)

        alpha      = self.gating(txt, img)
        fused      = torch.cat([alpha * img, (1 - alpha) * txt, alpha], dim=-1)
        logit      = self.classifier(fused).squeeze(-1)
        return logit, alpha.squeeze(-1)
