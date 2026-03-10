"""
model.py
========
Unified Adaptive Framework for Multimodal Data Fusion
with Static and Dynamic Modality Reweighting

Architecture:
    CLIP Image Encoder --+
                         +--> Cross-Modal Attention --> Dynamic Gating --> Classifier --> 0/1
    CLIP Text Encoder  --+

Phase 1: StaticGatingNetwork  -- one learned scalar alpha for all samples
Phase 2: DynamicGatingNetwork -- per-sample alpha computed from input (current)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel


# -----------------------------------------------------------------
# BLOCK 1: Cross-Modal Attention
# -----------------------------------------------------------------

class CrossModalAttention(nn.Module):
    """
    Bidirectional cross-modal attention.
    - Image attends to Text  : "what text context is relevant to what I see?"
    - Text  attends to Image : "what visual context is relevant to what I say?"

    Each direction has:
        MultiheadAttention -> Residual + LayerNorm -> FFN -> Residual + LayerNorm
    """
    def __init__(self, embed_dim=512, num_heads=8, ffn_dim=1024, dropout=0.1):
        super().__init__()

        self.img_to_text_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.text_to_img_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )

        self.norm_img_1  = nn.LayerNorm(embed_dim)
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

        self.norm_img_2  = nn.LayerNorm(embed_dim)
        self.norm_text_2 = nn.LayerNorm(embed_dim)

    def forward(self, text_feat, img_feat):
        t = text_feat.unsqueeze(1)
        i = img_feat.unsqueeze(1)

        img_attn,  _ = self.img_to_text_attn(query=i, key=t, value=t)
        text_attn, _ = self.text_to_img_attn(query=t, key=i, value=i)

        img_out  = self.norm_img_1 (i + img_attn ).squeeze(1)
        text_out = self.norm_text_1(t + text_attn).squeeze(1)

        img_out  = self.norm_img_2 (img_out  + self.ffn_img (img_out ))
        text_out = self.norm_text_2(text_out + self.ffn_text(text_out))

        return text_out, img_out


# -----------------------------------------------------------------
# BLOCK 2a: Static Gating Network (Phase 1 -- kept for reference)
# -----------------------------------------------------------------

class StaticGatingNetwork(nn.Module):
    """
    Phase 1 -- Static Reweighting.
    Learns ONE fixed alpha scalar for the entire dataset.
    Kept here for ablation comparison only.
    """
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, text_feat, img_feat):
        alpha = torch.clamp(self.alpha, 0.0, 1.0)
        fused = alpha * img_feat + (1 - alpha) * text_feat
        return fused, alpha.unsqueeze(0).expand(text_feat.size(0), -1).expand(-1, text_feat.size(1))


# -----------------------------------------------------------------
# BLOCK 2b: Dynamic Gating Network (Phase 2 -- active)
# -----------------------------------------------------------------

class DynamicGatingNetwork(nn.Module):
    """
    Phase 2 -- Dynamic Reweighting.

    For EACH sample, computes a per-dimension alpha in (0,1):
    - alpha close to 1.0 --> trust image more for this sample
    - alpha close to 0.0 --> trust text more for this sample

    Gate is computed from the concatenation of both modality features,
    so the model learns: "given THIS image and THIS text, which should
    I trust more?"

    Requires entropy regularization during training to prevent collapse.
    See train_dynamic.py for entropy loss implementation.
    """
    def __init__(self, embed_dim=512, dropout=0.3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),   # (B, 1024) -> (B, 512)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, embed_dim),        # (B, 512) -- per dimension
            nn.Sigmoid()                      # alpha in (0, 1)
        )

    def forward(self, text_feat, img_feat):
        """
        Args:
            text_feat : (B, 512)
            img_feat  : (B, 512)
        Returns:
            fused     : (B, 512)
            alpha     : (B, 512) -- per-sample, per-dimension weights
        """
        combined = torch.cat([text_feat, img_feat], dim=-1)  # (B, 1024)
        alpha    = self.gate(combined)                        # (B, 512)
        fused    = alpha * img_feat + (1 - alpha) * text_feat # (B, 512)
        return fused, alpha


# -----------------------------------------------------------------
# BLOCK 3: Full Model
# -----------------------------------------------------------------

class AdaptiveFusionModel(nn.Module):
    """
    Full pipeline:
    1. CLIP encodes image -> img_feat  (512)
    2. CLIP encodes text  -> text_feat (512)
    3. L2 normalize both
    4. CrossModalAttention (bidirectional + FFN)
    5. Re-normalize after attention
    6. DynamicGatingNetwork -- per-sample alpha (Phase 2)
    7. Classifier on [fused || text_feat || img_feat] -> logit
    """
    def __init__(
        self,
        embed_dim   = 512,
        num_heads   = 8,
        ffn_dim     = 1024,
        dropout     = 0.1,
        freeze_clip = True
    ):
        super().__init__()

        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        if freeze_clip:
            for param in self.clip.parameters():
                param.requires_grad = False

        self.cross_attn = CrossModalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout
        )

        # Phase 2: Dynamic Gating (replaced StaticGatingNetwork)
        self.gating = DynamicGatingNetwork(
            embed_dim=embed_dim,
            dropout=0.3
        )

        # Classifier: fused(512) + text(512) + img(512) = 1536
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 3, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        # Step 1: Encode
        text_out  = self.clip.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_feat = self.clip.text_projection(text_out.pooler_output)   # (B, 512)

        img_out   = self.clip.vision_model(pixel_values=pixel_values)
        img_feat  = self.clip.visual_projection(img_out.pooler_output)  # (B, 512)

        # Step 2: L2 Normalize
        text_feat = F.normalize(text_feat, dim=-1)
        img_feat  = F.normalize(img_feat,  dim=-1)

        # Step 3: Cross-Modal Attention
        text_feat, img_feat = self.cross_attn(text_feat, img_feat)

        # Step 4: Re-normalize
        text_feat = F.normalize(text_feat, dim=-1)
        img_feat  = F.normalize(img_feat,  dim=-1)

        # Step 5: Dynamic Gating
        fused, alpha = self.gating(text_feat, img_feat)

        # Step 6: Classify
        combined = torch.cat([fused, text_feat, img_feat], dim=-1)  # (B, 1536)
        logit    = self.classifier(combined).squeeze(1)              # (B,)

        return logit, alpha