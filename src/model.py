"""
model.py
========
Unified Adaptive Framework for Multimodal Data Fusion
with Static and Dynamic Modality Reweighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel


class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, ffn_dim=1024, dropout=0.1):
        super().__init__()
        self.img_to_text_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.text_to_img_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm_img_1 = nn.LayerNorm(embed_dim)
        self.norm_text_1 = nn.LayerNorm(embed_dim)
        self.ffn_img = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim), nn.Dropout(dropout)
        )
        self.ffn_text = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim), nn.Dropout(dropout)
        )
        self.norm_img_2 = nn.LayerNorm(embed_dim)
        self.norm_text_2 = nn.LayerNorm(embed_dim)

    def forward(self, text_feat, img_feat):
        t = text_feat.unsqueeze(1)
        i = img_feat.unsqueeze(1)
        img_attn, _ = self.img_to_text_attn(query=i, key=t, value=t)
        text_attn, _ = self.text_to_img_attn(query=t, key=i, value=i)
        img_out = self.norm_img_1(i + img_attn).squeeze(1)
        text_out = self.norm_text_1(t + text_attn).squeeze(1)
        img_out = self.norm_img_2(img_out + self.ffn_img(img_out))
        text_out = self.norm_text_2(text_out + self.ffn_text(text_out))
        return text_out, img_out


class StaticGatingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, text_feat, img_feat):
        alpha = torch.clamp(self.alpha, 0.0, 1.0)
        return alpha


class DynamicGatingNetwork(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, text_feat, img_feat):
        delta = torch.abs(img_feat - text_feat)
        prod = img_feat * text_feat
        combined = torch.cat([img_feat, text_feat, delta, prod], dim=-1)
        logits = self.net(combined)
        alpha = torch.sigmoid(logits)
        return alpha


class AdaptiveFusionModel(nn.Module):
    def __init__(
        self,
        embed_dim=512,
        num_heads=8,
        ffn_dim=1024,
        dropout=0.1,
        classifier_dropout=0.3,
        freeze_clip=True,
        use_dynamic=True
    ):
        super().__init__()
        self.use_dynamic = use_dynamic
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        if freeze_clip:
            for param in self.clip.parameters():
                param.requires_grad = False

        self.cross_attn = CrossModalAttention(embed_dim=embed_dim, num_heads=num_heads, ffn_dim=ffn_dim, dropout=dropout)
        self.gating = DynamicGatingNetwork(embed_dim=embed_dim) if self.use_dynamic else StaticGatingNetwork()

        classifier_in_dim = embed_dim * 2 + 1
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, 256),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(64, 1)
        )

    def forward(self, input_ids, attention_mask, pixel_values, return_debug=False):
        text_out = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = self.clip.text_projection(text_out.pooler_output)
        img_out = self.clip.vision_model(pixel_values=pixel_values)
        img_feat = self.clip.visual_projection(img_out.pooler_output)

        text_feat = F.normalize(text_feat, dim=-1)
        img_feat = F.normalize(img_feat, dim=-1)
        text_pre_attn = text_feat
        img_pre_attn = img_feat

        text_feat, img_feat = self.cross_attn(text_feat, img_feat)
        text_feat = F.normalize(text_feat, dim=-1)
        img_feat = F.normalize(img_feat, dim=-1)

        if self.use_dynamic:
            alpha = self.gating(text_feat, img_feat)
        else:
            alpha = self.gating(text_feat, img_feat).view(1, 1).expand(text_feat.size(0), 1)

        img_branch = alpha * img_feat
        text_branch = (1.0 - alpha) * text_feat
        fused = torch.cat([img_branch, text_branch], dim=-1)
        fused = torch.cat([fused, alpha], dim=-1)

        logit = self.classifier(fused).squeeze(1)
        alpha_out = alpha.squeeze(-1)

        if not return_debug:
            return logit, alpha_out

        debug = {
            "attn_delta_text": (text_feat - text_pre_attn).norm(dim=-1).mean().detach(),
            "attn_delta_img": (img_feat - img_pre_attn).norm(dim=-1).mean().detach(),
        }
        return logit, alpha_out, debug
