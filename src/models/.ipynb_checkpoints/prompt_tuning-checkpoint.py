from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn


class VPT(nn.Module):

    def __init__(self, vit: nn.Module, prompt_length: int, embed_dim: int) -> None:
        super().__init__()
        self.vit = vit
        self.prompt_length = prompt_length
        self.embed_dim = embed_dim
        self.prompt = nn.Parameter(torch.zeros(1, prompt_length, embed_dim))
        nn.init.normal_(self.prompt, mean=0.0, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patch_tokens = self.vit.patch_embed(images)
        B = patch_tokens.size(0)

        prompt = self.prompt.expand(B, -1, -1)

        tokens = torch.cat([prompt, patch_tokens], dim=1)  # (B, prompt_length + N, D)

        if hasattr(self.vit, 'interpolate_pos_encoding'):
            tokens = self.vit.interpolate_pos_encoding(tokens)

        x = self.vit.pos_drop(tokens)
        for block in self.vit.blocks:
            x = block(x)

        if hasattr(self.vit, 'norm'):
            x = self.vit.norm(x)
        return x
