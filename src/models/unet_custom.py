from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ibn import IBN


def get_norm(norm_type: str, num_features: int) -> nn.Module:
    if norm_type == "ibn":
        return IBN(num_features)

    return nn.BatchNorm2d(num_features)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_type: str = "bn") -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            get_norm(norm_type, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            get_norm(norm_type, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DoubleConvAdapter(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_type: str = "bn") -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)

        self.adapter1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.adapter2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

        nn.init.zeros_(self.adapter1.weight)
        nn.init.zeros_(self.adapter2.weight)

        self.norm1 = get_norm(norm_type, out_channels)
        self.norm2 = get_norm(norm_type, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x) + self.adapter1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out) + self.adapter2(out)
        out = self.norm2(out)
        out = self.relu(out)
        return out


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_type: str = "bn", use_adapter: bool = False) -> None:
        super().__init__()
        ConvBlock = DoubleConvAdapter if use_adapter else DoubleConv
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels, norm_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return self.conv(x)


class Up(nn.Module):
    def __init__(
            self,
            in_channels: int,
            skip_channels: int,
            out_channels: int,
            norm_type: str = "bn",
            bilinear: bool = True,
            use_adapter: bool = False,
    ) -> None:
        super().__init__()
        self.bilinear = bilinear
        ConvBlock = DoubleConvAdapter if use_adapter else DoubleConv

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            up_channels = in_channels
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            up_channels = out_channels

        self.conv = ConvBlock(up_channels + skip_channels, out_channels, norm_type)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class CustomUNet(nn.Module):
    def __init__(
            self,
            n_channels: int,
            n_classes: int,
            norm_type: str = "bn",
            bilinear: bool = True,
            use_adapter: bool = False,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.norm_type = norm_type
        self.use_adapter = use_adapter
        self.bilinear = bilinear

        ConvBlock = DoubleConvAdapter if use_adapter else DoubleConv

        self.inc = ConvBlock(n_channels, 64, norm_type)
        self.down1 = Down(64, 128, norm_type, use_adapter)
        self.down2 = Down(128, 256, norm_type, use_adapter)
        self.down3 = Down(256, 512, norm_type, use_adapter)
        self.down4 = Down(512, 1024, norm_type, use_adapter)

        self.up1 = Up(1024, 512, 512, norm_type, bilinear, use_adapter)
        self.up2 = Up(512, 256, 256, norm_type, bilinear, use_adapter)
        self.up3 = Up(256, 128, 128, norm_type, bilinear, use_adapter)
        self.up4 = Up(128, 64, 64, norm_type, bilinear, use_adapter)
        self.outc = OutConv(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNetMidFusion(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, norm_type: str = "bn") -> None:
        super().__init__()
        assert n_channels == 4, "UNetMidFusion expects 4 input channels (RGB + NIR)"
        self.norm_type = norm_type

        self.rgb_conv = DoubleConv(3, 32, norm_type)
        self.nir_conv = DoubleConv(1, 32, norm_type)

        self.fuse_conv = nn.Conv2d(64, 64, kernel_size=1)

        self.unet = CustomUNet(n_channels=64, n_classes=n_classes, norm_type=norm_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb = x[:, :3, :, :]
        nir = x[:, 3:4, :, :]

        rgb_feat = self.rgb_conv(rgb)
        nir_feat = self.nir_conv(nir)
        fused = torch.cat([rgb_feat, nir_feat], dim=1)  # (B,64,H,W)
        fused = self.fuse_conv(fused)
        logits = self.unet(fused)
        return logits


class UNetAdapter(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, norm_type: str = "bn") -> None:
        super().__init__()
        self.model = CustomUNet(
            n_channels=n_channels,
            n_classes=n_classes,
            norm_type=norm_type,
            bilinear=True,
            use_adapter=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
