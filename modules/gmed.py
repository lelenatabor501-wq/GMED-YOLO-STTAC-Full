"""Custom modules used by the paper's GMED-YOLO detector."""

import numpy as np
import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


__all__ = [
    "CSPOmniKernel",
    "ConvEdgeFusion",
    "FGM",
    "MultiScaleEdgeInfoGenerator",
    "OmniKernel",
    "SPDConv",
    "SelectScale",
    "SkipConvEdgeFusion",
    "SobelConv",
]


class FGM(nn.Module):
    """Frequency-domain gating used internally by OmniKernel."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(dim, dim * 2, 3, 1, 1, groups=dim)
        self.dwconv1 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.dwconv2 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)
        x2_fft = torch.fft.fft2(x2, norm="backward")
        out = torch.fft.ifft2(x1 * x2_fft, dim=(-2, -1), norm="backward")
        return torch.abs(out) * self.alpha + x * self.beta


class OmniKernel(nn.Module):
    """Multi-branch large-kernel block used by CSPOmniKernel."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        kernel_size = 31
        padding = kernel_size // 2
        self.in_conv = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.GELU())
        self.out_conv = nn.Conv2d(dim, dim, 1)
        self.dw_13 = nn.Conv2d(
            dim, dim, kernel_size=(1, kernel_size), padding=(0, padding), groups=dim
        )
        self.dw_31 = nn.Conv2d(
            dim, dim, kernel_size=(kernel_size, 1), padding=(padding, 0), groups=dim
        )
        self.dw_33 = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, groups=dim)
        self.activation = nn.ReLU()
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fac_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.fac_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fgm = FGM(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.in_conv(x)
        frequency_attention = self.fac_conv(self.fac_pool(out))
        frequency_features = torch.fft.fft2(out, norm="backward")
        frequency_features = frequency_attention * frequency_features
        frequency_features = torch.abs(
            torch.fft.ifft2(frequency_features, dim=(-2, -1), norm="backward")
        )
        spatial_attention = self.conv(self.pool(frequency_features))
        gated_features = self.fgm(spatial_attention * frequency_features)
        out = x + self.dw_13(out) + self.dw_31(out) + self.dw_33(out) + self.dw_11(out) + gated_features
        return self.out_conv(self.activation(out))


class SPDConv(nn.Module):
    """Space-to-depth convolution that preserves fine-grained P2 information."""

    def __init__(self, in_channels: int, out_channels: int, dimension: int = 1) -> None:
        super().__init__()
        self.dimension = dimension
        self.conv = Conv(in_channels * 4, out_channels, k=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        space_to_depth = torch.cat(
            [x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]],
            dim=self.dimension,
        )
        return self.conv(space_to_depth)


class CSPOmniKernel(nn.Module):
    """Cross-stage partial OmniKernel module (CSPOKM in the manuscript)."""

    def __init__(self, dim: int, expansion: float = 0.25) -> None:
        super().__init__()
        self.expansion = expansion
        self.cv1 = Conv(dim, dim, 1)
        self.cv2 = Conv(dim, dim, 1)
        self.omni_kernel = OmniKernel(int(dim * self.expansion))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        omni_channels = int(self.cv1.conv.out_channels * self.expansion)
        identity_channels = self.cv1.conv.out_channels - omni_channels
        omni_branch, identity = torch.split(self.cv1(x), [omni_channels, identity_channels], dim=1)
        return self.cv2(torch.cat((self.omni_kernel(omni_branch), identity), dim=1))


class SobelConv(nn.Module):
    """Fixed first-order Sobel operator used to extract shallow P2 edge responses."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        sobel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        kernel_y = torch.tensor(sobel, dtype=torch.float32).unsqueeze(0).expand(channels, 1, 1, 3, 3)
        kernel_x = torch.tensor(sobel.T, dtype=torch.float32).unsqueeze(0).expand(channels, 1, 1, 3, 3)
        self.sobel_x = nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.sobel_y = nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.sobel_x.weight.data = kernel_x.clone()
        self.sobel_y.weight.data = kernel_y.clone()
        self.sobel_x.requires_grad_(False)
        self.sobel_y.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_3d = x[:, :, None, :, :]
        return (self.sobel_x(x_3d) + self.sobel_y(x_3d))[:, :, 0]


class MultiScaleEdgeInfoGenerator(nn.Module):
    """Generate Edge_P3, Edge_P4, and Edge_P5 from shallow P2 features."""

    def __init__(self, in_channels: int, out_channels: list[int]) -> None:
        super().__init__()
        self.sobel = SobelConv(in_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.projections = nn.ModuleList(Conv(in_channels, channels, 1) for channels in out_channels)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        pooled_features = []
        edge_features = self.sobel(x)
        for projection in self.projections:
            edge_features = self.max_pool(edge_features)
            pooled_features.append(projection(edge_features))
        return pooled_features


class ConvEdgeFusion(nn.Module):
    """Fuse same-scale edge and convolutional features at the neck input."""

    def __init__(self, in_channels: list[int], out_channels: int) -> None:
        super().__init__()
        hidden_channels = out_channels // 2
        self.channel_fusion = Conv(sum(in_channels), hidden_channels, k=1)
        self.feature_extraction = Conv(hidden_channels, hidden_channels, k=3)
        self.output_projection = Conv(hidden_channels, out_channels, k=1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        fused = self.channel_fusion(torch.cat(features, dim=1))
        return self.output_projection(self.feature_extraction(fused))


# Skip-CEF applies the same fusion operation at a second, later connection in the model graph.
SkipConvEdgeFusion = ConvEdgeFusion


class SelectScale(nn.Module):
    """Select one scale from the multi-scale edge feature list."""

    def __init__(self, index: int) -> None:
        super().__init__()
        self.index = index

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        return features[self.index]
