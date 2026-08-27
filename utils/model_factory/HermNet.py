"""HermNet full architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Complex-domain operations
# =============================================================================

def iq_to_complex(x: torch.Tensor) -> torch.Tensor:
    """Convert an I/Q tensor [B, 2, L] to a complex tensor [B, 1, L]."""
    return torch.complex(x[:, 0, :], x[:, 1, :]).unsqueeze(1)


class ComplexConv1d(nn.Module):
    """Complex-valued 1D convolution."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True):
        super().__init__()
        self.conv_r = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False,
        )
        self.conv_i = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False,
        )
        if bias:
            self.bias_r = nn.Parameter(torch.zeros(out_channels))
            self.bias_i = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias_r", None)
            self.register_parameter("bias_i", None)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        real = self.conv_r(z.real) - self.conv_i(z.imag)
        imag = self.conv_r(z.imag) + self.conv_i(z.real)
        if self.bias_r is not None:
            real = real + self.bias_r.view(1, -1, 1)
            imag = imag + self.bias_i.view(1, -1, 1)
        return torch.complex(real, imag)


# =============================================================================
# Hermitian covariance attention
# =============================================================================

class HermitianCovarianceAttention(nn.Module):
    """Second-order channel attention using full Hermitian covariance."""

    def __init__(self, channels, reduction=8, dropout=0.0,
                 sqrt_iter=3, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.sqrt_iter = sqrt_iter
        self.eps = eps

        hidden = max(2, channels // reduction)
        self.diag_proj = nn.Linear(channels, hidden, bias=False)
        self.offdiag_proj = nn.Linear(
            channels * (channels - 1), hidden, bias=False
        )
        self.fuse = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(2 * hidden, channels, bias=False),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)

        upper_i, upper_j = torch.triu_indices(channels, channels, offset=1)
        self.register_buffer("upper_i", upper_i, persistent=False)
        self.register_buffer("upper_j", upper_j, persistent=False)

    @staticmethod
    def _newton_schulz_sqrt(
            matrix: torch.Tensor, num_iter: int) -> torch.Tensor:
        batch_size, channels, _ = matrix.shape
        dtype, device = matrix.dtype, matrix.device
        trace = matrix.diagonal(
            dim1=-2, dim2=-1
        ).real.sum(dim=-1).clamp(min=1e-6)
        matrix_norm = matrix / trace.view(batch_size, 1, 1).to(dtype)

        identity = torch.eye(
            channels, dtype=dtype, device=device
        ).expand(batch_size, channels, channels)
        y = matrix_norm
        z = identity.clone()
        for _ in range(num_iter):
            update = 0.5 * (3.0 * identity - z @ y)
            y = y @ update
            z = update @ z
        return y * trace.sqrt().view(batch_size, 1, 1).to(dtype)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size, channels, length = z.shape

        covariance = torch.einsum(
            "bcl,bdl->bcd", z, z.conj()
        ) / length
        trace = covariance.diagonal(
            dim1=-2, dim2=-1
        ).real.sum(dim=-1)
        regularizer = (
            self.eps * trace / channels
        ).view(batch_size, 1, 1).to(covariance.dtype)
        identity = torch.eye(
            channels, dtype=covariance.dtype, device=covariance.device
        ).expand(batch_size, channels, channels)
        covariance = covariance + regularizer * identity

        covariance_sqrt = self._newton_schulz_sqrt(
            covariance, self.sqrt_iter
        )

        diagonal = covariance_sqrt.diagonal(
            dim1=-2, dim2=-1
        ).real
        diagonal = F.normalize(diagonal, dim=-1, eps=self.eps)
        diagonal_feature = self.diag_proj(diagonal)

        upper = covariance_sqrt[:, self.upper_i, self.upper_j]
        offdiagonal = torch.cat([upper.real, upper.imag], dim=-1)
        offdiagonal = F.normalize(offdiagonal, dim=-1, eps=self.eps)
        offdiagonal_feature = self.offdiag_proj(offdiagonal)

        attention_feature = torch.cat(
            [diagonal_feature, offdiagonal_feature], dim=-1
        )
        weights = self.dropout(
            self.fuse(attention_feature)
        ).unsqueeze(-1)
        return z * weights


# =============================================================================
# Amplitude-phase fusion
# =============================================================================

class AmplitudePhaseFusion(nn.Module):
    """Fuse magnitude and phase into real-valued temporal features."""

    def __init__(self, in_channels, fused_dim, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels * 2, fused_dim, kernel_size,
                padding=kernel_size // 2, bias=False,
            ),
            nn.BatchNorm1d(fused_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                fused_dim, fused_dim, kernel_size,
                padding=kernel_size // 2, bias=False,
            ),
            nn.BatchNorm1d(fused_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        magnitude_phase = torch.cat(
            [torch.abs(z), torch.angle(z)], dim=1
        )
        return self.conv(magnitude_phase)


# =============================================================================
# Multi-scale gated temporal modeling
# =============================================================================

class MultiScaleGatedBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4, dropout=0.1, scales=(3, 5, 7)):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.norm = nn.LayerNorm(dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                dim, dim, kernel_size=kernel_size,
                padding=kernel_size // 2, groups=dim, bias=False,
            )
            for kernel_size in scales
        ])
        self.conv_merge = nn.Conv1d(
            dim * len(scales), dim, kernel_size=1, bias=False
        )
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.gate_proj = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)

        temporal = x.transpose(1, 2)
        multi_scale = torch.cat(
            [conv(temporal) for conv in self.convs], dim=1
        )
        multi_scale = self.conv_merge(multi_scale).transpose(1, 2)

        hidden = self.fc1(multi_scale)
        content, gate_input = hidden.chunk(2, dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_input))
        output = self.dropout(self.fc2(content * gate))
        return residual + output


class MultiScaleGatedTemporal(nn.Module):
    def __init__(self, channels, num_layers=2, mlp_ratio=4,
                 scales=(3, 5, 7), dropout=0.1):
        super().__init__()
        self.blocks = nn.Sequential(*[
            MultiScaleGatedBlock(
                channels,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                scales=scales,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


# =============================================================================
# Full HermNet
# =============================================================================

class HermNet(nn.Module):
    """
    Full HermNet pipeline:

        IQ
        -> complex convolutional frontend with two full HCA modules
        -> amplitude-phase fusion
        -> multi-scale gated temporal modeling
        -> global temporal average
        -> classifier
    """

    def __init__(self, num_classes=10, d_model=16, fused_dim=8,
                 num_temporal_layers=3, dropout=0.2, eta=2):
        super().__init__()

        self.frontend = nn.Sequential(
            ComplexConv1d(
                1, d_model, kernel_size=5, stride=2, padding=2
            ),
            ComplexConv1d(
                d_model, d_model, kernel_size=5, stride=2, padding=2
            ),
            HermitianCovarianceAttention(
                d_model, reduction=eta, dropout=dropout
            ),
            ComplexConv1d(
                d_model, d_model, kernel_size=5, stride=2, padding=2
            ),
            HermitianCovarianceAttention(
                d_model, reduction=eta, dropout=dropout
            ),
        )
        self.ap_fusion = AmplitudePhaseFusion(d_model, fused_dim)
        self.temporal = MultiScaleGatedTemporal(
            channels=fused_dim,
            num_layers=num_temporal_layers,
            mlp_ratio=4,
            scales=(3, 5, 7),
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the classifier-input representation [B, fused_dim]."""
        z = iq_to_complex(x)
        z = self.frontend(z)
        features = self.ap_fusion(z).transpose(1, 2)
        features = self.temporal(features)
        return features.mean(dim=1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for feature-extraction and analysis code."""
        return self.extract_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))
