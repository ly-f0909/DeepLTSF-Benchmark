"""TiDE forecaster with target-only RevIN and future-known covariate conditioning."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.features import NUM_COVARIATES, TARGET_IDX


class RevIN(nn.Module):
    """Reversible instance normalization applied to the target series only."""

    def __init__(self, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(1))
            self.beta = nn.Parameter(torch.zeros(1))

    def compute_stats(self, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = target.mean(dim=1, keepdim=True).detach()
        stdev = torch.sqrt(target.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        return mean, stdev

    def normalize_with_stats(
        self,
        target: torch.Tensor,
        mean: torch.Tensor,
        stdev: torch.Tensor,
    ) -> torch.Tensor:
        normed = (target - mean) / stdev
        if self.affine:
            normed = normed * self.gamma + self.beta
        return normed

    def normalize(self, target: torch.Tensor) -> torch.Tensor:
        mean, stdev = self.compute_stats(target)
        self.mean = mean
        self.stdev = stdev
        return self.normalize_with_stats(target, mean, stdev)

    def denormalize(self, target: torch.Tensor) -> torch.Tensor:
        out = target
        if self.affine:
            out = (out - self.beta) / self.gamma
        return out * self.stdev + self.mean


class ResidualBlock(nn.Module):
    """Dense residual block with LayerNorm and dropout."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.norm(x))


class ForecastModel(nn.Module):
    """
    TiDE dense encoder-decoder with target-only RevIN.

    Inputs:
        x_past:   [B, seq_len, num_features]  historical covariates + target
        x_future: [B, pred_len, num_covariates] known future covariates (no target)
    Output:
        [B, pred_len] target forecasts in original scale
    """

    def __init__(
        self,
        seq_len: int = 336,
        pred_len: int = 24,
        num_features: int = NUM_COVARIATES + 1,
        num_covariates: int = NUM_COVARIATES,
        target_idx: int = TARGET_IDX,
        hidden_dim: int = 256,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dropout: float = 0.2,
        use_revin: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_features = num_features
        self.num_covariates = num_covariates
        self.target_idx = target_idx
        self.hidden_dim = hidden_dim
        self.use_revin = use_revin

        if use_revin:
            self.revin = RevIN(affine=True)

        self.past_feat_proj = nn.Linear(num_covariates + 1, hidden_dim)
        self.future_feat_proj = nn.Linear(num_covariates, hidden_dim)

        encoder_in = seq_len * hidden_dim
        encoder_layers: list[nn.Module] = [
            nn.Linear(encoder_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        for _ in range(num_encoder_layers):
            encoder_layers.append(ResidualBlock(hidden_dim, hidden_dim * 2, dropout))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_in = hidden_dim + pred_len * hidden_dim
        decoder_layers: list[nn.Module] = [
            nn.Linear(decoder_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        for _ in range(num_decoder_layers):
            decoder_layers.append(ResidualBlock(hidden_dim, hidden_dim * 2, dropout))
        decoder_layers.append(nn.Linear(hidden_dim, pred_len))
        self.decoder = nn.Sequential(*decoder_layers)

        self.global_skip = nn.Linear(seq_len, pred_len)

    def _split_past(self, x_past: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        past_target = x_past[:, :, self.target_idx]
        past_cov = torch.cat(
            [x_past[:, :, : self.target_idx], x_past[:, :, self.target_idx + 1 :]],
            dim=-1,
        )
        return past_cov, past_target

    def forward(
        self,
        x_past: torch.Tensor,
        x_future: torch.Tensor,
        revin_mean: torch.Tensor | None = None,
        revin_stdev: torch.Tensor | None = None,
    ) -> torch.Tensor:
        past_cov, past_target = self._split_past(x_past)

        if self.use_revin:
            if revin_mean is not None and revin_stdev is not None:
                past_target_norm = self.revin.normalize_with_stats(
                    past_target, revin_mean, revin_stdev
                )
                self.revin.mean = revin_mean
                self.revin.stdev = revin_stdev
            else:
                past_target_norm = self.revin.normalize(past_target)
        else:
            past_target_norm = past_target

        past_mixed = torch.cat([past_cov, past_target_norm.unsqueeze(-1)], dim=-1)
        past_hidden = self.past_feat_proj(past_mixed)
        encoded = self.encoder(past_hidden.reshape(x_past.size(0), -1))

        future_hidden = self.future_feat_proj(x_future)
        mlp_out = self.decoder(
            torch.cat([encoded, future_hidden.reshape(x_past.size(0), -1)], dim=-1)
        )

        skip_out = self.global_skip(past_target_norm)
        out_norm = mlp_out + skip_out

        if self.use_revin:
            return self.revin.denormalize(out_norm)

        return out_norm
