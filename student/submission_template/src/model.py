"""TiDE forecaster with RevIN and future-known covariate conditioning."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    """Reversible instance normalization with learnable affine parameters."""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        if mode == "denorm":
            return self._denormalize(x)
        raise ValueError(f"Unsupported RevIN mode: {mode!r}")

    def _get_statistics(self, x: torch.Tensor) -> None:
        self.mean = x.mean(dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.gamma + self.beta
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = (x - self.beta) / self.gamma
        return x * self.stdev + self.mean


class ResidualBlock(nn.Module):
    """Dense residual MLP block used in TiDE encoder/decoder."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.net(x))


class ForecastModel(nn.Module):
    """
    TiDE-style dense encoder-decoder that conditions on future-known covariates.

    Inputs:
        past_target: [B, seq_len]
        past_cov:    [B, seq_len, num_covariates]
        future_cov:  [B, pred_len, num_covariates]
    Output:
        [B, pred_len] target forecasts
    """

    def __init__(
        self,
        seq_len: int = 336,
        pred_len: int = 24,
        num_covariates: int = 22,
        hidden_dim: int = 256,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dropout: float = 0.1,
        use_revin: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_covariates = num_covariates
        self.hidden_dim = hidden_dim
        self.use_revin = use_revin

        if use_revin:
            self.revin = RevIN(num_features=1, affine=True)

        self.past_feat_proj = nn.Linear(1 + num_covariates, hidden_dim)
        self.future_feat_proj = nn.Linear(num_covariates, hidden_dim)

        encoder_in = seq_len * hidden_dim
        encoder_layers: list[nn.Module] = [
            nn.Linear(encoder_in, hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(num_encoder_layers):
            encoder_layers.append(ResidualBlock(hidden_dim, hidden_dim * 2, dropout))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_in = hidden_dim + pred_len * hidden_dim
        decoder_layers: list[nn.Module] = [
            nn.Linear(decoder_in, hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(num_decoder_layers):
            decoder_layers.append(ResidualBlock(hidden_dim, hidden_dim * 2, dropout))
        decoder_layers.append(nn.Linear(hidden_dim, pred_len))
        self.decoder = nn.Sequential(*decoder_layers)

        # Residual temporal skip from lookback target to horizon.
        self.temporal_decoder = nn.Linear(seq_len, pred_len)

    def forward(
        self,
        past_target: torch.Tensor,
        past_cov: torch.Tensor,
        future_cov: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_revin:
            past_target = self.revin(past_target.unsqueeze(-1), mode="norm").squeeze(-1)

        past_mixed = torch.cat([past_target.unsqueeze(-1), past_cov], dim=-1)
        past_hidden = self.past_feat_proj(past_mixed)
        encoded = self.encoder(past_hidden.reshape(past_target.size(0), -1))

        future_hidden = self.future_feat_proj(future_cov)
        decoded = self.decoder(torch.cat([encoded, future_hidden.reshape(past_target.size(0), -1)], dim=-1))
        out = decoded + self.temporal_decoder(past_target)

        if self.use_revin:
            out = self.revin(out.unsqueeze(-1), mode="denorm").squeeze(-1)

        return out
