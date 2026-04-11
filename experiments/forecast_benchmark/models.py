from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class PPNConfig:
    dim: int
    context_len: int
    horizon: int
    hidden_dim: int = 128
    tau_hidden_dim: int = 32
    dropout: float = 0.0
    use_batch_norm: bool = False
    activation: str = "silu"


class IntrinsicTimeModule(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 32, output_dim: int | None = None):
        super().__init__()
        self.output_dim = int(output_dim) if output_dim is not None else int(dim)
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.output_dim),
        )
        self.softplus = nn.Softplus()
        self.epsilon = 1e-3

    def forward(self, x_t: torch.Tensor, x_prev: torch.Tensor) -> torch.Tensor:
        z = torch.cat([x_t, x_prev], dim=-1)
        tau = self.softplus(self.net(z)) + self.epsilon
        return tau


class MLPBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        activation: str = "silu",
    ):
        super().__init__()

        if activation == "silu":
            act_fn = nn.SiLU
        elif activation == "relu":
            act_fn = nn.ReLU
        elif activation == "gelu":
            act_fn = nn.GELU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        layers = [nn.Linear(input_dim, hidden_dim)]
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(act_fn())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(act_fn())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _repeat_steps(tensor: torch.Tensor, horizon: int) -> torch.Tensor:
    return tensor.unsqueeze(1).expand(tensor.shape[0], horizon, tensor.shape[-1])


class PPNForecastModel(nn.Module):
    def __init__(self, config: PPNConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.context_len = config.context_len
        self.horizon = config.horizon

        self.tau_module = IntrinsicTimeModule(config.dim, hidden_dim=max(16, config.tau_hidden_dim), output_dim=config.dim)
        self.context_encoder = MLPBlock(
            input_dim=config.context_len * config.dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.hidden_dim,
            dropout=config.dropout,
            use_batch_norm=config.use_batch_norm,
            activation=config.activation,
        )
        self.tau_path_head = MLPBlock(
            input_dim=config.hidden_dim + 2 * config.dim + 1,
            hidden_dim=config.hidden_dim,
            output_dim=config.dim,
            dropout=config.dropout,
            use_batch_norm=config.use_batch_norm,
            activation=config.activation,
        )
        self.prediction_head = MLPBlock(
            input_dim=config.hidden_dim + 4 * config.dim + 1,
            hidden_dim=config.hidden_dim,
            output_dim=config.dim,
            dropout=config.dropout,
            use_batch_norm=config.use_batch_norm,
            activation=config.activation,
        )

    def forward(
        self,
        context: torch.Tensor,
        decoder_input: Optional[torch.Tensor] = None,
        horizon: Optional[int] = None,
        context_mark: Optional[torch.Tensor] = None,
        target_mark: Optional[torch.Tensor] = None,
        use_intrinsic: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del decoder_input, context_mark, target_mark
        batch_size, seq_len, dim = context.shape
        if seq_len != self.context_len:
            raise ValueError(f"Expected context length {self.context_len}, got {seq_len}")
        if dim != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {dim}")

        pred_horizon = int(horizon) if horizon is not None else self.horizon
        if pred_horizon != self.horizon:
            raise ValueError(f"Model horizon is fixed to {self.horizon}, got {pred_horizon}")

        x_t = context[:, -1, :]
        x_prev = context[:, -2, :]
        x_prev2 = context[:, -3, :] if seq_len >= 3 else x_prev
        recent_delta = x_t - x_prev

        if use_intrinsic:
            tau_t = self.tau_module(x_t, x_prev)
            tau_prev = self.tau_module(x_prev, x_prev2)
            rho_t = tau_t / (tau_prev + 1e-6)
        else:
            tau_t = torch.ones((batch_size, 1), device=context.device, dtype=context.dtype)
            rho_t = torch.ones((batch_size, 1), device=context.device, dtype=context.dtype)

        context_flat = context.reshape(batch_size, -1)
        context_embedding = self.context_encoder(context_flat)

        u = torch.linspace(0, 1, self.horizon + 1, device=context.device, dtype=context.dtype)[1:]
        u = u.unsqueeze(0).unsqueeze(-1)

        context_rep = _repeat_steps(context_embedding, self.horizon)
        x_t_rep = _repeat_steps(x_t, self.horizon)
        delta_rep = _repeat_steps(recent_delta, self.horizon)
        tau_seed_rep = _repeat_steps(tau_t, self.horizon)
        rho_rep = _repeat_steps(rho_t, self.horizon)
        u_rep = u.expand(batch_size, -1, -1)

        tau_features = torch.cat([context_rep, x_t_rep, delta_rep, u_rep], dim=-1)
        tau_features_flat = tau_features.reshape(batch_size * self.horizon, -1)
        tau_increments = self.tau_path_head(tau_features_flat)
        tau_increments = tau_increments.reshape(batch_size, self.horizon, self.dim)
        tau_increments = torch.nn.functional.softplus(tau_increments) + 1e-3
        tau_path = tau_seed_rep + torch.cumsum(tau_increments, dim=1)

        features = torch.cat([context_rep, x_t_rep, delta_rep, tau_path, rho_rep, u_rep], dim=-1)
        features_flat = features.reshape(batch_size * self.horizon, -1)

        deltas_flat = self.prediction_head(features_flat)
        deltas = deltas_flat.reshape(batch_size, self.horizon, self.dim)

        predictions = x_t.unsqueeze(1) + deltas

        aux = {
            "tau": tau_t,
            "rho": rho_t,
            "tau_path": tau_path,
            "x_t": x_t,
            "context_embedding": context_embedding,
        }
        return predictions, aux


def compute_weighted_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    weight_scheme: str = "linear",
) -> Tuple[torch.Tensor, torch.Tensor]:
    horizon = predictions.shape[1]

    if weight_scheme == "linear":
        weights = torch.arange(1, horizon + 1, device=predictions.device, dtype=predictions.dtype)
    elif weight_scheme == "quadratic":
        weights = torch.arange(1, horizon + 1, device=predictions.device, dtype=predictions.dtype) ** 2
    elif weight_scheme == "uniform":
        weights = torch.ones(horizon, device=predictions.device, dtype=predictions.dtype)
    else:
        raise ValueError(f"Unknown weight_scheme: {weight_scheme}")

    mse_per_step = ((predictions - targets) ** 2).mean(dim=(0, 2))
    weights = weights / weights.sum()
    weighted_loss = (mse_per_step * weights).sum()
    return weighted_loss, mse_per_step


def weighted_mse(predictions: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return compute_weighted_mse(predictions, targets, weight_scheme="linear")


def build_model(name: str, input_dim: int, context_len: int, horizon: int) -> nn.Module:
    if name.lower() != "ppn":
        raise ValueError(f"Unsupported model: {name}")

    cfg = PPNConfig(
        dim=int(input_dim),
        context_len=int(context_len),
        horizon=int(horizon),
        hidden_dim=128,
        tau_hidden_dim=32,
        dropout=0.0,
        use_batch_norm=False,
        activation="silu",
    )
    return PPNForecastModel(cfg)
