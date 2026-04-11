from __future__ import annotations

import torch
import torch.nn as nn


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
        return self.softplus(self.net(torch.cat([x_t, x_prev], dim=-1))) + self.epsilon


class MLPBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _repeat_steps(tensor: torch.Tensor, horizon: int) -> torch.Tensor:
    return tensor.unsqueeze(1).expand(tensor.shape[0], horizon, tensor.shape[-1])


class Model(nn.Module):
    """TSLib-compatible PPN model wrapper."""

    def __init__(self, configs):
        super().__init__()
        self.dim = int(configs.enc_in)
        self.context_len = int(configs.seq_len)
        self.horizon = int(configs.pred_len)
        self.hidden_dim = int(getattr(configs, "d_model", 128))
        self.dropout = float(getattr(configs, "dropout", 0.0))

        self.tau_module = IntrinsicTimeModule(self.dim, hidden_dim=32, output_dim=self.dim)
        self.context_encoder = MLPBlock(
            input_dim=self.context_len * self.dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            dropout=self.dropout,
        )
        self.tau_path_head = MLPBlock(
            input_dim=self.hidden_dim + 2 * self.dim + 1,
            hidden_dim=self.hidden_dim,
            output_dim=self.dim,
            dropout=self.dropout,
        )
        self.prediction_head = MLPBlock(
            input_dim=self.hidden_dim + 4 * self.dim + 1,
            hidden_dim=self.hidden_dim,
            output_dim=self.dim,
            dropout=self.dropout,
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        del x_mark_enc, x_dec, x_mark_dec, mask

        batch_size, seq_len, dim = x_enc.shape
        if seq_len != self.context_len or dim != self.dim:
            raise ValueError(
                f"PPN input mismatch: expected [B,{self.context_len},{self.dim}], got [B,{seq_len},{dim}]"
            )

        x_t = x_enc[:, -1, :]
        x_prev = x_enc[:, -2, :]
        x_prev2 = x_enc[:, -3, :] if seq_len >= 3 else x_prev
        recent_delta = x_t - x_prev

        tau_t = self.tau_module(x_t, x_prev)
        tau_prev = self.tau_module(x_prev, x_prev2)
        rho_t = tau_t / (tau_prev + 1e-6)

        context_embedding = self.context_encoder(x_enc.reshape(batch_size, -1))

        u = torch.linspace(0, 1, self.horizon + 1, device=x_enc.device, dtype=x_enc.dtype)[1:]
        u = u.unsqueeze(0).unsqueeze(-1)

        context_rep = _repeat_steps(context_embedding, self.horizon)
        x_t_rep = _repeat_steps(x_t, self.horizon)
        delta_rep = _repeat_steps(recent_delta, self.horizon)
        tau_seed_rep = _repeat_steps(tau_t, self.horizon)
        rho_rep = _repeat_steps(rho_t, self.horizon)
        u_rep = u.expand(batch_size, -1, -1)

        tau_features = torch.cat([context_rep, x_t_rep, delta_rep, u_rep], dim=-1)
        tau_increments = self.tau_path_head(tau_features.reshape(batch_size * self.horizon, -1))
        tau_increments = tau_increments.reshape(batch_size, self.horizon, self.dim)
        tau_increments = torch.nn.functional.softplus(tau_increments) + 1e-3
        tau_path = tau_seed_rep + torch.cumsum(tau_increments, dim=1)

        features = torch.cat([context_rep, x_t_rep, delta_rep, tau_path, rho_rep, u_rep], dim=-1)
        deltas = self.prediction_head(features.reshape(batch_size * self.horizon, -1))
        deltas = deltas.reshape(batch_size, self.horizon, self.dim)

        pred = x_t.unsqueeze(1) + deltas

        # TSLib forecast pipeline expects output length = label_len + pred_len.
        # The model now generates the whole horizon directly from a tau trajectory,
        # without bootstrap rollouts.
        return pred
