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


def _parse_tau_scales(scale_text: str) -> list[int]:
    scales = []
    for piece in str(scale_text).split(','):
        piece = piece.strip()
        if not piece:
            continue
        scale = int(piece)
        if scale > 0 and scale not in scales:
            scales.append(scale)
    return scales if scales else [1, 2, 4]


class Model(nn.Module):
    """TSLib-compatible PPN model wrapper."""

    def __init__(self, configs):
        super().__init__()
        self.dim = int(configs.enc_in)
        self.context_len = int(configs.seq_len)
        self.horizon = int(configs.pred_len)
        self.hidden_dim = int(getattr(configs, "d_model", 128))
        self.dropout = float(getattr(configs, "dropout", 0.0))
        self.use_patch_embed = int(getattr(configs, "ppn_use_patch_embed", 0)) == 1
        self.patch_len = int(getattr(configs, "ppn_patch_len", 8))
        self.patch_stride = int(getattr(configs, "ppn_patch_stride", 4))
        self.use_var_tau_gate = int(getattr(configs, "ppn_use_var_tau_gate", 0)) == 1
        self.var_tau_gate_scale = float(getattr(configs, "ppn_var_tau_gate_scale", 0.25))
        self.use_multi_scale_tau = int(getattr(configs, "ppn_use_multi_scale_tau", 0)) == 1
        self.tau_scales = _parse_tau_scales(getattr(configs, "ppn_tau_scales", "1,2,4"))
        self.use_horizon_residual = int(getattr(configs, "ppn_use_horizon_residual", 0)) == 1
        self.horizon_residual_scale = float(getattr(configs, "ppn_horizon_residual_scale", 0.1))

        self.tau_module = IntrinsicTimeModule(self.dim, hidden_dim=32, output_dim=self.dim)
        self.context_encoder = MLPBlock(
            input_dim=self.context_len * self.dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            dropout=self.dropout,
        )
        self.patch_context_encoder = MLPBlock(
            input_dim=self.dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            dropout=self.dropout,
        )
        self.patch_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.SiLU(),
        )
        self.var_tau_gate = nn.Sequential(
            nn.Linear(2 * self.dim, self.dim),
            nn.Sigmoid(),
        )
        self.multi_tau_gate = nn.Linear(self.hidden_dim + self.dim, len(self.tau_scales))
        self.horizon_encoder = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
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
        self.horizon_residual_head = MLPBlock(
            input_dim=self.hidden_dim * 2 + 5 * self.dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.dim,
            dropout=self.dropout,
        )

    def _encode_patch_context(self, x_enc: torch.Tensor) -> torch.Tensor:
        if not self.use_patch_embed or self.context_len < self.patch_len:
            return torch.zeros(x_enc.shape[0], self.hidden_dim, device=x_enc.device, dtype=x_enc.dtype)

        # Build lightweight patch statistics without changing the tau-driven prediction path.
        patches = x_enc.unfold(dimension=1, size=self.patch_len, step=max(1, self.patch_stride))
        # [B, num_patches, D, patch_len] -> average over patch length, then over patches.
        patch_feat = patches.mean(dim=-1).mean(dim=1)
        return self.patch_context_encoder(patch_feat)

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

        context_embedding = self.context_encoder(x_enc.reshape(batch_size, -1))

        if self.use_multi_scale_tau:
            scale_tau_t = []
            scale_tau_prev = []
            for scale in self.tau_scales:
                curr_idx = seq_len - 1
                prev_idx = max(0, curr_idx - scale)
                prev2_idx = max(0, curr_idx - 2 * scale)
                x_curr_s = x_enc[:, curr_idx, :]
                x_prev_s = x_enc[:, prev_idx, :]
                x_prev2_s = x_enc[:, prev2_idx, :]
                scale_tau_t.append(self.tau_module(x_curr_s, x_prev_s))
                scale_tau_prev.append(self.tau_module(x_prev_s, x_prev2_s))

            scale_logits = self.multi_tau_gate(torch.cat([context_embedding, x_t], dim=-1))
            scale_weights = torch.softmax(scale_logits, dim=-1).unsqueeze(-1)
            tau_t = torch.zeros(batch_size, self.dim, device=x_enc.device, dtype=x_enc.dtype)
            tau_prev = torch.zeros(batch_size, self.dim, device=x_enc.device, dtype=x_enc.dtype)
            for idx, tau_candidate in enumerate(scale_tau_t):
                tau_t = tau_t + scale_weights[:, idx, :] * tau_candidate
            for idx, tau_candidate in enumerate(scale_tau_prev):
                tau_prev = tau_prev + scale_weights[:, idx, :] * tau_candidate
        else:
            tau_t = self.tau_module(x_t, x_prev)
            tau_prev = self.tau_module(x_prev, x_prev2)

        if self.use_var_tau_gate:
            gate = self.var_tau_gate(torch.cat([x_t, x_prev], dim=-1))
            tau_scale = 1.0 + self.var_tau_gate_scale * (2.0 * gate - 1.0)
            tau_t = tau_t * tau_scale
            tau_prev = tau_prev * tau_scale

        rho_t = tau_t / (tau_prev + 1e-6)

        patch_embedding = self._encode_patch_context(x_enc)
        if self.use_patch_embed:
            context_embedding = self.patch_fusion(torch.cat([context_embedding, patch_embedding], dim=-1))

        u = torch.linspace(0, 1, self.horizon + 1, device=x_enc.device, dtype=x_enc.dtype)[1:]
        u = u.unsqueeze(0).unsqueeze(-1)

        context_rep = _repeat_steps(context_embedding, self.horizon)
        x_t_rep = _repeat_steps(x_t, self.horizon)
        delta_rep = _repeat_steps(recent_delta, self.horizon)
        tau_seed_rep = _repeat_steps(tau_t, self.horizon)
        rho_rep = _repeat_steps(rho_t, self.horizon)
        u_rep = u.expand(batch_size, -1, -1)
        horizon_rep = self.horizon_encoder(u_rep.reshape(batch_size * self.horizon, 1))
        horizon_rep = horizon_rep.reshape(batch_size, self.horizon, self.hidden_dim)

        tau_features = torch.cat([context_rep, x_t_rep, delta_rep, u_rep], dim=-1)
        tau_increments = self.tau_path_head(tau_features.reshape(batch_size * self.horizon, -1))
        tau_increments = tau_increments.reshape(batch_size, self.horizon, self.dim)
        tau_increments = torch.nn.functional.softplus(tau_increments) + 1e-3
        tau_path = tau_seed_rep + torch.cumsum(tau_increments, dim=1)

        features = torch.cat([context_rep, x_t_rep, delta_rep, tau_path, rho_rep, u_rep], dim=-1)
        deltas = self.prediction_head(features.reshape(batch_size * self.horizon, -1))
        deltas = deltas.reshape(batch_size, self.horizon, self.dim)

        if self.use_horizon_residual:
            residual_features = torch.cat([context_rep, x_t_rep, delta_rep, tau_path, rho_rep, horizon_rep, deltas], dim=-1)
            residual = self.horizon_residual_head(residual_features.reshape(batch_size * self.horizon, -1))
            residual = residual.reshape(batch_size, self.horizon, self.dim)
            deltas = deltas + self.horizon_residual_scale * residual

        pred = x_t.unsqueeze(1) + deltas

        # TSLib forecast pipeline expects output length = label_len + pred_len.
        # The model now generates the whole horizon directly from a tau trajectory,
        # without bootstrap rollouts.
        return pred
