from __future__ import annotations

import torch
import torch.nn as nn


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


class StructuralEvidenceEncoder(nn.Module):
    """Build causal structural evidence stream from raw sequence dynamics."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        input_dim = 3 * dim + 2
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        x_prev = torch.cat([x[:, :1, :], x[:, :-1, :]], dim=1)
        dx = x - x_prev
        dx_prev = torch.cat([dx[:, :1, :], dx[:, :-1, :]], dim=1)
        d2x = dx - dx_prev

        # Two lightweight structural channels beyond derivatives.
        local_energy = (dx * dx).mean(dim=-1, keepdim=True)
        local_curvature = torch.abs(d2x).mean(dim=-1, keepdim=True)

        feat = torch.cat([x, dx, d2x, local_energy, local_curvature], dim=-1)
        return self.encoder(feat)


class Model(nn.Module):
    """Axiomatic tau-space PPN: structure -> clock -> tau-domain forecast -> projection."""

    def __init__(self, configs):
        super().__init__()
        self.dim = int(configs.enc_in)
        self.context_len = int(configs.seq_len)
        self.horizon = int(configs.pred_len)
        self.hidden_dim = int(getattr(configs, "d_model", 128))
        self.dropout = float(getattr(configs, "dropout", 0.0))

        # Axiomatic tau-space controls.
        self.use_tau_space_predictor = int(getattr(configs, "ppn_use_tau_space_predictor", 1)) == 1
        self.tau_global_scale = float(getattr(configs, "ppn_tau_global_scale", 1.0))
        self.tau_cross_adjust_scale = float(getattr(configs, "ppn_tau_cross_adjust_scale", 0.2))
        self.use_tau_cross_adjust_gate = int(getattr(configs, "ppn_use_tau_cross_adjust_gate", 1)) == 1
        self.use_bidirectional_tau_coupling = int(getattr(configs, "ppn_use_bidirectional_tau_coupling", 0)) == 1

        # Keep compatibility with existing residual branch experiments.
        self.use_horizon_residual = int(getattr(configs, "ppn_use_horizon_residual", 0)) == 1
        self.use_horizon_residual_gate = int(getattr(configs, "ppn_use_horizon_residual_gate", 0)) == 1
        self.horizon_residual_scale = float(getattr(configs, "ppn_horizon_residual_scale", 0.1))

        self.structure_encoder = StructuralEvidenceEncoder(self.dim, self.hidden_dim, self.dropout)
        self.context_encoder = MLPBlock(
            input_dim=self.context_len * self.dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            dropout=self.dropout,
        )

        # Causal monotonic clock (A1, A2, A7).
        self.clock_delta_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.softplus = nn.Softplus()

        # Lift / project heads (A3, A6).
        self.lift_head = MLPBlock(
            input_dim=self.dim + self.hidden_dim + 1,
            hidden_dim=self.hidden_dim,
            output_dim=self.dim,
            dropout=self.dropout,
        )
        self.reconstruct_head = MLPBlock(
            input_dim=self.dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.dim,
            dropout=self.dropout,
        )

        # Forecast in tau-space with explicit global branch and cross-adjustment.
        self.horizon_encoder = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.future_tau_inc_head = MLPBlock(
            input_dim=self.hidden_dim + self.hidden_dim + 1,
            hidden_dim=self.hidden_dim,
            output_dim=1,
            dropout=self.dropout,
        )
        self.global_mapping_head = MLPBlock(
            input_dim=self.hidden_dim + self.dim + self.hidden_dim + 1,
            hidden_dim=self.hidden_dim,
            output_dim=self.dim,
            dropout=self.dropout,
        )
        self.tau_space_head = MLPBlock(
            input_dim=self.hidden_dim + self.dim + 2,
            hidden_dim=self.hidden_dim,
            output_dim=self.dim,
            dropout=self.dropout,
        )
        self.cross_adjust_head = MLPBlock(
            input_dim=self.hidden_dim * 2 + 5 * self.dim + 2,
            hidden_dim=self.hidden_dim,
            output_dim=self.dim,
            dropout=self.dropout,
        )
        self.cross_adjust_gate = nn.Sequential(
            nn.Linear(self.hidden_dim + 2, self.dim),
            nn.Sigmoid(),
        )
        self.tau_from_global_gate = nn.Sequential(
            nn.Linear(self.hidden_dim + self.dim + 1, self.dim),
            nn.Sigmoid(),
        )
        self.global_from_tau_gate = nn.Sequential(
            nn.Linear(self.hidden_dim + self.dim + 1, self.dim),
            nn.Sigmoid(),
        )

        self.horizon_residual_head = MLPBlock(
            input_dim=self.hidden_dim * 2 + 5 * self.dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.dim,
            dropout=self.dropout,
        )
        self.horizon_residual_gate = nn.Sequential(
            nn.Linear(self.hidden_dim + self.dim + 1, self.dim),
            nn.Sigmoid(),
        )

        # Exposed auxiliary losses consumed by training loop.
        self.last_tau_smooth_loss = torch.tensor(0.0)
        self.last_tau_monotonic_loss = torch.tensor(0.0)
        self.last_tau_recon_loss = torch.tensor(0.0)
        self.last_tau_flatness_loss = torch.tensor(0.0)

    def _build_clock(self, evidence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # evidence: [B, L, H]
        raw_delta = self.clock_delta_head(evidence)
        delta_tau = self.softplus(raw_delta) + 1e-3
        tau = torch.cumsum(delta_tau, dim=1)
        return tau, delta_tau

    def _build_tau_losses(
        self,
        x_enc: torch.Tensor,
        z_enc: torch.Tensor,
        delta_tau_hist: torch.Tensor,
        delta_tau_future: torch.Tensor,
    ) -> None:
        device = x_enc.device
        dtype = x_enc.dtype

        # Monotonic loss: should stay near zero due to softplus, but still monitored.
        mono_hist = torch.relu(1e-4 - delta_tau_hist).mean()
        mono_future = torch.relu(1e-4 - delta_tau_future).mean() if delta_tau_future.numel() > 0 else torch.zeros((), device=device, dtype=dtype)
        self.last_tau_monotonic_loss = mono_hist + mono_future

        smooth_terms = []
        if delta_tau_hist.shape[1] > 1:
            smooth_terms.append(torch.mean(torch.abs(delta_tau_hist[:, 1:, :] - delta_tau_hist[:, :-1, :])))
        if delta_tau_future.shape[1] > 1:
            smooth_terms.append(torch.mean(torch.abs(delta_tau_future[:, 1:, :] - delta_tau_future[:, :-1, :])))
        if smooth_terms:
            self.last_tau_smooth_loss = sum(smooth_terms) / float(len(smooth_terms))
        else:
            self.last_tau_smooth_loss = torch.zeros((), device=device, dtype=dtype)

        # Reconstruction consistency in lifted space.
        x_rec = self.reconstruct_head(z_enc.reshape(-1, self.dim)).reshape_as(x_enc)
        self.last_tau_recon_loss = ((x_rec - x_enc) ** 2).mean()

        # Flatness ratio in tau-space dynamics (A4).
        if x_enc.shape[1] >= 3:
            d2x = x_enc[:, 2:, :] - 2 * x_enc[:, 1:-1, :] + x_enc[:, :-2, :]
            d2z = z_enc[:, 2:, :] - 2 * z_enc[:, 1:-1, :] + z_enc[:, :-2, :]
            num = torch.mean(torch.abs(d2z))
            den = torch.mean(torch.abs(d2x)) + 1e-6
            self.last_tau_flatness_loss = num / den
        else:
            self.last_tau_flatness_loss = torch.zeros((), device=device, dtype=dtype)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        del x_mark_enc, x_dec, x_mark_dec, mask

        batch_size, seq_len, dim = x_enc.shape
        if seq_len != self.context_len or dim != self.dim:
            raise ValueError(
                f"PPN input mismatch: expected [B,{self.context_len},{self.dim}], got [B,{seq_len},{dim}]"
            )

        context_embedding = self.context_encoder(x_enc.reshape(batch_size, -1))
        evidence = self.structure_encoder(x_enc)
        evidence_summary = evidence.mean(dim=1)

        tau_hist, delta_tau_hist = self._build_clock(evidence)
        tau_hist_norm = tau_hist / (tau_hist[:, -1:, :] + 1e-6)

        z_input = torch.cat([x_enc, evidence, tau_hist_norm], dim=-1)
        z_enc = self.lift_head(z_input.reshape(batch_size * seq_len, -1)).reshape(batch_size, seq_len, self.dim)

        x_t = x_enc[:, -1, :]
        x_prev = x_enc[:, -2, :] if seq_len >= 2 else x_t
        recent_delta = x_t - x_prev

        u = torch.linspace(0, 1, self.horizon + 1, device=x_enc.device, dtype=x_enc.dtype)[1:]
        u = u.unsqueeze(0).unsqueeze(-1)
        u_rep = u.expand(batch_size, -1, -1)

        context_rep = context_embedding.unsqueeze(1).expand(batch_size, self.horizon, self.hidden_dim)
        evidence_rep = evidence_summary.unsqueeze(1).expand(batch_size, self.horizon, self.hidden_dim)
        x_t_rep = x_t.unsqueeze(1).expand(batch_size, self.horizon, self.dim)
        delta_rep = recent_delta.unsqueeze(1).expand(batch_size, self.horizon, self.dim)

        horizon_rep = self.horizon_encoder(u_rep.reshape(batch_size * self.horizon, 1))
        horizon_rep = horizon_rep.reshape(batch_size, self.horizon, self.hidden_dim)

        # Future tau trajectory: non-bootstrap one-shot path.
        tau_seed = tau_hist[:, -1:, :].expand(batch_size, self.horizon, 1)
        future_tau_feat = torch.cat([context_rep, evidence_rep, u_rep], dim=-1)
        delta_tau_future = self.softplus(
            self.future_tau_inc_head(future_tau_feat.reshape(batch_size * self.horizon, -1)).reshape(batch_size, self.horizon, 1)
        ) + 1e-3
        tau_future = tau_seed + torch.cumsum(delta_tau_future, dim=1)
        tau_ratio = tau_future / (tau_seed + 1e-6)

        # Global mapping branch (required to prevent tau-only self-bootstrap).
        global_feat = torch.cat([context_rep, x_t_rep, horizon_rep, u_rep], dim=-1)
        global_delta = self.global_mapping_head(global_feat.reshape(batch_size * self.horizon, -1))
        global_delta = global_delta.reshape(batch_size, self.horizon, self.dim)
        global_pred = x_t_rep + self.tau_global_scale * global_delta

        # Tau-space branch.
        tau_feat = torch.cat([context_rep, x_t_rep, tau_ratio, u_rep], dim=-1)
        tau_delta = self.tau_space_head(tau_feat.reshape(batch_size * self.horizon, -1))
        tau_delta = tau_delta.reshape(batch_size, self.horizon, self.dim)

        # Bidirectional coupling: global dynamics constrain tau amplitude and vice versa.
        if self.use_bidirectional_tau_coupling:
            tau_gate_feat = torch.cat([horizon_rep, global_delta, u_rep], dim=-1)
            tau_gate = self.tau_from_global_gate(tau_gate_feat.reshape(batch_size * self.horizon, -1))
            tau_gate = tau_gate.reshape(batch_size, self.horizon, self.dim)
            tau_delta = tau_delta * tau_gate

            global_gate_feat = torch.cat([horizon_rep, tau_delta, u_rep], dim=-1)
            global_gate = self.global_from_tau_gate(global_gate_feat.reshape(batch_size * self.horizon, -1))
            global_gate = global_gate.reshape(batch_size, self.horizon, self.dim)
            global_pred = global_pred + self.tau_cross_adjust_scale * global_gate * tau_delta

        if self.use_tau_space_predictor:
            pred = global_pred + tau_delta

            cross_feat = torch.cat(
                [context_rep, horizon_rep, x_t_rep, delta_rep, global_delta, tau_delta, global_pred, tau_ratio, u_rep],
                dim=-1,
            )
            cross_delta = self.cross_adjust_head(cross_feat.reshape(batch_size * self.horizon, -1))
            cross_delta = cross_delta.reshape(batch_size, self.horizon, self.dim)

            if self.use_tau_cross_adjust_gate:
                gate_feat = torch.cat([horizon_rep, tau_ratio, u_rep], dim=-1)
                gate = self.cross_adjust_gate(gate_feat.reshape(batch_size * self.horizon, -1))
                gate = gate.reshape(batch_size, self.horizon, self.dim)
                cross_delta = cross_delta * gate

            pred = pred + self.tau_cross_adjust_scale * cross_delta
        else:
            pred = global_pred

        if self.use_horizon_residual:
            residual_features = torch.cat(
                [context_rep, x_t_rep, delta_rep, pred, global_delta, horizon_rep, tau_delta],
                dim=-1,
            )
            residual = self.horizon_residual_head(residual_features.reshape(batch_size * self.horizon, -1))
            residual = residual.reshape(batch_size, self.horizon, self.dim)

            if self.use_horizon_residual_gate:
                residual_gate_features = torch.cat([horizon_rep, tau_ratio, u_rep], dim=-1)
                residual_gate = self.horizon_residual_gate(residual_gate_features.reshape(batch_size * self.horizon, -1))
                residual_gate = residual_gate.reshape(batch_size, self.horizon, self.dim)
                residual = residual * residual_gate

            pred = pred + self.horizon_residual_scale * residual

        self._build_tau_losses(x_enc, z_enc, delta_tau_hist, delta_tau_future)
        return pred
