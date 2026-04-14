from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """Build causal structural evidence stream from sequence dynamics."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        feat_dim = 5 * input_dim + 2
        num_heads = max(1, min(4, hidden_dim // 32))
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        x_prev = torch.cat([x[:, :1, :], x[:, :-1, :]], dim=1)
        dx = x - x_prev
        dx_prev = torch.cat([dx[:, :1, :], dx[:, :-1, :]], dim=1)
        d2x = dx - dx_prev

        x_short = F.avg_pool1d(x.transpose(1, 2), kernel_size=3, stride=1, padding=1).transpose(1, 2)
        x_long = F.avg_pool1d(x.transpose(1, 2), kernel_size=7, stride=1, padding=3).transpose(1, 2)
        local_energy = (dx * dx).mean(dim=-1, keepdim=True)
        local_curvature = torch.abs(d2x).mean(dim=-1, keepdim=True)

        feat = torch.cat([x, dx, d2x, x_short, x_long, local_energy, local_curvature], dim=-1)
        hidden = self.encoder(feat)

        attn_mask = torch.triu(
            torch.ones(hidden.shape[1], hidden.shape[1], device=hidden.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_out, _ = self.attn(hidden, hidden, hidden, attn_mask=attn_mask, need_weights=False)
        hidden = self.norm1(hidden + attn_out)
        hidden = self.norm2(hidden + self.ffn(hidden))
        return hidden


@dataclass
class TauMechanismConfig:
    obs_dim: int
    hidden_dim: int
    horizon: int
    dropout: float = 0.0
    max_integration_steps: int = 0
    tau_stop_eps: float = 1e-4
    use_transformer_evolver: bool = False
    use_residual_refine: bool = True
    tau_coarse_kernel: int = 4
    tau_coarse_stride: int = 4
    use_multi_scale_transport: bool = False
    tau_transport_scales: tuple[int, ...] = (1, 3, 5)


class TauMechanismCore(nn.Module):
    """Plug-and-play tau mechanism: clock -> lift -> evolve -> project.

    This module is model-agnostic. Any backbone that can provide:
    1) observed sequence x_hist: [B, L, C]
    2) optional context vector ctx: [B, H]
    can attach this mechanism as a forecasting head.
    """

    def __init__(self, cfg: TauMechanismConfig):
        super().__init__()
        self.obs_dim = int(cfg.obs_dim)
        self.hidden_dim = int(cfg.hidden_dim)
        self.horizon = int(cfg.horizon)
        self.dropout = float(cfg.dropout)
        self.max_steps = int(cfg.max_integration_steps) if int(cfg.max_integration_steps) > 0 else max(self.horizon * 3, 16)
        self.tau_stop_eps = float(cfg.tau_stop_eps)
        self.use_transformer_evolver = bool(cfg.use_transformer_evolver)
        self.use_residual_refine = bool(cfg.use_residual_refine)
        self.tau_coarse_kernel = max(2, int(cfg.tau_coarse_kernel))
        self.tau_coarse_stride = max(1, int(cfg.tau_coarse_stride))
        self.use_multi_scale_transport = bool(cfg.use_multi_scale_transport)
        scales = [max(1, int(s)) for s in tuple(cfg.tau_transport_scales)]
        self.tau_transport_scales = tuple(s if s % 2 == 1 else s + 1 for s in scales)

        self.softplus = nn.Softplus()
        self.evidence_encoder = StructuralEvidenceEncoder(self.obs_dim, self.hidden_dim, self.dropout)
        self.context_encoder = MLPBlock(
            input_dim=self.obs_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            dropout=self.dropout,
        )

        self.clock_delta_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.clock_coarse_delta_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.clock_fusion_gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.lift_head = MLPBlock(
            input_dim=self.obs_dim + self.hidden_dim + 1,
            hidden_dim=self.hidden_dim,
            output_dim=self.obs_dim,
            dropout=self.dropout,
        )
        self.project_head = MLPBlock(
            input_dim=self.obs_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.obs_dim,
            dropout=self.dropout,
        )

        self.tau_target_head = MLPBlock(
            input_dim=self.hidden_dim + self.hidden_dim + self.obs_dim,
            hidden_dim=self.hidden_dim,
            output_dim=1,
            dropout=self.dropout,
        )
        self.future_tau_inc_head = MLPBlock(
            input_dim=self.hidden_dim + self.hidden_dim + self.obs_dim + 1,
            hidden_dim=self.hidden_dim,
            output_dim=1,
            dropout=self.dropout,
        )
        self.channel_tau_scale_head = MLPBlock(
            input_dim=self.hidden_dim + self.hidden_dim + self.obs_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.obs_dim,
            dropout=self.dropout,
        )
        self.tau_state_init = MLPBlock(
            input_dim=self.obs_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            dropout=self.dropout,
        )
        self.tau_state_decode = MLPBlock(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.obs_dim,
            dropout=self.dropout,
        )
        self.tau_dynamics_head = MLPBlock(
            input_dim=self.hidden_dim + self.obs_dim + 2,
            hidden_dim=self.hidden_dim,
            output_dim=self.obs_dim,
            dropout=self.dropout,
        )
        if self.use_multi_scale_transport:
            self.transport_gate = MLPBlock(
                input_dim=self.hidden_dim + self.hidden_dim + self.obs_dim,
                hidden_dim=self.hidden_dim,
                output_dim=len(self.tau_transport_scales),
                dropout=self.dropout,
            )

        if self.use_transformer_evolver:
            self.step_proj = nn.Linear(self.hidden_dim + self.hidden_dim + 2, self.hidden_dim)
            self.step_attn = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=max(1, min(4, self.hidden_dim // 32)),
                dim_feedforward=self.hidden_dim * 2,
                dropout=self.dropout,
                batch_first=True,
                activation="gelu",
            )
            self.step_norm = nn.LayerNorm(self.hidden_dim)
        else:
            self.step_cell = nn.GRUCell(self.hidden_dim + self.hidden_dim + 2, self.hidden_dim)

    def _build_clock(self, evidence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fine_delta_raw = self.clock_delta_head(evidence)
        fine_delta = self.softplus(fine_delta_raw) + 1e-3

        coarse = F.avg_pool1d(
            evidence.transpose(1, 2),
            kernel_size=self.tau_coarse_kernel,
            stride=self.tau_coarse_stride,
            ceil_mode=True,
        ).transpose(1, 2)
        coarse_delta_raw_small = self.clock_coarse_delta_head(coarse)
        coarse_delta_small = self.softplus(coarse_delta_raw_small) + 1e-3
        coarse_delta_raw = F.interpolate(
            coarse_delta_raw_small.transpose(1, 2),
            size=evidence.shape[1],
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        coarse_delta = F.interpolate(
            coarse_delta_small.transpose(1, 2),
            size=evidence.shape[1],
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

        gate_context = torch.cat([evidence.mean(dim=1), coarse.mean(dim=1)], dim=-1)
        coarse_gate = self.clock_fusion_gate(gate_context).unsqueeze(1)
        delta_tau = fine_delta + coarse_gate * coarse_delta
        delta_tau_raw = fine_delta_raw + coarse_gate * coarse_delta_raw
        tau = torch.cumsum(delta_tau, dim=1)
        return tau, delta_tau, delta_tau_raw

    @staticmethod
    def _interpolate_tau_path(tau_nodes: torch.Tensor, z_nodes: torch.Tensor, tau_query: torch.Tensor) -> torch.Tensor:
        dist = torch.abs(tau_nodes.unsqueeze(1).unsqueeze(1) - tau_query.unsqueeze(-1))
        weights = torch.softmax(-dist / 0.15, dim=-1)
        return torch.einsum("b h d s, b s d -> b h d", weights, z_nodes)

    def _multi_scale_transport(
        self,
        tau_nodes: torch.Tensor,
        z_nodes: torch.Tensor,
        tau_query: torch.Tensor,
        context_state: torch.Tensor,
        evidence_summary: torch.Tensor,
        z_last: torch.Tensor,
    ) -> torch.Tensor:
        base_path = self._interpolate_tau_path(tau_nodes, z_nodes, tau_query)
        if not self.use_multi_scale_transport or len(self.tau_transport_scales) <= 1:
            return base_path

        path_list = [base_path]
        for scale in self.tau_transport_scales[1:]:
            smooth_nodes = F.avg_pool1d(
                z_nodes.transpose(1, 2),
                kernel_size=scale,
                stride=1,
                padding=scale // 2,
            ).transpose(1, 2)
            path_list.append(self._interpolate_tau_path(tau_nodes, smooth_nodes, tau_query))

        gate_feat = torch.cat([context_state, evidence_summary, z_last], dim=-1)
        gate = torch.softmax(self.transport_gate(gate_feat), dim=-1).unsqueeze(1).unsqueeze(1)
        stacked = torch.stack(path_list, dim=-1)
        return torch.sum(stacked * gate, dim=-1)

    def forward(self, x_hist: torch.Tensor, context_state: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        # x_hist: [B, L, C]
        batch_size = x_hist.shape[0]

        evidence = self.evidence_encoder(x_hist)
        tau_hist, delta_tau_hist, delta_tau_hist_raw = self._build_clock(evidence)
        tau_hist_norm = tau_hist / (tau_hist[:, -1:, :] + 1e-6)

        z_input = torch.cat([x_hist, evidence, tau_hist_norm], dim=-1)
        z_hist = self.lift_head(z_input.reshape(batch_size * x_hist.shape[1], -1)).reshape(batch_size, x_hist.shape[1], self.obs_dim)

        z_last = z_hist[:, -1, :]
        tau_last = tau_hist[:, -1, 0:1]
        tau_scale = delta_tau_hist.mean(dim=1).clamp_min(1e-3)

        if context_state is None:
            context_state = self.context_encoder(x_hist[:, -1, :])
        assert context_state is not None

        evidence_summary = evidence.mean(dim=1)
        target_feat = torch.cat([context_state, evidence_summary, z_last], dim=-1)
        target_ratio = self.softplus(self.tau_target_head(target_feat)) + 1e-3
        channel_tau_scale = self.softplus(self.channel_tau_scale_head(target_feat)) + 1e-3
        tau_target = tau_last + target_ratio * tau_scale * float(self.horizon) * channel_tau_scale

        tau_nodes = []
        z_nodes = []
        delta_tau_future_list = []
        delta_tau_future_raw_list = []
        step_tokens = []

        tau_curr = tau_last
        tau_hidden = self.tau_state_init(z_last)
        z_state = z_last

        for _ in range(self.max_steps):
            active = (tau_curr + self.tau_stop_eps < tau_target.max(dim=-1, keepdim=True).values).to(x_hist.dtype)
            if not bool(active.any()):
                break

            tau_curr_norm = tau_curr / (tau_last + 1e-6)
            delta_feat = torch.cat([context_state, evidence_summary, z_state, tau_curr_norm], dim=-1)
            delta_tau_raw = self.future_tau_inc_head(delta_feat)
            delta_tau = self.softplus(delta_tau_raw) + 1e-3
            delta_tau = delta_tau * active
            delta_tau_raw = delta_tau_raw * active

            tau_next = active * (tau_curr + delta_tau) + (1.0 - active) * tau_curr
            delta_tau_norm = delta_tau / (tau_scale + 1e-6)
            tau_next_norm = tau_next / (tau_last + 1e-6)
            step_feat = torch.cat([context_state, evidence_summary, delta_tau_norm, tau_next_norm], dim=-1)

            if self.use_transformer_evolver:
                step_token = self.step_proj(step_feat)
                step_tokens.append(step_token)
                step_seq = torch.stack(step_tokens, dim=1)
                step_len = step_seq.shape[1]
                causal_mask = torch.triu(
                    torch.ones(step_len, step_len, device=x_hist.device, dtype=torch.bool),
                    diagonal=1,
                )
                step_context = self.step_attn(step_seq, src_mask=causal_mask)
                tau_hidden = self.step_norm(step_context[:, -1, :] + tau_hidden)
            else:
                tau_hidden = self.step_cell(step_feat, tau_hidden)

            z_roll = self.tau_state_decode(tau_hidden)
            if self.use_residual_refine:
                z_dyn_feat = torch.cat([context_state, z_roll, delta_tau, tau_next], dim=-1)
                dz = self.tau_dynamics_head(z_dyn_feat)
                z_next = z_roll + delta_tau * dz * channel_tau_scale
            else:
                z_next = z_roll

            z_state = active * z_next + (1.0 - active) * z_state
            tau_curr = tau_next

            tau_nodes.append(tau_curr)
            z_nodes.append(z_state)
            delta_tau_future_list.append(delta_tau)
            delta_tau_future_raw_list.append(delta_tau_raw)

        if tau_nodes:
            tau_nodes = torch.stack(tau_nodes, dim=1).squeeze(-1)
            z_nodes = torch.stack(z_nodes, dim=1)
            delta_tau_future = torch.stack(delta_tau_future_list, dim=1)
            delta_tau_future_raw = torch.stack(delta_tau_future_raw_list, dim=1)
        else:
            tau_nodes = tau_last.expand(batch_size, 1)
            z_nodes = z_last.unsqueeze(1)
            delta_tau_future = torch.zeros(batch_size, 1, 1, device=x_hist.device, dtype=x_hist.dtype)
            delta_tau_future_raw = torch.zeros(batch_size, 1, 1, device=x_hist.device, dtype=x_hist.dtype)

        tau_query_ratio = torch.linspace(0, 1, self.horizon + 1, device=x_hist.device, dtype=x_hist.dtype)[1:]
        tau_query_ratio = tau_query_ratio.view(1, self.horizon, 1)
        tau_query = tau_last.unsqueeze(1) + tau_query_ratio * (tau_target.unsqueeze(1) - tau_last.unsqueeze(1))
        z_future = self._multi_scale_transport(tau_nodes, z_nodes, tau_query, context_state, evidence_summary, z_last)
        x_future = self.project_head(z_future.reshape(batch_size * self.horizon, self.obs_dim)).reshape(batch_size, self.horizon, self.obs_dim)

        return {
            "pred": x_future,
            "tau_hist": tau_hist,
            "delta_tau_hist": delta_tau_hist,
            "delta_tau_hist_raw": delta_tau_hist_raw,
            "delta_tau_future": delta_tau_future,
            "delta_tau_future_raw": delta_tau_future_raw,
            "z_hist": z_hist,
            "z_future": z_future,
            "evidence": evidence,
        }


class TauForecastAdapter(nn.Module):
    """Adapter to graft TauMechanismCore onto arbitrary backbones.

    Expected backbone output: history hidden states [B, L, H_backbone].
    This adapter projects hidden states to pseudo-observation space, applies
    tau mechanism, and optionally fuses backbone direct head with tau head.
    """

    def __init__(self, backbone_hidden_dim: int, output_dim: int, tau_cfg: TauMechanismConfig, fuse_alpha: float = 0.0):
        super().__init__()
        self.hidden_to_obs = nn.Linear(backbone_hidden_dim, output_dim)
        self.tau_core = TauMechanismCore(tau_cfg)
        self.fuse_alpha = float(fuse_alpha)

    def forward(self, hidden_hist: torch.Tensor, direct_pred: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        x_hist = self.hidden_to_obs(hidden_hist)
        tau_out = self.tau_core(x_hist)

        if direct_pred is not None and self.fuse_alpha > 0:
            tau_out["pred"] = (1.0 - self.fuse_alpha) * tau_out["pred"] + self.fuse_alpha * direct_pred
        return tau_out
