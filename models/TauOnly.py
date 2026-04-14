from __future__ import annotations

import torch
import torch.nn as nn

from .tau_mechanism import MLPBlock, TauMechanismConfig, TauMechanismCore


class Model(nn.Module):
    """Tau-is-all-you-need baseline.

    Forecasting relies only on learned tau coordinates and tau-space dynamics.
    No external backbone is used.
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.enc_in = int(configs.enc_in)
        self.hidden_dim = int(getattr(configs, "d_model", 128))
        self.dropout = float(getattr(configs, "dropout", 0.0))
        self.anchor_base_scale = float(getattr(configs, "tauonly_anchor_scale", 1.0))
        self.anchor_active_scale = float(self.anchor_base_scale)

        self.context_encoder = MLPBlock(
            input_dim=self.seq_len * self.enc_in,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            dropout=self.dropout,
        )
        self.anchor_gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.enc_in),
            nn.Sigmoid(),
        )

        self.reconstruct_head = MLPBlock(
            input_dim=self.enc_in,
            hidden_dim=self.hidden_dim,
            output_dim=self.enc_in,
            dropout=self.dropout,
        )
        self.project_head = MLPBlock(
            input_dim=self.enc_in,
            hidden_dim=self.hidden_dim,
            output_dim=self.enc_in,
            dropout=self.dropout,
        )
        self.obs_to_latent_head = MLPBlock(
            input_dim=self.enc_in,
            hidden_dim=self.hidden_dim,
            output_dim=self.enc_in,
            dropout=self.dropout,
        )
        self.tau_contrast_proj = MLPBlock(
            input_dim=self.enc_in,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            dropout=self.dropout,
        )
        self.evidence_contrast_proj = MLPBlock(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            dropout=self.dropout,
        )

        tau_cfg = TauMechanismConfig(
            obs_dim=self.enc_in,
            hidden_dim=self.hidden_dim,
            horizon=self.pred_len,
            dropout=self.dropout,
            max_integration_steps=int(getattr(configs, "ppn_max_tau_integration_steps", 0)),
            tau_stop_eps=float(getattr(configs, "ppn_tau_stop_eps", 1e-4)),
            use_transformer_evolver=int(getattr(configs, "ppn_use_transformer", 0)) == 1,
            use_residual_refine=int(getattr(configs, "ppn_tau_use_residual_refine", 1)) == 1,
            tau_coarse_kernel=max(2, int(getattr(configs, "ppn_tau_coarse_kernel", 4))),
            tau_coarse_stride=max(1, int(getattr(configs, "ppn_tau_coarse_stride", 4))),
            use_multi_scale_transport=bool(getattr(configs, "tau_use_multi_scale_transport", 1)),
            tau_transport_scales=tuple(
                int(s) for s in str(getattr(configs, "tau_transport_scales", "1,3,5")).split(",") if str(s).strip()
            ),
        )
        self.tau_core = TauMechanismCore(tau_cfg)
        self.tau_module = self.tau_core

        self.last_tau_smooth_loss = torch.tensor(0.0)
        self.last_tau_traj_smooth_loss = torch.tensor(0.0)
        self.last_tau_proj_cycle_loss = torch.tensor(0.0)
        self.last_tau_contrast_loss = torch.tensor(0.0)
        self.last_tau_monotonic_loss = torch.tensor(0.0)
        self.last_tau_recon_loss = torch.tensor(0.0)
        self.last_tau_flatness_loss = torch.tensor(0.0)

    @staticmethod
    def _info_nce_loss(anchor: torch.Tensor, positive: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
        anchor = torch.nn.functional.normalize(anchor, dim=-1)
        positive = torch.nn.functional.normalize(positive, dim=-1)
        logits = anchor @ positive.t() / temperature
        labels = torch.arange(anchor.shape[0], device=anchor.device)
        loss_a = torch.nn.functional.cross_entropy(logits, labels)
        loss_b = torch.nn.functional.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_a + loss_b)

    def _build_tau_losses(
        self,
        x_enc: torch.Tensor,
        z_enc: torch.Tensor,
        delta_tau_hist: torch.Tensor,
        delta_tau_future: torch.Tensor,
        delta_tau_hist_raw: torch.Tensor | None = None,
        delta_tau_future_raw: torch.Tensor | None = None,
        z_future: torch.Tensor | None = None,
        x_future: torch.Tensor | None = None,
        evidence_summary: torch.Tensor | None = None,
    ) -> None:
        device = x_enc.device
        dtype = x_enc.dtype

        if delta_tau_hist_raw is not None:
            mono_hist = torch.nn.functional.softplus(-delta_tau_hist_raw).mean()
        else:
            mono_hist = torch.relu(1e-4 - delta_tau_hist).mean()

        if delta_tau_future_raw is not None and delta_tau_future_raw.numel() > 0:
            mono_future = torch.nn.functional.softplus(-delta_tau_future_raw).mean()
        elif delta_tau_future.numel() > 0:
            mono_future = torch.relu(1e-4 - delta_tau_future).mean()
        else:
            mono_future = torch.zeros((), device=device, dtype=dtype)
        self.last_tau_monotonic_loss = mono_hist + mono_future

        smooth_terms = []
        if delta_tau_hist.shape[1] > 1:
            smooth_terms.append(torch.mean(torch.abs(delta_tau_hist[:, 1:, :] - delta_tau_hist[:, :-1, :])))
        if delta_tau_future.shape[1] > 1:
            smooth_terms.append(torch.mean(torch.abs(delta_tau_future[:, 1:, :] - delta_tau_future[:, :-1, :])))
        self.last_tau_smooth_loss = sum(smooth_terms) / float(len(smooth_terms)) if smooth_terms else torch.zeros((), device=device, dtype=dtype)

        if z_future is not None and z_future.shape[1] > 1:
            self.last_tau_traj_smooth_loss = torch.mean(torch.abs(z_future[:, 1:, :] - z_future[:, :-1, :]))
        else:
            self.last_tau_traj_smooth_loss = torch.zeros((), device=device, dtype=dtype)

        if z_future is not None and x_future is not None:
            z_from_x = self.obs_to_latent_head(x_future.reshape(-1, self.enc_in)).reshape_as(z_future)
            x_cycle = self.project_head(z_from_x.reshape(-1, self.enc_in)).reshape_as(x_future)
            z_cycle = self.obs_to_latent_head(self.project_head(z_future.reshape(-1, self.enc_in))).reshape_as(z_future)
            self.last_tau_proj_cycle_loss = ((z_from_x - z_future) ** 2).mean() + ((x_cycle - x_future) ** 2).mean() + ((z_cycle - z_future) ** 2).mean()
        else:
            self.last_tau_proj_cycle_loss = torch.zeros((), device=device, dtype=dtype)

        if z_future is not None and evidence_summary is not None:
            tau_summary = self.tau_contrast_proj(z_future.mean(dim=1))
            evidence_summary_proj = self.evidence_contrast_proj(evidence_summary)
            self.last_tau_contrast_loss = self._info_nce_loss(tau_summary, evidence_summary_proj)
        else:
            self.last_tau_contrast_loss = torch.zeros((), device=device, dtype=dtype)

        x_rec = self.reconstruct_head(z_enc.reshape(-1, self.enc_in)).reshape_as(x_enc)
        self.last_tau_recon_loss = ((x_rec - x_enc) ** 2).mean()

        if x_enc.shape[1] >= 3:
            d2x = x_enc[:, 2:, :] - 2 * x_enc[:, 1:-1, :] + x_enc[:, :-2, :]
            d2z = z_enc[:, 2:, :] - 2 * z_enc[:, 1:-1, :] + z_enc[:, :-2, :]
            self.last_tau_flatness_loss = torch.mean(torch.abs(d2z)) / (torch.mean(torch.abs(d2x)) + 1e-6)
        else:
            self.last_tau_flatness_loss = torch.zeros((), device=device, dtype=dtype)

    def _compute_simple_tau(self, x_t: torch.Tensor, x_t1: torch.Tensor, x_t2: torch.Tensor | None = None) -> torch.Tensor:
        v_t = x_t - x_t1
        if x_t2 is not None:
            v_t1 = x_t1 - x_t2
            v_t_norm = torch.norm(v_t, dim=-1, keepdim=True).clamp_min(1e-6)
            v_t1_norm = torch.norm(v_t1, dim=-1, keepdim=True).clamp_min(1e-6)
            return v_t_norm / (v_t1_norm + 1e-6)
        return torch.norm(v_t, dim=-1, keepdim=True).clamp_min(1e-3)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        del x_mark_enc, x_dec, x_mark_dec, mask

        if self.task_name not in {"long_term_forecast", "short_term_forecast"}:
            raise ValueError("TauOnly currently supports forecasting tasks only")

        # RevIN-style normalization improves tau-only optimization stability.
        mean = x_enc.mean(dim=1, keepdim=True)
        std = torch.sqrt(torch.var(x_enc - mean, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = (x_enc - mean) / std

        context = self.context_encoder(x_norm.reshape(x_norm.shape[0], -1))
        out = self.tau_core(x_norm, context_state=context)
        pred_norm = out["pred"]

        evidence_summary = out["evidence"].mean(dim=1)
        self._build_tau_losses(
            x_enc=x_norm,
            z_enc=out["z_hist"],
            delta_tau_hist=out["delta_tau_hist"],
            delta_tau_future=out["delta_tau_future"],
            delta_tau_hist_raw=out.get("delta_tau_hist_raw"),
            delta_tau_future_raw=out.get("delta_tau_future_raw"),
            z_future=out["z_future"],
            x_future=pred_norm,
            evidence_summary=evidence_summary,
        )

        anchor = x_norm[:, -1:, :].repeat(1, self.pred_len, 1)
        gate = self.anchor_gate(context).unsqueeze(1)
        pred_norm = pred_norm + self.anchor_active_scale * gate * anchor

        return pred_norm * std + mean
