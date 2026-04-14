from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tau_mechanism import MLPBlock, TauMechanismConfig, TauMechanismCore


class Model(nn.Module):
    """PPN wrapper over reusable tau mechanism core.

    This class keeps the original training-loop contract (forward shape and
    exposed last_tau_* losses) while delegating tau forecasting dynamics to
    TauMechanismCore for easier cross-model grafting.
    """

    def __init__(self, configs):
        super().__init__()
        self.dim = int(configs.enc_in)
        self.context_len = int(configs.seq_len)
        self.horizon = int(configs.pred_len)
        self.hidden_dim = int(getattr(configs, "d_model", 128))
        self.dropout = float(getattr(configs, "dropout", 0.0))

        tau_cfg = TauMechanismConfig(
            obs_dim=self.dim,
            hidden_dim=self.hidden_dim,
            horizon=self.horizon,
            dropout=self.dropout,
            max_integration_steps=int(getattr(configs, "ppn_max_tau_integration_steps", 0)),
            tau_stop_eps=float(getattr(configs, "ppn_tau_stop_eps", 1e-4)),
            use_transformer_evolver=int(getattr(configs, "ppn_use_transformer", 0)) == 1,
            use_residual_refine=int(getattr(configs, "ppn_tau_use_residual_refine", 1)) == 1,
            tau_coarse_kernel=max(2, int(getattr(configs, "ppn_tau_coarse_kernel", 4))),
            tau_coarse_stride=max(1, int(getattr(configs, "ppn_tau_coarse_stride", 4))),
        )
        self.tau_core = TauMechanismCore(tau_cfg)

        # Keep the original full-context embedding behavior from PPN.
        self.context_encoder = MLPBlock(
            input_dim=self.context_len * self.dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            dropout=self.dropout,
        )

        # Auxiliary heads retained for the original training loop losses.
        self.reconstruct_head = MLPBlock(
            input_dim=self.dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.dim,
            dropout=self.dropout,
        )
        self.project_head = MLPBlock(
            input_dim=self.dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.dim,
            dropout=self.dropout,
        )
        self.obs_to_latent_head = MLPBlock(
            input_dim=self.dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.dim,
            dropout=self.dropout,
        )
        self.tau_contrast_proj = MLPBlock(
            input_dim=self.dim,
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

        # Exposed auxiliary losses consumed by training loop.
        self.last_tau_smooth_loss = torch.tensor(0.0)
        self.last_tau_traj_smooth_loss = torch.tensor(0.0)
        self.last_tau_proj_cycle_loss = torch.tensor(0.0)
        self.last_tau_contrast_loss = torch.tensor(0.0)
        self.last_tau_monotonic_loss = torch.tensor(0.0)
        self.last_tau_recon_loss = torch.tensor(0.0)
        self.last_tau_flatness_loss = torch.tensor(0.0)

    @staticmethod
    def _info_nce_loss(anchor: torch.Tensor, positive: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        logits = anchor @ positive.t() / temperature
        labels = torch.arange(anchor.shape[0], device=anchor.device)
        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_a + loss_b)

    def _build_tau_losses(
        self,
        x_enc: torch.Tensor,
        z_enc: torch.Tensor,
        delta_tau_hist: torch.Tensor,
        delta_tau_future: torch.Tensor,
        z_future: torch.Tensor | None = None,
        x_future: torch.Tensor | None = None,
        evidence_summary: torch.Tensor | None = None,
    ) -> None:
        device = x_enc.device
        dtype = x_enc.dtype

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

        if z_future is not None and z_future.shape[1] > 1:
            self.last_tau_traj_smooth_loss = torch.mean(torch.abs(z_future[:, 1:, :] - z_future[:, :-1, :]))
        else:
            self.last_tau_traj_smooth_loss = torch.zeros((), device=device, dtype=dtype)

        if z_future is not None and x_future is not None:
            z_from_x = self.obs_to_latent_head(x_future.reshape(-1, self.dim)).reshape_as(z_future)
            x_cycle = self.project_head(z_from_x.reshape(-1, self.dim)).reshape_as(x_future)
            z_cycle = self.obs_to_latent_head(self.project_head(z_future.reshape(-1, self.dim))).reshape_as(z_future)
            self.last_tau_proj_cycle_loss = ((z_from_x - z_future) ** 2).mean() + ((x_cycle - x_future) ** 2).mean() + ((z_cycle - z_future) ** 2).mean()
        else:
            self.last_tau_proj_cycle_loss = torch.zeros((), device=device, dtype=dtype)

        if z_future is not None and evidence_summary is not None:
            tau_summary = self.tau_contrast_proj(z_future.mean(dim=1))
            evidence_summary_proj = self.evidence_contrast_proj(evidence_summary)
            self.last_tau_contrast_loss = self._info_nce_loss(tau_summary, evidence_summary_proj)
        else:
            self.last_tau_contrast_loss = torch.zeros((), device=device, dtype=dtype)

        x_rec = self.reconstruct_head(z_enc.reshape(-1, self.dim)).reshape_as(x_enc)
        self.last_tau_recon_loss = ((x_rec - x_enc) ** 2).mean()

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
        tau_out = self.tau_core(x_enc, context_state=context_embedding)
        pred = tau_out["pred"]

        evidence_summary = tau_out["evidence"].mean(dim=1)
        self._build_tau_losses(
            x_enc=x_enc,
            z_enc=tau_out["z_hist"],
            delta_tau_hist=tau_out["delta_tau_hist"],
            delta_tau_future=tau_out["delta_tau_future"],
            z_future=tau_out["z_future"],
            x_future=pred,
            evidence_summary=evidence_summary,
        )
        return pred
