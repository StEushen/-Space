from __future__ import annotations

import importlib

import torch
import torch.nn as nn

from .tau_mechanism import TauMechanismConfig, TauMechanismCore


class Model(nn.Module):
    """Generic SOTA + tau-space fusion model.

    Set `--tau_base_model` to any model filename under models/ (without .py),
    e.g. DLinear, PatchTST, Transformer.
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = int(configs.pred_len)
        self.enc_in = int(configs.enc_in)

        self.base_model_name = str(getattr(configs, "tau_base_model", "DLinear"))
        if self.base_model_name.lower() in {"taufusion", "ppn"}:
            raise ValueError("tau_base_model cannot be TauFusion or PPN")

        module = importlib.import_module(f"models.{self.base_model_name}")
        if not hasattr(module, "Model"):
            raise ValueError(f"models.{self.base_model_name} has no Model class")
        self.base_model = module.Model(configs)

        tau_cfg = TauMechanismConfig(
            obs_dim=self.enc_in,
            hidden_dim=int(getattr(configs, "d_model", 128)),
            horizon=self.pred_len,
            dropout=float(getattr(configs, "dropout", 0.0)),
            max_integration_steps=int(getattr(configs, "ppn_max_tau_integration_steps", 0)),
            tau_stop_eps=float(getattr(configs, "ppn_tau_stop_eps", 1e-4)),
            use_transformer_evolver=int(getattr(configs, "ppn_use_transformer", 0)) == 1,
            use_residual_refine=int(getattr(configs, "ppn_tau_use_residual_refine", 1)) == 1,
            tau_coarse_kernel=max(2, int(getattr(configs, "ppn_tau_coarse_kernel", 4))),
            tau_coarse_stride=max(1, int(getattr(configs, "ppn_tau_coarse_stride", 4))),
        )
        self.tau_core = TauMechanismCore(tau_cfg)

        # Learnable fusion weight with safe bounds.
        init_alpha = float(getattr(configs, "tau_fuse_alpha", 0.3))
        init_alpha = min(0.95, max(0.05, init_alpha))
        self.fuse_logit = nn.Parameter(torch.logit(torch.tensor(init_alpha, dtype=torch.float32)))

    def _fuse_alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.fuse_logit)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name not in {"long_term_forecast", "short_term_forecast"}:
            return self.base_model(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)

        base_pred = self.base_model(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        tau_out = self.tau_core(x_enc)
        tau_pred = tau_out["pred"]

        alpha = self._fuse_alpha().to(base_pred.dtype)
        fused = (1.0 - alpha) * base_pred + alpha * tau_pred
        return fused
