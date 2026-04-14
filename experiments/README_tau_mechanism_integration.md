# Tau Mechanism Integration Guide

This note explains how to extract the tau mechanism from PPN and graft it onto other models.

## 1) What Was Extracted

File: models/tau_mechanism.py

Main classes:
- TauMechanismCore: clock -> lift -> evolve -> project
- TauForecastAdapter: attach tau mechanism to any backbone hidden history

## 2) Minimal Integration Pattern

Assume your backbone returns hidden history with shape [B, L, H].

```python
from models.tau_mechanism import TauMechanismConfig, TauForecastAdapter

class MyModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.backbone = ...
        tau_cfg = TauMechanismConfig(
            obs_dim=configs.c_out,
            hidden_dim=configs.d_model,
            horizon=configs.pred_len,
            dropout=getattr(configs, "dropout", 0.0),
            max_integration_steps=getattr(configs, "ppn_max_tau_integration_steps", 0),
            tau_stop_eps=getattr(configs, "ppn_tau_stop_eps", 1e-4),
            use_transformer_evolver=(int(getattr(configs, "ppn_use_transformer", 0)) == 1),
            use_residual_refine=(int(getattr(configs, "ppn_tau_use_residual_refine", 1)) == 1),
            tau_coarse_kernel=getattr(configs, "ppn_tau_coarse_kernel", 4),
            tau_coarse_stride=getattr(configs, "ppn_tau_coarse_stride", 4),
        )
        self.tau_adapter = TauForecastAdapter(
            backbone_hidden_dim=configs.d_model,
            output_dim=configs.c_out,
            tau_cfg=tau_cfg,
            fuse_alpha=0.0,
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        hidden_hist = self.backbone.encode(x_enc, x_mark_enc)
        tau_out = self.tau_adapter(hidden_hist)
        pred = tau_out["pred"]
        return pred
```

## 3) Output Contract

TauMechanismCore returns a dict:
- pred: [B, horizon, C]
- tau_hist: [B, L, 1]
- delta_tau_hist: [B, L, 1]
- delta_tau_future: [B, S, 1]
- z_hist: [B, L, C]
- z_future: [B, horizon, C]
- evidence: [B, L, H]

This contract makes it easy to add tau losses in training loops.

## 4) Training Loss Hook (Recommended)

Use your main prediction loss as base, then add optional tau terms from outputs:
- monotonic regularization from delta_tau_hist/delta_tau_future
- tau smoothness from delta differences
- reconstruction or projection-cycle on z_hist/z_future if needed

Keep lambda values small at first and tune after confirming baseline stability.

## 5) Why This Is Graftable

- The mechanism does not depend on PPN-specific class names.
- Input is only hidden history tensor [B, L, H] after a linear adapter.
- Compatible with Transformer, RNN, CNN, Mamba-like backbones.
