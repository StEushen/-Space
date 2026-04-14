from __future__ import annotations

import os
import torch
import torch.nn as nn

from .TauOnly import Model as TauOnlyModel


class Model(nn.Module):
    """Fixed tau-clock + weak GRU predictor.

    Workflow:
    1) Use TauOnly or PatchTST as a tau-coordinate teacher (optionally frozen).
    2) Extract tau-space latent trajectory z_hist.
    3) Train a lightweight GRU decoder in tau space for forecasting.
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = int(configs.pred_len)
        self.enc_in = int(configs.enc_in)

        self.teacher_model_name = str(getattr(configs, "tau_teacher_model", "TauOnly")).strip() or "TauOnly"
        teacher_kind = self.teacher_model_name.lower()
        if teacher_kind in {"tauonly", "tauclock", "tauclockfixedgru"}:
            self.teacher = TauOnlyModel(configs)
            self.teacher_feature_dim = self.enc_in
            self.teacher_feature_kind = "tauonly"
        elif teacher_kind == "patchtst":
            from .PatchTST import Model as PatchTSTModel

            self.teacher = PatchTSTModel(configs)
            self.teacher_feature_dim = int(getattr(configs, "d_model", 128))
            self.teacher_feature_kind = "patchtst"
        else:
            raise ValueError(f"Unsupported tau_teacher_model: {self.teacher_model_name}")

        self.teacher_trainable = int(getattr(configs, "tau_teacher_trainable", 0)) == 1

        teacher_ckpt = str(getattr(configs, "tau_teacher_checkpoint", "")).strip()
        if teacher_ckpt:
            if os.path.isdir(teacher_ckpt):
                teacher_ckpt = os.path.join(teacher_ckpt, "checkpoint.pth")
            if not os.path.exists(teacher_ckpt):
                raise FileNotFoundError(f"tau teacher checkpoint not found: {teacher_ckpt}")
            state_dict = self._normalize_state_dict(torch.load(teacher_ckpt, map_location="cpu"))
            strict = bool(getattr(configs, "tau_teacher_strict", 1))
            msg = self.teacher.load_state_dict(state_dict, strict=strict)
            print(f"Loaded tau teacher checkpoint: {teacher_ckpt}")
            if not strict:
                print(f"tau teacher missing keys: {len(msg.missing_keys)}, unexpected keys: {len(msg.unexpected_keys)}")

        if not self.teacher_trainable:
            for p in self.teacher.parameters():
                p.requires_grad = False
            self.teacher.eval()

        hidden_dim = int(getattr(configs, "tau_student_hidden_dim", 128))
        n_layers = max(1, int(getattr(configs, "tau_student_num_layers", 1)))
        self.teacher_feature_norm = nn.LayerNorm(self.teacher_feature_dim)
        self.init_proj = nn.Linear(self.teacher_feature_dim, hidden_dim)
        self.decoder_cell = nn.GRUCell(self.teacher_feature_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, self.teacher_feature_dim)
        self.output_proj = nn.Identity() if self.teacher_feature_dim == self.enc_in else nn.Linear(self.teacher_feature_dim, self.enc_in)
        self.dropout = nn.Dropout(float(getattr(configs, "dropout", 0.0)))

    @staticmethod
    def _normalize_state_dict(raw_state):
        if isinstance(raw_state, dict):
            for key in ("state_dict", "model_state_dict", "model", "net"):
                nested = raw_state.get(key)
                if isinstance(nested, dict):
                    raw_state = nested
                    break
        if isinstance(raw_state, dict) and raw_state:
            if all(str(key).startswith("module.") for key in raw_state.keys()):
                return {str(key)[7:]: value for key, value in raw_state.items()}
        return raw_state

    def _teacher_tauonly_features(self, x_norm: torch.Tensor) -> torch.Tensor:
        context_encoder = getattr(self.teacher, "context_encoder")
        tau_core = getattr(self.teacher, "tau_core")
        context = context_encoder(x_norm.reshape(x_norm.shape[0], -1))
        out = tau_core(x_norm, context_state=context)
        return out["z_hist"]

    def _teacher_patchtst_features(self, x_norm: torch.Tensor) -> torch.Tensor:
        x_patch = x_norm.permute(0, 2, 1)
        patch_embedding = getattr(self.teacher, "patch_embedding")
        encoder = getattr(self.teacher, "encoder")
        enc_out, n_vars = patch_embedding(x_patch)
        enc_out, _ = encoder(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        return enc_out.mean(dim=1)

    def _teacher_features(self, x_norm: torch.Tensor) -> torch.Tensor:
        if self.teacher_feature_kind == "patchtst":
            return self._teacher_patchtst_features(x_norm)
        return self._teacher_tauonly_features(x_norm)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        del x_mark_enc, x_dec, x_mark_dec, mask

        if self.task_name not in {"long_term_forecast", "short_term_forecast"}:
            raise ValueError("TauClockFixedGRU currently supports forecasting tasks only")

        mean = x_enc.mean(dim=1, keepdim=True)
        std = torch.sqrt(torch.var(x_enc - mean, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = (x_enc - mean) / std

        if self.teacher_trainable:
            teacher_features = self._teacher_features(x_norm)
        else:
            with torch.no_grad():
                teacher_features = self._teacher_features(x_norm)

        teacher_features = self.teacher_feature_norm(teacher_features)

        h = self.init_proj(teacher_features[:, -1, :])
        prev = teacher_features[:, -1, :]
        preds = []
        for _ in range(self.pred_len):
            h = self.decoder_cell(prev, h)
            h = self.dropout(h)
            latent_step = self.readout(h)
            obs_step = self.output_proj(latent_step)
            preds.append(obs_step.unsqueeze(1))
            prev = latent_step

        pred_norm = torch.cat(preds, dim=1)
        return pred_norm * std + mean
