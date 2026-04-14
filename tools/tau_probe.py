import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_provider.data_factory import data_provider
from models.PPN import Model as PPNModel


def build_args(parsed):
    return SimpleNamespace(
        task_name="long_term_forecast",
        data=parsed.data,
        root_path=parsed.root_path,
        data_path=parsed.data_path,
        features="M",
        target="OT",
        freq="h",
        embed="timeF",
        seasonal_patterns="Monthly",
        seq_len=parsed.seq_len,
        label_len=parsed.label_len,
        pred_len=parsed.pred_len,
        batch_size=parsed.batch_size,
        num_workers=parsed.num_workers,
        enc_in=parsed.enc_in,
        dec_in=parsed.enc_in,
        c_out=parsed.enc_in,
        d_model=parsed.d_model,
        dropout=0.1,
        ppn_use_patch_embed=0,
        ppn_patch_len=8,
        ppn_patch_stride=4,
        ppn_use_var_tau_gate=0,
        ppn_var_tau_gate_scale=0.25,
        ppn_use_multi_scale_tau=0,
        ppn_tau_scales="1,2,4",
        ppn_use_horizon_residual=parsed.use_hres,
        ppn_use_horizon_residual_gate=parsed.use_hres_gate,
        ppn_horizon_residual_scale=0.1,
        augmentation_ratio=0,
    )


def corr(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="ETTh1")
    parser.add_argument("--root_path", default="./dataset/ETT-small/")
    parser.add_argument("--data_path", default="ETTh1.csv")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--enc_in", type=int, default=7)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=40)
    parser.add_argument("--use_hres", type=int, default=1)
    parser.add_argument("--use_hres_gate", type=int, default=0)
    args = parser.parse_args()

    cfg = build_args(args)

    model = PPNModel(cfg)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()

    _, train_loader = data_provider(cfg, "train")

    tau_vals = []
    rho_vals = []
    accel_vals = []
    delta_vals = []

    with torch.no_grad():
        for bi, (batch_x, _, _, _) in enumerate(train_loader):
            if bi >= args.max_batches:
                break
            x = batch_x.float()
            if x.shape[1] < 4:
                continue

            x_t = x[:, -1, :]
            x_prev = x[:, -2, :]
            x_prev2 = x[:, -3, :]
            x_prev3 = x[:, -4, :]

            tau_t = model.tau_module(x_t, x_prev)
            tau_prev = model.tau_module(x_prev, x_prev2)
            rho_t = tau_t / (tau_prev + 1e-6)

            a_t = x_t - 2.0 * x_prev + x_prev2
            a_prev = x_prev - 2.0 * x_prev2 + x_prev3
            accel = torch.norm(a_t - a_prev, dim=-1)
            delta = torch.norm(x_t - x_prev, dim=-1)

            tau_vals.extend(tau_t.mean(dim=-1).cpu().numpy().tolist())
            rho_vals.extend(rho_t.mean(dim=-1).cpu().numpy().tolist())
            accel_vals.extend(accel.cpu().numpy().tolist())
            delta_vals.extend(delta.cpu().numpy().tolist())

    tau_vals = np.asarray(tau_vals)
    rho_vals = np.asarray(rho_vals)
    accel_vals = np.asarray(accel_vals)
    delta_vals = np.asarray(delta_vals)

    q75 = np.quantile(accel_vals, 0.75)
    q25 = np.quantile(accel_vals, 0.25)
    high_tau = tau_vals[accel_vals >= q75]
    low_tau = tau_vals[accel_vals <= q25]

    print("tau_probe_report")
    print(f"samples={tau_vals.size}")
    print(f"tau_mean={tau_vals.mean():.6f} tau_std={tau_vals.std():.6f}")
    print(f"rho_mean={rho_vals.mean():.6f} rho_std={rho_vals.std():.6f}")
    print(f"corr_tau_accel={corr(tau_vals, accel_vals):.6f}")
    print(f"corr_tau_delta={corr(tau_vals, delta_vals):.6f}")
    print(f"tau_high_accel_mean={high_tau.mean():.6f} tau_low_accel_mean={low_tau.mean():.6f}")


if __name__ == "__main__":
    main()
