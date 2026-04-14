from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

LOG = Path("result_long_term_forecast.txt")
OUT = Path("logs/baseline_scores_summary.txt")
CSV_OUT = Path("logs/baseline_scores_all.csv")

run_line = re.compile(r"^long_term_forecast_(?P<run>.+?)\s*$")
metric_line = re.compile(r"^mse:(?P<mse>[0-9.]+),\s*mae:(?P<mae>[0-9.]+),\s*dtw:")
dataset_re = re.compile(r"_(ETTh1|ETTh2|ETTm1|ETTm2|Weather|Electricity|Traffic|Exchange|ILI)(?:_|$)")
seed_re = re.compile(r"_(?:seed|s)(\d+)(?:_|$)")
seed_after_horizon_re = re.compile(r"_(?:96|192|336|720|24|36|48|60)_(\d+)_DLinear")
horizon_re = re.compile(r"_(?:pl|h)(\d+)(?:_|$)")

lines = LOG.read_text(encoding="utf-8", errors="ignore").splitlines()
entries = []
pending_run = None

for raw in lines:
    line = raw.strip()
    if not line:
        continue
    m_run = run_line.match(line)
    if m_run:
        pending_run = m_run.group("run")
        continue
    if pending_run is not None:
        m_metric = metric_line.match(line)
        if not m_metric:
            continue

        run = pending_run
        pending_run = None
        mse = float(m_metric.group("mse"))
        mae = float(m_metric.group("mae"))

        model = ""
        if "DLinear" in run:
            model = "DLinear"
        elif "PatchTST" in run:
            model = "PatchTST"
        elif "iTransformer" in run:
            model = "iTransformer"
        elif "PPN" in run:
            model = "PPN"

        dataset = ""
        m_ds = dataset_re.search(run)
        if m_ds:
            dataset = m_ds.group(1)

        horizon = ""
        m_h = horizon_re.search(run)
        if m_h:
            horizon = m_h.group(1)

        seed = ""
        m_s = seed_re.search(run)
        if m_s:
            seed = m_s.group(1)
        else:
            m_s2 = seed_after_horizon_re.search(run)
            if m_s2:
                seed = m_s2.group(1)

        entries.append({"run": run, "model": model, "dataset": dataset, "horizon": horizon, "seed": seed, "mse": mse, "mae": mae})

# Deduplicate repeated run+metric pairs while preserving order.
seen = set()
unique = []
for item in entries:
    key = (item["run"], item["mse"], item["mae"])
    if key in seen:
        continue
    seen.add(key)
    unique.append(item)

by_ds = defaultdict(list)
for item in unique:
    if item["dataset"]:
        by_ds[item["dataset"]].append(item)

lines = []
lines.append("Baseline score extraction")
lines.append("=" * 80)
lines.append(f"Total entries: {len(unique)}")
lines.append("")

for ds in sorted(by_ds.keys()):
    lines.append(f"[{ds}]")
    items = by_ds[ds]
    by_h = defaultdict(list)
    for item in items:
        by_h[item["horizon"]].append(item)
    for h in sorted(by_h.keys(), key=lambda x: int(x) if x.isdigit() else 9999):
        vals = by_h[h]
        mse_mean = sum(v["mse"] for v in vals) / len(vals)
        mae_mean = sum(v["mae"] for v in vals) / len(vals)
        lines.append(f"  h={h:<4} n={len(vals):<2} mean_mse={mse_mean:.6f} mean_mae={mae_mean:.6f}")
        for v in vals[:5]:
            lines.append(f"    seed={v['seed']:<4} mse={v['mse']:.6f} mae={v['mae']:.6f}")
        if len(vals) > 5:
            lines.append(f"    ... {len(vals)-5} more")
    lines.append("")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines), encoding="utf-8")
CSV_OUT.write_text(
    "run,model,dataset,horizon,seed,mse,mae\n"
    + "\n".join(
        f"{item['run']},{item['model']},{item['dataset']},{item['horizon']},{item['seed']},{item['mse']:.10f},{item['mae']:.10f}"
        for item in unique
    ),
    encoding="utf-8",
)
print(OUT)
print(CSV_OUT)
print(f"entries={len(unique)}")
