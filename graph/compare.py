
#!/usr/bin/env python3
"""
Compare training runs across circuits/ansatz by plotting loss/MMD/KL/grad_norm/step_time
and by exporting a summary CSV.

Expected per-run directory layout (auto-discovered under --root):
  <root>/<run_id>/
    final_params.npy
    grad_norm.npy
    grads.npy
    kl.npy
    mmd.npy
    loss.npy
    params.npy
    step_time.npy      # seconds or milliseconds; auto-detected

Example:
  python compare_training_runs.py \
      --root data/results/Qubits8 \
      --out  figures/Qubits8 \
      --ema  0.9 \
      --metrics loss mmd kl grad_norm step_time

Notes:
- Uses ONLY numpy & matplotlib (no seaborn).
- Each chart is a single-axes figure (no subplots).
- If step_time looks like seconds (median < 0.1), it will be converted to milliseconds.
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


METRIC_FILES = {
    "loss": "loss.npy",
    "mmd": "mmd.npy",
    "kl": "kl.npy",
    "grad_norm": "grad_norm.npy",
    "step_time": "step_time.npy",
}
OPTIONAL_FILES = {"grads": "grads.npy", "params": "params.npy", "final_params": "final_params.npy"}


def ema(arr: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average with coefficient alpha in (0,1].
    Larger alpha -> less smoothing (more weight on current point).
    """
    if alpha <= 0 or alpha > 1:
        return arr
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
    return out


def load_metric(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        x = np.load(path)
        # Ensure 1D time series if possible
        if x.ndim > 1:
            # If it's shape (n_steps, ...), try to take a scalar view like mean across trailing dims
            try:
                return np.mean(x, axis=tuple(range(1, x.ndim)))
            except Exception:
                return None
        return x
    except Exception:
        return None


def detect_and_convert_step_time(step_time: np.ndarray) -> Tuple[np.ndarray, str]:
    """Heuristic: if times look like seconds (very small), convert to ms."""
    med = np.median(step_time)
    # threshold: if < 0.2 it's probably seconds
    if med < 0.2:
        return step_time * 1e3, "ms"
    # if values already large, assume ms
    return step_time, "ms" if med > 2.0 else "ms"  # default to ms either way for display consistency


def discover_runs(root: Path) -> List[Path]:
    runs = []
    for sub in sorted(root.glob("*")):
        if not sub.is_dir():
            continue
        # consider a directory a "run" if it contains at least one core metric file
        if any((sub / fname).exists() for fname in METRIC_FILES.values()):
            runs.append(sub)
    return runs


def load_run(run_dir: Path) -> Dict[str, np.ndarray]:
    data = {}
    for k, fname in METRIC_FILES.items():
        arr = load_metric(run_dir / fname)
        if arr is not None:
            data[k] = arr

    # Optional large arrays; don't coerce shapes
    for k, fname in OPTIONAL_FILES.items():
        fpath = run_dir / fname
        if fpath.exists():
            try:
                data[k] = np.load(fpath, mmap_mode="r")
            except Exception:
                pass

    # Normalize step_time to ms (1-D)
    if "step_time" in data:
        st = data["step_time"]
        if st.ndim > 1:
            st = np.mean(st, axis=tuple(range(1, st.ndim)))
        st_ms, unit = detect_and_convert_step_time(st.astype(np.float64))
        data["step_time"] = st_ms
        data["_step_time_unit"] = unit
    return data


def align_length(arrs: List[np.ndarray]) -> int:
    """Find minimum common length across arrays to align for plotting."""
    if not arrs:
        return 0
    return int(min(len(a) for a in arrs if a is not None))


def summarize_run(run_id: str, d: Dict[str, np.ndarray]) -> Dict[str, float]:
    out = {"run_id": run_id}
    def safe(v, fn, default=np.nan):
        try:
            return float(fn(v)) if v is not None else default
        except Exception:
            return default

    # Core metrics
    loss = d.get("loss", None)
    mmd = d.get("mmd", None)
    kl = d.get("kl", None)
    step_time = d.get("step_time", None)

    out["n_steps"] = len(loss) if loss is not None else (
        len(mmd) if mmd is not None else (len(kl) if kl is not None else 0)
    )

    if loss is not None:
        out["final_loss"] = float(loss[-1])
        out["best_loss"] = float(np.min(loss))
        out["best_loss_step"] = int(np.argmin(loss))
    else:
        out["final_loss"] = np.nan
        out["best_loss"] = np.nan
        out["best_loss_step"] = -1

    if mmd is not None:
        out["final_mmd"] = float(mmd[-1]); out["best_mmd"] = float(np.min(mmd)); out["best_mmd_step"] = int(np.argmin(mmd))
    else:
        out["final_mmd"] = np.nan; out["best_mmd"] = np.nan; out["best_mmd_step"] = -1

    if kl is not None:
        out["final_kl"] = float(kl[-1]); out["best_kl"] = float(np.min(kl)); out["best_kl_step"] = int(np.argmin(kl))
    else:
        out["final_kl"] = np.nan; out["best_kl"] = np.nan; out["best_kl_step"] = -1

    if step_time is not None:
        out["avg_step_time_ms"] = float(np.mean(step_time))
        out["p50_step_time_ms"] = float(np.percentile(step_time, 50))
        out["p95_step_time_ms"] = float(np.percentile(step_time, 95))
        out["total_time_ms"] = float(np.sum(step_time))
        if loss is not None:
            k = out["best_loss_step"]
            out["time_to_best_loss_ms"] = float(np.sum(step_time[:k+1])) if k >= 0 else np.nan
    else:
        out["avg_step_time_ms"] = np.nan; out["p50_step_time_ms"] = np.nan; out["p95_step_time_ms"] = np.nan; out["total_time_ms"] = np.nan; out["time_to_best_loss_ms"] = np.nan

    return out


def plot_metric_over_steps(runs: Dict[str, Dict[str, np.ndarray]], metric: str, out_dir: Path, ema_coeff: float = 1.0):
    # gather series
    series = []
    labels = []
    for run_id, d in runs.items():
        arr = d.get(metric, None)
        if arr is None:
            continue
        series.append(arr)
        labels.append(run_id)
    if not series:
        print(f"[warn] No data for metric '{metric}', skipping plot.")
        return

    L = align_length(series)
    series = [s[:L] for s in series]

    plt.figure(figsize=(9, 5))
    for y, label in zip(series, labels):
        y_plot = ema(y, ema_coeff) if ema_coeff and ema_coeff <= 1.0 else y
        plt.plot(np.arange(len(y_plot)), y_plot, label=label)

    plt.xlabel("Step")
    ylabel = metric if metric != "step_time" else "Step time (ms)"
    plt.ylabel(ylabel)
    plt.title(f"{metric} vs. step (EMA={ema_coeff})" if ema_coeff and ema_coeff <= 1.0 and ema_coeff < 1.0 else f"{metric} vs. step")
    plt.legend(loc="best")
    out_path = out_dir / f"{metric}_vs_step.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[save] {out_path}")


def plot_step_time_box(runs: Dict[str, Dict[str, np.ndarray]], out_dir: Path):
    # One box per run
    data = []
    labels = []
    for run_id, d in runs.items():
        st = d.get("step_time", None)
        if st is None:
            continue
        data.append(st)
        labels.append(run_id)
    if not data:
        print("[warn] No step_time found for boxplot.")
        return

    plt.figure(figsize=(9, 5))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel("Step time (ms)")
    plt.title("Step time distribution per run")
    plt.tight_layout()
    out_path = out_dir / "step_time_boxplot.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[save] {out_path}")


def write_summary_csv(summaries: List[Dict[str, float]], out_path: Path):
    # Collect all keys
    keys = sorted({k for s in summaries for k in s.keys()})
    with out_path.open("w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for s in summaries:
            row = []
            for k in keys:
                v = s.get(k, "")
                row.append(str(v))
            f.write(",".join(row) + "\n")
    print(f"[save] {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True, help="Root directory containing per-run subfolders.")
    p.add_argument("--out", type=Path, required=True, help="Output directory for figures & summary.")
    p.add_argument("--runs", type=str, nargs="*", default=None, help="Specific run directory names (under --root) to include. Defaults to all discovered runs.")
    p.add_argument("--labels", type=str, nargs="*", default=None, help="Optional labels for runs (same length as --runs).")
    p.add_argument("--ema", type=float, default=1.0, help="EMA coefficient in (0,1]; smaller = stronger smoothing. 1.0 disables.")
    p.add_argument("--metrics", type=str, nargs="*", default=["loss", "mmd", "kl", "grad_norm", "step_time"], help="Which metrics to plot over steps.")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Discover or filter runs
    discovered = discover_runs(args.root)
    if args.runs:
        selected = []
        for name in args.runs:
            path = args.root / name
            if not path.exists():
                print(f"[warn] selected run not found: {name}")
            else:
                selected.append(path)
        run_dirs = selected
    else:
        run_dirs = discovered

    if not run_dirs:
        print("[error] No runs found.")
        return

    # Load runs
    runs_data: Dict[str, Dict[str, np.ndarray]] = {}
    for run_dir in run_dirs:
        run_id = run_dir.name
        runs_data[run_id] = load_run(run_dir)

    # Labels
    if args.labels and len(args.labels) == len(run_dirs):
        label_map = {d.name: lab for d, lab in zip(run_dirs, args.labels)}
        runs_data = {label_map.get(run_id, run_id): data for run_id, data in runs_data.items()}

    # Summary
    summaries = [summarize_run(run_id, d) for run_id, d in runs_data.items()]
    write_summary_csv(summaries, args.out / "summary.csv")

    # Plots
    for metric in args.metrics:
        plot_metric_over_steps(runs_data, metric, args.out, ema_coeff=args.ema)

    # Extra: step-time boxplot
    if "step_time" in args.metrics:
        plot_step_time_box(runs_data, args.out)

    print("[done] Figures & summary written to", args.out)


if __name__ == "__main__":
    main()