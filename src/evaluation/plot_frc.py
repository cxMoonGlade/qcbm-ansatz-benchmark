from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
metrics_path = ROOT / "data" / "results" / "metrics.json"
out_dir = ROOT / "data" / "results" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

with open(metrics_path) as f:
    rows = json.load(f)
df = pd.DataFrame(rows).sort_values("ansatz")

# pretty labels
labels = [f"ansatz{int(a)}" for a in df["ansatz"]]
x = range(len(labels))

plt.figure(figsize=(8,4.8))
w = 0.25
plt.bar([i- w for i in x], df["Fidelity"], width=w, label="Fidelity")
plt.bar([i     for i in x], df["Rate"],     width=w, label="Rate (normalized)")
plt.bar([i+ w for i in x], df["Coverage"], width=w, label="Coverage (normalized)")
plt.xticks(x, labels)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("FRC metrics by ansatz")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "frc_metrics.png", dpi=180)
print("[saved]", out_dir / "frc_metrics.png")
