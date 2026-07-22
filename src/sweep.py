"""Grid sweep over alpha (ablation) and beta (addition), measuring refusal
rate on held-out harmful/harmless sets at each setting. Saves CSVs and a
crossover plot to the given results directory.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

from .ablate import make_hooks
from .metrics import refusal_rate


def run_alpha_sweep(model, tok, direction, add_layer, harmful_eval, harmless_eval, alphas):
    rows = []
    for a in alphas:
        hooks_fn = lambda a=a: make_hooks(model, direction, add_layer, alpha=a, beta=0.0)
        hr = refusal_rate(model, tok, harmful_eval, hooks_fn=hooks_fn)
        br = refusal_rate(model, tok, harmless_eval, hooks_fn=hooks_fn)
        rows.append({"alpha": a, "harmful_refusal_rate": hr, "harmless_refusal_rate": br})
        print(f"alpha={a:.2f}  harmful_rr={hr:.2f}  harmless_rr={br:.2f}")
    return pd.DataFrame(rows)


def run_beta_sweep(model, tok, direction, add_layer, harmful_eval, harmless_eval, betas):
    rows = []
    for b in betas:
        hooks_fn = lambda b=b: make_hooks(model, direction, add_layer, alpha=0.0, beta=b)
        hr = refusal_rate(model, tok, harmful_eval, hooks_fn=hooks_fn)
        br = refusal_rate(model, tok, harmless_eval, hooks_fn=hooks_fn)
        rows.append({"beta": b, "harmful_refusal_rate": hr, "harmless_refusal_rate": br})
        print(f"beta={b:.2f}  harmful_rr={hr:.2f}  harmless_rr={br:.2f}")
    return pd.DataFrame(rows)


def save_results(alpha_df, beta_df, results_dir, model_alias):
    os.makedirs(results_dir, exist_ok=True)
    alpha_path = os.path.join(results_dir, f"alpha_sweep_{model_alias}.csv")
    beta_path = os.path.join(results_dir, f"beta_sweep_{model_alias}.csv")
    plot_path = os.path.join(results_dir, f"crossover_{model_alias}.png")

    alpha_df.to_csv(alpha_path, index=False)
    beta_df.to_csv(beta_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(alpha_df["alpha"], alpha_df["harmful_refusal_rate"], marker="o", label="harmful refusal rate")
    axes[0].plot(alpha_df["alpha"], alpha_df["harmless_refusal_rate"], marker="o", label="harmless refusal rate")
    axes[0].set_xlabel("alpha (ablation strength)")
    axes[0].set_ylabel("refusal rate")
    axes[0].set_title(f"Ablation sweep — {model_alias}")
    axes[0].legend()

    axes[1].plot(beta_df["beta"], beta_df["harmless_refusal_rate"], marker="o", label="harmless refusal rate")
    axes[1].plot(beta_df["beta"], beta_df["harmful_refusal_rate"], marker="o", label="harmful refusal rate")
    axes[1].set_xlabel("beta (addition strength)")
    axes[1].set_ylabel("refusal rate")
    axes[1].set_title(f"Addition sweep — {model_alias}")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.show()

    print(f"saved: {alpha_path}, {beta_path}, {plot_path}")
    return alpha_path, beta_path, plot_path
