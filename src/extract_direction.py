"""Extract the refusal direction: sweep layers to pick the best one, then
compute the final difference-in-means direction at that layer.

Usage (from a notebook, after loading a config):

    from src.extract_direction import select_layer, extract_direction

    best_layer, layer_scores = select_layer(model, tok, cfg, harmful_extract,
                                             harmless_extract, harmful_eval)
    r = extract_direction(model, tok, harmful_extract, harmless_extract, best_layer)
"""

import torch

from .ablate import make_hooks
from .metrics import refusal_rate
from .model_utils import last_token_acts


def _direction_at_layer(model, tok, harmful_extract, harmless_extract, layer):
    mean_h = last_token_acts(model, tok, harmful_extract, layer).mean(0)
    mean_b = last_token_acts(model, tok, harmless_extract, layer).mean(0)
    r = mean_h - mean_b
    return (r / r.norm()).to(model.dtype)


def select_layer(model, tok, cfg, harmful_extract, harmless_extract, harmful_eval, verbose=True):
    """Try candidate layers, extracting a direction from the extraction set at
    each, and score by refusal-suppression on a small slice of the held-out
    harmful eval set. Returns (best_layer, list_of_(layer, score)).
    """
    num_layers = len(model.model.layers)
    val_n = cfg.get("layer_sweep_val_n", 16)
    val_subset = harmful_eval[:val_n] if len(harmful_eval) >= val_n else harmful_eval

    candidate_layers = list(range(cfg.get("layer_sweep_start", 4), num_layers))
    scores = []

    for L in candidate_layers:
        r_L = _direction_at_layer(model, tok, harmful_extract, harmless_extract, L)
        hooks_fn = lambda r_L=r_L: make_hooks(model, r_L, add_layer=L, alpha=1.0, beta=0.0)
        rr = refusal_rate(model, tok, val_subset, hooks_fn=hooks_fn, max_new=40)
        scores.append((L, rr))
        if verbose:
            print(f"layer {L:2d}: refusal rate under full ablation = {rr:.2f}")

    best_layer = min(scores, key=lambda x: x[1])[0]
    if verbose:
        print(f"\nSelected layer: {best_layer}")
    return best_layer, scores


def extract_direction(model, tok, harmful_extract, harmless_extract, layer):
    """Final direction at the chosen layer, using the full extraction set."""
    return _direction_at_layer(model, tok, harmful_extract, harmless_extract, layer)
