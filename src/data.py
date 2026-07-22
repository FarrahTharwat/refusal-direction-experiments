"""Data loading for refusal-direction experiments.

Loads harmful prompts (AdvBench) and harmless prompts (Alpaca), and splits
each into a direction-extraction set and a disjoint held-out evaluation set.
"""

import pandas as pd
from datasets import load_dataset

ADVBENCH_URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"


def load_harmful_pool(n, seed=0):
    """Sample n harmful instructions from AdvBench."""
    adv = pd.read_csv(ADVBENCH_URL)
    return adv["goal"].sample(n, random_state=seed).tolist()


def load_harmless_pool(n):
    """Take the first n standalone Alpaca instructions (no extra 'input' context)."""
    alp = load_dataset("tatsu-lab/alpaca", split="train")
    out = []
    for r in alp:
        if r["input"] == "":
            out.append(r["instruction"])
        if len(out) >= n:
            break
    return out


def build_splits(cfg):
    """Returns (harmful_extract_pool, harmful_eval_pool, harmless_extract_pool, harmless_eval_pool).

    These are *pools* — the harmful ones still need refusal-filtering
    (see metrics.filter_refused) before use, since not every AdvBench prompt
    is guaranteed to be refused by a given model.
    """
    pool_n = (cfg["extract_n"] + cfg["eval_n"]) * cfg["pool_multiplier"]

    harmful_pool = load_harmful_pool(pool_n, seed=cfg["seed"])
    harmless_pool = load_harmless_pool(pool_n)

    harmful_extract_pool = harmful_pool[: cfg["extract_n"]]
    harmful_eval_pool = harmful_pool[cfg["extract_n"] :]

    harmless_extract_pool = harmless_pool[: cfg["extract_n"]]
    harmless_eval_pool = harmless_pool[cfg["extract_n"] :]

    return harmful_extract_pool, harmful_eval_pool, harmless_extract_pool, harmless_eval_pool
