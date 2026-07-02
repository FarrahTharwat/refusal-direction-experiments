# Refusal Direction — A Reproduction

A from-scratch reproduction of **"Refusal in Language Models Is Mediated by a
Single Direction"** (Arditi et al., 2024), built for my own learning while
reading the paper.

- Paper: https://arxiv.org/abs/2406.11717
- Original authors' code: https://github.com/andyrdt/refusal_direction

The method is entirely theirs; this repo is my own re-implementation to
understand it end to end.

## What it does

Working on **Qwen1.5-1.8B-Chat**, this notebook:

1. Loads harmful prompts (AdvBench) and harmless prompts (Alpaca), then filters
   the harmful set down to prompts the model actually refuses.
2. Extracts a **refusal direction** as the difference-in-means between harmful
   and harmless last-token activations at a chosen layer.
3. Applies **directional ablation** — projecting that direction out of the
   residual stream with a forward hook — and the model then answers prompts it
   previously refused.

## Notes

- An interpretability / learning exercise. No modified model weights are
  published here — only the code.
- Direction vectors and model files are intentionally kept out of version
  control (`.gitignore`).

## Credit

All credit for the method and original findings goes to Arditi, Obeso, Syed,
Paleka, Panickssery, Gurnee, and Nanda.
