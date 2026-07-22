# refusal-ablation

A reproduction of **"Refusal in Language Models Is Mediated by a Single Direction"**
(Arditi et al., 2024), extended with a held-out evaluation split, a layer-selection
sweep, and an alpha/beta ablation-vs-addition sweep with a crossover plot.

- Paper: https://arxiv.org/abs/2406.11717
- Original authors' code: https://github.com/andyrdt/refusal_direction

The core method (difference-in-means direction, directional ablation and addition
via forward hooks) is entirely theirs. This repo re-implements it end to end, adds
a proper train/eval split (no more testing on the data the direction was built
from), and adds quantitative refusal-rate measurement across a strength sweep,
rather than eyeballing single examples.

## What it does

1. Loads harmful prompts (AdvBench) and harmless prompts (Alpaca), splitting each
   into a **direction-extraction set** (32/32) and a **disjoint held-out eval set**
   (configurable per model, see `configs/`).
2. Extracts a **refusal direction** as the difference-in-means between harmful and
   harmless last-token residual-stream activations — the layer is *selected* via a
   validation sweep, not hardcoded.
3. Applies **directional ablation** (removes the direction from every layer's
   output) and **directional addition** (injects the direction at a single layer)
   at configurable strengths, alpha and beta.
4. Sweeps alpha and beta across a grid, measuring refusal rate on the held-out
   harmful and harmless sets at each setting — producing a crossover plot showing
   where harmful refusal collapses and where harmless refusal starts appearing.

## Structure

```
configs/     one YAML per model — swap the config, not the code, to try a new model
src/         core library: data loading, model/generation utils, direction
             extraction + layer sweep, ablation/addition hooks, sweep + plotting
notebooks/   thin orchestration notebook (demo.ipynb) — wires src/ together
             for a given config; run this in Colab
results/     CSVs + crossover plots land here after a run
data/        small hand-picked harmful/harmless prompt pairs, used only for
             blog-post screenshots (the real eval uses AdvBench/Alpaca directly)
```

## Running it

Open `notebooks/demo.ipynb` in Colab (GPU runtime), point `CONFIG_PATH` at a
config in `configs/`, and run top to bottom. Swapping models is just changing
that one path — everything downstream re-derives itself.

## Results

*(fill in after running: crossover plot + 1-2 sentences on where the crossover
happened for this model)*

## Notes

- Interpretability / evaluation exercise — no modified model weights are
  published, only code and (once run) CSV/plot artifacts.
- Direction vectors and model checkpoints are intentionally kept out of version
  control (see `.gitignore`).

## Credit

All credit for the method and original findings goes to Arditi, Obeso, Syed,
Paleka, Panickssery, Gurnee, and Nanda.
