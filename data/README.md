# data/

The real evaluation pulls directly from AdvBench (harmful) and Alpaca (harmless)
at runtime — see `src/data.py`. Nothing needs to be hand-authored for the actual
experiment.

This folder is for a **small curated subset**, used only for cleaner screenshots
/ GIFs in the blog post (AdvBench prompts read as repetitive and dry on screen).

To build it: after running `notebooks/demo.ipynb`, pick ~15-20 prompts from
`harmful_extract` / `harmless_extract` (or `harmful_eval` / `harmless_eval`) that
produced the clearest before/after contrast, and save them here as
`curated_harmful.jsonl` / `curated_harmless.jsonl` (one JSON object per line,
e.g. `{"prompt": "..."}`). Pulling from prompts the pipeline already validated
(rather than writing new ones) keeps the "legit, reproducible" framing intact.
