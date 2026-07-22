"""Model loading and generation, with pluggable forward hooks."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_DTYPE_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def load_model(cfg):
    dtype = _DTYPE_MAP[cfg.get("dtype", "float16")]
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], torch_dtype=dtype).to("cuda").eval()
    return model, tok


@torch.no_grad()
def generate(model, tok, prompt, hooks=None, max_new=80):
    """Generate a completion for a single user prompt.

    `hooks`: optional list of (module, hook_fn) pairs, registered as forward
    hooks for the duration of this call only.
    """
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_dict=False,
        return_tensors="pt",
    ).to("cuda")

    handles = []
    if hooks:
        handles = [module.register_forward_hook(fn) for module, fn in hooks]

    try:
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    finally:
        for h in handles:
            h.remove()

    return tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True)


@torch.no_grad()
def last_token_acts(model, tok, prompts, layer):
    """Residual-stream activation at the last prompt token, for each prompt, at `layer`."""
    outs = []
    for p in prompts:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": p}], add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")
        hs = model(**ids, output_hidden_states=True).hidden_states
        outs.append(hs[layer][0, -1].float())
    return torch.stack(outs)
