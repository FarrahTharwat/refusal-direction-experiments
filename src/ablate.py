"""Hook builders for directional ablation (alpha) and directional addition (beta).

Ablation removes the refusal direction's component from every layer's output
(matches Arditi et al.'s "ablate everywhere" method). Addition injects the
direction at a single layer only — applying it everywhere over-amplifies the
signal and mostly just breaks coherence rather than inducing clean refusal.
"""


def _ablate_hook(direction, alpha):
    def hook(module, inp, out):
        w = out[0] if isinstance(out, tuple) else out
        w = w - alpha * (w @ direction).unsqueeze(-1) * direction
        return (w,) + out[1:] if isinstance(out, tuple) else w

    return hook


def _add_hook(direction, beta):
    def hook(module, inp, out):
        w = out[0] if isinstance(out, tuple) else out
        w = w + beta * direction
        return (w,) + out[1:] if isinstance(out, tuple) else w

    return hook


def make_hooks(model, direction, add_layer, alpha=0.0, beta=0.0):
    """Build the (module, hook_fn) list for a given alpha/beta setting.

    `add_layer`: index into model.model.layers where addition is applied.
    Should normally be the same layer the direction was extracted from.
    """
    hooks = []
    if alpha != 0:
        ah = _ablate_hook(direction, alpha)
        hooks += [(l, ah) for l in model.model.layers]  # ablation: every layer
    if beta != 0:
        bh = _add_hook(direction, beta)
        hooks += [(model.model.layers[add_layer], bh)]  # addition: one layer only
    return hooks
