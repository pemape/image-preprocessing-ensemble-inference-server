"""Resolution of Grad-CAM target layers from dotted paths."""

import torch.nn as nn

from .contracts import ExplainabilityError


def resolve_target_layer(model: nn.Module, path: str) -> nn.Module:
    """
    Resolve a dotted path (e.g. ``"features.-1"`` or ``"bn4"``) to a sub-module.

    Integer tokens index into containers (``nn.Sequential`` supports negative
    indices); any other token is resolved as an attribute.
    """
    module = model
    for token in path.split("."):
        try:
            if token.lstrip("-").isdigit():
                module = module[int(token)]
            else:
                module = getattr(module, token)
        except (AttributeError, IndexError, TypeError, KeyError) as exc:
            raise ExplainabilityError(
                f"Cannot resolve target layer '{path}' (failed at '{token}')"
            ) from exc

    if not isinstance(module, nn.Module):
        raise ExplainabilityError(f"Target layer '{path}' is not a torch module")
    return module
