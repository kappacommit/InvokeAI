"""Utilities for patching the AnimaTransformer for attention-couple regional prompting."""

from contextlib import contextmanager
from typing import Callable, Optional

import torch

from invokeai.backend.anima.regional_prompting import AnimaAttentionCoupleExtension
from invokeai.backend.util.logging import InvokeAILogger

logger = InvokeAILogger.get_logger(__name__)


def _patched_cross_attn_forward(
    original_forward: Callable[..., torch.Tensor],
    coupling: AnimaAttentionCoupleExtension,
    bypass_warned: list,
):
    """Create a patched forward for a DiT block's cross-attention CosmosAttention.

    When coupling is enabled (the denoise loop turns it on for the conditional
    pass only), the image-token queries are repeated once per slot, each copy
    attends to its own slot's context with the ORIGINAL unmasked attention math,
    and the outputs are blended back to a single batch element with the slots'
    spatial weights. Everything else — self-attention, MLPs, the uncond pass —
    runs the model unmodified.

    Args:
        original_forward: The original CosmosAttention.forward method (bound to self).
        coupling: The attention-couple extension holding slot contexts, blend
            weights, and the per-call `coupling_enabled` flag.
        bypass_warned: Shared one-element flag list; set to [True] after warning
            once about coupling bypassed due to a geometry mismatch.
    """

    def forward(x, context=None, rope_emb=None):
        # Self-attention calls (context=None) and non-coupled passes run unmodified.
        if context is None or not coupling.coupling_enabled:
            return original_forward(x, context, rope_emb=rope_emb)

        num_slots, img_seq_len = coupling.token_weights.shape[0], coupling.token_weights.shape[1]
        if x.shape[0] != 1 or x.shape[-2] != img_seq_len:
            # The denoise loop only enables coupling for calls whose geometry was
            # derived from the same width/height fields as the blend weights, so a
            # mismatch here indicates an unexpected token-grid disagreement. Warn
            # once rather than silently generating with the wrong conditioning.
            if not bypass_warned[0]:
                bypass_warned[0] = True
                logger.warning(
                    "Anima attention couple bypassed: expected query shape (1, %d, dim), got %s. "
                    "Regional prompts are being applied without spatial blending.",
                    img_seq_len,
                    tuple(x.shape),
                )
            return original_forward(x, context, rope_emb=rope_emb)

        # (1, S, D) -> (N, S, D): each slot sees the same queries but its own context.
        out = original_forward(x.expand(num_slots, -1, -1), coupling.slot_contexts, rope_emb=rope_emb)
        weights = coupling.token_weights.to(device=out.device, dtype=out.dtype)
        return (out * weights).sum(dim=0, keepdim=True)

    return forward


@contextmanager
def patch_anima_for_attention_couple(
    transformer,
    coupling: Optional[AnimaAttentionCoupleExtension],
):
    """Context manager to temporarily patch the Anima transformer for attention couple.

    Patches the cross-attention of every DiT block to consult the coupling
    extension on each call. Whether coupling is actually applied is controlled by
    the extension's `coupling_enabled` flag, which the denoise loop toggles per
    pass (cond only).

    Args:
        transformer: The AnimaTransformer instance.
        coupling: The attention-couple extension. If None, no patching occurs.

    Yields:
        The (possibly patched) transformer.
    """
    if coupling is None:
        yield transformer
        return

    original_forwards: list[tuple] = []
    bypass_warned = [False]
    try:
        for block in transformer.blocks:
            original_forwards.append((block, block.cross_attn.forward))
            block.cross_attn.forward = _patched_cross_attn_forward(block.cross_attn.forward, coupling, bypass_warned)
        yield transformer
    finally:
        for block, cross_attn_forward in original_forwards:
            block.cross_attn.forward = cross_attn_forward
