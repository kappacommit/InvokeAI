"""Regional prompting for Anima via attention couple.

Anima's DiT uses separate cross-attention in each block: image tokens attend to
the 512-token context produced by the LLM Adapter. Regional guidance runs each
prompt through the LLM Adapter separately, then every DiT block's cross-attention
runs once per slot — the image-token queries are repeated, and each copy attends
to its own slot's context with a normal, unmasked softmax — and the attention
outputs are blended with the slots' spatial weights. Self-attention is untouched,
so the remaining layers of the same forward pass re-harmonize the mixture.

Ported from the Anima region support in Acly's comfyui-tooling-nodes (region.py,
adapted from pamparamm/ComfyUI-ppm and laksjdjf's attention_couple), which
krita-ai-diffusion uses for Anima regions.

Coupling runs for the entire sampling schedule. Blend weights: region slots use
their masks; base (unmasked) prompts cover the area outside every region mask;
per token, the weights are normalized to a convex combination.
"""

from typing import Optional

import torch
import torchvision

from invokeai.backend.anima.conditioning_data import AnimaRegionalTextConditioning
from invokeai.backend.util.mask import to_standard_float_mask


def build_regional_blend_weights(
    image_masks: list[torch.Tensor | None],
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Build per-slot convex blend weights on a flattened spatial grid.

    Args:
        image_masks: One entry per conditioning slot. Region slots carry a mask of
            shape (1, 1, seq_len) with values in [0, 1]; base (unmasked) prompts
            carry None and are assigned the area outside every region mask.
        seq_len: Number of tokens in the flattened spatial grid.
        device: Device for the weight tensor.

    Returns:
        Weights of shape (num_slots, seq_len), float32, summing to 1 over slots
        for every token. Overlapping region masks share a token's weight
        proportionally. Tokens covered by no slot (possible only when no base
        prompt is present) are distributed uniformly across all slots.
    """
    flat_masks: list[torch.Tensor | None] = []
    for mask in image_masks:
        if mask is None:
            flat_masks.append(None)
            continue
        flat = mask.reshape(-1).to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
        if flat.shape[0] != seq_len:
            raise ValueError(
                f"Regional mask has {flat.shape[0]} tokens but the spatial grid has {seq_len} tokens. "
                "The mask was likely prepared for a different resolution."
            )
        flat_masks.append(flat)

    covered = torch.zeros(seq_len, device=device, dtype=torch.float32)
    for flat in flat_masks:
        if flat is not None:
            covered = torch.maximum(covered, flat)
    uncovered = (1.0 - covered).clamp(min=0.0)

    weights = torch.stack([uncovered if flat is None else flat for flat in flat_masks], dim=0)
    weight_sum = weights.sum(dim=0, keepdim=True)
    uniform = torch.full_like(weights, 1.0 / len(flat_masks))
    return torch.where(weight_sum > 1e-6, weights / weight_sum.clamp(min=1e-6), uniform)


class AnimaAttentionCoupleExtension:
    """Holds the per-slot contexts and blend weights for attention-couple mode.

    The transformer's cross-attention is patched once for the whole denoise loop
    (see anima_transformer_patch.py); the denoise loop toggles `coupling_enabled`
    per transformer call — on for the conditional pass, off for the uncond pass.
    """

    def __init__(self, slot_contexts: torch.Tensor, token_weights: torch.Tensor):
        # (num_slots, ctx_len, ctx_dim) — one LLM Adapter context per slot.
        self.slot_contexts = slot_contexts
        # (num_slots, img_seq_len, 1) — convex blend weights per image token.
        self.token_weights = token_weights
        self.coupling_enabled = False

    @classmethod
    def from_regional_conditioning(
        cls,
        regional_text_conditioning: AnimaRegionalTextConditioning,
        img_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "AnimaAttentionCoupleExtension":
        """Create the extension from pre-processed regional conditioning.

        Args:
            regional_text_conditioning: Concatenated per-slot LLM Adapter contexts
                with masks. Slots must all have the same context length (the LLM
                Adapter pads every prompt to 512 tokens).
            img_seq_len: Number of image tokens (T * H_patches * W_patches).
            device: Device for the tensors.
            dtype: Dtype for the slot contexts (the inference dtype).
        """
        ranges = regional_text_conditioning.context_ranges
        slot_lengths = {r.end - r.start for r in ranges}
        if len(slot_lengths) != 1:
            raise ValueError(
                f"Attention couple requires equal-length slot contexts, got lengths "
                f"{sorted(r.end - r.start for r in ranges)}."
            )
        slot_contexts = torch.stack(
            [regional_text_conditioning.context_embeds[r.start : r.end] for r in ranges], dim=0
        ).to(device=device, dtype=dtype)

        token_weights = build_regional_blend_weights(
            regional_text_conditioning.image_masks, img_seq_len, device
        ).unsqueeze(-1)

        return cls(slot_contexts=slot_contexts, token_weights=token_weights)


def preprocess_regional_prompt_mask(
    mask: Optional[torch.Tensor],
    target_height: int,
    target_width: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Preprocess a regional prompt mask to match a target spatial grid.

    Args:
        mask: Input mask tensor. If None, returns a mask of all ones.
        target_height: Height of the target grid (the image token grid).
        target_width: Width of the target grid.
        dtype: Target dtype for the mask.
        device: Target device for the mask.

    Returns:
        Processed mask of shape (1, 1, target_height * target_width).
    """
    seq_len = target_height * target_width

    if mask is None:
        return torch.ones((1, 1, seq_len), dtype=dtype, device=device)

    mask = to_standard_float_mask(mask, out_dtype=dtype)

    tf = torchvision.transforms.Resize(
        (target_height, target_width),
        interpolation=torchvision.transforms.InterpolationMode.NEAREST_EXACT,
    )

    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 3:
        mask = mask.unsqueeze(0)

    resized_mask = tf(mask)
    return resized_mask.flatten(start_dim=2).to(device=device)
