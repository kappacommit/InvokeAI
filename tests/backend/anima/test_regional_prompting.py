"""Tests for Anima regional prompting (attention couple)."""

import pytest
import torch

from invokeai.backend.anima.anima_transformer import CosmosAttention
from invokeai.backend.anima.anima_transformer_patch import patch_anima_for_attention_couple
from invokeai.backend.anima.conditioning_data import AnimaRegionalTextConditioning
from invokeai.backend.anima.regional_prompting import (
    AnimaAttentionCoupleExtension,
    build_regional_blend_weights,
    preprocess_regional_prompt_mask,
)
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import Range

DEVICE = torch.device("cpu")


def _mask(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).view(1, 1, -1)


class TestBuildRegionalBlendWeights:
    def test_disjoint_regions_with_base(self):
        # 4 tokens: region A covers 0-1, region B covers 2, token 3 is uncovered.
        weights = build_regional_blend_weights(
            [None, _mask([1, 1, 0, 0]), _mask([0, 0, 1, 0])], seq_len=4, device=DEVICE
        )
        expected = torch.tensor(
            [
                [0, 0, 0, 1],  # base: uncovered area only
                [1, 1, 0, 0],
                [0, 0, 1, 0],
            ],
            dtype=torch.float32,
        )
        assert torch.allclose(weights, expected)

    def test_weights_are_convex(self):
        weights = build_regional_blend_weights(
            [None, _mask([1, 0.5, 0, 0]), _mask([0, 0.5, 0.25, 0])], seq_len=4, device=DEVICE
        )
        assert torch.allclose(weights.sum(dim=0), torch.ones(4))
        assert (weights >= 0).all()

    def test_overlapping_regions_share_proportionally(self):
        weights = build_regional_blend_weights([_mask([1, 1]), _mask([1, 0])], seq_len=2, device=DEVICE)
        expected = torch.tensor([[0.5, 1.0], [0.5, 0.0]], dtype=torch.float32)
        assert torch.allclose(weights, expected)

    def test_uncovered_without_base_is_uniform(self):
        weights = build_regional_blend_weights([_mask([1, 0]), _mask([0, 0])], seq_len=2, device=DEVICE)
        expected = torch.tensor([[1.0, 0.5], [0.0, 0.5]], dtype=torch.float32)
        assert torch.allclose(weights, expected)

    def test_multiple_base_slots_share_uncovered(self):
        weights = build_regional_blend_weights([None, None, _mask([1, 0])], seq_len=2, device=DEVICE)
        expected = torch.tensor([[0.0, 0.5], [0.0, 0.5], [1.0, 0.0]], dtype=torch.float32)
        assert torch.allclose(weights, expected)

    def test_base_does_not_dilute_covered_tokens(self):
        # Tokens fully covered by a region get zero base weight.
        weights = build_regional_blend_weights([None, _mask([1, 1])], seq_len=2, device=DEVICE)
        assert torch.allclose(weights[0], torch.zeros(2))

    def test_wrong_seq_len_raises(self):
        with pytest.raises(ValueError, match="tokens"):
            build_regional_blend_weights([_mask([1, 0, 0])], seq_len=4, device=DEVICE)


class TestPreprocessRegionalPromptMask:
    def test_none_mask_returns_ones(self):
        mask = preprocess_regional_prompt_mask(None, 4, 6, torch.float32, DEVICE)
        assert mask.shape == (1, 1, 24)
        assert torch.all(mask == 1.0)

    def test_resizes_to_target_grid(self):
        raw = torch.zeros(8, 8)
        raw[:, 4:] = 1.0  # right half
        mask = preprocess_regional_prompt_mask(raw, 4, 4, torch.float32, DEVICE)
        assert mask.shape == (1, 1, 16)
        grid = mask.view(4, 4)
        assert torch.all(grid[:, :2] == 0.0)
        assert torch.all(grid[:, 2:] == 1.0)

    def test_odd_target_grid(self):
        # ceil-division token grids can be odd; the mask must match exactly.
        raw = torch.ones(130, 128)
        mask = preprocess_regional_prompt_mask(raw, 65, 64, torch.float32, DEVICE)
        assert mask.shape == (1, 1, 65 * 64)


class TestAnimaAttentionCoupleExtension:
    def test_from_regional_conditioning_stacks_slots(self):
        ctx_dim = 8
        embeds = torch.arange(2 * 4 * ctx_dim, dtype=torch.float32).view(8, ctx_dim)
        regional = AnimaRegionalTextConditioning(
            context_embeds=embeds,
            image_masks=[None, _mask([1, 1, 0, 0])],
            context_ranges=[Range(start=0, end=4), Range(start=4, end=8)],
        )
        ext = AnimaAttentionCoupleExtension.from_regional_conditioning(
            regional, img_seq_len=4, device=DEVICE, dtype=torch.float32
        )
        assert ext.slot_contexts.shape == (2, 4, ctx_dim)
        assert torch.allclose(ext.slot_contexts[0], embeds[:4])
        assert torch.allclose(ext.slot_contexts[1], embeds[4:])
        assert ext.token_weights.shape == (2, 4, 1)
        assert not ext.coupling_enabled

    def test_unequal_slot_lengths_raise(self):
        embeds = torch.zeros(7, 8)
        regional = AnimaRegionalTextConditioning(
            context_embeds=embeds,
            image_masks=[None, _mask([1, 0])],
            context_ranges=[Range(start=0, end=4), Range(start=4, end=7)],
        )
        with pytest.raises(ValueError, match="equal-length"):
            AnimaAttentionCoupleExtension.from_regional_conditioning(
                regional, img_seq_len=2, device=DEVICE, dtype=torch.float32
            )


class _FakeBlock:
    def __init__(self, cross_attn: CosmosAttention):
        self.cross_attn = cross_attn


class _FakeTransformer:
    def __init__(self, blocks: list[_FakeBlock]):
        self.blocks = blocks


def _make_coupling(num_slots: int, img_seq_len: int, ctx_len: int, ctx_dim: int) -> AnimaAttentionCoupleExtension:
    slot_contexts = torch.randn(num_slots, ctx_len, ctx_dim)
    masks = []
    for i in range(num_slots):
        m = torch.zeros(img_seq_len)
        m[i::num_slots] = 1.0  # interleaved coverage, fully covering the grid
        masks.append(m.view(1, 1, -1))
    token_weights = build_regional_blend_weights(masks, img_seq_len, DEVICE).unsqueeze(-1)
    return AnimaAttentionCoupleExtension(slot_contexts=slot_contexts, token_weights=token_weights)


class TestAttentionCouplePatch:
    @torch.no_grad()
    def test_disabled_coupling_is_passthrough(self):
        torch.manual_seed(0)
        attn = CosmosAttention(query_dim=16, context_dim=8, n_heads=2, head_dim=8)
        transformer = _FakeTransformer([_FakeBlock(attn)])
        coupling = _make_coupling(num_slots=2, img_seq_len=6, ctx_len=4, ctx_dim=8)

        x = torch.randn(1, 6, 16)
        context = torch.randn(1, 4, 8)
        expected = attn.forward(x, context)

        with patch_anima_for_attention_couple(transformer, coupling):
            assert coupling.coupling_enabled is False
            result = transformer.blocks[0].cross_attn.forward(x, context)

        assert torch.allclose(result, expected)

    @torch.no_grad()
    def test_enabled_coupling_blends_per_slot_outputs(self):
        torch.manual_seed(0)
        attn = CosmosAttention(query_dim=16, context_dim=8, n_heads=2, head_dim=8)
        transformer = _FakeTransformer([_FakeBlock(attn)])
        coupling = _make_coupling(num_slots=2, img_seq_len=6, ctx_len=4, ctx_dim=8)

        x = torch.randn(1, 6, 16)
        nominal_context = torch.randn(1, 4, 8)

        # Expected: per-slot unmasked attention outputs, blended by token weights.
        per_slot = torch.cat(
            [attn.forward(x, coupling.slot_contexts[i : i + 1]) for i in range(2)],
            dim=0,
        )
        expected = (per_slot * coupling.token_weights).sum(dim=0, keepdim=True)

        with patch_anima_for_attention_couple(transformer, coupling):
            coupling.coupling_enabled = True
            result = transformer.blocks[0].cross_attn.forward(x, nominal_context)

        assert result.shape == (1, 6, 16)
        assert torch.allclose(result, expected, atol=1e-6)

    @torch.no_grad()
    def test_self_attention_call_is_passthrough(self):
        torch.manual_seed(0)
        attn = CosmosAttention(query_dim=16, context_dim=None, n_heads=2, head_dim=8)
        transformer = _FakeTransformer([_FakeBlock(attn)])
        coupling = _make_coupling(num_slots=2, img_seq_len=6, ctx_len=4, ctx_dim=16)

        x = torch.randn(1, 6, 16)
        expected = attn.forward(x, None)

        with patch_anima_for_attention_couple(transformer, coupling):
            coupling.coupling_enabled = True
            result = transformer.blocks[0].cross_attn.forward(x, None)

        assert torch.allclose(result, expected)

    @torch.no_grad()
    def test_geometry_mismatch_bypasses_coupling(self):
        torch.manual_seed(0)
        attn = CosmosAttention(query_dim=16, context_dim=8, n_heads=2, head_dim=8)
        transformer = _FakeTransformer([_FakeBlock(attn)])
        coupling = _make_coupling(num_slots=2, img_seq_len=6, ctx_len=4, ctx_dim=8)

        # Query length 5 != img_seq_len 6 -> bypass.
        x = torch.randn(1, 5, 16)
        context = torch.randn(1, 4, 8)
        expected = attn.forward(x, context)

        with patch_anima_for_attention_couple(transformer, coupling):
            coupling.coupling_enabled = True
            result = transformer.blocks[0].cross_attn.forward(x, context)

        assert torch.allclose(result, expected)

    @torch.no_grad()
    def test_patch_restores_original_forward(self):
        attn = CosmosAttention(query_dim=16, context_dim=8, n_heads=2, head_dim=8)
        transformer = _FakeTransformer([_FakeBlock(attn)])
        coupling = _make_coupling(num_slots=2, img_seq_len=6, ctx_len=4, ctx_dim=8)

        original_forward = attn.forward
        with patch_anima_for_attention_couple(transformer, coupling):
            assert attn.forward is not original_forward
        assert attn.forward == original_forward

    @torch.no_grad()
    def test_none_coupling_is_noop(self):
        attn = CosmosAttention(query_dim=16, context_dim=8, n_heads=2, head_dim=8)
        transformer = _FakeTransformer([_FakeBlock(attn)])
        original_forward = attn.forward
        with patch_anima_for_attention_couple(transformer, None):
            assert attn.forward == original_forward
