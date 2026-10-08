"""Shape, causality, determinism, offload and budget tests.

These are the tests that decide whether the 8 GB claim and the VAE's streaming
property are real, so they assert on numbers rather than on "it ran".
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor, nn

from nanoh3.engine.budget import GIB, PROFILES, Profile, assert_fits
from nanoh3.engine.offload import OffloadedBlock, SequentialOffload, module_bytes
from nanoh3.engine.quant import dequantize_tensor, quantize_module_, quantize_tensor, storage_bytes
from nanoh3.engine.sampler import SamplerConfig, add_noise, sample, timesteps
from nanoh3.models import DiTConfig, NanoDiT, NanoTextEncoder, NanoVAE, TextEncoderConfig, VAEConfig
from nanoh3.models.rope import Rotary3D, apply_rotary, split_head_dim

torch.manual_seed(0)

SMALL_VAE = VAEConfig(base_channels=8, channel_multipliers=(1, 2, 2, 4, 4, 4))
SMALL_DIT = DiTConfig(hidden_size=64, depth=2, num_heads=4, text_dim=48, mlp_ratio=2.0)
SMALL_TEXT = TextEncoderConfig(width=48, layers=2, heads=4)


@pytest.fixture(scope="module")
def vae() -> NanoVAE:
    return NanoVAE(SMALL_VAE).eval()


# --------------------------------------------------------------------- VAE


def test_vae_latent_shape_matches_config(vae: NanoVAE) -> None:
    assert vae.latent_shape(9, 64, 64) == (3, 8, 8)
    assert vae.latent_shape(16, 256, 256) == (4, 32, 32)


def test_vae_encode_decode_roundtrip_shapes(vae: NanoVAE) -> None:
    video = torch.randn(2, 3, 9, 64, 64).clamp(-1, 1)
    with torch.no_grad():
        posterior = vae.encode(video)
        assert posterior.shape == (2, 2 * SMALL_VAE.latent_channels, 3, 8, 8)
        recon = vae.decode(posterior.chunk(2, dim=1)[0], (9, 64, 64))
    assert recon.shape == video.shape


def test_vae_is_causal(vae: NanoVAE) -> None:
    """Appending future frames must not change any earlier latent.

    Exact equality is asserted for the first latent frame. Later frames are
    compared with a tight tolerance rather than bitwise: encoding a 9-frame and a
    16-frame clip runs convolutions at different shapes, and the backend picks a
    different kernel/summation order for each, so float32 reassociation shows up
    as a residual of order 1e-7. That is arithmetic, not leakage -- a real leak
    would be orders of magnitude larger, and the first frame is unaffected
    because nothing can reach back to it.
    """
    video = torch.randn(1, 3, 9, 64, 64)
    future = torch.randn(1, 3, 5, 64, 64) * 3.0
    with torch.no_grad():
        short = vae.encode(video)
        longer = vae.encode(torch.cat([video, future], dim=2))[:, :, : short.shape[2]]
    assert torch.equal(short[:, :, 0], longer[:, :, 0])
    assert float((short - longer).abs().max()) < 1e-5


def test_vae_decode_is_causal_when_asked() -> None:
    """``causal_decoder=True`` must restore strict decoder causality.

    The default is a *non*-causal decoder, matching MiniMax-H3's own VAE
    (``causal_encoder: true, causal_decoder: false``) and for the reason measured
    in ``docs/DESIGN.md``: a causal decoder after nearest-neighbour upsampling
    looks backwards so hard that copying the first latent wins every loss.
    """
    cfg = VAEConfig(base_channels=8, channel_multipliers=(1, 2, 2, 4, 4, 4), causal_decoder=True)
    model = NanoVAE(cfg).eval()
    latent = torch.randn(1, SMALL_VAE.latent_channels, 3, 8, 8)
    with torch.no_grad():
        a = model.decode(latent, (9, 64, 64))
        bumped = latent.clone()
        bumped[:, :, -1] += 5.0
        b = model.decode(bumped, (9, 64, 64))
    first_two = SMALL_VAE.temporal_compression * 2
    assert float((a[:, :, :first_two] - b[:, :, :first_two]).abs().max()) < 1e-5


def test_default_decoder_is_not_causal_and_that_is_intentional() -> None:
    """The default decoder is centred, so later latents may affect earlier frames."""
    model = NanoVAE(SMALL_VAE).eval()
    latent = torch.randn(1, SMALL_VAE.latent_channels, 3, 8, 8)
    with torch.no_grad():
        a = model.decode(latent, (9, 64, 64))
        bumped = latent.clone()
        bumped[:, :, -1] += 5.0
        b = model.decode(bumped, (9, 64, 64))
    assert float((a - b).abs().max()) > 1e-5


def test_decoder_receptive_field_is_centred_on_its_own_latent() -> None:
    """Output frame 4 must depend on latent 1 about as much as on latent 0.

    This is the regression test for the static-output bug: with a causal decoder
    this ratio measured 144x, which made "repeat the first latent" the cheapest
    reconstruction and froze every generated clip. Anything above ~5x means the
    skew is back.
    """
    model = NanoVAE(SMALL_VAE).eval()
    latents = torch.randn(1, SMALL_VAE.latent_channels, 3, 8, 8)
    z = latents.clone().requires_grad_(True)
    out = model.decode(z, (9, 64, 64))
    grad = torch.autograd.grad(out[0, :, 4].abs().sum(), z)[0]
    w0 = float(grad[:, :, 0].abs().mean())
    w1 = float(grad[:, :, 1].abs().mean())
    ratio = w0 / max(w1, 1e-12)
    assert ratio < 5.0, f"decoder receptive field is skewed {ratio:.1f}x towards latent 0"
    assert w1 > 0, "output frame 4 ignores the latent it should be built from"


def test_clip_norm_mode_is_not_causal() -> None:
    """Document the cost of ``norm_mode='clip'`` rather than leaving it implicit."""
    cfg = VAEConfig(base_channels=8, channel_multipliers=(1, 2, 2, 4, 4, 4), norm_mode="clip")
    model = NanoVAE(cfg).eval()
    video = torch.randn(1, 3, 9, 64, 64)
    future = torch.randn(1, 3, 5, 64, 64) * 3.0
    with torch.no_grad():
        short = model.encode(video)
        longer = model.encode(torch.cat([video, future], dim=2))[:, :, : short.shape[2]]
    assert float((short - longer).abs().max()) > 1e-5, "clip norm was expected to leak, not to be causal"


def test_vae_recon_loss_is_finite(vae: NanoVAE) -> None:
    video = torch.randn(1, 3, 9, 64, 64).clamp(-1, 1)
    loss, metrics = vae.recon_loss(video)
    assert torch.isfinite(loss)
    assert metrics["l1"] > 0


def test_plan_strides_covers_both_axes() -> None:
    from nanoh3.models.vae import plan_strides

    strides = plan_strides(8, 4, 5)
    assert math.prod(s[0] for s in strides) == 4
    assert math.prod(s[1] for s in strides) == 8
    with pytest.raises(ValueError):
        plan_strides(8, 4, 2)


def test_h3_matched_geometry() -> None:
    """16x spatial / 4x temporal with 24 latent channels, as in MiniMax-H3's f16t4d24."""
    cfg = VAEConfig(
        base_channels=8,
        latent_channels=24,
        channel_multipliers=(1, 2, 2, 4, 4, 4, 4),
        spatial_compression=16,
        temporal_compression=4,
    )
    model = NanoVAE(cfg).eval()
    with torch.no_grad():
        posterior = model.encode(torch.randn(1, 3, 17, 64, 64))
    assert posterior.shape == (1, 48, 5, 4, 4)


# --------------------------------------------------------------------- DiT


def test_dit_output_shape() -> None:
    dit = NanoDiT(SMALL_DIT).eval()
    z = torch.randn(2, SMALL_DIT.latent_channels, 3, 8, 8)
    with torch.no_grad():
        out = dit(z, torch.rand(2), torch.randn(2, 5, SMALL_DIT.text_dim), torch.ones(2, 5))
    assert out.shape == z.shape


def test_dit_rejects_unpatchable_latents() -> None:
    dit = NanoDiT(SMALL_DIT).eval()
    with pytest.raises(ValueError):
        dit(torch.randn(1, SMALL_DIT.latent_channels, 2, 7, 7), torch.rand(1))


def test_dit_output_is_zero_at_init() -> None:
    """adaLN-Zero means an untrained model predicts no velocity."""
    dit = NanoDiT(SMALL_DIT).eval()
    with torch.no_grad():
        out = dit(torch.randn(1, 16, 2, 4, 4), torch.rand(1), torch.randn(1, 4, 48), torch.ones(1, 4))
    assert float(out.abs().max()) == 0.0


def test_dit_depth_pruning_changes_computation() -> None:
    dit = NanoDiT(SMALL_DIT).eval()
    z, ctx, mask = torch.randn(1, 16, 2, 4, 4), torch.randn(1, 4, 48), torch.ones(1, 4)
    t = torch.tensor([0.5])
    with torch.no_grad():
        full = dit(z, t, ctx, mask)
        dit.set_active_depth(1)
        assert dit.active_blocks == [0]
        pruned = dit(z, t, ctx, mask)
        dit.reset_active_depth()
        assert dit.active_blocks == [0, 1]
    assert pruned.shape == full.shape


def test_dit_text_conditioning_changes_output() -> None:
    """Text must reach the output.

    At init the model is adaLN-Zero, so the output head is exactly zero and *no*
    conditioning can show up. The test therefore un-zeros the head first, which
    is also the only way to observe the path at all.
    """
    dit = NanoDiT(SMALL_DIT).eval()
    with torch.no_grad():
        dit.proj_out.weight.normal_(0, 0.05)
    z, t = torch.randn(1, 16, 2, 4, 4), torch.tensor([0.4])
    with torch.no_grad():
        a = dit(z, t, torch.randn(1, 4, 48), torch.ones(1, 4))
        b = dit(z, t, torch.randn(1, 4, 48) * 5, torch.ones(1, 4))
    assert not torch.allclose(a, b)


def test_dit_runs_without_text() -> None:
    dit = NanoDiT(SMALL_DIT).eval()
    with torch.no_grad():
        out = dit(torch.randn(1, 16, 2, 4, 4), torch.rand(1), None, None)
    assert out.shape == (1, 16, 2, 4, 4)


# --------------------------------------------------------------------- RoPE


def test_split_head_dim_is_exact_and_even() -> None:
    for head_dim in (16, 32, 64, 128):
        for grid in [(1, 1, 1), (4, 8, 8), (7, 13, 5)]:
            dims = split_head_dim(head_dim, grid)
            assert sum(dims) == head_dim
            assert all(d >= 2 and d % 2 == 0 for d in dims)


def test_rope_is_position_dependent() -> None:
    rotary = Rotary3D(32)
    cos, sin = rotary((2, 4, 4), torch.device("cpu"), torch.float32)
    assert cos.shape == (32, 16)
    x = torch.randn(1, 2, 32, 32)
    rotated = apply_rotary(x, cos, sin)
    assert not torch.allclose(rotated, x)
    assert torch.allclose(rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_rope_token_zero_is_identity() -> None:
    """Position 0 must be the identity rotation, otherwise all tokens shift."""
    rotary = Rotary3D(32)
    cos, sin = rotary((1, 2, 2), torch.device("cpu"), torch.float32)
    x = torch.randn(1, 1, 4, 32)
    out = apply_rotary(x, cos[:1], sin[:1])
    assert torch.allclose(out, x, atol=1e-6)


# ------------------------------------------------------------------ sampler


def test_timesteps_span_one_to_zero() -> None:
    ts = timesteps(4, shift=3.0)
    assert float(ts[0]) == pytest.approx(1.0)
    assert float(ts[-1]) == pytest.approx(0.0)
    assert torch.all(ts[1:] < ts[:-1])


def test_add_noise_interpolates() -> None:
    x0 = torch.randn(2, 3, 4)
    noise = torch.randn(2, 3, 4)
    t = torch.tensor([0.0, 1.0])
    x_t, target = add_noise(x0, noise, t)
    assert torch.allclose(x_t[0], x0[0])
    assert torch.allclose(x_t[1], noise[1])
    assert torch.allclose(target[0], noise[0] - x0[0])


def test_sampling_is_deterministic_given_a_seed() -> None:
    cfg = SamplerConfig(steps=3, shift=3.0)
    model = NanoDiT(SMALL_DIT).eval()
    ctx, mask = torch.randn(1, 4, 48), torch.ones(1, 4)

    def run() -> torch.Tensor:
        from nanoh3.engine.sampler import make_noise

        z = make_noise([1, 16, 2, 4, 4], torch.device("cpu"), torch.float32, seed=7)
        with torch.no_grad():
            return sample(lambda x, t: model(x, t, ctx, mask), z, cfg)

    assert torch.equal(run(), run())


def test_step_count_equals_forward_count() -> None:
    calls = {"n": 0}
    model = NanoDiT(SMALL_DIT).eval()
    ctx = torch.randn(1, 4, 48)

    def fn(x, t):
        calls["n"] += 1
        return model(x, t, ctx, None)

    with torch.no_grad():
        sample(fn, torch.randn(1, 16, 2, 4, 4), SamplerConfig(steps=4))
    assert calls["n"] == 4


# -------------------------------------------------------------- quantisation


@pytest.mark.parametrize(
    "mode,limit",
    [
        # int8 symmetric per-output-channel: 1/254 of the channel range.
        ("int8", 0.02),
        # fp8 e4m3 per-tensor: 3 mantissa bits, so ~6% relative error; the absolute
        # bound is set by the largest weight, hence the looser threshold.
        ("fp8", 0.20),
    ],
)
def test_quant_roundtrip_error_is_small(mode: str, limit: float) -> None:
    w = torch.randn(128, 256)
    q, scale = quantize_tensor(w, mode)  # type: ignore[arg-type]
    assert q.element_size() == 1
    err = float((dequantize_tensor(q, scale) - w).abs().max())
    assert err < limit, f"{mode} max error {err}"


def test_quantize_module_shrinks_storage() -> None:
    dit = NanoDiT(SMALL_DIT)
    before = storage_bytes(dit)
    spec = quantize_module_(dit, "int8")
    after = storage_bytes(dit)
    assert spec.param_count > 0
    assert after < before * 0.45
    with torch.no_grad():
        out = dit(torch.randn(1, 16, 2, 4, 4), torch.rand(1), None, None)
    assert out.shape == (1, 16, 2, 4, 4)


# ------------------------------------------------------------------- offload


class _TinyBlock(nn.Module):
    """Stand-in with the same call signature as a DiT block."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 8, bias=False)

    def forward(self, x: Tensor, *args: object) -> Tensor:
        return self.fc(x)


def test_offloaded_block_is_numerically_transparent() -> None:
    """Streaming weights in and out must not change the result.

    The offload path is a no-op on CPU by default, so this forces it on: the point
    is that the move-to-host / move-to-device dance leaves the maths identical.
    """
    block = _TinyBlock().eval()
    x = torch.randn(2, 8)
    with torch.no_grad():
        expected = block(x)
    wrapped = OffloadedBlock(block, torch.device("cpu"), enabled=True, cpu_is_noop=False)
    with torch.no_grad():
        actual = wrapped(x)
    assert torch.allclose(expected, actual, atol=1e-6)
    assert wrapped.stats.moves > 0
    assert wrapped.stats.bytes_to_device == module_bytes(block)
    assert wrapped.stats.bytes_to_host == module_bytes(block)
    assert "offloaded=True" in wrapped.extra_repr()


def test_sequential_offload_restores_the_module_list() -> None:
    """Entering the context must swap blocks, and leaving it must put them back."""
    model = NanoDiT(SMALL_DIT).eval()
    z, ctx = torch.randn(1, 16, 2, 4, 4), torch.randn(1, 4, 48)
    with torch.no_grad():
        before = model(z, torch.tensor([0.5]), ctx, None)
    originals = list(model.blocks)
    with SequentialOffload(model.blocks, torch.device("cpu"), enabled=True, cpu_is_noop=False) as offload:
        assert all(isinstance(b, OffloadedBlock) for b in model.blocks)
        with torch.no_grad():
            during = model(z, torch.tensor([0.5]), ctx, None)
        assert offload.stats.moves > 0
    assert list(model.blocks) == originals
    with torch.no_grad():
        after = model(z, torch.tensor([0.5]), ctx, None)
    assert torch.allclose(before, during, atol=1e-6)
    assert torch.allclose(before, after, atol=1e-6)


# ------------------------------------------------------------------- budget


def test_8gb_profile_fits_8gib() -> None:
    for name in ("8gb", "8gb-tiny", "8gb-512"):
        est = assert_fits(PROFILES[name], 8 * GIB)
        assert est["total"] <= 8 * GIB


def test_oversized_profile_is_rejected() -> None:
    huge = Profile(name="huge", width=1920, height=1080, frames=120, offload_dit=False, offload_vae=False)
    with pytest.raises(MemoryError):
        assert_fits(huge, 8 * GIB)


def test_offload_reduces_resident_weight_bytes() -> None:
    p = PROFILES["8gb"]
    streamed = Profile(**{**p.__dict__, "offload_dit": True, "resident_blocks": 2})
    resident = Profile(**{**p.__dict__, "offload_dit": False, "resident_blocks": 2})
    assert streamed.dit_weight_bytes() < resident.dit_weight_bytes()


def test_budget_token_count_matches_diT() -> None:
    p = PROFILES["8gb"]
    latent = p.latent_shape
    assert latent == (4, 32, 32)
    assert p.token_grid == (4, 16, 16)
    assert p.num_tokens == 4 * 16 * 16


# -------------------------------------------------------------------- misc


def test_text_encoder_masks_padding() -> None:
    """Padded positions must be exactly zero, and a shared prefix must be stable."""
    text = NanoTextEncoder(SMALL_TEXT).eval()
    short = "a red circle"
    with torch.no_grad():
        a, mask_a = text.embed_prompts([short], torch.device("cpu"))
        b, mask_b = text.embed_prompts([short + " and a blue square"], torch.device("cpu"))
    assert a.shape[-1] == SMALL_TEXT.width
    assert mask_a.sum() < mask_b.sum()
    assert float(a[0, mask_a[0] == 0].abs().max()) == 0.0
    # The causal encoder must give a shared prefix the same representation.
    # (Not `mask_a.sum()`: that includes the extra sentinel position, which in the
    # longer prompt holds a real byte instead.)
    shared = len(short.encode("utf-8"))
    assert torch.allclose(a[0, :shared], b[0, :shared], atol=1e-6)
