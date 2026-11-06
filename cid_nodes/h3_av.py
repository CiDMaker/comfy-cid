"""Shared constants and AV latent helpers for the MiniMax H3 nodes (keyframes, prefix, save/load, decode).

MiniMax H3 latents are ``comfy.nested_tensor.NestedTensor`` pairs:

* video ``[B, 24, T, H/16, W/16]``; latent frame ``k`` covers
  ``FRAME_PER_TOKEN[k % 5]`` pixel frames, so ``T = 5k + 2`` tokens cover
  ``17k + 5`` frames (2 -> 5, 7 -> 22, 57 -> 192)
* audio ``[B, 32, 2, Ta]`` at 40 latent ticks per second, i.e.
  ``FRAME_RESCALE = 5/3`` ticks per pixel frame

A latent dict may also carry ``noise_mask``: a NestedTensor pair
``(video_mask, audio_mask)`` built by the core Concat AV Latent node, or a
single tensor on a non-nested latent. Masks keep whatever shape their maker
gave them until sampling resizes them (``comfy.utils.reshape_mask``), so the
audio mask of the upscale subgraph is a 64x64 SolidMask, not latent-shaped.

Everything that touches tensors imports torch lazily; the frame/tick math is
plain Python so it can be tested anywhere.
"""

import math

CATEGORY = "CiD/video"

FPS = 24
AUDIO_LATENT_FPS = 40
FRAME_RESCALE = 5.0 / 3.0            # audio latent ticks per pixel frame
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)    # comfy.ldm.minimax.model.FRAME_PER_TOKEN
VAE_DOWNSAMPLE = 16
VIDEO_CHANNELS = 24
AUDIO_CHANNELS = 32


# --- frame / token / tick math -------------------------------------------

def frames_from_tokens(tokens):
    """Pixel frames covered by the first ``tokens`` latent frames (7 -> 22, 57 -> 192)."""
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(tokens))


def tokens_from_frames(frames):
    """Latent frames for an aligned (17k+5) frame count; the core ``video_latent_t``."""
    return 2 if frames <= 5 else ((frames - 5) // 17) * 5 + 2


def is_valid_prefix_tokens(tokens):
    """A prefix of P tokens taken from the *end* of a 5k+2 latent keeps the
    (1,4,4,4,4) pattern at the head of the new latent only when
    (T_prev - P) % 5 == 0, i.e. P = 5m + 2 (2, 7, 12, ...)."""
    return tokens >= 2 and tokens % 5 == 2


def validate_prefix_tokens(tokens):
    if is_valid_prefix_tokens(tokens):
        return tokens
    lower = max(2, ((tokens - 2) // 5) * 5 + 2)
    upper = lower + 5
    raise ValueError(
        "prefix_tokens must be 5m+2 (2, 7, 12, ...), got {}; use {} ({} frames) or {} ({} frames)".format(
            tokens, lower, frames_from_tokens(lower), upper, frames_from_tokens(upper)))


def audio_prefix_ticks(frames):
    """Audio latent ticks that cover ``frames`` pixel frames, rounded up
    (22 -> 37, 5 -> 9, 39 -> 65). 37 for the default 7-token prefix is the
    value validated by the K/L runs."""
    return int(math.ceil(frames * FRAME_RESCALE - 1e-6))


def audio_total_ticks(frames):
    """Audio latent length the core gives a ``frames``-frame latent (``temporal_shape``)."""
    return int(round(frames / FPS * AUDIO_LATENT_FPS))


# --- AV latent dict helpers (torch / comfy imported lazily) --------------

def is_nested(t):
    return bool(getattr(t, "is_nested", False))


def av_tensors(samples):
    """(video, audio) of an AV NestedTensor; raises for anything else."""
    if not is_nested(samples) or len(samples.tensors) != 2:
        raise ValueError("expected a MiniMax H3 AV latent (nested video+audio), got {}".format(
            type(samples).__name__ if not is_nested(samples) else "{} streams".format(len(samples.tensors))))
    video, audio = samples.tensors
    if video.ndim != 5 or video.shape[1] != VIDEO_CHANNELS:
        raise ValueError("expected a video latent [B, 24, T, H/16, W/16], got shape {}".format(tuple(video.shape)))
    if audio.ndim != 4 or audio.shape[1] != AUDIO_CHANNELS:
        raise ValueError("expected an audio latent [B, 32, 2, Ta], got shape {}".format(tuple(audio.shape)))
    return video, audio


def split_av(latent):
    """Split an AV latent dict into its parts.

    Returns ``(video, audio, video_mask, audio_mask)``; the masks are ``None``
    when the dict has no ``noise_mask``. A nested mask is unbound into its two
    streams; a plain tensor mask is taken as the video mask.
    """
    video, audio = av_tensors(latent["samples"])
    video_mask = audio_mask = None
    mask = latent.get("noise_mask")
    if mask is not None:
        if is_nested(mask):
            parts = mask.unbind()
            video_mask = parts[0]
            audio_mask = parts[1] if len(parts) > 1 else None
        else:
            video_mask = mask
    return video, audio, video_mask, audio_mask


def join_av(video, audio, video_mask=None, audio_mask=None, base=None):
    """Build an AV latent dict from its parts.

    ``base`` (a latent dict) supplies the other keys (batch_index, ...). When
    either mask is given the result carries a nested ``noise_mask`` with the
    missing stream filled with ones, exactly like the core Concat AV Latent
    node; with no masks the key is dropped.
    """
    import torch
    import comfy.nested_tensor

    out = dict(base) if base else {}
    out["samples"] = comfy.nested_tensor.NestedTensor((video, audio))
    if video_mask is None and audio_mask is None:
        out.pop("noise_mask", None)
    else:
        if video_mask is None:
            video_mask = torch.ones_like(video)
        if audio_mask is None:
            audio_mask = torch.ones_like(audio)
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
    return out


def video_geometry(video):
    """(frames, tokens, width, height) of a video latent tensor."""
    tokens = video.shape[2]
    return frames_from_tokens(tokens), tokens, video.shape[-1] * VAE_DOWNSAMPLE, video.shape[-2] * VAE_DOWNSAMPLE
