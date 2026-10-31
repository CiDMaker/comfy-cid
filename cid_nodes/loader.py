"""One node that loads a diffusion model, a text encoder and two VAEs.

Replaces the "Loader" subgraph (UNETLoader + CLIPLoader + 2 x VAELoader).
Built for video models with separate video and audio VAEs, e.g. MiniMax
H3, hence the VAE names; nothing else here is model specific.

Naming follows the core loaders: inputs reuse the core widget names
(``unet_name``, ``weight_dtype``, ``clip_name``), CLIP-only options get a
``clip_`` prefix to disambiguate, and outputs are upper-case like
CheckpointLoaderSimple's ``MODEL`` / ``CLIP`` / ``VAE``.

The node does not load anything itself. Combo lists are read from the core
loader classes' INPUT_TYPES at graph-build time and the actual loading is
delegated to those classes, so new CLIP types, VAE approximations, fp8 modes
and so on appear here as soon as ComfyUI adds them.
"""

import nodes  # ComfyUI core; available to custom nodes at import time

CATEGORY = "CiD/loaders"

_PREFERRED_CLIP_TYPE = "minimax"


def _combo(spec, default=None, tooltip=None):
    """Rebuild a core combo spec as a plain (choices, options) tuple.

    Core specs are either ``(choices,)`` or ``(choices, options)``. Options
    such as ``{"advanced": True}`` are dropped on purpose: in this node every
    input is a first-class control, the way the subgraph exposed them.
    """
    choices = list(spec[0])
    options = {}
    if default is not None and default in choices:
        options["default"] = default
    if tooltip:
        options["tooltip"] = tooltip
    return (choices, options)


class CiDModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        unet = nodes.UNETLoader.INPUT_TYPES()["required"]
        clip = nodes.CLIPLoader.INPUT_TYPES()
        vae = nodes.VAELoader.INPUT_TYPES()["required"]
        return {
            "required": {
                "unet_name": _combo(unet["unet_name"]),
                "weight_dtype": _combo(unet["weight_dtype"]),
                "clip_name": _combo(clip["required"]["clip_name"]),
                "clip_type": _combo(clip["required"]["type"], default=_PREFERRED_CLIP_TYPE),
                "clip_device": _combo(clip["optional"]["device"]),
                "video_vae_name": _combo(vae["vae_name"]),
                "audio_vae_name": _combo(vae["vae_name"]),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "VAE")
    RETURN_NAMES = ("MODEL", "CLIP", "VIDEO_VAE", "AUDIO_VAE")
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "UNETLoader + CLIPLoader + two VAELoaders in one node. "
        "Changing any input reloads all four models; keep separate core "
        "loaders if you swap one piece often."
    )

    def load(self, unet_name, weight_dtype, clip_name, clip_type, clip_device, video_vae_name, audio_vae_name):
        (model,) = nodes.UNETLoader().load_unet(unet_name, weight_dtype)
        (clip,) = nodes.CLIPLoader().load_clip(clip_name, type=clip_type, device=clip_device)
        vae_loader = nodes.VAELoader()
        (video_vae,) = vae_loader.load_vae(video_vae_name)
        (audio_vae,) = vae_loader.load_vae(audio_vae_name)
        return (model, clip, video_vae, audio_vae)
