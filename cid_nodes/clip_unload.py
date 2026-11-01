"""Pass-through barrier that unloads the text encoder once encoding is done.

Sits between the CLIP Text Encode nodes and the sampler: it takes the CLIP
plus the encoded positive/negative conditioning, returns the conditioning
untouched, and on the way moves that one CLIP's weights off the GPU. Because
the conditioning inputs can only be ready after every text encode has run,
ComfyUI's dependency order guarantees the unload happens after encoding and
before anything downstream (the sampler) starts.

What "unload" means here: ``comfy.model_management.unload_model_and_clones``
detaches the CLIP's ModelPatcher (and its clones) to its offload device,
usually CPU RAM, then ``soft_empty_cache`` returns the freed blocks to the
driver. The CLIP object stays alive and cached; if a later run re-encodes a
changed prompt, ComfyUI moves the weights back to the GPU by itself. Nothing
is re-read from disk and no other model (diffusion model, VAEs, ControlNets)
is touched, so the combined loader's outputs stay cached as usual.

Cases where there is nothing to free are skipped instead of failing the run:

* load and offload device are both CPU (``clip_device = cpu`` on the loader):
  nothing is on the accelerator, silently skipped
* load and offload device are the same accelerator (ComfyUI started with
  ``--gpu-only``): the weights have nowhere to go, skipped with a warning

Only the ``negative`` input is optional. When it is left unconnected the
``negative`` output is the positive conditioning run through the core
ConditioningZeroOut node (zeroed embeddings, same shapes and metadata), so
samplers with a required negative such as KSampler keep working and the
workflow needs no separate Conditioning Zero Out node. Delegating to the
core class keeps the set of zeroed keys in sync with ComfyUI.

ComfyUI is imported lazily inside the execute path so the module and its
device logic stay testable without ComfyUI installed.
"""

import inspect
import logging

CATEGORY = "CiD/loaders"

UNLOAD = "unload"
SKIP_CPU = "skip_cpu"
SKIP_SAME_DEVICE = "skip_same_device"


def _is_cpu_like(device):
    name = str(device).lower()
    return name == "cpu" or name.startswith("meta")


def unload_action(load_device, offload_device):
    """Decide what to do for a patcher with these devices.

    UNLOAD when the weights can move from load_device to a different offload
    device; SKIP_CPU when nothing is on an accelerator; SKIP_SAME_DEVICE when
    both are the same accelerator (``--gpu-only``) so unloading frees nothing.
    """
    if load_device is None or offload_device is None:
        return UNLOAD
    if load_device != offload_device:
        return UNLOAD
    if _is_cpu_like(load_device):
        return SKIP_CPU
    return SKIP_SAME_DEVICE


def _unload_patcher(patcher):
    import comfy.model_management as mm

    unload = getattr(mm, "unload_model_and_clones", None)
    if not callable(unload):
        raise RuntimeError("Text Encoder Unload needs comfy.model_management.unload_model_and_clones; "
                           "update ComfyUI")
    try:
        if "all_devices" in inspect.signature(unload).parameters:
            unload(patcher, all_devices=True)
        else:
            unload(patcher)
    finally:
        mm.soft_empty_cache(force=True)


def zero_out(conditioning):
    """Zeroed copy of a conditioning, exactly what the core Conditioning Zero Out node produces."""
    import nodes  # ComfyUI core

    (zeroed,) = nodes.ConditioningZeroOut().zero_out(conditioning)
    return zeroed


def unload_clip(clip):
    """Unload one CLIP's patcher. Returns the action taken (UNLOAD / SKIP_*)."""
    patcher = getattr(clip, "patcher", None)
    if patcher is None:
        raise ValueError("the CLIP input has no .patcher; connect the CLIP output of a loader node")
    action = unload_action(getattr(patcher, "load_device", None), getattr(patcher, "offload_device", None))
    if action == SKIP_SAME_DEVICE:
        logging.warning("Text Encoder Unload: CLIP load and offload device are both %s, nothing to free "
                        "(ComfyUI started with --gpu-only?)", patcher.load_device)
    elif action == UNLOAD:
        _unload_patcher(patcher)
    return action


class CiDTextEncoderUnload:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "The text encoder to unload. Use the same CLIP output the "
                                             "text encode nodes above are connected to"}),
                "positive": ("CONDITIONING", {"tooltip": "Passed through unchanged"}),
            },
            "optional": {
                "negative": ("CONDITIONING", {"tooltip": "Passed through unchanged. Unconnected: the negative "
                                                         "output is the positive run through Conditioning "
                                                         "Zero Out"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    OUTPUT_TOOLTIPS = ("The positive input, unchanged. Wire the sampler here so it runs after the unload",
                       "The negative input, unchanged; a zeroed-out copy of positive when `negative` "
                       "is unconnected")
    FUNCTION = "unload"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Passes positive/negative conditioning through unchanged and moves the connected CLIP "
                   "off the GPU once all text encoding feeding it has finished. Wire the sampler to the "
                   "outputs so it starts with the text encoder's VRAM already freed. Without a negative "
                   "input the negative output is a zeroed-out copy of positive (as Conditioning Zero Out). "
                   "Only this CLIP is unloaded; the diffusion model and VAEs are not touched.")

    def unload(self, clip, positive, negative=None):
        if negative is None:
            negative = zero_out(positive)
        unload_clip(clip)
        return (positive, negative)
