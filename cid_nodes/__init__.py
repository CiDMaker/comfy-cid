"""Backend nodes for comfy-cid.

The ``h3_*`` modules share ``h3_av`` (MiniMax H3 AV latent helpers); the
segment-chaining nodes (prefix, save/load, decode & save) are designed in
docs/h3-chain-plan.md.

Each module here is one node class that replaces a subgraph the author kept
rebuilding by hand. The math nodes have no ComfyUI imports and can be
unit-tested standalone, and the H3 keyframe node keeps its torch and
ComfyUI imports inside the execute path for the same reason, as does the
text encoder unload node; the loader delegates to the core loader classes
so its combo lists and loading behaviour stay in sync with ComfyUI itself.
"""

from .clip_unload import CiDTextEncoderUnload
from .h3_decode_save import CiDH3DecodeSave
from .h3_keyframes import CiDH3ResizeKeyframes
from .h3_latent_io import CiDH3LoadAVLatent, CiDH3SaveAVLatent, CiDH3SaveAVLatentPassthrough
from .h3_prefix import CiDH3LatentPrefix
from .loader import CiDModelLoader
from .size_params import CiDSizeParams
from .video_frames import CiDVideoFrames

NODE_CLASS_MAPPINGS = {
    "CiD_ModelLoader": CiDModelLoader,
    "CiD_TextEncoderUnload": CiDTextEncoderUnload,
    "CiD_VideoFrames": CiDVideoFrames,
    "CiD_SizeParams": CiDSizeParams,
    "CiD_H3ResizeKeyframes": CiDH3ResizeKeyframes,
    "CiD_H3LatentPrefix": CiDH3LatentPrefix,
    "CiD_H3SaveAVLatent": CiDH3SaveAVLatent,
    "CiD_H3SaveAVLatentPassthrough": CiDH3SaveAVLatentPassthrough,
    "CiD_H3LoadAVLatent": CiDH3LoadAVLatent,
    "CiD_H3DecodeSave": CiDH3DecodeSave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CiD_ModelLoader": "Model Loader (UNET + CLIP + 2 VAE) [CiD]",
    "CiD_TextEncoderUnload": "Text Encoder Unload [CiD]",
    "CiD_VideoFrames": "Video Frames from Duration [CiD]",
    "CiD_SizeParams": "Size Parameters [CiD]",
    "CiD_H3ResizeKeyframes": "Resize H3 Keyframes to Latent [CiD]",
    "CiD_H3LatentPrefix": "H3 Latent Prefix [CiD]",
    "CiD_H3SaveAVLatent": "H3 Save AV Latent [CiD]",
    "CiD_H3SaveAVLatentPassthrough": "H3 Save AV Latent (passthrough) [CiD]",
    "CiD_H3LoadAVLatent": "H3 Load AV Latent [CiD]",
    "CiD_H3DecodeSave": "H3 Decode & Save [CiD]",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
