"""Backend nodes for comfy-cid.

Each module here is one node class that replaces a subgraph the author kept
rebuilding by hand. The math nodes have no ComfyUI imports and can be
unit-tested standalone, and the H3 keyframe node keeps its torch and
ComfyUI imports inside the execute path for the same reason; the loader
delegates to the core loader classes so its combo lists and loading
behaviour stay in sync with ComfyUI itself.
"""

from .h3_keyframes import CiDH3ResizeKeyframes
from .loader import CiDModelLoader
from .size_params import CiDSizeParams
from .video_frames import CiDVideoFrames

NODE_CLASS_MAPPINGS = {
    "CiD_ModelLoader": CiDModelLoader,
    "CiD_VideoFrames": CiDVideoFrames,
    "CiD_SizeParams": CiDSizeParams,
    "CiD_H3ResizeKeyframes": CiDH3ResizeKeyframes,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CiD_ModelLoader": "Model Loader (UNET + CLIP + 2 VAE) [CiD]",
    "CiD_VideoFrames": "Video Frames from Duration [CiD]",
    "CiD_SizeParams": "Size Parameters [CiD]",
    "CiD_H3ResizeKeyframes": "Resize H3 Keyframes to Latent [CiD]",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
