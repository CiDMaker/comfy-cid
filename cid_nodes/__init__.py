"""Backend nodes for comfy-cid.

Each module here is one node class that replaces a subgraph the author kept
rebuilding by hand. The math nodes have no ComfyUI imports and can be
unit-tested standalone; the loader delegates to the core loader classes so
its combo lists and loading behaviour stay in sync with ComfyUI itself.
"""

from .loader import CiDModelLoader
from .size_params import CiDSizeParams
from .video_frames import CiDVideoFrames

NODE_CLASS_MAPPINGS = {
    "CiD_ModelLoader": CiDModelLoader,
    "CiD_VideoFrames": CiDVideoFrames,
    "CiD_SizeParams": CiDSizeParams,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CiD_ModelLoader": "Model Loader (UNET + CLIP + 2 VAE) [CiD]",
    "CiD_VideoFrames": "Video Frames from Duration [CiD]",
    "CiD_SizeParams": "Size Parameters [CiD]",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
