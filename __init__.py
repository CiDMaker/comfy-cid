"""comfy-cid

ComfyUI extension with two parts:

* ./js  - frontend-only "Export (API, normalized)" command that re-numbers
          the API export's nodes to "1".."N" and rewrites link references.
* ./cid_nodes - backend nodes that replace subgraphs the author kept
          rebuilding: a combined model/CLIP/2xVAE loader, a text encoder
          unload barrier, a duration->frames node, an aspect-ratio/megapixel
          size calculator, an H3 keyframe resizer, the H3 segment-chain nodes and
          S3 (Garage) upload nodes.
"""

from .cid_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
