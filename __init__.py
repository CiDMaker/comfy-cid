"""comfy-normalizedexport

Frontend-only ComfyUI extension. It adds an "Export (API, normalized)"
command that re-numbers the nodes of the API-format export to "1".."N"
(in a user-selected order) and rewrites every link reference accordingly.

No canvas nodes are registered; the package exists only to serve ./js.
"""

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
