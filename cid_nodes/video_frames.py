"""Frame count from a duration in seconds.

Replaces the "Video Duration" subgraph, which held two expressions with
a = duration, b = fps:

    default : floor(a * b + 1)
    h3      : max(5, round(a * b)) + (5 - (max(5, round(a * b)) % 17)) % 17

"default" is the plain frame count for most video models: one frame per
1/fps plus the first frame. "h3" rounds duration*fps to an integer, clamps
it to at least 5, then bumps it up to the next value congruent to 5 modulo
17 (5, 22, 39, 56, ...). Its three constants are exposed as advanced inputs
with those defaults; they are ignored in default mode.

``round`` is Python's built-in, as in ComfyMathExpression, so exact .5
products round to the even neighbour.
"""

import math

CATEGORY = "CiD/video"

MODES = ["default", "h3"]


def frames_default(duration, fps):
    return math.floor(duration * fps + 1)


def frames_h3(duration, fps, frame_step=17, frame_offset=5, min_frames=5):
    n = max(min_frames, round(duration * fps))
    return n + (frame_offset - n % frame_step) % frame_step


def frames_for_duration(duration, fps, mode="default", frame_step=17, frame_offset=5, min_frames=5):
    if mode == "h3":
        return frames_h3(duration, fps, frame_step, frame_offset, min_frames)
    if mode == "default":
        return frames_default(duration, fps)
    raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")


class CiDVideoFrames:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "duration": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 3600.0, "step": 0.01,
                                       "tooltip": "Seconds"}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 240}),
                "mode": (MODES, {"default": "default",
                                 "tooltip": "default: floor(duration*fps + 1). "
                                            "h3: round(duration*fps), at least min_frames, "
                                            "rounded up to frame_offset + k*frame_step"}),
            },
            "optional": {
                "frame_step": ("INT", {"default": 17, "min": 1, "max": 1024, "advanced": True,
                                       "tooltip": "h3 mode only"}),
                "frame_offset": ("INT", {"default": 5, "min": 0, "max": 1024, "advanced": True,
                                         "tooltip": "h3 mode only"}),
                "min_frames": ("INT", {"default": 5, "min": 1, "max": 100000, "advanced": True,
                                       "tooltip": "h3 mode only"}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "compute"
    CATEGORY = CATEGORY
    DESCRIPTION = ("default: floor(duration*fps + 1). "
                   "h3: round(duration*fps), at least min_frames, rounded up to frame_offset + k*frame_step.")

    def compute(self, duration, fps, mode="default", frame_step=17, frame_offset=5, min_frames=5):
        return (frames_for_duration(duration, fps, mode, frame_step, frame_offset, min_frames),)
