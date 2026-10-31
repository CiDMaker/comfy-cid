"""Width/height from aspect ratio and megapixels, plus a scaled pair.

Replaces the "H3 Parameters" subgraph. Expressions, with a = ratio_width,
b = ratio_height, c = megapixels, d = multiple_of, s = scale_by:

    width           = int((c * a / b) ** 0.5 * 1024 / d + 0.5) * d
    height          = int((c * b / a) ** 0.5 * 1024 / d + 0.5) * d
    upscaled_width  = int(width  * s / d + 0.5) * d
    upscaled_height = int(height * s / d + 0.5) * d

So width * height is about ``megapixels * 1024 * 1024`` pixels and both
sides are rounded half-up to a multiple of ``multiple_of``. scale_by and
multiple_of are passed through so one node can drive the rest of the
graph. Works for still images as well as video, so there is no duration
here; use ``CiD_VideoFrames`` for that.

Names follow core nodes where one exists: ``megapixels`` from
ImageScaleToTotalPixels, ``scale_by`` from ImageScaleBy / LatentUpscaleBy,
``width`` / ``height`` outputs as in GetImageSize.

Note: in the original subgraph the "height scaled" expression was wired to
the *width* output by mistake, so scaled_h equalled scaled_w. This node
computes it from height, which is what the title said it did.
"""

CATEGORY = "CiD/image"


def _snap(value, multiple_of):
    return int(value / multiple_of + 0.5) * multiple_of


def size_params(ratio_width, ratio_height, megapixels, multiple_of, scale_by):
    width = _snap((megapixels * ratio_width / ratio_height) ** 0.5 * 1024, multiple_of)
    height = _snap((megapixels * ratio_height / ratio_width) ** 0.5 * 1024, multiple_of)
    upscaled_width = _snap(width * scale_by, multiple_of)
    upscaled_height = _snap(height * scale_by, multiple_of)
    return width, height, upscaled_width, upscaled_height


class CiDSizeParams:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ratio_width": ("INT", {"default": 1, "min": 1, "max": 4096,
                                        "tooltip": "Aspect ratio numerator, e.g. 9 for 9:16"}),
                "ratio_height": ("INT", {"default": 1, "min": 1, "max": 4096,
                                         "tooltip": "Aspect ratio denominator, e.g. 16 for 9:16"}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 64.0, "step": 0.01,
                                         "tooltip": "Target width*height in units of 1024*1024 pixels"}),
                "multiple_of": ("INT", {"default": 16, "min": 1, "max": 1024,
                                        "tooltip": "Round every side to a multiple of this"}),
                "scale_by": ("FLOAT", {"default": 2.0, "min": 0.01, "max": 16.0, "step": 0.01,
                                       "tooltip": "Factor for upscaled_width / upscaled_height; passed through"}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "FLOAT", "INT")
    RETURN_NAMES = ("width", "height", "upscaled_width", "upscaled_height", "scale_by", "multiple_of")
    FUNCTION = "compute"
    CATEGORY = CATEGORY
    DESCRIPTION = "Width/height for an aspect ratio at a megapixel budget, snapped to multiple_of, plus an upscaled pair."

    def compute(self, ratio_width, ratio_height, megapixels, multiple_of, scale_by):
        width, height, upscaled_width, upscaled_height = size_params(
            ratio_width, ratio_height, megapixels, multiple_of, scale_by)
        return (width, height, upscaled_width, upscaled_height, scale_by, multiple_of)
