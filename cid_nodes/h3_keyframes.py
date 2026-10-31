"""Rebuild MiniMax H3 keyframe conditioning at another latent size.

Replaces the "H3 FLF to Video - Pass2" subgraph, which re-ran
MiniMaxH3ImageToVideo at the upscaled width/height only to get keyframe
latents of the right size, paying for a second Qwen3-VL-32B pass on the way.

What MiniMaxH3ImageToVideo puts into the conditioning:

* the text/vision embedding tensor (prompt + first/last frame as <Picture>
  tokens); nothing about it is tied to the canvas size
* ``minimax_keyframes``: ``[{"resolved_frame_index", "latent"}]`` with the
  first frame stretched and the last frame center-cropped to the canvas,
  then VAE-encoded

The DiT packs the keyframe rows on the *target* latent's spatial grid
(``PackedLayout``: rows = keyframe T x target frame rows), so the keyframe
latents must have exactly the target latent's H/W. That is the only part of
the conditioning that has to change after a latent upscale, and this node
changes only that.

Width, height and frame count are read from the latent itself (pixel size =
latent size x 16; frame count from the 1,4,4,4,4 frames-per-token pattern),
so there is no separate size input that could drift from what the upscaler
actually produced.

Per keyframe:

* a connected ``first_frame`` / ``last_frame`` is resized with the core
  node's rules (first: stretch, last: center crop) and VAE-encoded at the
  new size, replacing the existing keyframe at frame 0 / the last frame
* every other keyframe (e.g. anchors from "Add Guide for MiniMax H3") keeps
  its index and audio and gets its video latent bilinearly interpolated to
  the new grid; already matching latents are left untouched

Torch and ComfyUI are imported lazily so the index logic stays testable
without them.
"""

CATEGORY = "CiD/video"

VAE_DOWNSAMPLE = 16
LATENT_CHANNELS = 24
# comfy.ldm.minimax.model.FRAME_PER_TOKEN: pixel frames covered by latent frame k (k % 5)
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)


def frame_count_from_latent_t(latent_t):
    """Pixel frames for a video latent with ``latent_t`` frames (2 -> 5, 37 -> 124)."""
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(latent_t))


def merge_keyframes(existing, replacements, frame_count):
    """Decide the new keyframe list without touching any tensors.

    ``existing`` is the conditioning's current ``minimax_keyframes`` list;
    ``replacements`` maps a resolved frame index to a freshly encoded video
    latent. Returns ``(keyframes, to_resize)``: the new list in order, and the
    indices into it whose ``latent`` still has to be resized by the caller.

    An existing keyframe at a replaced index swaps its video latent for the
    new one and keeps anything else it carried (``audio_latent``).
    Replacements that match no existing keyframe are appended, lowest frame
    index first.
    """
    keyframes = []
    to_resize = []
    used = set()
    for kf in existing:
        idx = kf.get("resolved_frame_index", 0)
        if idx < 0 or idx >= frame_count:
            raise ValueError(
                "keyframe at frame {} does not fit in the latent's {} frames; the conditioning "
                "and the latent were made for different durations".format(idx, frame_count))
        new = dict(kf)
        if idx in replacements:
            new["latent"] = replacements[idx]
            used.add(idx)
        elif new.get("latent") is not None:
            to_resize.append(len(keyframes))
        keyframes.append(new)
    for idx in sorted(replacements):
        if idx not in used:
            keyframes.append({"resolved_frame_index": idx, "latent": replacements[idx]})
    return keyframes, to_resize


def _video_latent(latent):
    samples = latent["samples"]
    video = samples.tensors[0] if getattr(samples, "is_nested", False) else samples
    if video.ndim != 5 or video.shape[1] != LATENT_CHANNELS:
        raise ValueError("expected a MiniMax H3 (AV) latent, got shape {}".format(tuple(video.shape)))
    return video


def _resize_image(image, width, height, crop):
    # [B, H, W, C] -> [B, height, width, 3], same call as the core H3 nodes
    import comfy.utils
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _resize_latent(lt, h, w):
    # [B, C, T, H, W] -> [B, C, T, h, w], bilinear per frame
    if lt.shape[-2] == h and lt.shape[-1] == w:
        return lt
    import torch.nn.functional as F
    b, c, t = lt.shape[:3]
    flat = lt.movedim(2, 1).reshape(b * t, c, lt.shape[-2], lt.shape[-1])
    out = F.interpolate(flat.float(), size=(h, w), mode="bilinear", align_corners=False)
    return out.reshape(b, t, c, h, w).movedim(1, 2).to(lt.dtype).contiguous()


class CiDH3ResizeKeyframes:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {"tooltip": "Positive conditioning from MiniMax H3 Image to Video (first pass)"}),
                "latent": ("LATENT", {"tooltip": "The latent the next pass will sample, e.g. the upscaled AV latent. "
                                                 "Width, height and frame count are read from it"}),
            },
            "optional": {
                "vae": ("VAE", {"tooltip": "Video VAE. Needed when first_frame or last_frame is connected"}),
                "first_frame": ("IMAGE", {"tooltip": "Re-encoded at the latent's size (stretched) as the keyframe at frame 0"}),
                "last_frame": ("IMAGE", {"tooltip": "Re-encoded at the latent's size (center crop) as the keyframe at the last frame"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "resize"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Rebuilds the MiniMax H3 keyframe latents in a conditioning at the size of the given latent, "
                   "so a first-pass conditioning can drive a second pass on an upscaled latent without "
                   "re-running the text encoder. Connected frames are re-encoded with the VAE; "
                   "other keyframes are interpolated.")

    def resize(self, conditioning, latent, vae=None, first_frame=None, last_frame=None):
        video = _video_latent(latent)
        h, w = video.shape[-2], video.shape[-1]
        width, height = w * VAE_DOWNSAMPLE, h * VAE_DOWNSAMPLE
        frame_count = frame_count_from_latent_t(video.shape[2])

        replacements = {}
        if first_frame is not None or last_frame is not None:
            if vae is None:
                raise ValueError("connect the video VAE to re-encode first_frame / last_frame")
            if first_frame is not None:
                replacements[0] = vae.encode(_resize_image(first_frame[:1], width, height, "disabled"))
            if last_frame is not None:
                replacements[frame_count - 1] = vae.encode(_resize_image(last_frame[:1], width, height, "center"))

        out = []
        for tensor, extra in conditioning:
            existing = extra.get("minimax_keyframes") or []
            if not existing and not replacements:
                out.append([tensor, extra])
                continue
            keyframes, to_resize = merge_keyframes(existing, replacements, frame_count)
            for i in to_resize:
                keyframes[i]["latent"] = _resize_latent(keyframes[i]["latent"], h, w)
            out.append([tensor, {**extra, "minimax_keyframes": keyframes}])
        return (out,)
