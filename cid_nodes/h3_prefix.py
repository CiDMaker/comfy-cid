"""H3 Latent Prefix: pin the tail of the previous segment's AV latent to the
head of a new segment's AV latent and mask it out of sampling.

Reproduces the validated core-node recipe of ``run_mask2.py`` (nodes 930-938
for Pass1 video, 960-967 for audio, 950-957 for Pass2) in one node. torch and
comfy are imported lazily so this module imports without them.
"""

from . import h3_av as common


def _shape(t):
    return tuple(t.shape)


def _splice(prev, new, count, dim):
    """cat(prev[..last count along dim], new[count.. along dim]) on new's device/dtype."""
    import torch

    head = prev.narrow(dim, prev.shape[dim] - count, count).to(device=new.device, dtype=new.dtype)
    tail = new.narrow(dim, count, new.shape[dim] - count)
    return torch.cat((head, tail), dim=dim)


def _prefix_mask(shape, count, dim, device):
    """float32 ones of ``shape`` with the first ``count`` entries along ``dim`` zeroed."""
    import torch

    mask = torch.ones(shape, dtype=torch.float32, device=device)
    mask.narrow(dim, 0, count).zero_()
    return mask


def _prefix_video(prev_video, video, tokens):
    pv, nv = _shape(prev_video), _shape(video)
    if pv[0] != nv[0]:
        raise ValueError("batch mismatch: prev_latent video {} vs latent video {}".format(pv, nv))
    if pv[3:] != nv[3:]:
        raise ValueError(
            "resolution mismatch: prev_latent video {} is {}x{} latent px but latent video {} is {}x{}; "
            "prev_latent must come from the same pass at the same resolution".format(
                pv, pv[4], pv[3], nv, nv[4], nv[3]))
    if pv[2] < tokens:
        raise ValueError("prev_latent video {} has {} latent frames, fewer than prefix_tokens {}".format(pv, pv[2], tokens))
    if nv[2] <= tokens:
        raise ValueError("latent video {} has {} latent frames; it needs more than prefix_tokens {}".format(nv, nv[2], tokens))
    out = _splice(prev_video, video, tokens, 2)
    # what SetLatentNoiseMask makes of a [T, H, W] mask: [T, 1, H, W]
    mask = _prefix_mask((nv[2], 1, nv[3], nv[4]), tokens, 0, video.device)
    return out, mask


def _prefix_audio(prev_audio, audio, ticks):
    pa, na = _shape(prev_audio), _shape(audio)
    if pa[:-1] != na[:-1]:
        raise ValueError("audio batch/channel mismatch: prev_latent audio {} vs latent audio {}".format(pa, na))
    if pa[-1] < ticks:
        raise ValueError("prev_latent audio {} has {} ticks, fewer than the {} audio prefix ticks".format(pa, pa[-1], ticks))
    if na[-1] <= ticks:
        raise ValueError("latent audio {} has {} ticks; it needs more than the {} audio prefix ticks".format(na, na[-1], ticks))
    out = _splice(prev_audio, audio, ticks, audio.ndim - 1)
    # what SetLatentNoiseMask makes of a [2, Ta] mask: [1, 1, 2, Ta]
    mask = _prefix_mask((1, 1, na[2], na[3]), ticks, 3, audio.device)
    return out, mask


class CiDH3LatentPrefix:
    DESCRIPTION = (
        "Pins the last prefix_tokens latent frames (and matching audio) of the previous segment's sampler output "
        "to the start of the new segment's AV latent and masks them, so the sampler continues the previous "
        "segment without redrawing it. Use the same node before Pass1 and before Pass2. With prev_latent "
        "unconnected the latent passes through unchanged (single-cut workflow)."
    )
    CATEGORY = common.CATEGORY
    FUNCTION = "execute"
    RETURN_TYPES = ("LATENT", "INT", "FLOAT")
    RETURN_NAMES = ("latent", "prefix_frames", "prefix_seconds")
    OUTPUT_TOOLTIPS = (
        "AV latent with the prefix spliced in and noise_mask set; connect to the sampler's latent_image.",
        "Pixel frames covered by the prefix (7 tokens -> 22); drop this many frames when joining. 0 when prev_latent is unconnected.",
        "prefix_frames / 24 (22 frames -> 0.9167 s); drop this much audio when joining. 0.0 when prev_latent is unconnected.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": "The new segment's AV latent: the empty latent from MiniMax H3 Reference/Image to Video "
                               "(Pass1) or the upscaled AV latent (Pass2)."}),
                "prefix_tokens": ("INT", {
                    "default": 7, "min": 2, "max": 102, "step": 1,
                    "tooltip": "Video latent frames taken from the end of prev_latent. Must be 5m+2 (2, 7, 12, ...); "
                               "7 tokens = 22 frames."}),
                "audio": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Also prefix the audio stream (ceil(frames x 5/3) ticks). Ignored when the latent already "
                               "carries an audio noise mask (Pass2 after the upscale subgraph): that audio is kept as is."}),
            },
            "optional": {
                "prev_latent": ("LATENT", {
                    "tooltip": "The previous segment's sampler output AV latent from the same pass and resolution. Only "
                               "its samples are read. Leave unconnected for the first segment."}),
            },
        }

    def execute(self, latent, prefix_tokens=7, audio=True, prev_latent=None):
        if prev_latent is None:
            return (latent, 0, 0.0)

        tokens = common.validate_prefix_tokens(prefix_tokens)
        prev_video, prev_audio = common.av_tensors(prev_latent["samples"])
        video, new_audio, _, audio_mask = common.split_av(latent)

        out_video, video_mask = _prefix_video(prev_video, video, tokens)
        frames = common.frames_from_tokens(tokens)

        out_audio = new_audio
        if audio_mask is None and audio:
            out_audio, audio_mask = _prefix_audio(prev_audio, new_audio, common.audio_prefix_ticks(frames))
        # else: an incoming audio mask (Pass2) is kept untouched, or audio is off and
        # join_av fills the audio mask with ones like the core Concat AV Latent node.

        out = common.join_av(out_video, out_audio, video_mask, audio_mask, base=latent)
        return (out, frames, frames / float(common.FPS))
