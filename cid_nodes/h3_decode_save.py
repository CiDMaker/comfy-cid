"""H3 Decode & Save: AV latent -> mp4 on disk, path string out.

Replaces the four core nodes at the end of the H3 graph (VAEDecodeTiled 369,
VAEDecodeAudio 351, CreateVideo 354, SaveVideo 357) with one OUTPUT_NODE whose
only output is the absolute path of the written file. The decoded IMAGE never
leaves ``execute`` (not returned, not stored on the instance), so ComfyUI's
output cache does not keep 2.7-11 GB of frames alive per block (docs/h3-chain-plan.md 3.3).

Every step delegates to the core code so behaviour follows ComfyUI updates:

* video: ``nodes.VAEDecodeTiled().decode`` (tile/temporal mapping, nested
  unbind, batch flatten), or ``nodes.VAEDecode().decode`` when ``tiled`` is off,
* audio: ``comfy_extras.nodes_audio.vae_decode_audio`` (the body of
  VAEDecodeAudio: nested unbind, std normalisation, sample-rate lookup),
* video object: ``InputImpl.VideoFromComponents`` with CreateVideo's defaults
  (bit_depth "auto" -> 8 for sRGB, color_space sRGB),
* encode: ``VideoFromComponents.save_to(format=mp4, codec=h264, crf=...)``,
  which is exactly what SaveVideo calls; its ``crf`` argument is SaveVideo's
  "re-encode" knob (core default 18 when the graph leaves encoding on auto).

torch / comfy / nodes are imported inside ``execute`` so this module imports
without ComfyUI.
"""

import os

from . import h3_av as common

VIDEO_FORMAT = "mp4"
VIDEO_CODEC = "h264"


# --- pure helpers (no torch) ---------------------------------------------

def output_file_name(filename, counter, extension=VIDEO_FORMAT):
    """SaveVideo's file name pattern: ``{filename}_{counter:05}_.{ext}``."""
    return "{}_{:05}_.{}".format(filename, counter, extension)


def preview_ui(file, subfolder):
    """The ``ui`` dict SaveVideo returns (``ui.PreviewVideo([SavedResult(...)])``)."""
    return {"images": [{"filename": file, "subfolder": subfolder, "type": "output"}],
            "animated": (True,)}


def build_metadata(prompt=None, extra_pnginfo=None, disable_metadata=False):
    """SaveVideo's metadata dict: extra_pnginfo entries plus ``prompt``; None when empty/disabled."""
    if disable_metadata:
        return None
    metadata = {}
    if extra_pnginfo is not None:
        metadata.update(extra_pnginfo)
    if prompt is not None:
        metadata["prompt"] = prompt
    return metadata or None


# --- core delegation (torch / comfy imported lazily) ---------------------

def decode_video(vae, video, tiled, tile_size, overlap, temporal_size, temporal_overlap):
    """IMAGE [N, H, W, 3] exactly as VAEDecodeTiled (tiled) or VAEDecode would produce it."""
    import nodes
    if not tiled:
        return nodes.VAEDecode().decode(vae, {"samples": video})[0]
    return nodes.VAEDecodeTiled().decode(vae, {"samples": video}, tile_size, overlap,
                                         temporal_size, temporal_overlap)[0]


def decode_audio(audio_vae, audio, latent=None):
    """AUDIO dict {"waveform", "sample_rate"} exactly as VAEDecodeAudio would produce it."""
    from comfy_extras.nodes_audio import vae_decode_audio
    samples = {"samples": audio}
    if latent is not None and "sample_rate" in latent:   # VAEDecodeAudio honours it
        samples["sample_rate"] = latent["sample_rate"]
    return vae_decode_audio(audio_vae, samples)


def save_mp4(images, audio, fps, crf, path, metadata=None):
    """CreateVideo (sRGB, 8 bit) + SaveVideo's ``save_to`` as mp4/h264 at ``crf``."""
    from fractions import Fraction
    from comfy_api.latest import InputImpl, Types
    video = InputImpl.VideoFromComponents(
        Types.VideoComponents(images=images, audio=audio, frame_rate=Fraction(fps)),
        bit_depth=8, color_space="sRGB")
    video.save_to(path, format=Types.VideoContainer(VIDEO_FORMAT), codec=Types.VideoCodec(VIDEO_CODEC),
                  metadata=metadata, crf=crf)


class CiDH3DecodeSave:
    DESCRIPTION = (
        "Decode an H3 AV latent (tiled video decode + audio decode) and save it as an mp4 (h264/aac) "
        "in the output directory. Outputs only the file path: the decoded frames are freed right after "
        "saving instead of staying in the node cache. Replaces VAE Decode (Tiled) or VAE Decode + "
        "VAE Decode Audio + Create Video + Save Video.")
    CATEGORY = common.CATEGORY
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    OUTPUT_TOOLTIPS = ("Absolute path of the written mp4.",)
    FUNCTION = "execute"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "MiniMax H3 AV latent (nested video + audio)."}),
                "vae": ("VAE", {"tooltip": "Video VAE."}),
                "audio_vae": ("VAE", {"tooltip": "Audio VAE."}),
                "filename_prefix": ("STRING", {
                    "default": "CiD/h3_segment",
                    "tooltip": "Prefix of the saved file, relative to the output directory; may contain "
                               "subfolders and %date:...%-style variables like Save Video."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0,
                                  "tooltip": "Frame rate of the saved video."}),
                "crf": ("INT", {"default": 18, "min": 0, "max": 51, "step": 1,
                                "tooltip": "h264 quality: lower is higher quality and larger files."}),
                "tiled": ("BOOLEAN", {"default": True,
                                      "tooltip": "Decode the video in tiles (VAE Decode (Tiled) with the settings "
                                                 "below). Off: decode in one pass (VAE Decode), faster but needs "
                                                 "more memory."}),
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 32, "advanced": True}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32, "advanced": True}),
                "temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4, "advanced": True,
                                          "tooltip": "Amount of frames to decode at a time."}),
                "temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4, "advanced": True,
                                             "tooltip": "Amount of frames to overlap."}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    def execute(self, latent, vae, audio_vae, filename_prefix, fps, crf, tiled=True,
                tile_size=512, overlap=64, temporal_size=64, temporal_overlap=8, prompt=None, extra_pnginfo=None):
        if not isinstance(latent, dict) or "samples" not in latent:
            raise ValueError("H3 Decode & Save: expected a LATENT dict, got {}".format(type(latent).__name__))
        try:
            video, audio = common.av_tensors(latent["samples"])
        except ValueError as e:
            raise ValueError("H3 Decode & Save: {}".format(e)) from None

        import folder_paths
        import comfy.model_management
        from comfy.cli_args import args

        images = audio_out = None
        try:
            images = decode_video(vae, video, tiled, tile_size, overlap, temporal_size, temporal_overlap)
            audio_out = decode_audio(audio_vae, audio, latent)

            full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
                filename_prefix, folder_paths.get_output_directory(), images.shape[2], images.shape[1])
            file = output_file_name(filename, counter)
            path = os.path.abspath(os.path.join(full_output_folder, file))
            save_mp4(images, audio_out, fps, crf, path,
                     build_metadata(prompt, extra_pnginfo, args.disable_metadata))
        finally:
            del images, audio_out
            comfy.model_management.soft_empty_cache()

        return {"ui": preview_ui(file, subfolder), "result": (path,)}
