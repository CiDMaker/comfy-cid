"""Save / load a MiniMax H3 AV latent as one safetensors file (``.h3lat``).

The core SaveLatent cannot write a nested AV latent (it calls
``samples.contiguous()``, which ``NestedTensor`` lacks) and LoadLatent only
reads from ``input/``. These nodes store video and audio (plus any noise mask)
in a single file at an explicit path, so an approved segment's latent
survives restarts, ephemeral containers and host changes (docs/h3-chain-plan.md 3.2).

Two Save node ids share one implementation: ``CiDH3SaveAVLatent`` is an
output node (always runs; used after Pass1) and
``CiDH3SaveAVLatentPassthrough`` is not (runs only when something consumes
its output; used after Pass2 behind the lazy upscale switch).

Torch, safetensors and ComfyUI modules are imported lazily.
"""

import datetime
import hashlib
import json
import logging
import os
import uuid

from . import h3_av as common

FORMAT = "h3chain-av-latent"
FORMAT_VERSION = "1"
EXTENSION = ".h3lat"

log = logging.getLogger(__name__)


# --- path helpers (plain Python) ------------------------------------------

def with_extension(path):
    """``path`` with ``.h3lat`` appended when it has no extension."""
    return path if os.path.splitext(path)[1] else path + EXTENSION


def _clean_path(path):
    path = (path or "").strip()
    if not path:
        return ""
    return os.path.expanduser(path)


def resolve_save_path(path):
    """Absolute target path for Save: relative paths go under the output directory."""
    path = _clean_path(path)
    if not path:
        raise ValueError("H3 Save AV Latent: set 'path' (absolute, or relative to ComfyUI's output directory), "
                         "e.g. latents/cut6_p1.h3lat")
    path = with_extension(path)
    if not os.path.isabs(path):
        import folder_paths
        path = os.path.join(folder_paths.get_output_directory(), path)
    return os.path.abspath(path)


def load_candidates(path):
    """Paths Load tries, in order: absolute as-is; relative under output/ then input/.
    In each place the bare name comes first, then the name with ``.h3lat`` when it
    has no extension."""
    path = _clean_path(path)
    if not path:
        return []
    names = [path] if os.path.splitext(path)[1] else [path, path + EXTENSION]
    if os.path.isabs(path):
        return [os.path.abspath(n) for n in names]
    import folder_paths
    out = []
    for base in (folder_paths.get_output_directory(), folder_paths.get_input_directory()):
        out.extend(os.path.abspath(os.path.join(base, n)) for n in names)
    return out


def resolve_load_path(path):
    if not _clean_path(path):
        raise ValueError("H3 Load AV Latent: set 'path' to an .h3lat file (absolute, or relative to "
                         "ComfyUI's output or input directory)")
    tried = load_candidates(path)
    for p in tried:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError("H3 Load AV Latent: no file for '{}'; tried: {}".format(path, ", ".join(tried)))


# --- metadata --------------------------------------------------------------

def _comfy_version():
    try:
        import comfyui_version
        return str(getattr(comfyui_version, "__version__", ""))
    except Exception:
        return ""


def prompt_hash(prompt):
    if prompt is None:
        return ""
    return hashlib.sha256(json.dumps(prompt, sort_keys=True, default=str).encode("utf-8")).hexdigest()


# --- nodes -----------------------------------------------------------------

class CiDH3SaveAVLatent:
    OUTPUT_NODE = True
    CATEGORY = common.CATEGORY
    FUNCTION = "save"
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "path")
    OUTPUT_TOOLTIPS = (
        "The input latent, unchanged (same object).",
        "Absolute path of the written file.",
    )
    DESCRIPTION = (
        "Writes a MiniMax H3 AV latent (video + audio, and its noise mask if any) to one safetensors "
        ".h3lat file and passes the latent through. Output node: it always runs, so the latent is saved "
        "even when nothing downstream consumes it (use after Pass1). Refuses to replace an existing file "
        "when 'overwrite' is off, protecting an approved segment's latent from a re-run.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "MiniMax H3 AV latent (nested video + audio), e.g. a sampler output."}),
                "path": ("STRING", {"default": "", "tooltip": (
                    "Absolute path, or relative to ComfyUI's output directory; .h3lat is appended when there "
                    "is no extension. Parent directories are created.")}),
                "overwrite": ("BOOLEAN", {"default": True, "tooltip": (
                    "Replace an existing file (atomically). Off: fail instead, so an approved latent is never "
                    "replaced by a re-run.")}),
            },
            "optional": {
                "note": ("STRING", {"default": "", "tooltip": "Free text stored in the file metadata, e.g. 'cut6 p1 seed 6602'."}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    def save(self, latent, path, overwrite=True, note="", prompt=None, extra_pnginfo=None):
        target = resolve_save_path(path)
        if os.path.exists(target) and not overwrite:
            raise ValueError("H3 Save AV Latent: '{}' already exists and overwrite is off; "
                             "change 'path' or enable 'overwrite'".format(target))

        import safetensors.torch

        video, audio, video_mask, audio_mask = common.split_av(latent)
        frames, tokens, width, height = common.video_geometry(video)

        tensors = {}
        seen = set()
        for key, t in (("video", video), ("audio", audio),
                       ("video_noise_mask", video_mask), ("audio_noise_mask", audio_mask)):
            if t is None:
                continue
            t = t.detach().contiguous().cpu()
            ptr = t.untyped_storage().data_ptr()
            if ptr in seen:      # safetensors refuses tensors that share memory
                t = t.clone()
            seen.add(t.untyped_storage().data_ptr())
            tensors[key] = t

        metadata = {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "frames": str(frames),
            "tokens": str(tokens),
            "width": str(width),
            "height": str(height),
            "audio_ticks": str(audio.shape[-1]),
            "batch": str(video.shape[0]),
            "note": str(note or ""),
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "comfy_version": _comfy_version(),
            "prompt_hash": prompt_hash(prompt),
        }

        parent = os.path.dirname(target)
        os.makedirs(parent, exist_ok=True)
        tmp = os.path.join(parent, ".{}.{}.tmp".format(os.path.basename(target), uuid.uuid4().hex[:12]))
        try:
            safetensors.torch.save_file(tensors, tmp, metadata=metadata)
            if not overwrite and os.path.exists(target):
                raise ValueError("H3 Save AV Latent: '{}' appeared while saving and overwrite is off".format(target))
            os.replace(tmp, target)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

        return {"ui": {"text": [target]}, "result": (latent, target)}


class CiDH3SaveAVLatentPassthrough(CiDH3SaveAVLatent):
    OUTPUT_NODE = False
    DESCRIPTION = (
        "Same as H3 Save AV Latent, but not an output node: it runs only when its latent output is "
        "consumed. Use after Pass2 feeding the upscale switch's on_true input, so turning upscale off "
        "does not force Pass2 to run.")


class CiDH3LoadAVLatent:
    CATEGORY = common.CATEGORY
    FUNCTION = "load"
    RETURN_TYPES = ("LATENT", "INT", "INT", "INT")
    RETURN_NAMES = ("latent", "frames", "width", "height")
    OUTPUT_TOOLTIPS = (
        "MiniMax H3 AV latent (nested video + audio), without a noise mask.",
        "Pixel frames covered by the video latent.",
        "Video width in pixels.",
        "Video height in pixels.",
    )
    DESCRIPTION = (
        "Reads an .h3lat file written by H3 Save AV Latent and returns the AV latent (noise masks are not "
        "restored; H3 Latent Prefix builds its own) with its frame count and size taken from the tensors.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": "", "tooltip": (
                    "Absolute path, or relative to ComfyUI's output directory, then its input directory; "
                    ".h3lat is tried when the name has no extension.")}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, path):
        try:
            resolved = resolve_load_path(path)
        except (ValueError, FileNotFoundError):
            return path
        st = os.stat(resolved)
        return "{}|{}|{}".format(resolved, st.st_mtime_ns, st.st_size)

    def load(self, path):
        resolved = resolve_load_path(path)

        import safetensors
        import comfy.nested_tensor

        with safetensors.safe_open(resolved, framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
            fmt = metadata.get("format")
            if fmt is not None and fmt != FORMAT:
                raise ValueError("H3 Load AV Latent: '{}' is not an H3 Chain AV latent (format '{}', expected '{}')".format(
                    resolved, fmt, FORMAT))
            keys = set(f.keys())
            missing = [k for k in ("video", "audio") if k not in keys]
            if missing:
                raise ValueError("H3 Load AV Latent: '{}' has no {} tensor (found: {})".format(
                    resolved, " or ".join(missing), ", ".join(sorted(keys)) or "nothing"))
            video = f.get_tensor("video")
            audio = f.get_tensor("audio")

        samples = comfy.nested_tensor.NestedTensor((video, audio))
        common.av_tensors(samples)   # shape check
        frames, tokens, width, height = common.video_geometry(video)

        meta_frames = metadata.get("frames")
        if meta_frames is not None and meta_frames != str(frames):
            log.warning("H3 Load AV Latent: %s metadata says %s frames but the video latent has %d (%d tokens); "
                        "using the tensor", resolved, meta_frames, frames, tokens)

        return ({"samples": samples}, frames, width, height)
