"""Unit tests for cid_nodes. Run from the repo root:

    python3 -m unittest discover -s test -p 'test_*.py'

ComfyUI is not installed here, so a stub ``nodes`` module stands in for the
core loader classes. The math nodes need no stubs.
"""

import math
import sys
import types
import unittest

# --- stub ComfyUI's `nodes` module before importing the package --------------

_calls = []


class _StubUNETLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"unet_name": (["u1.safetensors", "u2.safetensors"],),
                             "weight_dtype": (["default", "fp8_e4m3fn"], {"advanced": True})}}

    def load_unet(self, unet_name, weight_dtype):
        _calls.append(("unet", unet_name, weight_dtype))
        return (f"MODEL<{unet_name}|{weight_dtype}>",)


class _StubCLIPLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip_name": (["c1.safetensors"],),
                             "type": (["stable_diffusion", "wan", "minimax"],)},
                "optional": {"device": (["default", "cpu"], {"advanced": True})}}

    def load_clip(self, clip_name, type="stable_diffusion", device="default"):
        _calls.append(("clip", clip_name, type, device))
        return (f"CLIP<{clip_name}|{type}|{device}>",)


class _StubVAELoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae_name": (["taesd", "v1.safetensors", "v2.safetensors"],)}}

    def load_vae(self, vae_name):
        _calls.append(("vae", vae_name))
        return (f"VAE<{vae_name}>",)


_stub = types.ModuleType("nodes")
_stub.UNETLoader = _StubUNETLoader
_stub.CLIPLoader = _StubCLIPLoader
_stub.VAELoader = _StubVAELoader
sys.modules.setdefault("nodes", _stub)

import cid_nodes  # noqa: E402
from cid_nodes.h3_keyframes import (CiDH3ResizeKeyframes, frame_count_from_latent_t,  # noqa: E402
                                    merge_keyframes)
from cid_nodes.loader import CiDModelLoader  # noqa: E402
from cid_nodes.size_params import CiDSizeParams, size_params  # noqa: E402
from cid_nodes.video_frames import MODES, CiDVideoFrames, frames_for_duration  # noqa: E402


# --- reference implementations: the subgraph expressions, verbatim -----------

def _ref_frames_h3(a, b):
    return max(5, round(a * b)) + (5 - (max(5, round(a * b)) % 17)) % 17


def _ref_frames_default(a, b):
    return math.floor(a * b + 1)


def _ref_width(a, b, c, d):
    return int((c * a / b) ** 0.5 * 1024 / d + 0.5) * d


def _ref_height(a, b, c, d):
    return int((c * b / a) ** 0.5 * 1024 / d + 0.5) * d


def _ref_scaled(a, b, c):
    return int(a * b / c + 0.5) * c


class TestMappings(unittest.TestCase):
    def test_every_class_has_display_name_and_contract(self):
        self.assertEqual(set(cid_nodes.NODE_CLASS_MAPPINGS), set(cid_nodes.NODE_DISPLAY_NAME_MAPPINGS))
        for name, cls in cid_nodes.NODE_CLASS_MAPPINGS.items():
            with self.subTest(node=name):
                spec = cls.INPUT_TYPES()
                self.assertIn("required", spec)
                self.assertTrue(callable(getattr(cls, cls.FUNCTION)))
                self.assertEqual(len(cls.RETURN_TYPES), len(cls.RETURN_NAMES))
                self.assertTrue(cls.CATEGORY.startswith("CiD"))


class TestVideoFrames(unittest.TestCase):
    DURATIONS = (0.0, 0.5, 1.0, 2.5, 3.3, 5.0, 5.04, 7.125, 10.0, 12.75)
    FPS = (8, 16, 24, 25, 30, 60)

    def test_h3_matches_subgraph_expression(self):
        for duration in self.DURATIONS:
            for fps in self.FPS:
                with self.subTest(duration=duration, fps=fps):
                    self.assertEqual(frames_for_duration(duration, fps, "h3"), _ref_frames_h3(duration, fps))

    def test_default_matches_subgraph_expression(self):
        for duration in self.DURATIONS:
            for fps in self.FPS:
                with self.subTest(duration=duration, fps=fps):
                    self.assertEqual(frames_for_duration(duration, fps, "default"),
                                     _ref_frames_default(duration, fps))

    def test_h3_result_is_5_mod_17_and_not_below_rounded_product(self):
        for duration in (0.1, 1.0, 4.0, 9.9):
            for fps in (12, 24, 30):
                n = frames_for_duration(duration, fps, "h3")
                self.assertEqual(n % 17, 5)
                self.assertGreaterEqual(n, max(5, round(duration * fps)))
                self.assertLess(n - max(5, round(duration * fps)), 17)

    def test_known_values(self):
        self.assertEqual(frames_for_duration(5.0, 24, "h3"), 124)   # 120 -> 124
        self.assertEqual(frames_for_duration(0.0, 24, "h3"), 5)     # clamp to min
        self.assertEqual(frames_for_duration(1.0, 24, "h3"), 39)    # 24 -> 39
        self.assertEqual(frames_for_duration(5.0, 24, "h3", frame_step=4, frame_offset=1), 121)
        self.assertEqual(frames_for_duration(5.0, 24, "default"), 121)
        self.assertEqual(frames_for_duration(0.0, 24, "default"), 1)
        self.assertEqual(frames_for_duration(5.04, 25, "default"), 127)  # 126.0 -> 127

    def test_default_mode_ignores_h3_constants(self):
        self.assertEqual(frames_for_duration(5.0, 24, "default", frame_step=4, frame_offset=1, min_frames=999),
                         frames_for_duration(5.0, 24, "default"))

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            frames_for_duration(5.0, 24, "nope")

    def test_combo_lists_all_modes_and_defaults_to_default(self):
        spec = CiDVideoFrames.INPUT_TYPES()["required"]["mode"]
        self.assertEqual(spec[0], MODES)
        self.assertEqual(spec[1]["default"], "default")
        self.assertEqual(MODES, ["default", "h3"])

    def test_node_wrapper(self):
        node = CiDVideoFrames()
        self.assertEqual(node.compute(5.0, 24), (121,))
        self.assertEqual(node.compute(5.0, 24, "default"), (121,))
        self.assertEqual(node.compute(5.0, 24, "h3"), (124,))
        self.assertEqual(node.compute(5.0, 24, "h3", 17, 5, 5), (124,))


class TestSizeParams(unittest.TestCase):
    CASES = [
        # ratio_width, ratio_height, megapixels, multiple_of, scale_by
        (9, 16, 0.5, 32, 2.0),
        (16, 9, 1.0, 16, 1.5),
        (1, 1, 1.0, 64, 2.0),
        (4, 3, 0.75, 8, 1.0),
        (21, 9, 2.0, 32, 0.5),
        (3, 4, 0.25, 1, 3.0),
    ]

    def test_width_height_match_subgraph(self):
        for a, b, c, d, s in self.CASES:
            with self.subTest(case=(a, b, c, d, s)):
                w, h, sw, sh = size_params(a, b, c, d, s)
                self.assertEqual(w, _ref_width(a, b, c, d))
                self.assertEqual(h, _ref_height(a, b, c, d))
                self.assertEqual(sw, _ref_scaled(w, s, d))
                # intentionally from height: the subgraph fed width here by mistake
                self.assertEqual(sh, _ref_scaled(h, s, d))

    def test_alignment_and_pixel_budget(self):
        for a, b, c, d, s in self.CASES:
            w, h, sw, sh = size_params(a, b, c, d, s)
            for v in (w, h, sw, sh):
                self.assertEqual(v % d, 0)
            self.assertAlmostEqual(w * h / (1024 * 1024), c, delta=c * 0.15)

    def test_known_values(self):
        self.assertEqual(size_params(9, 16, 0.5, 32, 2.0), (544, 960, 1088, 1920))
        self.assertEqual(size_params(1, 1, 1.0, 64, 2.0), (1024, 1024, 2048, 2048))
        self.assertEqual(size_params(1, 1, 1.0, 16, 2.0), (1024, 1024, 2048, 2048))  # node defaults

    def test_defaults(self):
        req = CiDSizeParams.INPUT_TYPES()["required"]
        defaults = {k: v[1]["default"] for k, v in req.items()}
        self.assertEqual(defaults, {"ratio_width": 1, "ratio_height": 1, "megapixels": 1.0,
                                    "multiple_of": 16, "scale_by": 2.0})
        out = CiDSizeParams().compute(**defaults)
        self.assertEqual(out, (1024, 1024, 2048, 2048, 2.0, 16))
        vreq = CiDVideoFrames.INPUT_TYPES()["required"]
        self.assertEqual((vreq["duration"][1]["default"], vreq["fps"][1]["default"]), (5.0, 24))

    def test_node_wrapper_passthroughs(self):
        out = CiDSizeParams().compute(9, 16, 0.5, 32, 2.0)
        self.assertEqual(out, (544, 960, 1088, 1920, 2.0, 32))
        self.assertEqual(list(CiDSizeParams.INPUT_TYPES()["required"]),
                         ["ratio_width", "ratio_height", "megapixels", "multiple_of", "scale_by"])
        self.assertEqual(CiDSizeParams.RETURN_NAMES,
                         ("width", "height", "upscaled_width", "upscaled_height", "scale_by", "multiple_of"))
        self.assertEqual(len(out), len(CiDSizeParams.RETURN_TYPES))


class TestModelLoader(unittest.TestCase):
    def setUp(self):
        _calls.clear()

    def test_input_types_come_from_core_loaders(self):
        req = CiDModelLoader.INPUT_TYPES()["required"]
        self.assertEqual(list(req), ["unet_name", "weight_dtype", "clip_name", "clip_type",
                                     "clip_device", "video_vae_name", "audio_vae_name"])
        self.assertEqual(req["unet_name"][0], ["u1.safetensors", "u2.safetensors"])
        self.assertEqual(req["weight_dtype"][0], ["default", "fp8_e4m3fn"])
        self.assertNotIn("advanced", req["weight_dtype"][1])
        self.assertEqual(req["clip_type"][1].get("default"), "minimax")
        self.assertEqual(req["clip_device"][0], ["default", "cpu"])
        self.assertEqual(req["video_vae_name"][0], req["audio_vae_name"][0])
        self.assertEqual(CiDModelLoader.RETURN_NAMES, ("MODEL", "CLIP", "VIDEO_VAE", "AUDIO_VAE"))

    def test_clip_type_default_only_if_available(self):
        original = _StubCLIPLoader.INPUT_TYPES
        _StubCLIPLoader.INPUT_TYPES = classmethod(
            lambda cls: {"required": {"clip_name": (["c"],), "type": (["sd3"],)},
                         "optional": {"device": (["default"],)}})
        try:
            req = CiDModelLoader.INPUT_TYPES()["required"]
            self.assertNotIn("default", req["clip_type"][1])
        finally:
            _StubCLIPLoader.INPUT_TYPES = original

    def test_load_delegates_in_order(self):
        out = CiDModelLoader().load("u1.safetensors", "fp8_e4m3fn", "c1.safetensors", "minimax",
                                    "cpu", "v1.safetensors", "v2.safetensors")
        self.assertEqual(out, ("MODEL<u1.safetensors|fp8_e4m3fn>", "CLIP<c1.safetensors|minimax|cpu>",
                               "VAE<v1.safetensors>", "VAE<v2.safetensors>"))
        self.assertEqual(_calls, [("unet", "u1.safetensors", "fp8_e4m3fn"),
                                  ("clip", "c1.safetensors", "minimax", "cpu"),
                                  ("vae", "v1.safetensors"), ("vae", "v2.safetensors")])


# --- H3 keyframes: duck-typed tensors, no torch here -------------------------

class _Shape:
    """Stands in for a tensor: only ndim / shape are read on the no-resize path."""

    def __init__(self, *shape):
        self.shape = tuple(shape)
        self.ndim = len(shape)


class _Nested:
    def __init__(self, *tensors):
        self.tensors = list(tensors)
        self.is_nested = True


class _Image(_Shape):
    def __getitem__(self, item):
        return self


class _VAE:
    def __init__(self, h, w):
        self.h, self.w = h, w
        self.calls = []

    def encode(self, image):
        self.calls.append(image)
        return _Shape(1, 24, 1, self.h, self.w)


class TestH3Keyframes(unittest.TestCase):
    def test_frame_count_matches_core_pattern(self):
        # comfy_extras.nodes_minimax_h3.video_latent_t inverted: 5 -> 2, 22 -> 7, 124 -> 37
        self.assertEqual(frame_count_from_latent_t(2), 5)
        self.assertEqual(frame_count_from_latent_t(7), 22)
        self.assertEqual(frame_count_from_latent_t(37), 124)
        self.assertEqual(frame_count_from_latent_t(1), 1)
        for t in range(2, 60, 5):
            self.assertEqual(frame_count_from_latent_t(t) % 17, 5)

    def test_merge_replaces_in_place_and_keeps_audio(self):
        existing = [{"resolved_frame_index": 0, "latent": "old0"},
                    {"resolved_frame_index": 60, "latent": "mid", "audio_latent": "a60"},
                    {"resolved_frame_index": 123, "latent": "old123", "audio_latent": "a123"}]
        kfs, to_resize = merge_keyframes(existing, {0: "new0", 123: "new123"}, 124)
        self.assertEqual(kfs, [{"resolved_frame_index": 0, "latent": "new0"},
                               {"resolved_frame_index": 60, "latent": "mid", "audio_latent": "a60"},
                               {"resolved_frame_index": 123, "latent": "new123", "audio_latent": "a123"}])
        self.assertEqual(to_resize, [1])
        self.assertEqual(existing[0]["latent"], "old0")  # input not mutated

    def test_merge_appends_missing_replacements_in_frame_order(self):
        kfs, to_resize = merge_keyframes([], {123: "last", 0: "first"}, 124)
        self.assertEqual(kfs, [{"resolved_frame_index": 0, "latent": "first"},
                               {"resolved_frame_index": 123, "latent": "last"}])
        self.assertEqual(to_resize, [])
        kfs, _ = merge_keyframes([{"resolved_frame_index": 0, "latent": "x"}], {123: "last"}, 124)
        self.assertEqual([k["resolved_frame_index"] for k in kfs], [0, 123])

    def test_merge_audio_only_keyframes_are_not_resized(self):
        kfs, to_resize = merge_keyframes([{"resolved_frame_index": 10, "audio_latent": "a"}], {}, 124)
        self.assertEqual(kfs, [{"resolved_frame_index": 10, "audio_latent": "a"}])
        self.assertEqual(to_resize, [])

    def test_merge_rejects_keyframes_outside_the_latent(self):
        with self.assertRaises(ValueError):
            merge_keyframes([{"resolved_frame_index": 124, "latent": "x"}], {}, 124)

    def test_node_contract(self):
        spec = CiDH3ResizeKeyframes.INPUT_TYPES()
        self.assertEqual(list(spec["required"]), ["conditioning", "latent"])
        self.assertEqual(list(spec["optional"]), ["vae", "first_frame", "last_frame"])
        self.assertEqual(CiDH3ResizeKeyframes.RETURN_TYPES, ("CONDITIONING",))
        self.assertEqual(CiDH3ResizeKeyframes.RETURN_NAMES, ("conditioning",))

    def _latent(self, t=37, h=96, w=168, nested=True):
        video = _Shape(1, 24, t, h, w)
        return {"samples": _Nested(video, _Shape(1, 32, 2, 200)) if nested else video}

    def test_reencodes_connected_frames_at_latent_size(self):
        vae = _VAE(96, 168)
        cond = [["emb", {"minimax_keyframes": [{"resolved_frame_index": 0, "latent": _Shape(1, 24, 1, 48, 84)},
                                               {"resolved_frame_index": 123, "latent": _Shape(1, 24, 1, 48, 84)}],
                         "other": 1}]]
        first, last = _Image(1, 720, 1280, 3), _Image(1, 1080, 1920, 3)

        resized = []

        def fake_resize_image(image, width, height, crop):
            resized.append((image, width, height, crop))
            return image

        import cid_nodes.h3_keyframes as mod
        original = mod._resize_image
        mod._resize_image = fake_resize_image
        try:
            (out,) = CiDH3ResizeKeyframes().resize(cond, self._latent(), vae=vae, first_frame=first, last_frame=last)
        finally:
            mod._resize_image = original

        self.assertEqual(resized, [(first, 2688, 1536, "disabled"), (last, 2688, 1536, "center")])
        self.assertEqual(vae.calls, [first, last])
        kfs = out[0][1]["minimax_keyframes"]
        self.assertEqual([k["resolved_frame_index"] for k in kfs], [0, 123])
        for k in kfs:
            self.assertEqual(k["latent"].shape[-2:], (96, 168))
        self.assertEqual(out[0][1]["other"], 1)
        self.assertEqual(out[0][0], "emb")
        # input conditioning untouched
        self.assertEqual(cond[0][1]["minimax_keyframes"][0]["latent"].shape[-2:], (48, 84))

    def test_plain_video_latent_and_first_frame_only(self):
        vae = _VAE(96, 168)
        cond = [["emb", {}]]
        import cid_nodes.h3_keyframes as mod
        original = mod._resize_image
        mod._resize_image = lambda image, width, height, crop: image
        try:
            (out,) = CiDH3ResizeKeyframes().resize(cond, self._latent(nested=False), vae=vae,
                                                   first_frame=_Image(1, 8, 8, 3))
        finally:
            mod._resize_image = original
        self.assertEqual([k["resolved_frame_index"] for k in out[0][1]["minimax_keyframes"]], [0])

    def test_matching_keyframes_pass_through_without_torch(self):
        kf = {"resolved_frame_index": 0, "latent": _Shape(1, 24, 1, 96, 168)}
        cond = [["emb", {"minimax_keyframes": [kf]}]]
        (out,) = CiDH3ResizeKeyframes().resize(cond, self._latent())
        self.assertIs(out[0][1]["minimax_keyframes"][0]["latent"], kf["latent"])

    def test_no_keyframes_and_no_frames_is_a_no_op(self):
        cond = [["emb", {"x": 1}]]
        (out,) = CiDH3ResizeKeyframes().resize(cond, self._latent())
        self.assertEqual(out, cond)

    def test_frames_without_vae_raise(self):
        with self.assertRaises(ValueError):
            CiDH3ResizeKeyframes().resize([["emb", {}]], self._latent(), first_frame=_Image(1, 8, 8, 3))

    def test_wrong_latent_raises(self):
        with self.assertRaises(ValueError):
            CiDH3ResizeKeyframes().resize([["emb", {}]], {"samples": _Shape(1, 4, 64, 64)})


if __name__ == "__main__":
    unittest.main()
