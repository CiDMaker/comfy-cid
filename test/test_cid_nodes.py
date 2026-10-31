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


if __name__ == "__main__":
    unittest.main()
