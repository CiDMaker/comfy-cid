import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cid_nodes import h3_av as common  # noqa: E402
from cid_nodes.h3_prefix import CiDH3LatentPrefix  # noqa: E402

try:
    import torch
    import comfy.nested_tensor
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


class Schema(unittest.TestCase):
    def test_input_types(self):
        it = CiDH3LatentPrefix.INPUT_TYPES()
        req, opt = it["required"], it["optional"]
        self.assertEqual(list(req), ["latent", "prefix_tokens", "audio"])
        self.assertEqual(list(opt), ["prev_latent"])
        self.assertEqual(req["latent"][0], "LATENT")
        self.assertEqual(opt["prev_latent"][0], "LATENT")
        kind, cfg = req["prefix_tokens"]
        self.assertEqual((kind, cfg["default"], cfg["min"], cfg["max"], cfg["step"]), ("INT", 7, 2, 102, 1))
        kind, cfg = req["audio"]
        self.assertEqual((kind, cfg["default"]), ("BOOLEAN", True))
        for spec in list(req.values()) + list(opt.values()):
            self.assertTrue(spec[1].get("tooltip"))

    def test_outputs(self):
        n = CiDH3LatentPrefix
        self.assertEqual(n.RETURN_TYPES, ("LATENT", "INT", "FLOAT"))
        self.assertEqual(n.RETURN_NAMES, ("latent", "prefix_frames", "prefix_seconds"))
        self.assertEqual(len(n.OUTPUT_TOOLTIPS), 3)
        self.assertEqual(n.CATEGORY, common.CATEGORY)
        self.assertTrue(hasattr(n, n.FUNCTION))
        self.assertTrue(n.DESCRIPTION)

    def test_pass_through_without_prev(self):
        sentinel = object()
        latent = {"samples": sentinel, "noise_mask": sentinel, "batch_index": [0]}
        # invalid prefix_tokens must not matter when nothing is spliced
        for tokens in (7, 8):
            out, frames, seconds = CiDH3LatentPrefix().execute(latent, tokens, True, None)
            self.assertIs(out, latent)
            self.assertIs(out["samples"], sentinel)
            self.assertEqual((frames, seconds), (0, 0.0))
        self.assertEqual(latent, {"samples": sentinel, "noise_mask": sentinel, "batch_index": [0]})


def _av(T, Ta, h=4, w=6, zero=False, gen=None):
    if zero:
        v = torch.zeros(1, 24, T, h, w)
        a = torch.zeros(1, 32, 2, Ta)
    else:
        v = torch.randn(1, 24, T, h, w, generator=gen)
        a = torch.randn(1, 32, 2, Ta, generator=gen)
    return v, a


def _nested(*ts):
    return comfy.nested_tensor.NestedTensor(ts)


@unittest.skipUnless(HAVE_TORCH, "needs torch and comfy (run scripts/gb10_test.sh prefix on gb10)")
class PrefixTensors(unittest.TestCase):
    P, A = 7, 37

    def setUp(self):
        self.gen = torch.Generator().manual_seed(1234)
        # prev: 12 tokens = 39 frames, 65 ticks; new: 57 tokens = 192 frames, 320 ticks
        self.pv, self.pa = _av(12, 65, gen=self.gen)
        self.prev = {"samples": _nested(self.pv, self.pa),
                     "noise_mask": _nested(torch.zeros_like(self.pv), torch.zeros_like(self.pa))}

    def run_node(self, latent, tokens=7, audio=True, prev=None):
        return CiDH3LatentPrefix().execute(latent, tokens, audio, self.prev if prev is None else prev)

    @staticmethod
    def video_mask(T, h, w, P):
        # CreateFadeMaskAdvanced "0:(0.0),P:(1.0)" interpolation none -> [T, h, w]; SetLatentNoiseMask -> [T, 1, h, w]
        m = torch.ones(T, h, w)
        m[:P] = 0.0
        return m.reshape((-1, 1, h, w))

    @staticmethod
    def audio_mask(Ta, A):
        # SolidMask(1, Ta x 2) * SolidMask(0, A x 2) at 0,0 -> [1, 2, Ta]; SetLatentNoiseMask -> [1, 1, 2, Ta]
        m = torch.ones(1, 2, Ta)
        m[:, :, :A] = 0.0
        return m.reshape((-1, 1, 2, Ta))

    def test_pass1_matches_runner_recipe(self):
        nv, na = _av(57, 320, zero=True)          # R2V's empty AV latent is zeros
        T, Ta, P, A = 57, 320, self.P, self.A
        out, frames, seconds = self.run_node({"samples": _nested(nv, na)})
        # 933-935: LatentCut(prev, t, -P, P) ++ new, cut to T; 936/937 mask
        ref_v = torch.cat((self.pv.narrow(2, 12 - P, P), nv), dim=2).narrow(2, 0, T)
        # 961-963: LatentCut(prev audio, x, -A, A) ++ new audio, cut to Ta; 964-967 mask
        ref_a = torch.cat((self.pa.narrow(3, 65 - A, A), na), dim=3).narrow(3, 0, Ta)
        s, m = out["samples"].tensors, out["noise_mask"].tensors
        self.assertTrue(torch.equal(s[0], ref_v))
        self.assertTrue(torch.equal(s[1], ref_a))
        self.assertTrue(torch.equal(m[0], self.video_mask(T, 4, 6, P)))
        self.assertTrue(torch.equal(m[1], self.audio_mask(Ta, A)))
        self.assertEqual(m[0].dtype, torch.float32)
        self.assertEqual(m[1].dtype, torch.float32)
        self.assertEqual((frames, round(seconds, 4)), (22, 0.9167))
        self.assertAlmostEqual(seconds, 22 / 24)

    def test_nonzero_new_keeps_positions(self):
        # the node keeps new[P:] (the Pass2 recipe); for the empty Pass1 latent this equals the runner's new[:T-P]
        nv, na = _av(12, 65, gen=self.gen)
        out, _, _ = self.run_node({"samples": _nested(nv, na)})
        s = out["samples"].tensors
        self.assertTrue(torch.equal(s[0], torch.cat((self.pv[:, :, -7:], nv[:, :, 7:]), dim=2)))
        self.assertTrue(torch.equal(s[1], torch.cat((self.pa[..., -37:], na[..., 37:]), dim=3)))
        self.assertEqual(out["noise_mask"].tensors[1].shape, (1, 1, 2, 65))

    def test_pass2_auto_keeps_audio_and_matches_runner(self):
        nv, na = _av(12, 65, gen=self.gen)
        solid = torch.zeros(1, 1, 64, 64)       # 356:347 SolidMask 64x64 0 through 356:348
        latent = {"samples": _nested(nv, na), "noise_mask": _nested(torch.ones_like(nv), solid), "extra": "kept"}
        P, T = self.P, 12
        for audio_flag in (True, False):
            out, frames, _ = self.run_node(latent, audio=audio_flag)
            # 950-957: LatentCut(prev, t, -P, P) ++ LatentCut(new, t, P, T-P); audio from 950 out1 as is
            ref_v = torch.cat((self.pv.narrow(2, 12 - P, P), nv.narrow(2, P, T - P)), dim=2)
            s, m = out["samples"].tensors, out["noise_mask"].tensors
            self.assertTrue(torch.equal(s[0], ref_v))
            self.assertTrue(torch.equal(m[0], self.video_mask(T, 4, 6, P)))
            self.assertIs(s[1], na)
            self.assertIs(m[1], solid)
            self.assertTrue(torch.equal(m[1], torch.zeros(1, 1, 64, 64)))
            self.assertEqual(out["extra"], "kept")
            self.assertEqual(frames, 22)
        # input untouched
        self.assertIs(latent["samples"].tensors[0], nv)
        self.assertTrue(torch.equal(latent["noise_mask"].tensors[0], torch.ones_like(nv)))

    def test_audio_off(self):
        nv, na = _av(57, 320, zero=True)
        out, _, _ = self.run_node({"samples": _nested(nv, na)}, audio=False)
        s, m = out["samples"].tensors, out["noise_mask"].tensors
        self.assertIs(s[1], na)
        self.assertEqual(m[1].shape, na.shape)
        self.assertTrue(torch.equal(m[1], torch.ones_like(na)))
        self.assertTrue(torch.equal(m[0], self.video_mask(57, 4, 6, 7)))

    def test_incoming_video_mask_replaced(self):
        nv, na = _av(12, 65, gen=self.gen)
        latent = {"samples": _nested(nv, na), "noise_mask": torch.full((12, 1, 4, 6), 0.5)}
        out, _, _ = self.run_node(latent)
        m = out["noise_mask"].tensors
        self.assertTrue(torch.equal(m[0], self.video_mask(12, 4, 6, 7)))
        self.assertTrue(torch.equal(m[1], self.audio_mask(65, 37)))

    def test_prefix_2_tokens(self):
        nv, na = _av(12, 65, gen=self.gen)
        out, frames, seconds = self.run_node({"samples": _nested(nv, na)}, tokens=2)
        self.assertEqual(frames, 5)
        self.assertAlmostEqual(seconds, 5 / 24)
        self.assertTrue(torch.equal(out["samples"].tensors[0][:, :, :2], self.pv[:, :, -2:]))
        self.assertTrue(torch.equal(out["noise_mask"].tensors[1], self.audio_mask(65, 9)))

    def test_dtype_follows_new(self):
        nv, na = _av(12, 65, gen=self.gen)
        out, _, _ = self.run_node({"samples": _nested(nv.half(), na.half())})
        s = out["samples"].tensors
        self.assertEqual((s[0].dtype, s[1].dtype), (torch.float16, torch.float16))
        self.assertTrue(torch.equal(s[0][:, :, :7], self.pv[:, :, -7:].half()))
        self.assertEqual(out["noise_mask"].tensors[0].dtype, torch.float32)

    def test_output_does_not_alias_prev(self):
        nv, na = _av(12, 65, gen=self.gen)
        out, _, _ = self.run_node({"samples": _nested(nv, na)})
        before = self.pv.clone()
        out["samples"].tensors[0].add_(1.0)
        self.assertTrue(torch.equal(self.pv, before))

    def test_errors(self):
        nv, na = _av(12, 65, gen=self.gen)
        new = {"samples": _nested(nv, na)}
        with self.assertRaisesRegex(ValueError, r"use 7 \(22 frames\) or 12 \(39 frames\)"):
            self.run_node(new, tokens=8)
        # prev shorter than P
        short = {"samples": _nested(*_av(7, 37, gen=self.gen))}
        with self.assertRaisesRegex(ValueError, "fewer than prefix_tokens 12"):
            self.run_node(new, tokens=12, prev=short)
        # new not longer than P
        with self.assertRaisesRegex(ValueError, "needs more than prefix_tokens 12"):
            self.run_node(new, tokens=12)
        # spatial mismatch
        big = {"samples": _nested(*_av(12, 65, h=8, w=12, gen=self.gen))}
        with self.assertRaisesRegex(ValueError, r"resolution mismatch.*\(1, 24, 12, 8, 12\).*\(1, 24, 12, 4, 6\)"):
            self.run_node(new, prev=big)
        # batch mismatch
        v2, a2 = _av(12, 65, gen=self.gen)
        two = {"samples": _nested(v2.repeat(2, 1, 1, 1, 1), a2.repeat(2, 1, 1, 1))}
        with self.assertRaisesRegex(ValueError, "batch mismatch"):
            self.run_node(new, prev=two)
        # prev audio shorter than the audio prefix
        pv, _ = _av(12, 65, gen=self.gen)
        short_audio = {"samples": _nested(pv, torch.zeros(1, 32, 2, 30))}
        with self.assertRaisesRegex(ValueError, "fewer than the 37 audio prefix ticks"):
            self.run_node(new, prev=short_audio)
        # ... which is fine with audio off
        self.run_node(new, audio=False, prev=short_audio)
        # non-nested samples on either side
        with self.assertRaisesRegex(ValueError, "expected a MiniMax H3 AV latent"):
            self.run_node(new, prev={"samples": pv})
        with self.assertRaisesRegex(ValueError, "expected a MiniMax H3 AV latent"):
            self.run_node({"samples": nv})


if __name__ == "__main__":
    unittest.main()
