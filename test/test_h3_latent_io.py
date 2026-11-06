import hashlib
import json
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cid_nodes import h3_latent_io as latent_io  # noqa: E402
from cid_nodes.h3_latent_io import (  # noqa: E402
    CiDH3LoadAVLatent, CiDH3SaveAVLatent, CiDH3SaveAVLatentPassthrough)

try:
    import torch
    import safetensors
    import safetensors.torch
    import comfy.nested_tensor
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


class NodeDefinitions(unittest.TestCase):
    def test_save_flags(self):
        self.assertIs(CiDH3SaveAVLatent.OUTPUT_NODE, True)
        self.assertIs(CiDH3SaveAVLatentPassthrough.OUTPUT_NODE, False)
        self.assertFalse(getattr(CiDH3LoadAVLatent, "OUTPUT_NODE", False))
        self.assertNotEqual(CiDH3SaveAVLatent.DESCRIPTION, CiDH3SaveAVLatentPassthrough.DESCRIPTION)
        self.assertIs(CiDH3SaveAVLatentPassthrough.save, CiDH3SaveAVLatent.save)

    def test_save_io(self):
        for cls in (CiDH3SaveAVLatent, CiDH3SaveAVLatentPassthrough):
            it = cls.INPUT_TYPES()
            self.assertEqual(list(it["required"]), ["latent", "path", "overwrite"])
            self.assertEqual(it["required"]["latent"][0], "LATENT")
            self.assertEqual(it["required"]["path"][0], "STRING")
            self.assertEqual(it["required"]["path"][1]["default"], "")
            self.assertEqual(it["required"]["overwrite"][0], "BOOLEAN")
            self.assertIs(it["required"]["overwrite"][1]["default"], True)
            self.assertEqual(it["optional"]["note"][0], "STRING")
            self.assertEqual(it["hidden"], {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"})
            self.assertEqual(cls.RETURN_TYPES, ("LATENT", "STRING"))
            self.assertEqual(cls.RETURN_NAMES, ("latent", "path"))
            self.assertEqual(cls.FUNCTION, "save")
            self.assertEqual(cls.CATEGORY, "CiD/video")

    def test_load_io(self):
        it = CiDH3LoadAVLatent.INPUT_TYPES()
        self.assertEqual(list(it["required"]), ["path"])
        self.assertEqual(it["required"]["path"][0], "STRING")
        self.assertEqual(CiDH3LoadAVLatent.RETURN_TYPES, ("LATENT", "INT", "INT", "INT"))
        self.assertEqual(CiDH3LoadAVLatent.RETURN_NAMES, ("latent", "frames", "width", "height"))
        self.assertEqual(CiDH3LoadAVLatent.FUNCTION, "load")
        self.assertEqual(CiDH3LoadAVLatent.CATEGORY, "CiD/video")

    def test_extension_rule(self):
        self.assertEqual(latent_io.with_extension("/a/cut6_p1"), "/a/cut6_p1.h3lat")
        self.assertEqual(latent_io.with_extension("latents/cut6_p1"), "latents/cut6_p1.h3lat")
        self.assertEqual(latent_io.with_extension("/a/cut6_p1.h3lat"), "/a/cut6_p1.h3lat")
        self.assertEqual(latent_io.with_extension("/a/cut6_p1.safetensors"), "/a/cut6_p1.safetensors")
        self.assertEqual(latent_io.resolve_save_path("/a/b/cut"), "/a/b/cut.h3lat")

    def test_load_candidates_absolute(self):
        self.assertEqual(latent_io.load_candidates("/a/cut"), ["/a/cut", "/a/cut.h3lat"])
        self.assertEqual(latent_io.load_candidates("/a/cut.h3lat"), ["/a/cut.h3lat"])

    def test_empty_path(self):
        for p in ("", "   "):
            with self.assertRaisesRegex(ValueError, "set 'path'"):
                CiDH3SaveAVLatent().save(latent={}, path=p)
            with self.assertRaisesRegex(ValueError, "set 'path'"):
                CiDH3LoadAVLatent().load(path=p)
        self.assertEqual(CiDH3LoadAVLatent.IS_CHANGED(""), "")

    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "nope")
            with self.assertRaises(FileNotFoundError) as cm:
                CiDH3LoadAVLatent().load(path=p)
            self.assertIn(p + ".h3lat", str(cm.exception))
            self.assertEqual(CiDH3LoadAVLatent.IS_CHANGED(p), p)

    def test_prompt_hash(self):
        self.assertEqual(latent_io.prompt_hash(None), "")
        self.assertEqual(latent_io.prompt_hash({"b": 1, "a": 2}), latent_io.prompt_hash({"a": 2, "b": 1}))


@unittest.skipUnless(HAVE_TORCH, "needs torch, safetensors and ComfyUI (run scripts/gb10_test.sh latent_io)")
class SaveLoadTensors(unittest.TestCase):
    PROMPT = {"1": {"class_type": "KSampler", "inputs": {"seed": 6602}}}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def make_latent(self, dtype=None, mask=True):
        from cid_nodes import h3_av as common
        g = torch.Generator().manual_seed(1234)
        video = torch.randn(1, 24, 12, 4, 6, generator=g)
        audio = torch.randn(1, 32, 2, 65, generator=g)
        if dtype is not None:
            video, audio = video.to(dtype), audio.to(dtype)
        if not mask:
            return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}
        vmask = torch.ones(12, 4, 6)
        vmask[:7] = 0.0
        return common.join_av(video, audio, video_mask=vmask, base={"batch_index": [0]})

    def read(self, path):
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            return f.metadata(), {k: f.get_tensor(k) for k in f.keys()}

    def roundtrip(self, dtype, cls=CiDH3SaveAVLatent):
        latent = self.make_latent(dtype)
        snapshot = dict(latent)
        target = os.path.join(self.dir, "sub", "cut6_p1")
        out = cls().save(latent=latent, path=target, overwrite=True, note="cut6 p1 seed 6602", prompt=self.PROMPT)
        written = target + ".h3lat"
        self.assertEqual(out["ui"], {"text": [written]})
        out_latent, out_path = out["result"]
        self.assertIs(out_latent, latent)
        self.assertEqual(latent, snapshot)
        self.assertTrue(os.path.isabs(out_path))
        self.assertEqual(out_path, written)
        self.assertTrue(os.path.isfile(written))
        self.assertEqual(os.listdir(os.path.dirname(written)), ["cut6_p1.h3lat"])  # no temp file left

        meta, tensors = self.read(written)
        self.assertEqual(set(tensors), {"video", "audio", "video_noise_mask", "audio_noise_mask"})
        self.assertTrue(torch.equal(tensors["video_noise_mask"], latent["noise_mask"].tensors[0]))
        self.assertTrue(torch.equal(tensors["audio_noise_mask"], torch.ones_like(latent["samples"].tensors[1])))
        self.assertEqual(meta["format"], "h3chain-av-latent")
        self.assertEqual(meta["format_version"], "1")
        self.assertEqual(meta["frames"], "39")
        self.assertEqual(meta["tokens"], "12")
        self.assertEqual(meta["width"], "96")
        self.assertEqual(meta["height"], "64")
        self.assertEqual(meta["audio_ticks"], "65")
        self.assertEqual(meta["batch"], "1")
        self.assertEqual(meta["note"], "cut6 p1 seed 6602")
        self.assertEqual(meta["prompt_hash"],
                         hashlib.sha256(json.dumps(self.PROMPT, sort_keys=True).encode()).hexdigest())
        self.assertIn("comfy_version", meta)
        self.assertTrue(meta["created"].endswith("+00:00"))
        self.assertNotIn("prompt", meta)

        loaded, frames, width, height = CiDH3LoadAVLatent().load(path=written)
        self.assertEqual(set(loaded), {"samples"})
        self.assertTrue(loaded["samples"].is_nested)
        lv, la = loaded["samples"].tensors
        v, a = latent["samples"].tensors
        self.assertEqual(lv.dtype, v.dtype)
        self.assertEqual(la.dtype, a.dtype)
        self.assertTrue(torch.equal(lv, v))
        self.assertTrue(torch.equal(la, a))
        self.assertEqual((frames, width, height), (39, 96, 64))
        # bare name resolves to the .h3lat file too
        self.assertEqual(CiDH3LoadAVLatent().load(path=target)[1:], (39, 96, 64))
        return written

    def test_roundtrip_float32(self):
        self.roundtrip(torch.float32)

    def test_roundtrip_bfloat16(self):
        self.roundtrip(torch.bfloat16)

    def test_passthrough_saves_identically(self):
        a = self.roundtrip(torch.float32, CiDH3SaveAVLatent)
        meta_a, ta = self.read(a)
        os.rename(a, os.path.join(self.dir, "first.h3lat"))
        b = self.roundtrip(torch.float32, CiDH3SaveAVLatentPassthrough)
        meta_b, tb = self.read(b)
        self.assertEqual(set(ta), set(tb))
        for k in ta:
            self.assertTrue(torch.equal(ta[k], tb[k]), k)
        meta_a.pop("created")
        meta_b.pop("created")
        self.assertEqual(meta_a, meta_b)

    def test_no_mask_no_prompt(self):
        latent = self.make_latent(mask=False)
        path = CiDH3SaveAVLatent().save(latent=latent, path=os.path.join(self.dir, "x.h3lat"),
                                          overwrite=True)["result"][1]
        meta, tensors = self.read(path)
        self.assertEqual(set(tensors), {"video", "audio"})
        self.assertEqual(meta["prompt_hash"], "")
        self.assertEqual(meta["note"], "")

    def test_overwrite_false(self):
        path = os.path.join(self.dir, "cut.h3lat")
        CiDH3SaveAVLatent().save(latent=self.make_latent(), path=path, overwrite=True)
        with open(path, "rb") as f:
            before = f.read()
        other = self.make_latent(torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "already exists"):
            CiDH3SaveAVLatent().save(latent=other, path=path, overwrite=False)
        with open(path, "rb") as f:
            self.assertEqual(f.read(), before)
        # overwrite=True replaces it
        CiDH3SaveAVLatent().save(latent=other, path=path, overwrite=True)
        self.assertEqual(CiDH3LoadAVLatent().load(path=path)[0]["samples"].tensors[0].dtype, torch.bfloat16)
        self.assertEqual(os.listdir(self.dir), ["cut.h3lat"])

    def test_load_wrong_format(self):
        path = os.path.join(self.dir, "other.safetensors")
        safetensors.torch.save_file({"video": torch.zeros(1, 24, 2, 1, 1), "audio": torch.zeros(1, 32, 2, 9)},
                                    path, metadata={"format": "something-else"})
        with self.assertRaisesRegex(ValueError, "not an H3 Chain AV latent"):
            CiDH3LoadAVLatent().load(path=path)

    def test_load_missing_audio(self):
        path = os.path.join(self.dir, "noaudio.h3lat")
        safetensors.torch.save_file({"video": torch.zeros(1, 24, 2, 1, 1)}, path,
                                    metadata={"format": "h3chain-av-latent"})
        with self.assertRaisesRegex(ValueError, "no audio tensor"):
            CiDH3LoadAVLatent().load(path=path)

    def test_load_without_metadata(self):
        path = os.path.join(self.dir, "bare.h3lat")
        safetensors.torch.save_file({"video": torch.zeros(1, 24, 7, 2, 3), "audio": torch.zeros(1, 32, 2, 37)}, path)
        _, frames, width, height = CiDH3LoadAVLatent().load(path=path)
        self.assertEqual((frames, width, height), (22, 48, 32))

    def test_load_frames_mismatch_warns(self):
        path = os.path.join(self.dir, "edited.h3lat")
        safetensors.torch.save_file({"video": torch.zeros(1, 24, 7, 2, 3), "audio": torch.zeros(1, 32, 2, 37)},
                                    path, metadata={"format": "h3chain-av-latent", "frames": "39"})
        with self.assertLogs(latent_io.log, "WARNING"):
            _, frames, _, _ = CiDH3LoadAVLatent().load(path=path)
        self.assertEqual(frames, 22)

    def test_is_changed(self):
        path = os.path.join(self.dir, "cut.h3lat")
        CiDH3SaveAVLatent().save(latent=self.make_latent(), path=path, overwrite=True)
        first = CiDH3LoadAVLatent.IS_CHANGED(path)
        self.assertTrue(first.startswith(path))
        self.assertEqual(CiDH3LoadAVLatent.IS_CHANGED(path), first)
        time.sleep(0.01)
        CiDH3SaveAVLatent().save(latent=self.make_latent(torch.bfloat16), path=path, overwrite=True)
        self.assertNotEqual(CiDH3LoadAVLatent.IS_CHANGED(path), first)


if __name__ == "__main__":
    unittest.main()
