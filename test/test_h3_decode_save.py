import importlib.util
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cid_nodes import h3_av as common  # noqa: E402
from cid_nodes.h3_decode_save import (  # noqa: E402
    CiDH3DecodeSave, build_metadata, output_file_name, preview_ui)


def _has(*mods):
    try:
        return all(importlib.util.find_spec(m) is not None for m in mods)
    except (ImportError, ValueError):
        return False


HAS_COMFY = _has("torch", "av", "comfy", "nodes", "folder_paths", "comfy_api")


class NodeSchema(unittest.TestCase):
    def test_inputs(self):
        types = CiDH3DecodeSave.INPUT_TYPES()
        req = types["required"]
        self.assertEqual(list(req), ["latent", "vae", "audio_vae", "filename_prefix", "fps", "crf", "tiled",
                                     "tile_size", "overlap", "temporal_size", "temporal_overlap"])
        self.assertEqual(req["tiled"], ("BOOLEAN", req["tiled"][1]))
        self.assertIs(req["tiled"][1]["default"], True)
        self.assertEqual(req["latent"][0], "LATENT")
        self.assertEqual(req["vae"][0], "VAE")
        self.assertEqual(req["audio_vae"][0], "VAE")
        self.assertEqual(req["filename_prefix"][1]["default"], "CiD/h3_segment")
        self.assertEqual(req["fps"][0], "FLOAT")          # CreateVideo's fps type
        self.assertEqual(req["fps"][1]["default"], 24)
        self.assertEqual((req["crf"][1]["default"], req["crf"][1]["min"], req["crf"][1]["max"]), (18, 0, 51))
        defaults = [req[k][1]["default"] for k in ("tile_size", "overlap", "temporal_size", "temporal_overlap")]
        self.assertEqual(defaults, [512, 64, 64, 8])       # graph node 369 widgets
        # ranges/steps copied from VAEDecodeTiled
        self.assertEqual({k: v for k, v in req["tile_size"][1].items() if k in ("min", "max", "step")},
                         {"min": 64, "max": 4096, "step": 32})
        self.assertEqual({k: v for k, v in req["temporal_overlap"][1].items() if k in ("min", "max", "step")},
                         {"min": 4, "max": 4096, "step": 4})
        self.assertEqual(types["hidden"], {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"})

    def test_outputs(self):
        self.assertEqual(CiDH3DecodeSave.RETURN_TYPES, ("STRING",))
        self.assertEqual(CiDH3DecodeSave.RETURN_NAMES, ("path",))
        self.assertIs(CiDH3DecodeSave.OUTPUT_NODE, True)
        self.assertEqual(CiDH3DecodeSave.CATEGORY, common.CATEGORY)
        self.assertTrue(hasattr(CiDH3DecodeSave, CiDH3DecodeSave.FUNCTION))


class PureHelpers(unittest.TestCase):
    def test_file_name(self):
        self.assertEqual(output_file_name("segment", 7), "segment_00007_.mp4")

    def test_preview_ui(self):
        self.assertEqual(preview_ui("a_00001_.mp4", "CiD"),
                         {"images": [{"filename": "a_00001_.mp4", "subfolder": "CiD", "type": "output"}],
                          "animated": (True,)})

    def test_metadata(self):
        self.assertIsNone(build_metadata())
        self.assertIsNone(build_metadata({"1": {}}, {"workflow": {}}, disable_metadata=True))
        self.assertEqual(build_metadata({"1": {}}, {"workflow": {"n": 1}}),
                         {"workflow": {"n": 1}, "prompt": {"1": {}}})

    def test_rejects_non_latent(self):
        node = CiDH3DecodeSave()
        with self.assertRaisesRegex(ValueError, "expected a LATENT dict"):
            node.execute("nope", None, None, "x", 24, 19, True, 512, 64, 64, 8)

        class Plain:          # a non-nested tensor stand-in
            is_nested = False
        with self.assertRaisesRegex(ValueError, "H3 Decode & Save: expected a MiniMax H3 AV latent"):
            node.execute({"samples": Plain()}, None, None, "x", 24, 19, True, 512, 64, 64, 8)


# --- gb10 only: real core classes, fake VAEs -----------------------------

FRAMES, SIZE, SAMPLE_RATE = 10, 64, 32000


class FakeVideoVAE:
    """Exposes what nodes.VAEDecodeTiled.decode reads."""

    def __init__(self):
        self.calls = []

    def spacial_compression_decode(self):
        return 16

    def temporal_compression_decode(self):
        return 4

    def decode(self, samples):
        import torch
        self.calls.append(dict(shape=tuple(samples.shape), plain=True))
        g = torch.Generator().manual_seed(0)
        return torch.rand(1, FRAMES, SIZE, SIZE, 3, generator=g)

    def decode_tiled(self, samples, tile_x=None, tile_y=None, overlap=None, tile_t=None, overlap_t=None):
        import torch
        self.calls.append(dict(shape=tuple(samples.shape), tile_x=tile_x, tile_y=tile_y, overlap=overlap,
                               tile_t=tile_t, overlap_t=overlap_t))
        # 5D [B, T, H, W, C] like a real video VAE; VAEDecodeTiled flattens it
        # textured frames so the encoder's crf has something to work on
        g = torch.Generator().manual_seed(0)
        return torch.rand(1, FRAMES, SIZE, SIZE, 3, generator=g)


class FakeAudioVAE:
    """Exposes what comfy_extras.nodes_audio.vae_decode_audio reads."""
    audio_sample_rate = SAMPLE_RATE

    def __init__(self):
        self.calls = []

    def decode(self, samples):
        import torch
        self.calls.append(tuple(samples.shape))
        length = SAMPLE_RATE * FRAMES // 24 + SAMPLE_RATE // 10
        t = torch.arange(length) / SAMPLE_RATE
        wave = 0.3 * torch.sin(2 * torch.pi * 440 * t)
        return torch.stack([wave, wave], dim=-1).unsqueeze(0)     # [1, L, 2]; core movedims to [1, 2, L]


@unittest.skipUnless(HAS_COMFY, "needs torch, PyAV and a ComfyUI checkout (run scripts/gb10_test.sh)")
class DecodeSaveSmoke(unittest.TestCase):
    def setUp(self):
        import folder_paths
        self.tmp = tempfile.mkdtemp(prefix="h3chain_decode_save_")
        self.prev_out = folder_paths.get_output_directory()
        folder_paths.set_output_directory(self.tmp)

    def tearDown(self):
        import folder_paths
        folder_paths.set_output_directory(self.prev_out)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _latent(self):
        import torch
        import comfy.nested_tensor
        video = torch.zeros(1, common.VIDEO_CHANNELS, 7, SIZE // 16, SIZE // 16)
        audio = torch.zeros(1, common.AUDIO_CHANNELS, 2, 37)
        return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}

    def test_untiled_uses_plain_vae_decode(self):
        import av
        vae, audio_vae = FakeVideoVAE(), FakeAudioVAE()
        out = CiDH3DecodeSave().execute(self._latent(), vae, audio_vae, "CiD/plain", 24.0, 19, False,
                                          512, 64, 64, 8)
        (path,) = out["result"]
        self.assertTrue(os.path.isfile(path))
        self.assertEqual(len(vae.calls), 1)
        self.assertTrue(vae.calls[0].get("plain"), vae.calls)      # VAEDecode, not decode_tiled
        with av.open(path) as c:
            self.assertEqual(sum(1 for _ in c.decode(video=0)), FRAMES)

    def test_decode_and_save(self):
        import av
        vae, audio_vae = FakeVideoVAE(), FakeAudioVAE()
        node = CiDH3DecodeSave()
        out = node.execute(self._latent(), vae, audio_vae, "CiD/seg", 24.0, 19, True, 512, 64, 64, 8,
                           prompt={"1": {"class_type": "X"}}, extra_pnginfo={"workflow": {"nodes": []}})

        self.assertEqual(set(out), {"ui", "result"})
        (path,) = out["result"]
        self.assertTrue(os.path.isabs(path))
        self.assertEqual(path, os.path.join(self.tmp, "CiD", "seg_00001_.mp4"))
        self.assertTrue(os.path.isfile(path))
        self.assertEqual(out["ui"], {"images": [{"filename": "seg_00001_.mp4", "subfolder": "CiD",
                                                 "type": "output"}], "animated": (True,)})
        self.assertFalse(any(getattr(v, "ndim", 0) == 4 for v in vars(node).values()))

        # the core decoders got the right streams and VAEDecodeTiled's tile mapping
        self.assertEqual(vae.calls, [dict(shape=(1, 24, 7, 4, 4), tile_x=32, tile_y=32, overlap=4,
                                          tile_t=16, overlap_t=2)])
        self.assertEqual(audio_vae.calls, [(1, 32, 2, 37)])

        with av.open(path) as c:
            vs, aus = c.streams.video[0], c.streams.audio[0]
            self.assertEqual((vs.codec_context.name, vs.width, vs.height), ("h264", SIZE, SIZE))
            self.assertEqual(vs.codec_context.pix_fmt, "yuv420p")
            self.assertEqual(vs.average_rate, 24)
            self.assertEqual(aus.codec_context.name, "aac")
            self.assertEqual(aus.sample_rate, SAMPLE_RATE)
            self.assertEqual(aus.channels if hasattr(aus, "channels") else aus.layout.nb_channels, 2)
            self.assertIn("prompt", c.metadata)
            frames = sum(1 for _ in c.decode(video=0))
        self.assertEqual(frames, FRAMES)

        # counter increments like SaveVideo
        out2 = node.execute(self._latent(), vae, audio_vae, "CiD/seg", 24.0, 19, True, 512, 64, 64, 8)
        self.assertEqual(os.path.basename(out2["result"][0]), "seg_00002_.mp4")

    def test_date_in_prefix(self):
        import datetime
        today = datetime.datetime.now().strftime("%y%m%d")
        path = CiDH3DecodeSave().execute(self._latent(), FakeVideoVAE(), FakeAudioVAE(),
                                         "Local/%date:yyMMdd%/h3", 24.0, 19, False, 512, 64, 64, 8)["result"][0]
        self.assertEqual(path, os.path.join(self.tmp, "Local", today, "h3_00001_.mp4"))
        self.assertTrue(os.path.isfile(path))

    def test_crf_reaches_encoder(self):
        import av
        node = CiDH3DecodeSave()
        sizes = {}
        for crf in (0, 51):
            path = node.execute(self._latent(), FakeVideoVAE(), FakeAudioVAE(), "crf{}/x".format(crf),
                                24.0, crf, True, 512, 64, 64, 8)["result"][0]
            with av.open(path) as c:
                sizes[crf] = sum(p.size for p in c.demux(video=0))
        self.assertGreater(sizes[0], sizes[51])

    def test_rejects_video_only_latent(self):
        import torch
        node = CiDH3DecodeSave()
        latent = {"samples": torch.zeros(1, 24, 7, 4, 4)}
        with self.assertRaisesRegex(ValueError, "H3 Decode & Save: expected a MiniMax H3 AV latent"):
            node.execute(latent, FakeVideoVAE(), FakeAudioVAE(), "x", 24.0, 19, True, 512, 64, 64, 8)


if __name__ == "__main__":
    unittest.main()
