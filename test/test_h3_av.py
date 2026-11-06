import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cid_nodes import h3_av as common  # noqa: E402


class FrameMath(unittest.TestCase):
    def test_frames_from_tokens(self):
        self.assertEqual(common.frames_from_tokens(2), 5)
        self.assertEqual(common.frames_from_tokens(7), 22)
        self.assertEqual(common.frames_from_tokens(12), 39)
        self.assertEqual(common.frames_from_tokens(57), 192)

    def test_tokens_from_frames_roundtrip(self):
        for tokens in (2, 7, 12, 37, 57):
            self.assertEqual(common.tokens_from_frames(common.frames_from_tokens(tokens)), tokens)

    def test_prefix_tokens_rule(self):
        for ok in (2, 7, 12, 17):
            self.assertEqual(common.validate_prefix_tokens(ok), ok)
        for bad in (0, 1, 3, 5, 6, 8, 10):
            with self.assertRaises(ValueError):
                common.validate_prefix_tokens(bad)
        with self.assertRaisesRegex(ValueError, "use 7 \\(22 frames\\) or 12 \\(39 frames\\)"):
            common.validate_prefix_tokens(9)

    def test_audio_ticks(self):
        self.assertEqual(common.audio_prefix_ticks(22), 37)   # the validated 7-token value
        self.assertEqual(common.audio_prefix_ticks(5), 9)
        self.assertEqual(common.audio_prefix_ticks(39), 65)
        self.assertEqual(common.audio_total_ticks(192), 320)
        self.assertEqual(common.audio_total_ticks(124), 207)


if __name__ == "__main__":
    unittest.main()
