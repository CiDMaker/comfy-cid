import datetime
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cid_nodes.filename_vars import expand_date, format_date  # noqa: E402


class FormatDate(unittest.TestCase):
    NOW = datetime.datetime(2026, 11, 7, 13, 5, 9)

    def test_tokens_match_frontend_format_date(self):
        self.assertEqual(format_date("yyMMdd", self.NOW), "261107")
        self.assertEqual(format_date("yyyy-MM-dd", self.NOW), "2026-11-07")
        self.assertEqual(format_date("hhmmss", self.NOW), "130509")
        self.assertEqual(format_date("d/M h:m:s", self.NOW), "7/11 13:5:9")       # single letters: no padding
        self.assertEqual(format_date("yyyy_MM_dd-hh_mm_ss", self.NOW), "2026_11_07-13_05_09")
        self.assertEqual(format_date("yyy", self.NOW), "yyy")                     # unknown token kept
        # like the frontend, every d/M/h/m/s letter inside %date:...% is a token (no word boundaries)
        self.assertEqual(format_date("seg", self.NOW), "9eg")

    def test_expand_date(self):
        self.assertEqual(expand_date("Local/%date:yyMMdd%/h3", self.NOW), "Local/261107/h3")
        self.assertEqual(expand_date("a/%date:yyyy-MM-dd%/b_%date:hhmmss%", self.NOW), "a/2026-11-07/b_130509")
        self.assertEqual(expand_date("plain", self.NOW), "plain")
        self.assertEqual(expand_date("%width%x%height%", self.NOW), "%width%x%height%")  # left for core
        self.assertEqual(expand_date("", self.NOW), "")

    def test_default_now_is_today(self):
        self.assertEqual(expand_date("%date:yyyy%"), str(datetime.datetime.now().year))


if __name__ == "__main__":
    unittest.main()
