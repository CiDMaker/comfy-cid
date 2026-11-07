"""``%date:...%`` expansion for filename_prefix / key strings.

The ComfyUI frontend expands ``%date:yyyy-MM-dd%`` only for a fixed list of
core nodes (SaveImage, SaveVideo, ... in its Comfy.SaveImageExtraOutput
extension); any other node receives the raw text, and the backend's
``folder_paths.get_save_image_path`` knows only ``%year%``-style tokens. So the
CiD nodes expand it themselves, with the frontend's ``formatDate`` rules:

* ``yyyy`` full year, ``yy`` last two digits,
* ``d``/``dd`` day, ``M``/``MM`` month, ``h``/``hh`` hour (24 h), ``m``/``mm``
  minute, ``s``/``ss`` second; the doubled form is zero-padded,
* anything else inside ``%date:...%`` (``-``, ``_``, letters) is kept as is.

Pure stdlib; no ComfyUI imports.
"""

import datetime
import re

_DATE_RE = re.compile(r"%date:(.*?)%")
_TOKEN_RE = re.compile(r"dd?|MM?|hh?|mm?|ss?|yyy?y?")
_FIELDS = {"d": "day", "M": "month", "h": "hour", "m": "minute", "s": "second"}


def format_date(fmt, now):
    """``yyMMdd`` -> ``261107``: the frontend's formatDate on one ``%date:...%`` body."""
    def token(m):
        t = m.group(0)
        if t == "yyyy":
            return str(now.year)
        if t == "yy":
            return "{:04d}".format(now.year)[2:]
        if t[0] in _FIELDS:
            return str(getattr(now, _FIELDS[t[0]])).zfill(len(t))
        return t
    return _TOKEN_RE.sub(token, fmt)


def expand_date(text, now=None):
    """Replace every ``%date:FORMAT%`` in ``text`` with the current local time formatted as FORMAT."""
    now = now or datetime.datetime.now()
    return _DATE_RE.sub(lambda m: format_date(m.group(1), now), text)
