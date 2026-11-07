"""Upload files / images to an S3-compatible bucket (Garage on the home server).

Two nodes share one implementation:

* ``CiDS3UploadFile``: takes a file path string (for example the ``path``
  output of H3 Decode & Save) and uploads that file as one object.
* ``CiDS3SaveImage``: writes an IMAGE batch as PNG through the core
  ``SaveImage`` (same file naming and metadata), then uploads every file.

Credentials never appear as node widgets (widgets are saved into workflows and
embedded as PNG/mp4 metadata). They come from, in order of precedence,

1. environment variables ``CID_S3_ENDPOINT``, ``CID_S3_REGION``,
   ``CID_S3_BUCKET``, ``CID_S3_ACCESS_KEY``, ``CID_S3_SECRET_KEY``
   (``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` are accepted too),
2. a JSON file: ``$CID_S3_CONFIG`` if set, else ``s3.json`` next to this
   package's repo root, else ``~/.config/comfy-cid/s3.json`` (see
   ``s3.example.json``).

``endpoint``, ``region`` and ``bucket`` can also be set on the node; an empty
widget falls back to the configuration above.

Garage specifics: path-style addressing (the ``s3.<domain>`` reverse proxy has
no wildcard for ``<bucket>.s3.<domain>``), region ``garage``, SigV4. boto3 is
imported lazily so this module (and its pure helpers) import without it.
"""

import datetime
import json
import logging
import mimetypes
import os
import posixpath
import re

CATEGORY = "CiD/io"

ENV_PREFIX = "CID_S3_"
CONFIG_ENV = "CID_S3_CONFIG"
CONFIG_FILE_NAME = "s3.json"
DEFAULT_REGION = "garage"
CONFIG_KEYS = ("endpoint", "region", "bucket", "access_key", "secret_key")

log = logging.getLogger(__name__)


# --- configuration (plain Python) -----------------------------------------

def config_candidates(env=None):
    """Config files tried in order: $CID_S3_CONFIG, <repo>/s3.json, ~/.config/comfy-cid/s3.json."""
    env = os.environ if env is None else env
    out = []
    explicit = (env.get(CONFIG_ENV) or "").strip()
    if explicit:
        out.append(os.path.expanduser(explicit))
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out.append(os.path.join(repo_root, CONFIG_FILE_NAME))
    out.append(os.path.join(os.path.expanduser("~"), ".config", "comfy-cid", CONFIG_FILE_NAME))
    return out


def load_config_file(paths):
    """First readable JSON file among ``paths`` as a dict (only known keys), plus its path."""
    for p in paths:
        if not os.path.isfile(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("S3 config {}: expected a JSON object".format(p))
        return {k: str(data[k]) for k in CONFIG_KEYS if data.get(k) not in (None, "")}, p
    return {}, None


def env_config(env=None):
    env = os.environ if env is None else env
    out = {}
    for k in CONFIG_KEYS:
        v = (env.get(ENV_PREFIX + k.upper()) or "").strip()
        if v:
            out[k] = v
    # the AWS pair is only a fallback, and only as a pair: half a credential is ignored
    aws = (env.get("AWS_ACCESS_KEY_ID") or "").strip(), (env.get("AWS_SECRET_ACCESS_KEY") or "").strip()
    if "access_key" not in out and "secret_key" not in out and all(aws):
        out["access_key"], out["secret_key"] = aws
    return out


def resolve_config(endpoint="", region="", bucket="", env=None, config_paths=None):
    """Merged settings: node widgets > environment > config file. Raises on anything missing."""
    env = os.environ if env is None else env
    file_cfg, used = load_config_file(config_candidates(env) if config_paths is None else config_paths)
    cfg = dict(file_cfg)
    cfg.update(env_config(env))
    for k, v in (("endpoint", endpoint), ("region", region), ("bucket", bucket)):
        v = (v or "").strip()
        if v:
            cfg[k] = v
    cfg.setdefault("region", DEFAULT_REGION)

    missing = [k for k in ("endpoint", "bucket", "access_key", "secret_key") if not cfg.get(k)]
    if missing:
        where = "config file {}".format(used) if used else "no config file found (tried: {})".format(
            ", ".join(config_candidates(env) if config_paths is None else config_paths))
        raise ValueError(
            "S3: missing {}. Set the node's endpoint/bucket, or {} env vars, or a JSON file "
            "(see s3.example.json); {}.".format(
                ", ".join(missing), ", ".join(ENV_PREFIX + k.upper() for k in missing), where))
    cfg["endpoint"] = normalize_endpoint(cfg["endpoint"])
    return cfg


def normalize_endpoint(endpoint):
    endpoint = endpoint.strip().rstrip("/")
    if not re.match(r"^https?://", endpoint):
        endpoint = "https://" + endpoint
    return endpoint


# --- object keys and URLs --------------------------------------------------

_DATE_RE = re.compile(r"%date:(.*?)%")
_DATE_MAP = (("yyyy", "%Y"), ("yy", "%y"), ("MM", "%m"), ("dd", "%d"), ("hh", "%H"), ("mm", "%M"), ("ss", "%S"))


def expand_date(text, now=None):
    """ComfyUI-style ``%date:yyyy-MM-dd%`` substitution (same tokens as Save Image)."""
    now = now or datetime.datetime.now()

    def sub(m):
        fmt = m.group(1)
        for token, code in _DATE_MAP:
            fmt = fmt.replace(token, code)
        return now.strftime(fmt)
    return _DATE_RE.sub(sub, text)


def build_key(prefix, name, now=None):
    """``prefix/name`` as a clean S3 key: ``%date:...%`` expanded, slashes collapsed, no leading slash."""
    name = (name or "").strip().strip("/")
    if not name:
        raise ValueError("S3: object name is empty")
    parts = [expand_date(prefix or "", now).strip().strip("/"), expand_date(name, now)]
    key = "/".join(p for p in parts if p)
    key = posixpath.normpath(key.replace("\\", "/"))
    if key.startswith("../") or key == ".." or key.startswith("/"):
        raise ValueError("S3: invalid object key '{}'".format(key))
    return key


def object_url(endpoint, bucket, key):
    """Path-style URL of an object (what Garage behind ``s3.<domain>`` serves)."""
    from urllib.parse import quote
    return "{}/{}/{}".format(normalize_endpoint(endpoint), bucket, quote(key, safe="/"))


def s3_uri(bucket, key):
    return "s3://{}/{}".format(bucket, key)


def guess_content_type(path):
    ctype, _ = mimetypes.guess_type(path)
    return ctype or "application/octet-stream"


# --- client (boto3 imported lazily) ---------------------------------------

def make_client(cfg):
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        raise ImportError("S3 nodes need boto3: run `pip install -r custom_nodes/comfy-cid/requirements.txt` "
                          "with ComfyUI's Python") from None
    opts = dict(signature_version="s3v4", s3={"addressing_style": "path"},
                retries={"max_attempts": 3, "mode": "standard"})
    try:   # botocore >= 1.36 adds CRC checksums by default; only send them when the API requires it
        config = Config(request_checksum_calculation="when_required",
                        response_checksum_validation="when_required", **opts)
    except TypeError:
        config = Config(**opts)
    return boto3.client("s3", endpoint_url=cfg["endpoint"], region_name=cfg.get("region") or DEFAULT_REGION,
                        aws_access_key_id=cfg["access_key"], aws_secret_access_key=cfg["secret_key"],
                        config=config)


def object_exists(client, bucket, key):
    from botocore.exceptions import ClientError
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = str(e.response.get("Error", {}).get("Code", ""))
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def upload_file(client, path, bucket, key, content_type=None, overwrite=True, presign_seconds=0, metadata=None):
    """Upload one local file (multipart for large files via boto3) and return (url, s3_uri, key)."""
    if not os.path.isfile(path):
        raise FileNotFoundError("S3: no such file: {}".format(path))
    if not overwrite and object_exists(client, bucket, key):
        raise FileExistsError("S3: s3://{}/{} already exists (turn 'overwrite' on to replace it)".format(bucket, key))
    extra = {"ContentType": content_type or guess_content_type(path)}
    if metadata:
        extra["Metadata"] = {str(k): str(v) for k, v in metadata.items()}
    client.upload_file(path, bucket, key, ExtraArgs=extra)
    endpoint = client.meta.endpoint_url
    if presign_seconds and presign_seconds > 0:
        url = client.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key},
                                            ExpiresIn=int(presign_seconds))
    else:
        url = object_url(endpoint, bucket, key)
    return url, s3_uri(bucket, key), key


def _common_inputs():
    return {
        "key_prefix": ("STRING", {
            "default": "comfyui/%date:yyyy-MM-dd%",
            "tooltip": "Folder part of the object key. May contain %date:yyyy-MM-dd% like Save Image. "
                       "Empty: objects go to the bucket root."}),
        "bucket": ("STRING", {"default": "",
                              "tooltip": "Target bucket. Empty: CID_S3_BUCKET or the config file's bucket."}),
        "endpoint": ("STRING", {"default": "",
                                "tooltip": "S3 endpoint URL, e.g. https://s3.cidbase.dev. Empty: CID_S3_ENDPOINT "
                                           "or the config file's endpoint."}),
        "overwrite": ("BOOLEAN", {"default": False,
                                  "tooltip": "Off: fail instead of replacing an object that already exists."}),
        "presign_seconds": ("INT", {"default": 0, "min": 0, "max": 604800, "step": 60,
                                    "tooltip": "0: 'url' is the plain object URL (needs a public bucket or "
                                               "credentials). >0: 'url' is a presigned GET link valid this many "
                                               "seconds (max 7 days)."}),
        "delete_local": ("BOOLEAN", {"default": False,
                                     "tooltip": "Delete the local file after a successful upload."}),
    }


# --- nodes -----------------------------------------------------------------

class CiDS3UploadFile:
    DESCRIPTION = (
        "Uploads a file (by path, e.g. the 'path' output of H3 Decode & Save) to an S3-compatible bucket "
        "such as Garage. Credentials come from CID_S3_* environment variables or s3.json, never from the "
        "workflow. Outputs the object URL (plain or presigned), the s3:// URI and the key.")
    CATEGORY = CATEGORY
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("url", "s3_uri", "key")
    OUTPUT_TOOLTIPS = ("HTTP URL of the object (presigned when presign_seconds > 0).",
                       "s3://bucket/key", "Object key inside the bucket.")
    FUNCTION = "execute"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        req = {
            "path": ("STRING", {"forceInput": True, "tooltip": "Local path of the file to upload."}),
            "object_name": ("STRING", {"default": "",
                                       "tooltip": "File name inside key_prefix. Empty: the local file's name. "
                                                  "May contain %date:...%."}),
        }
        req.update(_common_inputs())
        return {"required": req}

    def execute(self, path, object_name="", key_prefix="", bucket="", endpoint="", overwrite=False,
                presign_seconds=0, delete_local=False):
        path = os.path.expanduser((path or "").strip())
        if not path:
            raise ValueError("S3 Upload File: 'path' is empty")
        if not os.path.isfile(path):
            raise FileNotFoundError("S3 Upload File: no such file: {}".format(path))
        cfg = resolve_config(endpoint, "", bucket)
        key = build_key(key_prefix, object_name or os.path.basename(path))
        client = make_client(cfg)
        url, uri, key = upload_file(client, path, cfg["bucket"], key, overwrite=overwrite,
                                    presign_seconds=presign_seconds)
        log.info("S3 Upload File: %s -> %s", path, uri)
        if delete_local:
            os.remove(path)
        return {"ui": {"text": [uri]}, "result": (url, uri, key)}


class CiDS3SaveImage:
    DESCRIPTION = (
        "Saves an IMAGE batch as PNG exactly like Save Image (same naming and metadata, written to the "
        "output directory), then uploads each file to an S3-compatible bucket such as Garage. Outputs one "
        "URL / s3:// URI / key per line. Credentials come from CID_S3_* environment variables or s3.json.")
    CATEGORY = CATEGORY
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("urls", "s3_uris", "keys")
    OUTPUT_TOOLTIPS = ("One object URL per line (presigned when presign_seconds > 0).",
                       "One s3://bucket/key per line.", "One object key per line.")
    FUNCTION = "execute"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        req = {
            "images": ("IMAGE", {"tooltip": "Images to save and upload."}),
            "filename_prefix": ("STRING", {"default": "CiD/s3",
                                           "tooltip": "Save Image's filename_prefix (output directory, may "
                                                      "contain subfolders and %date:...%)."}),
        }
        req.update(_common_inputs())
        return {"required": req, "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}}

    def execute(self, images, filename_prefix="CiD/s3", key_prefix="", bucket="", endpoint="", overwrite=False,
                presign_seconds=0, delete_local=False, prompt=None, extra_pnginfo=None):
        import folder_paths
        import nodes
        cfg = resolve_config(endpoint, "", bucket)
        client = make_client(cfg)
        saved = nodes.SaveImage().save_images(images, filename_prefix, prompt, extra_pnginfo)
        out_dir = folder_paths.get_output_directory()
        urls, uris, keys = [], [], []
        for item in saved["ui"]["images"]:
            local = os.path.join(out_dir, item["subfolder"], item["filename"])
            key = build_key(key_prefix, item["filename"])
            url, uri, key = upload_file(client, local, cfg["bucket"], key, content_type="image/png",
                                        overwrite=overwrite, presign_seconds=presign_seconds)
            urls.append(url)
            uris.append(uri)
            keys.append(key)
            if delete_local:
                os.remove(local)
        log.info("S3 Save Image: %d file(s) -> s3://%s/%s", len(keys), cfg["bucket"], key_prefix)
        ui = {"text": uris}
        if not delete_local:
            ui["images"] = saved["ui"]["images"]
        return {"ui": ui, "result": ("\n".join(urls), "\n".join(uris), "\n".join(keys))}
