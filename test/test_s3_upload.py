"""Tests for the S3 upload nodes. Pure helpers and the upload flow run anywhere
(a fake client stands in for boto3). Set CID_S3_TEST=1 with a working s3.json /
CID_S3_* environment to also run a real round trip against the bucket."""

import datetime
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cid_nodes import s3_upload as s3  # noqa: E402
from cid_nodes.s3_upload import CiDS3SaveImage, CiDS3UploadFile  # noqa: E402

HAS_BOTO3 = importlib.util.find_spec("boto3") is not None


class NodeSchema(unittest.TestCase):
    def test_upload_file(self):
        req = CiDS3UploadFile.INPUT_TYPES()["required"]
        self.assertEqual(list(req), ["path", "object_name", "key_prefix", "bucket", "endpoint", "overwrite",
                                     "presign_seconds", "delete_local"])
        self.assertEqual(req["path"][0], "STRING")
        self.assertIs(req["path"][1]["forceInput"], True)
        self.assertIs(req["overwrite"][1]["default"], False)
        self.assertIs(req["delete_local"][1]["default"], False)
        self.assertEqual((req["presign_seconds"][1]["default"], req["presign_seconds"][1]["max"]), (0, 604800))
        self.assertEqual(CiDS3UploadFile.RETURN_TYPES, ("STRING", "STRING", "STRING"))
        self.assertEqual(CiDS3UploadFile.RETURN_NAMES, ("url", "s3_uri", "key"))
        self.assertIs(CiDS3UploadFile.OUTPUT_NODE, True)
        self.assertEqual(CiDS3UploadFile.CATEGORY, "CiD/io")
        self.assertTrue(hasattr(CiDS3UploadFile, CiDS3UploadFile.FUNCTION))

    def test_save_image(self):
        types = CiDS3SaveImage.INPUT_TYPES()
        req = types["required"]
        self.assertEqual(list(req)[:2], ["images", "filename_prefix"])
        self.assertEqual(req["images"][0], "IMAGE")
        self.assertEqual(req["filename_prefix"][1]["default"], "CiD/s3")
        self.assertEqual(types["hidden"], {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"})
        self.assertEqual(CiDS3SaveImage.RETURN_NAMES, ("urls", "s3_uris", "keys"))
        self.assertIs(CiDS3SaveImage.OUTPUT_NODE, True)

    def test_registered(self):
        from cid_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        self.assertIs(NODE_CLASS_MAPPINGS["CiD_S3UploadFile"], CiDS3UploadFile)
        self.assertIs(NODE_CLASS_MAPPINGS["CiD_S3SaveImage"], CiDS3SaveImage)
        self.assertEqual(NODE_DISPLAY_NAME_MAPPINGS["CiD_S3UploadFile"], "S3 Upload File [CiD]")


class Keys(unittest.TestCase):
    NOW = datetime.datetime(2026, 11, 7, 13, 5, 9)

    def test_expand_date(self):
        self.assertEqual(s3.expand_date("a/%date:yyyy-MM-dd%/b_%date:hhmmss%", self.NOW), "a/2026-11-07/b_130509")
        self.assertEqual(s3.expand_date("plain", self.NOW), "plain")

    def test_build_key(self):
        self.assertEqual(s3.build_key("comfyui/%date:yyyy-MM-dd%", "clip_00001_.mp4", self.NOW),
                         "comfyui/2026-11-07/clip_00001_.mp4")
        self.assertEqual(s3.build_key("", "x.png", self.NOW), "x.png")
        self.assertEqual(s3.build_key("/a//b/", "/c.png", self.NOW), "a/b/c.png")
        self.assertEqual(s3.build_key("a\\b", "c.png", self.NOW), "a/b/c.png")
        with self.assertRaisesRegex(ValueError, "empty"):
            s3.build_key("a", "", self.NOW)
        with self.assertRaisesRegex(ValueError, "invalid object key"):
            s3.build_key("..", "../x.png", self.NOW)

    def test_urls(self):
        self.assertEqual(s3.object_url("https://s3.cidbase.dev/", "comfystudio", "a b/c.mp4"),
                         "https://s3.cidbase.dev/comfystudio/a%20b/c.mp4")
        self.assertEqual(s3.object_url("s3.cidbase.dev", "b", "k"), "https://s3.cidbase.dev/b/k")
        self.assertEqual(s3.s3_uri("b", "k/x"), "s3://b/k/x")

    def test_content_type(self):
        self.assertEqual(s3.guess_content_type("x.mp4"), "video/mp4")
        self.assertEqual(s3.guess_content_type("x.png"), "image/png")
        self.assertEqual(s3.guess_content_type("x.h3lat"), "application/octet-stream")


class ConfigResolution(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="cid_s3_")
        self.cfg_path = os.path.join(self.tmp, "s3.json")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, data):
        with open(self.cfg_path, "w") as f:
            json.dump(data, f)

    def test_file_then_env_then_widget(self):
        self.write({"endpoint": "s3.file.example", "bucket": "filebucket", "access_key": "AKF", "secret_key": "SKF"})
        cfg = s3.resolve_config(env={}, config_paths=[self.cfg_path])
        self.assertEqual(cfg, {"endpoint": "https://s3.file.example", "bucket": "filebucket", "region": "garage",
                               "access_key": "AKF", "secret_key": "SKF"})
        env = {"CID_S3_BUCKET": "envbucket", "CID_S3_REGION": "us-east-1", "AWS_SECRET_ACCESS_KEY": "ignored"}
        cfg = s3.resolve_config(env=env, config_paths=[self.cfg_path])
        self.assertEqual((cfg["bucket"], cfg["region"], cfg["secret_key"]), ("envbucket", "us-east-1", "SKF"))
        cfg = s3.resolve_config(endpoint="http://localhost:3900", bucket=" widget ", env=env,
                                config_paths=[self.cfg_path])
        self.assertEqual((cfg["endpoint"], cfg["bucket"]), ("http://localhost:3900", "widget"))

    def test_env_only_with_aws_names(self):
        env = {"CID_S3_ENDPOINT": "https://s3.x", "CID_S3_BUCKET": "b",
               "AWS_ACCESS_KEY_ID": "AK", "AWS_SECRET_ACCESS_KEY": "SK"}
        cfg = s3.resolve_config(env=env, config_paths=[os.path.join(self.tmp, "missing.json")])
        self.assertEqual((cfg["access_key"], cfg["secret_key"], cfg["region"]), ("AK", "SK", "garage"))

    def test_missing_reports_what_and_where(self):
        with self.assertRaisesRegex(ValueError, r"missing endpoint, bucket, access_key, secret_key.*no config file"):
            s3.resolve_config(env={}, config_paths=[os.path.join(self.tmp, "missing.json")])
        self.write({"endpoint": "e", "bucket": "b", "access_key": "a"})
        with self.assertRaisesRegex(ValueError, r"missing secret_key.*CID_S3_SECRET_KEY.*s3\.json"):
            s3.resolve_config(env={}, config_paths=[self.cfg_path])

    def test_candidates(self):
        env = {"CID_S3_CONFIG": "~/custom.json"}
        c = s3.config_candidates(env)
        self.assertEqual(c[0], os.path.expanduser("~/custom.json"))
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.assertEqual(c[1], os.path.join(repo_root, "s3.json"))
        self.assertTrue(c[2].endswith(os.path.join(".config", "comfy-cid", "s3.json")))
        self.assertEqual(len(s3.config_candidates({})), 2)


class FakeClient:
    """What upload_file / object_exists call on a boto3 S3 client."""

    def __init__(self, endpoint="https://s3.test", existing=()):
        self.meta = type("Meta", (), {"endpoint_url": endpoint})()
        self.existing = set(existing)
        self.uploads = []
        self.presigned = []

    def head_object(self, Bucket, Key):
        if (Bucket, Key) in self.existing:
            return {}
        from botocore.exceptions import ClientError
        raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")

    def upload_file(self, path, bucket, key, ExtraArgs=None):
        self.uploads.append((path, bucket, key, ExtraArgs))
        self.existing.add((bucket, key))

    def generate_presigned_url(self, op, Params, ExpiresIn):
        self.presigned.append((op, Params, ExpiresIn))
        return "https://s3.test/{}/{}?X-Amz-Expires={}".format(Params["Bucket"], Params["Key"], ExpiresIn)


class UploadFlow(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="cid_s3_")
        self.path = os.path.join(self.tmp, "clip_00001_.mp4")
        with open(self.path, "wb") as f:
            f.write(b"\x00" * 16)
        self.cfg = {"endpoint": "https://s3.test", "region": "garage", "bucket": "bk",
                    "access_key": "AK", "secret_key": "SK"}

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_upload_file_helper(self):
        c = FakeClient()
        url, uri, key = s3.upload_file(c, self.path, "bk", "p/clip.mp4")
        self.assertEqual((url, uri, key), ("https://s3.test/bk/p/clip.mp4", "s3://bk/p/clip.mp4", "p/clip.mp4"))
        self.assertEqual(c.uploads, [(self.path, "bk", "p/clip.mp4", {"ContentType": "video/mp4"})])
        url, _, _ = s3.upload_file(c, self.path, "bk", "p/clip.mp4", overwrite=True, presign_seconds=3600)
        self.assertIn("X-Amz-Expires=3600", url)
        self.assertEqual(c.presigned[0][2], 3600)

    @unittest.skipUnless(HAS_BOTO3, "overwrite check raises a botocore ClientError")
    def test_no_overwrite(self):
        c = FakeClient(existing=[("bk", "p/clip.mp4")])
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            s3.upload_file(c, self.path, "bk", "p/clip.mp4", overwrite=False)
        self.assertEqual(c.uploads, [])

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            s3.upload_file(FakeClient(), os.path.join(self.tmp, "nope.mp4"), "bk", "k")

    def test_node_execute(self):
        c = FakeClient()
        node = CiDS3UploadFile()
        with mock.patch.object(s3, "resolve_config", return_value=dict(self.cfg)) as rc, \
                mock.patch.object(s3, "make_client", return_value=c):
            out = node.execute(self.path, "", "comfyui/%date:yyyy%", bucket="other", endpoint="",
                               overwrite=False, presign_seconds=0, delete_local=True)
        rc.assert_called_once_with("", "", "other")
        year = str(datetime.datetime.now().year)
        self.assertEqual(out["result"], ("https://s3.test/bk/comfyui/{}/clip_00001_.mp4".format(year),
                                         "s3://bk/comfyui/{}/clip_00001_.mp4".format(year),
                                         "comfyui/{}/clip_00001_.mp4".format(year)))
        self.assertEqual(out["ui"], {"text": [out["result"][1]]})
        self.assertFalse(os.path.exists(self.path), "delete_local removes the file after upload")

    def test_node_keeps_file_when_upload_fails(self):
        class Failing(FakeClient):
            def upload_file(self, *a, **k):
                raise RuntimeError("boom")
        with mock.patch.object(s3, "resolve_config", return_value=dict(self.cfg)), \
                mock.patch.object(s3, "make_client", return_value=Failing()):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                CiDS3UploadFile().execute(self.path, "x.mp4", "", delete_local=True)
        self.assertTrue(os.path.exists(self.path))

    def test_node_rejects_empty_or_missing_path(self):
        with self.assertRaisesRegex(ValueError, "'path' is empty"):
            CiDS3UploadFile().execute("  ")
        with self.assertRaises(FileNotFoundError):
            CiDS3UploadFile().execute(os.path.join(self.tmp, "nope"))


@unittest.skipUnless(HAS_BOTO3, "builds a real boto3 client (no network)")
class ClientConfig(unittest.TestCase):
    def test_make_client(self):
        c = s3.make_client({"endpoint": "https://s3.cidbase.dev", "region": "garage",
                            "access_key": "AK", "secret_key": "SK"})
        self.assertEqual(c.meta.endpoint_url, "https://s3.cidbase.dev")
        self.assertEqual(c.meta.region_name, "garage")
        self.assertEqual(c.meta.config.s3["addressing_style"], "path")
        self.assertEqual(c.meta.config.signature_version, "s3v4")
        if hasattr(c.meta.config, "request_checksum_calculation"):      # botocore >= 1.36
            self.assertEqual(c.meta.config.request_checksum_calculation, "when_required")
        retries = c.meta.config.retries   # botocore rewrites max_attempts -> total_max_attempts (3 or 3+1 by version)
        self.assertIn(retries.get("max_attempts", retries.get("total_max_attempts")), (3, 4))
        self.assertEqual(retries["mode"], "standard")
        # a missing-dependency message, not a bare ImportError, when boto3 is absent
        with mock.patch.dict(sys.modules, {"boto3": None}):
            with self.assertRaisesRegex(ImportError, "requirements.txt"):
                s3.make_client({"endpoint": "e", "access_key": "a", "secret_key": "s"})


@unittest.skipUnless(os.environ.get("CID_S3_TEST") == "1" and HAS_BOTO3,
                     "set CID_S3_TEST=1 (and configure s3.json / CID_S3_*) for a real round trip")
class LiveRoundTrip(unittest.TestCase):
    def test_put_get_delete(self):
        import uuid
        cfg = s3.resolve_config()
        client = s3.make_client(cfg)
        tmp = tempfile.mkdtemp(prefix="cid_s3_live_")
        path = os.path.join(tmp, "roundtrip.txt")
        payload = uuid.uuid4().hex.encode()
        with open(path, "wb") as f:
            f.write(payload)
        key = s3.build_key("comfy-cid-test/%date:yyyy-MM-dd%", "roundtrip-{}.txt".format(payload[:8].decode()))
        try:
            url, uri, key = s3.upload_file(client, path, cfg["bucket"], key, presign_seconds=300)
            self.assertTrue(s3.object_exists(client, cfg["bucket"], key))
            body = client.get_object(Bucket=cfg["bucket"], Key=key)
            self.assertEqual(body["Body"].read(), payload)
            self.assertEqual(body["ContentType"], "text/plain")
            from urllib.request import urlopen
            with urlopen(url, timeout=20) as r:          # presigned GET through the reverse proxy
                self.assertEqual(r.read(), payload)
            with self.assertRaises(FileExistsError):
                s3.upload_file(client, path, cfg["bucket"], key, overwrite=False)
        finally:
            client.delete_object(Bucket=cfg["bucket"], Key=key)
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
        self.assertFalse(s3.object_exists(client, cfg["bucket"], key))


if __name__ == "__main__":
    unittest.main()
