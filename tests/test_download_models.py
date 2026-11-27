"""Offline tests for the model preparation utility; no checkpoints or GPU needed."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from scripts import download_models as dm


class Response(io.BytesIO):
    def __init__(self, content, status=200, headers=None):
        super().__init__(content)
        self.status = status
        self.headers = headers or {"Content-Length": str(len(content))}


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_http_range_resume(self):
        path = self.root / "model.part"
        path.write_bytes(b"abc")
        response = Response(b"def", 206, {"Content-Length": "3", "Content-Range": "bytes 3-5/6"})
        with patch.object(dm.urllib.request, "urlopen", return_value=response) as call:
            dm.download("https://example.org/model", path)
        self.assertEqual(call.call_args.args[0].get_header("Range"), "bytes=3-")
        self.assertEqual(path.read_bytes(), b"abcdef")

    def test_http_range_ignored_restarts(self):
        path = self.root / "model.part"
        path.write_bytes(b"abc")
        with patch.object(dm.urllib.request, "urlopen", return_value=Response(b"abcdef")):
            dm.download("https://example.org/model", path)
        self.assertEqual(path.read_bytes(), b"abcdef")

    def test_wrong_range_preserves_partial(self):
        path = self.root / "model.part"
        path.write_bytes(b"abc")
        response = Response(b"ef", 206, {"Content-Range": "bytes 4-5/6"})
        with patch.object(dm.urllib.request, "urlopen", return_value=response):
            with self.assertRaises(OSError):
                dm.download("https://example.org/model", path)
        self.assertEqual(path.read_bytes(), b"abc")

    def test_short_download_preserves_partial(self):
        path = self.root / "model.part"
        with patch.object(dm.urllib.request, "urlopen", return_value=Response(b"abc", headers={"Content-Length": "6"})):
            with self.assertRaises(OSError):
                dm.download("https://example.org/model", path)
        self.assertEqual(path.read_bytes(), b"abc")

    def test_encoder_publish_and_skip(self):
        path = self.root / "clip.pt"
        content = b"checkpoint"
        with patch.dict(dm.CHECKPOINT_HASHES, {"CLIP": hashlib.sha256(content).hexdigest()}):
            with patch.object(dm.urllib.request, "urlopen", return_value=Response(content)) as call:
                dm.prepare_encoder("CLIP", path)
                dm.prepare_encoder("CLIP", path)
                dm.prepare_encoder("CLIP", path, check_only=True)
                self.assertEqual(call.call_count, 1)
        self.assertEqual(path.read_bytes(), content)
        self.assertFalse(path.with_name("clip.pt.part").exists())

    def test_bad_encoder_not_published_or_overwritten(self):
        path = self.root / "clip.pt"
        with patch.object(dm.urllib.request, "urlopen", return_value=Response(b"bad")):
            with self.assertRaises(ValueError):
                dm.prepare_encoder("CLIP", path)
        self.assertFalse(path.exists())
        path.write_bytes(b"existing")
        with patch.object(dm.urllib.request, "urlopen") as call:
            with self.assertRaises(ValueError):
                dm.prepare_encoder("CLIP", path)
            call.assert_not_called()
        self.assertEqual(path.read_bytes(), b"existing")

    def test_skyclip_stream_only_target(self):
        archive = self.root / "SkyCLIP_ViT_B32_top50pct.zip"
        path = self.root / "epoch_20.pt"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("checkpoints/epoch_20.pt", b"target")
            z.writestr("../../unexpected", b"other")
        with patch.dict(dm.CHECKPOINT_HASHES, {"SkyCLIP-50": hashlib.sha256(b"target").hexdigest()}):
            dm.prepare_encoder("SkyCLIP-50", path)
        self.assertEqual(path.read_bytes(), b"target")
        self.assertEqual(set(p.name for p in self.root.iterdir()), {archive.name, path.name})

    def test_skyclip_ambiguous_epoch_rejected(self):
        archive = self.root / "sky.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("a/epoch_20.pt", b"a")
            z.writestr("b/epoch_20.pt", b"b")
        with self.assertRaises(ValueError):
            dm.extract_skyclip(archive, self.root / "partial")

    def fake_hub(self):
        names = ["config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json",
                 "model.safetensors.index.json", "model-00001-of-00001.safetensors"]
        content = {name: name.encode() for name in names}
        siblings = []
        for name, data in content.items():
            lfs = SimpleNamespace(sha256=hashlib.sha256(data).hexdigest()) if name.endswith(".safetensors") else None
            blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            siblings.append(SimpleNamespace(rfilename=name, size=len(data), lfs=lfs, blob_id=blob))
        info = SimpleNamespace(sha=dm.MODEL_REVISION, siblings=siblings)
        calls = []
        def snapshot_download(**kwargs):
            calls.append(kwargs)
            for name in kwargs["allow_patterns"]:
                (Path(kwargs["local_dir"]) / name).write_bytes(content[name])
        module = SimpleNamespace(HfApi=lambda: SimpleNamespace(model_info=lambda *a, **k: info),
                                 snapshot_download=snapshot_download)
        return module, calls, content, info

    def test_qwen_pinned_download_resume_and_verify(self):
        module, calls, content, _ = self.fake_hub()
        (self.root / "config.json").write_bytes(content["config.json"])
        with patch.dict(sys.modules, {"huggingface_hub": module}):
            dm.prepare_qwen(self.root)
            dm.prepare_qwen(self.root, check_only=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["revision"], dm.MODEL_REVISION)
        self.assertNotIn("config.json", calls[0]["allow_patterns"])

    def test_qwen_rejects_bad_version_and_bad_content(self):
        module, calls, content, info = self.fake_hub()
        with patch.dict(sys.modules, {"huggingface_hub": module}):
            info.sha = "wrong"
            with self.assertRaises(ValueError):
                dm.prepare_qwen(self.root)
            info.sha = dm.MODEL_REVISION
            (self.root / "config.json").write_bytes(b"x" * len(content["config.json"]))
            with self.assertRaises(ValueError):
                dm.prepare_qwen(self.root)
        self.assertFalse(calls)

    def test_qwen_rejects_extra_files_and_missing_check_only(self):
        module, calls, _, _ = self.fake_hub()
        with patch.dict(sys.modules, {"huggingface_hub": module}):
            with self.assertRaises(ValueError):
                dm.prepare_qwen(self.root, check_only=True)
            (self.root / "pytorch_model.bin").write_bytes(b"unexpected")
            with self.assertRaises(ValueError):
                dm.prepare_qwen(self.root)
        self.assertFalse(calls)


if __name__ == "__main__":
    unittest.main()
