#!/usr/bin/env python3
"""Prepare the exact FocusEarth checkpoints in the configured local paths."""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from focusearth.configuration import load_configuration
from focusearth.io_utils import file_sha256
from focusearth.models.image_text_encoders import CHECKPOINT_HASHES
from focusearth.models.qwen3_vl import MODEL_ID, MODEL_REVISION

URLS = {
    "CLIP": "https://openaipublic.azureedge.net/clip/models/"
    + CHECKPOINT_HASHES["CLIP"] + "/ViT-B-32.pt",
    "GeoRSCLIP": "https://huggingface.co/Zilun/GeoRSCLIP/resolve/main/ckpt/RS5M_ViT-B-32.pt",
    "RemoteCLIP": "https://huggingface.co/chendelong/RemoteCLIP/resolve/main/RemoteCLIP-ViT-B-32.pt",
    "SkyCLIP-50": "https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/ckpt/SkyCLIP_ViT_B32_top50pct.zip",
}
CHUNK = 8 << 20


def require_space(directory: Path, size: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(directory).free < size:
        raise OSError(f"Insufficient disk space in {directory}: need {size:,} additional bytes")


def download(url: str, partial: Path) -> None:
    """Resume a partial HTTP download when Range is supported; otherwise restart."""
    partial.parent.mkdir(parents=True, exist_ok=True)
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Accept-Encoding": "identity", "User-Agent": "FocusEarth/model-preparation"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    try:
        response = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)
    except urllib.error.HTTPError as exc:
        # An interrupted process may have already received the final byte.
        if exc.code == 416 and offset and exc.headers.get("Content-Range") == f"bytes */{offset}":
            exc.close()
            return
        raise
    with response:
        status = response.status
        length = response.headers.get("Content-Length")
        if status == 206:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if not match:
                raise OSError("Invalid Content-Range from download server")
            start, end, total = map(int, match.groups())
            if start != offset or end != total - 1 or end < start:
                raise OSError("Unexpected partial response; refusing to append")
            remaining = total - offset
            if length is not None and int(length) != remaining:
                raise OSError("Inconsistent range length")
            mode = "ab"
        elif status == 200:
            offset = 0
            if length is None:
                raise OSError("Missing Content-Length; cannot check download size")
            remaining = int(length)
            total = remaining
            mode = "wb"
        else:
            raise OSError(f"Unexpected HTTP status {status}")
        if remaining <= 0:
            raise OSError("Empty download")
        require_space(partial.parent, remaining)
        print(f"  {partial.name}: {offset:,}/{total:,} bytes", flush=True)
        written = 0
        with partial.open(mode) as handle:
            while block := response.read(CHUNK):
                if written + len(block) > remaining:
                    raise OSError("Response exceeds expected size")
                handle.write(block)
                written += len(block)
                if written % (128 << 20) < len(block) or written == remaining:
                    print(f"  {offset + written:,}/{total:,} bytes", flush=True)
        if written != remaining:
            raise OSError(f"Incomplete download: {written:,}/{remaining:,} bytes; rerun to resume")


def verify_encoder(name: str, path: Path) -> None:
    if not path.is_file() or file_sha256(path) != CHECKPOINT_HASHES[name]:
        raise ValueError(f"{name}: missing or incorrect SHA-256 at {path}; move incorrect files aside before retrying")


def extract_skyclip(archive: Path, partial: Path) -> None:
    # Never extract arbitrary archive paths; stream only the unique epoch-20 file.
    with zipfile.ZipFile(archive) as source:
        matches = [item for item in source.infolist()
                   if not item.is_dir() and PurePosixPath(item.filename).name == "epoch_20.pt"]
        if len(matches) != 1:
            raise ValueError("SkyCLIP archive does not contain a unique epoch_20.pt; inspect the official archive")
        require_space(partial.parent, matches[0].file_size)
        with source.open(matches[0]) as src, partial.open("wb") as dst:
            shutil.copyfileobj(src, dst, CHUNK)


def prepare_encoder(name: str, path: Path, check_only: bool = False) -> None:
    if path.exists() or check_only:
        verify_encoder(name, path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".part")
        if name == "SkyCLIP-50":
            archive = path.parent / "SkyCLIP_ViT_B32_top50pct.zip"
            if not archive.exists():
                pending = archive.with_name(archive.name + ".part")
                download(URLS[name], pending)
                with zipfile.ZipFile(pending):
                    pass  # Verify the archive directory before publishing the download.
                pending.rename(archive)
            extract_skyclip(archive, partial)
        elif not partial.is_file() or file_sha256(partial) != CHECKPOINT_HASHES[name]:
            download(URLS[name], partial)
        verify_encoder(name, partial)
        # Publish only verified weights. Never replace a user's existing checkpoint.
        if path.exists():
            raise FileExistsError(f"Destination appeared during download: {path}")
        partial.rename(path)
    print(f"[OK] {name}: {path}", flush=True)


def verify_hub_file(path: Path, sibling) -> None:
    if not path.is_file() or path.stat().st_size != sibling.size:
        raise ValueError(f"Missing or incorrect file size: {path}")
    if sibling.lfs is not None:
        actual = file_sha256(path)
        expected = sibling.lfs.sha256
    else:
        # Git blob identity includes the header, not just the file bytes.
        digest = hashlib.sha1(f"blob {sibling.size}\0".encode())
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(CHUNK), b""):
                digest.update(block)
        actual, expected = digest.hexdigest(), sibling.blob_id
    if not expected or actual != expected:
        raise ValueError(f"Qwen file identity mismatch: {path}; move incorrect files aside before retrying")


def prepare_qwen(directory: Path, check_only: bool = False) -> None:
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION, files_metadata=True)
    if info.sha != MODEL_REVISION or not info.siblings:
        raise ValueError("Cannot establish the pinned Qwen snapshot identity")
    files = {entry.rfilename: entry for entry in info.siblings}
    required = {"config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json",
                "model.safetensors.index.json"}
    if not required <= files.keys() or not any(n.endswith(".safetensors") for n in files):
        raise ValueError("Pinned Qwen snapshot metadata is incomplete")
    missing = []
    for name, entry in files.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name or entry.size is None:
            raise ValueError(f"Invalid snapshot metadata: {name}")
        path = directory / name
        if path.exists():
            verify_hub_file(path, entry)
        else:
            missing.append(name)
    if directory.exists():
        extras = [p for p in directory.rglob("*") if p.is_file()
                  and p.relative_to(directory).parts[0] != ".cache"
                  and p.relative_to(directory).as_posix() not in files]
        if extras:
            raise ValueError(f"Unexpected files in Qwen snapshot: {extras[:5]}; use a clean model directory")
    if missing:
        if check_only:
            raise ValueError(f"Incomplete Qwen snapshot: {', '.join(missing)}")
        size = sum(files[name].size for name in missing)
        print(f"Qwen: {size:,} bytes of missing files (before cache/temporary overhead)", flush=True)
        require_space(directory, size)
        snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, local_dir=directory,
                          allow_patterns=missing, max_workers=2)
        for name in missing:
            verify_hub_file(directory / name, files[name])
    print(f"[OK] {MODEL_ID}@{MODEL_REVISION}: {directory}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/mme_realworld_rs.yaml"))
    parser.add_argument("--only", choices=[*URLS, "Qwen"], help="prepare just one model")
    parser.add_argument("--check-only", action="store_true", help="verify without downloading weights (Qwen needs online metadata)")
    args = parser.parse_args(argv)
    try:
        config = load_configuration(args.config)
        for name, value in config["scoring_encoders"]["checkpoints"].items():
            if args.only is None or args.only == name:
                prepare_encoder(name, Path(value), args.check_only)
        if args.only in (None, "Qwen"):
            prepare_qwen(Path(config["answer_model"]["checkpoint"]), args.check_only)
    except Exception as exc:
        print(f"Model preparation failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted. Rerun the command to reuse downloaded files and partial downloads.", file=sys.stderr)
        return 130
    print("Selected checkpoints verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
