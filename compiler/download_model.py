"""Download the pinned Qwen3 checkpoint and record hashes for offline validation."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from prepare import DEFAULT_MODEL, verify_model


REPO = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
FILES = (
    "config.json", "generation_config.json", "merges.txt", "model.safetensors",
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "README.md",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    manifest_path = args.model_dir / "model_manifest.json"
    if manifest_path.exists():
        manifest = verify_model(args.model_dir)
        if manifest["repo_id"] != REPO or manifest["revision"] != REVISION:
            raise ValueError("Existing checkpoint has a different identity; choose another --model-dir")
        if set(manifest["files"]) != set(FILES):
            raise ValueError("Existing manifest does not contain the expected checkpoint files")
        return
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise ValueError("Hub resolved an unexpected checkpoint revision")
    snapshot_download(REPO, revision=REVISION, local_dir=args.model_dir,
                      allow_patterns=list(FILES), max_workers=4)
    remote = {entry.rfilename: entry for entry in info.siblings}
    files = {}
    for name in FILES:
        path = args.model_dir / name
        with path.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        entry = remote[name]
        if entry.size is not None and path.stat().st_size != entry.size:
            raise ValueError(f"Hub size mismatch: {name}")
        if entry.lfs is not None and sha != entry.lfs.sha256:
            raise ValueError(f"Hub SHA-256 mismatch: {name}")
        files[name] = {"bytes": path.stat().st_size, "sha256": sha}
    manifest_path.write_text(json.dumps(
        {"repo_id": REPO, "revision": REVISION, "files": files}, indent=2) + "\n")
    verify_model(args.model_dir)


if __name__ == "__main__":
    main()
