"""Verified Hub file access, checkpoint publication and offline token helpers."""

from contextlib import contextmanager
import hashlib
from pathlib import Path
import re
import shutil
import tempfile

from compiler.common.artifacts import require


@contextmanager
def checkpoint_directory(directory):
    """Publish a complete checkpoint only after preparation succeeds."""
    directory = Path(directory)
    if directory.exists() or directory.is_symlink():
        raise FileExistsError(f"Output already exists: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{directory.name}.prepare-", dir=directory.parent
    ) as temporary:
        staging = Path(temporary)
        yield staging
        if directory.exists() or directory.is_symlink():
            raise FileExistsError(f"Output already exists: {directory}")
        staging.rename(directory)


class Hub:
    """Fetch a pinned revision and verify files against published Hub metadata."""

    def __init__(self, model, revision, cache_dir=None):
        from huggingface_hub import HfApi

        self.model = model
        self.revision = revision
        self.cache_dir = cache_dir
        info = HfApi().model_info(model, revision=revision, files_metadata=True)
        require(info.sha == revision, "Hub revision differs from pinned checkpoint")
        self.files = {item.rfilename: item for item in info.siblings}
        self.verified = {}

    def _download(self, name):
        from huggingface_hub import hf_hub_download

        relative = Path(name)
        require(
            not relative.is_absolute() and ".." not in relative.parts and name in self.files,
            f"Checkpoint file is missing or invalid: {name}",
        )
        return Path(
            hf_hub_download(
                self.model,
                name,
                revision=self.revision,
                cache_dir=self.cache_dir,
            )
        )

    def _verify(self, path, name):
        metadata = self.files[name]
        size = path.stat().st_size
        require(
            type(metadata.size) is int and size == metadata.size,
            f"Hub file size mismatch: {name}",
        )
        lfs = metadata.lfs
        expected = (
            (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None))
            if lfs is not None
            else metadata.blob_id
        )
        width = 64 if lfs is not None else 40
        require(
            isinstance(expected, str) and re.fullmatch(f"[0-9a-f]{{{width}}}", expected),
            f"Hub checksum metadata is missing: {name}",
        )
        digest = hashlib.sha256()
        blob = None if lfs is not None else hashlib.sha1(f"blob {size}\0".encode())
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
                if blob is not None:
                    blob.update(chunk)
        actual = digest.hexdigest() if blob is None else blob.hexdigest()
        require(actual == expected, f"Hub checksum mismatch: {name}")
        descriptor = dict(bytes=size, sha256=digest.hexdigest())
        self.verified[name] = descriptor
        return descriptor

    def fetch(self, name):
        path = self._download(name)
        self._verify(path, name)
        return path

    def copy(self, name, destination):
        path = self._download(name)
        shutil.copyfile(path, destination)
        return self._verify(Path(destination), name)


def load_tokenizer(model_dir):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_dir, local_files_only=True)


def encode_prompt(tokenizer, prompt):
    require(bool(prompt.strip()), "Prompt must not be empty")
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=False,
    )
    require(bool(ids), "Prompt must contain tokens")
    return ids


def write_ids(path, ids):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output:
        output.write(" ".join(map(str, ids)) + "\n")
