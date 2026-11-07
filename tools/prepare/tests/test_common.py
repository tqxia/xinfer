"""CPU contracts for verified preparation, publication and offline input tools."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

from compiler.common.artifacts import MODEL_MANIFEST_NAME
from tools.prepare import cli, common


MODEL = "Qwen/fixture"
REVISION = "1" * 40


def descriptor(data):
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def file_info(name, data, *, lfs=False):
    return SimpleNamespace(
        rfilename=name,
        size=len(data),
        lfs=SimpleNamespace(sha256=hashlib.sha256(data).hexdigest()) if lfs else None,
        blob_id=hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest(),
    )


class HubFiles(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = b"original checkpoint bytes\n"
        self.source = self.root / "cached-file"
        self.source.write_bytes(self.data)

    def install(self, files, revision=REVISION):
        api = SimpleNamespace(
            model_info=Mock(return_value=SimpleNamespace(sha=revision, siblings=files))
        )
        download = Mock(return_value=str(self.source))
        module = SimpleNamespace(HfApi=Mock(return_value=api), hf_hub_download=download)
        replacement = patch.dict(sys.modules, huggingface_hub=module)
        replacement.start()
        self.addCleanup(replacement.stop)
        return api, download

    def test_fetch_verifies_git_blob_and_requests_the_pinned_metadata(self):
        api, download = self.install([file_info("config.json", self.data)])
        cache = self.root / "cache"
        hub = common.Hub(MODEL, REVISION, cache_dir=cache)
        self.assertEqual(hub.fetch("config.json"), self.source)
        api.model_info.assert_called_once_with(MODEL, revision=REVISION, files_metadata=True)
        download.assert_called_once_with(MODEL, "config.json", revision=REVISION, cache_dir=cache)
        self.assertEqual(hub.verified, {"config.json": descriptor(self.data)})
        self.assertEqual(self.source.read_bytes(), self.data)

    def test_copy_verifies_lfs_metadata_from_both_supported_hub_representations(self):
        for lfs in (
            SimpleNamespace(sha256=hashlib.sha256(self.data).hexdigest()),
            dict(sha256=hashlib.sha256(self.data).hexdigest()),
        ):
            with self.subTest(lfs=lfs):
                entry = file_info("model.safetensors", self.data, lfs=True)
                entry.lfs = lfs
                _, download = self.install([entry])
                hub = common.Hub(MODEL, REVISION)
                destination = self.root / "model.safetensors"
                actual = hub.copy("model.safetensors", destination)
                self.assertEqual(actual, descriptor(self.data))
                self.assertEqual(hub.verified["model.safetensors"], actual)
                self.assertEqual(destination.read_bytes(), self.data)
                download.assert_called_once_with(
                    MODEL, "model.safetensors", revision=REVISION, cache_dir=None
                )

    def test_wrong_revision_is_rejected_before_any_file_download(self):
        _, download = self.install([file_info("config.json", self.data)], revision="2" * 40)
        with self.assertRaisesRegex(ValueError, "revision differs"):
            common.Hub(MODEL, REVISION)
        download.assert_not_called()

    def test_missing_files_and_traversal_are_rejected_before_download(self):
        names = ("missing.json", "../escape", "/absolute", "nested/../../escape")
        entries = [file_info(name, self.data) for name in names[1:]]
        _, download = self.install(entries)
        hub = common.Hub(MODEL, REVISION)
        for name in names:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "missing or invalid"):
                hub.fetch(name)
        download.assert_not_called()
        self.assertEqual(hub.verified, {})

    def test_corrupt_files_and_missing_checksum_metadata_are_not_recorded_as_verified(self):
        for mutation, message in (
            ("size", "size mismatch"),
            ("size_type", "size mismatch"),
            ("lfs_digest", "checksum mismatch"),
            ("blob_digest", "checksum mismatch"),
            ("missing_lfs", "checksum metadata is missing"),
            ("missing_blob", "checksum metadata is missing"),
        ):
            with self.subTest(mutation=mutation):
                entry = file_info("file", self.data, lfs=mutation in ("lfs_digest", "missing_lfs"))
                if mutation == "size":
                    entry.size += 1
                elif mutation == "size_type":
                    entry.size = True
                elif mutation == "lfs_digest":
                    entry.lfs.sha256 = "0" * 64
                elif mutation == "blob_digest":
                    entry.blob_id = "0" * 40
                elif mutation == "missing_lfs":
                    entry.lfs = {}
                else:
                    entry.blob_id = None
                self.install([entry])
                hub = common.Hub(MODEL, REVISION)
                with self.assertRaisesRegex(ValueError, message):
                    hub.fetch("file")
                self.assertEqual(hub.verified, {})

    def test_failed_verified_copy_cannot_publish_a_partial_checkpoint(self):
        entry = file_info("model.safetensors", self.data, lfs=True)
        entry.lfs.sha256 = "0" * 64
        self.install([entry])
        hub = common.Hub(MODEL, REVISION)
        output = self.root / "checkpoint"
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            with common.checkpoint_directory(output) as staging:
                hub.copy("model.safetensors", staging / "model.safetensors")
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.iterdir()), [self.source])
        self.assertEqual(self.source.read_bytes(), self.data)


class Publication(unittest.TestCase):
    def test_success_publishes_once_and_existing_output_is_preserved(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "checkpoint"
            with common.checkpoint_directory(output) as staging:
                self.assertFalse(output.exists())
                (staging / "complete").write_text("prepared")
            self.assertEqual((output / "complete").read_text(), "prepared")
            self.assertEqual(list(output.parent.iterdir()), [output])
            with self.assertRaises(FileExistsError):
                with common.checkpoint_directory(output):
                    self.fail("Existing checkpoint was accepted")
            self.assertEqual((output / "complete").read_text(), "prepared")


class OfflineTokens(unittest.TestCase):
    def test_tokenizer_loading_is_local_only(self):
        tokenizer = object()
        load = Mock(return_value=tokenizer)
        module = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=load))
        with patch.dict(sys.modules, transformers=module):
            self.assertIs(common.load_tokenizer(Path("checkpoint")), tokenizer)
        load.assert_called_once_with(Path("checkpoint"), local_files_only=True)

    def test_chat_template_flags_and_prompt_boundaries(self):
        tokenizer = SimpleNamespace(apply_chat_template=Mock(return_value=[7, 8]))
        self.assertEqual(common.encode_prompt(tokenizer, "Hello"), [7, 8])
        tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "Hello"}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )
        for text in ("", " \n"):
            tokenizer.apply_chat_template.reset_mock()
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "must not be empty"):
                common.encode_prompt(tokenizer, text)
            tokenizer.apply_chat_template.assert_not_called()
        tokenizer.apply_chat_template.return_value = []
        with self.assertRaisesRegex(ValueError, "Prompt must contain tokens"):
            common.encode_prompt(tokenizer, "Hello")
        for count in (512, 513, 8192, 8193, 16384):
            tokenizer.apply_chat_template.return_value = [7] * count
            self.assertEqual(len(common.encode_prompt(tokenizer, "Hello")), count)


class PreparationCLI(unittest.TestCase):
    def checkpoint(self, directory, model, *, qwen3=False):
        manifest = {"repo_id" if qwen3 else "model": model, "revision": REVISION}
        (directory / MODEL_MANIFEST_NAME).write_text(json.dumps(manifest))
        (directory / "config.json").write_text(json.dumps(dict(max_position_embeddings=32768)))
        return manifest

    def test_download_routes_every_model_and_honors_explicit_output_and_cache(self):
        for model in cli.MODELS:
            for explicit in (False, True):
                with self.subTest(model=model, explicit=explicit):
                    hooks = SimpleNamespace(
                        default_model_dir=Mock(return_value=Path("default/model")),
                        prepare=Mock(return_value={"revision": REVISION}),
                    )
                    args = ["download", "--model", model]
                    if explicit:
                        args += ["--output", "custom/model", "--cache-dir", "custom/cache"]
                    with (
                        patch.object(cli, "model_hooks", return_value=hooks) as route,
                        redirect_stdout(io.StringIO()),
                    ):
                        cli.main(args)
                    route.assert_called_once_with(model)
                    hooks.prepare.assert_called_once_with(
                        Path("custom/model" if explicit else "default/model"),
                        model,
                        cache_dir=Path("custom/cache") if explicit else None,
                    )
                    if explicit:
                        hooks.default_model_dir.assert_not_called()
                    else:
                        hooks.default_model_dir.assert_called_once_with(model)

    def test_verify_identifies_both_manifest_forms_without_loading_a_tokenizer(self):
        for model, family in cli.MODELS.items():
            with self.subTest(model=model), TemporaryDirectory() as temporary:
                directory = Path(temporary)
                manifest = self.checkpoint(directory, model, qwen3=family == "qwen3")
                hooks = SimpleNamespace(verify_model=Mock(return_value=manifest))
                with (
                    patch.object(cli, "model_hooks", return_value=hooks) as route,
                    patch.object(cli, "load_tokenizer") as tokenizer,
                    redirect_stdout(io.StringIO()) as output,
                ):
                    cli.main(["verify", "--model-dir", str(directory)])
                route.assert_called_once_with(model)
                hooks.verify_model.assert_called_once_with(directory)
                tokenizer.assert_not_called()
                self.assertIn(f"Verified {model} at {REVISION}", output.getvalue())

    def test_unknown_models_and_malformed_manifests_fail_before_hooks(self):
        with patch.object(cli, "model_hooks") as hooks, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as stopped:
                cli.main(["download", "--model", "Qwen/unknown"])
            self.assertEqual(stopped.exception.code, 2)
            hooks.assert_not_called()
        for manifest in ([], {"model": "Qwen/unknown"}, {"repo_id": []}):
            with self.subTest(manifest=manifest), TemporaryDirectory() as temporary:
                directory = Path(temporary)
                (directory / MODEL_MANIFEST_NAME).write_text(json.dumps(manifest))
                with patch.object(cli, "model_hooks") as hooks:
                    with self.assertRaises(ValueError):
                        cli.main(["verify", "--model-dir", str(directory)])
                hooks.assert_not_called()

    def test_encode_uses_shared_offline_helpers_and_refuses_existing_output(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.checkpoint(directory, next(iter(cli.MODELS)), qwen3=True)
            output = directory / "inputs/prompt.ids"
            tokenizer = SimpleNamespace(apply_chat_template=Mock(return_value=[4, 5]))
            args = [
                "encode",
                "--model-dir",
                str(directory),
                "--prompt",
                "Hello",
                "--output",
                str(output),
            ]
            with (
                patch.object(cli, "load_tokenizer", return_value=tokenizer) as load,
                redirect_stdout(io.StringIO()),
            ):
                cli.main(args)
                load.assert_called_once_with(directory)
                self.assertEqual(output.read_text(), "4 5\n")
                load.reset_mock()
                with self.assertRaisesRegex(ValueError, "Output already exists"):
                    cli.main(args)
                load.assert_not_called()
            with self.assertRaises(FileExistsError):
                common.write_ids(output, [9])
            self.assertEqual(output.read_text(), "4 5\n")

    def test_encode_uses_declared_model_ceiling_and_rejects_retired_prompt_cap(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.checkpoint(directory, next(iter(cli.MODELS)), qwen3=True)
            (directory / "config.json").write_text(
                json.dumps(dict(text_config=dict(max_position_embeddings=10000)))
            )
            tokenizer = SimpleNamespace(apply_chat_template=Mock(return_value=[4] * 8193))

            def arguments(name, extra=()):
                return [
                    "encode",
                    "--model-dir",
                    str(directory),
                    "--prompt",
                    "Hello",
                    "--output",
                    str(directory / name),
                    *extra,
                ]

            with (
                patch.object(cli, "load_tokenizer", return_value=tokenizer),
                redirect_stdout(io.StringIO()),
            ):
                cli.main(arguments("long.ids"))
                self.assertEqual(len((directory / "long.ids").read_text().split()), 8193)
                with (
                    patch("sys.stderr", new_callable=io.StringIO),
                    self.assertRaises(SystemExit) as stopped,
                ):
                    cli.main(arguments("capped.ids", ["--max-prompt-length", "128"]))
                self.assertEqual(stopped.exception.code, 2)
                self.assertFalse((directory / "capped.ids").exists())
                tokenizer.apply_chat_template.return_value = [4] * 10001
                with self.assertRaisesRegex(ValueError, "Expected 1..10000"):
                    cli.main(arguments("oversized.ids"))
                self.assertFalse((directory / "oversized.ids").exists())

    def test_decode_rejects_invalid_ids_before_calling_the_tokenizer_decoder(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.checkpoint(directory, next(iter(cli.MODELS)), qwen3=True)
            source = directory / "generated.ids"
            tokenizer = MagicMock()
            tokenizer.__len__.return_value = 10
            tokenizer.decode.return_value = "decoded text"
            args = ["decode", "--model-dir", str(directory), "--input", str(source)]
            with patch.object(cli, "load_tokenizer", return_value=tokenizer):
                source.write_text("1 2\n")
                with redirect_stdout(io.StringIO()) as output:
                    cli.main(args)
                self.assertEqual(output.getvalue(), "decoded text\n")
                tokenizer.decode.assert_called_once_with([1, 2], skip_special_tokens=True)
                for text in ("-1", "10", "1 nope"):
                    tokenizer.decode.reset_mock()
                    source.write_text(text)
                    with self.subTest(ids=text), self.assertRaises(ValueError):
                        cli.main(args)
                    tokenizer.decode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
