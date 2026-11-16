import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("record", Path(__file__).with_name("record.py"))
record = importlib.util.module_from_spec(spec)
spec.loader.exec_module(record)


def investigation():
    """Small linked graph for record checks; all paths and results are synthetic."""
    timestamp = "2026-01-01T00:00:00Z"

    def evidence(kind, subjects, **fields):
        return {
            "created_at": timestamp,
            "kind": kind,
            "scope": "model",
            "subjects": subjects,
            "status": "completed",
            "artifacts": ["result"],
            **fields,
        }

    baseline = {"revision": "r0", "export": "export-r0"}
    candidate = {"revision": "r2", "export": "export-r2"}
    return {
        "schema_version": 2,
        "id": "test",
        "created_at": timestamp,
        "objective": "Test record links",
        "context": {key: key for key in ("checkpoint", "workload", "environment")},
        "baseline_revision": "r0",
        "artifacts": {
            key: {"kind": key, "uri": f"{key}.json"}
            for key in (
                "checkpoint",
                "workload",
                "environment",
                "kernel-workload",
                "source",
                "result",
            )
        },
        "candidates": {
            "c1": {
                "created_at": timestamp,
                "baseline_revision": "r0",
                "origin": "Baseline profile",
                "initial_readiness": "Prototype",
                "hypothesis": "Fuse the intermediate because its measured traffic is significant",
                "prediction": "Lower native latency; no repeatable gain would refute the benefit",
                "motivation": ["profile-r0"],
            }
        },
        "revisions": {
            key: {
                "created_at": timestamp,
                "candidate": None if key == "r0" else "c1",
                "parents": [] if key == "r0" else ["r0"],
                "change": key,
                "source": {"commit": None, "artifacts": ["source"]},
            }
            for key in ("r0", "r2")
        },
        "evidence": {
            "export-r0": evidence("export", {"implementation": {"revision": "r0"}}),
            "export-r2": evidence("export", {"implementation": {"revision": "r2"}}),
            "profile-r0": evidence("nsys", {"implementation": baseline}),
            "check-r2": evidence(
                "validation",
                {"implementation": {"revision": "r2"}},
                scope="kernel",
                context={"workload": "kernel-workload"},
            ),
            "model-r2": evidence("validation", {"implementation": candidate}),
            "paired-r2": evidence(
                "paired_benchmark",
                {"baseline": baseline, "candidate": candidate},
                depends_on=["model-r2"],
            ),
        },
        "decisions": {
            "d1": {
                "created_at": timestamp,
                "revision": "r2",
                "evidence": ["check-r2", "model-r2", "paired-r2"],
                "outcome": "keep",
                "readiness": "Validated",
                "acceptance": {"local": "passed", "model": "passed"},
                "assessment": "Observed the predicted lower latency; required checks pass",
                "next_action": "Use r2 as the next baseline",
                "next_baseline": "r2",
            }
        },
    }


class RecordTests(unittest.TestCase):
    def test_decision_requires_assessment(self):
        data = investigation()
        del data["decisions"]["d1"]["assessment"]
        with self.assertRaisesRegex(ValueError, "assessment"):
            record.check_links(data)

    def test_old_record_version_is_rejected(self):
        data = investigation()
        data["schema_version"] = 1
        with self.assertRaisesRegex(ValueError, "schema_version"):
            record.check_links(data)

    def test_each_schema_validates_its_record_type(self):
        data = investigation()
        cases = {"investigation": data}
        for collection, schema_name in (
            ("artifacts", "artifact"),
            ("candidates", "candidate"),
            ("revisions", "revision"),
            ("evidence", "evidence"),
            ("decisions", "decision"),
        ):
            cases[schema_name] = next(iter(data[collection].values()))
        cases["file-manifest"] = {"files": {"log": {"uri": "run.log", "sha256": "0" * 64}}}
        for name, value in cases.items():
            with self.subTest(schema=name):
                record.validate(value, name)
                with self.assertRaises(ValueError):
                    record.validate({**value, "unexpected": True}, name)
        # The investigation schema must enforce the referenced entry schemas too.
        data["candidates"]["c1"]["hypothesis"] = ""
        with self.assertRaisesRegex(ValueError, "candidates/c1/hypothesis"):
            record.validate(data)

    def test_partial_context_and_optional_fields(self):
        data = investigation()
        evidence = data["evidence"]["check-r2"]
        self.assertEqual(evidence["context"], {"workload": "kernel-workload"})
        evidence.pop("summary", None)
        evidence.pop("depends_on", None)
        record.check_links(data)
        # The investigation itself still supplies the full default context.
        del data["context"]["environment"]
        with self.assertRaisesRegex(ValueError, "environment"):
            record.check_links(data)

    def test_failed_and_skipped_steps_need_an_explanation(self):
        for status in ("failed", "not_run"):
            with self.subTest(status=status):
                data = investigation()
                evidence = data["evidence"]["check-r2"]
                evidence["status"] = status
                evidence.pop("summary", None)
                with self.assertRaisesRegex(ValueError, "summary"):
                    record.check_links(data)
                evidence["summary"] = "Command failed; see the retained log."
                record.check_links(data)

    def test_rejects_bad_links_cycles_and_export_identity(self):
        changes = (
            lambda d: d["revisions"]["r2"]["parents"].append("missing"),
            lambda d: d["revisions"]["r0"]["parents"].append("r2"),
            lambda d: d["evidence"]["model-r2"]["subjects"]["implementation"].update(
                export="export-r0"
            ),
            lambda d: d["evidence"]["check-r2"]["context"].update(workload="missing"),
        )
        for change in changes:
            with self.subTest(change=change):
                data = investigation()
                change(data)
                with self.assertRaises(ValueError):
                    record.check_links(data)

    def test_two_exports_of_one_revision_remain_distinct(self):
        data = investigation()
        data["evidence"]["rebuild"] = copy.deepcopy(data["evidence"]["export-r2"])
        data["evidence"]["paired-r2"]["subjects"] = {
            "baseline": {"revision": "r2", "export": "export-r2"},
            "candidate": {"revision": "r2", "export": "rebuild"},
        }
        record.check_links(data)

    def test_bundle_and_verify_changed_or_missing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "run #1.log"
            source.write_text("finished\n")
            output = root / "files.json"
            artifact = record.bundle(output, [f"log={source}"])
            manifest = record.read_json(output)
            self.assertEqual(manifest["files"]["log"]["uri"], "run%20%231.log")
            data = {"artifacts": {"run": {**artifact, "uri": "files.json"}}}
            counts = record.check_files(data, root)
            self.assertEqual(counts["hashes_checked"], 2)
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                record.bundle(output, [f"log={source}"])
            self.assertEqual(output.read_bytes(), original)
            source.write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "Hash mismatch"):
                record.check_files(data, root)
            source.unlink()
            with self.assertRaisesRegex(ValueError, "Missing file"):
                record.check_files(data, root)

    def test_bundle_rejects_duplicate_roles_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "run.log"
            source.touch()
            output = root / "files.json"
            with self.assertRaisesRegex(ValueError, "unique role"):
                record.bundle(output, [f"log={source}", f"log={source}"])
            self.assertFalse(output.exists())

    def test_file_pointers_and_remote_gaps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "report.json").write_text(json.dumps({"a/b": [{"~key": 3}]}))
            data = {
                "artifacts": {
                    "report": {
                        "kind": "report",
                        "uri": "report.json",
                        "pointer": "/a~1b/0/~0key",
                    },
                    "remote": {"kind": "log", "uri": "ssh://gpu/run.log"},
                }
            }
            counts = record.check_files(data, root)
            self.assertEqual(counts["files_checked"], 1)
            self.assertEqual(counts["remote_files_unchecked"], 1)
            data["artifacts"]["report"]["pointer"] = "/missing"
            with self.assertRaisesRegex(ValueError, "Invalid pointer"):
                record.check_files(data, root)


if __name__ == "__main__":
    unittest.main()
