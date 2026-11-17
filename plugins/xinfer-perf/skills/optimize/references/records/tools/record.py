"""Inspect, bundle and check linked investigation records."""

import argparse
import hashlib
import json
import os
from datetime import datetime
from graphlib import TopologicalSorter
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

SCHEMAS = Path(__file__).resolve().parents[1] / "schemas"
COLLECTIONS = ("artifacts", "candidates", "revisions", "evidence", "decisions")


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def bundle(output, files):
    """Create one immutable inventory; file arguments are role=path pairs."""
    output = Path(output).resolve()
    entries = {}
    for item in files:
        role, separator, name = item.partition("=")
        if not role or not separator or not name or role in entries:
            raise ValueError(f"Expected a unique role=path: {item}")
        path = Path(name).resolve()
        entries[role] = {
            "uri": quote(os.path.relpath(path, output.parent), safe="/"),
            "sha256": digest(path),
        }
    if not entries:
        raise ValueError("A bundle needs at least one file")
    # Do not replace a manifest already referenced by an investigation.
    with output.open("x") as stream:
        stream.write(json.dumps({"files": entries}, indent=2) + "\n")
    return {"kind": "file_manifest", "uri": output.as_uri(), "sha256": digest(output)}


def validate(data, schema_name="investigation"):
    from jsonschema import Draft202012Validator, FormatChecker
    from referencing import Registry, Resource

    registry = Registry().with_resources(
        (path.as_uri(), Resource.from_contents(read_json(path)))
        for path in SCHEMAS.glob("*.schema.json")
    )
    schema = {"$ref": (SCHEMAS / f"{schema_name}.schema.json").as_uri()}
    validator = Draft202012Validator(schema, registry=registry, format_checker=FormatChecker())
    error = next(validator.iter_errors(data), None)
    if error:
        location = "/".join(map(str, error.absolute_path)) or "/"
        raise ValueError(f"{location}: {error.message}")


def check_links(record):
    validate(record)
    edges = {(collection, key): set() for collection in COLLECTIONS for key in record[collection]}
    root = ("investigation", record["id"])
    edges[root] = set()

    def link(node, collection, key):
        target = (collection, key)
        if target not in edges:
            raise ValueError(f"{node}: unresolved {target}")
        edges[node].add(target)

    def links(node, collection, keys):
        for key in keys:
            link(node, collection, key)

    links(root, "artifacts", record["context"].values())
    link(root, "revisions", record["baseline_revision"])
    for key, candidate in record["candidates"].items():
        node = ("candidates", key)
        link(node, "revisions", candidate["baseline_revision"])
        links(node, "evidence", candidate["motivation"])
        if "follows_decision" in candidate:
            link(node, "decisions", candidate["follows_decision"])
    for key, revision in record["revisions"].items():
        node = ("revisions", key)
        if revision["candidate"] is not None:
            link(node, "candidates", revision["candidate"])
        links(node, "revisions", revision["parents"])
        links(node, "artifacts", revision["source"]["artifacts"])
        links(node, "evidence", revision.get("motivation", []))
        if "follows_decision" in revision:
            link(node, "decisions", revision["follows_decision"])
    for key, evidence in record["evidence"].items():
        node = ("evidence", key)
        context = {**record["context"], **evidence.get("context", {})}
        links(node, "artifacts", context.values())
        links(node, "artifacts", evidence["artifacts"])
        links(node, "evidence", evidence.get("depends_on", []))
        for subject in evidence["subjects"].values():
            link(node, "revisions", subject["revision"])
            if "export" in subject:
                link(node, "evidence", subject["export"])
                exported = record["evidence"][subject["export"]]
                revisions = {value["revision"] for value in exported["subjects"].values()}
                if exported["kind"] != "export" or revisions != {subject["revision"]}:
                    raise ValueError(f"{key}: export does not identify the subject revision")
    for key, decision in record["decisions"].items():
        node = ("decisions", key)
        link(node, "revisions", decision["revision"])
        links(node, "evidence", decision["evidence"])
        if "next_baseline" in decision:
            link(node, "revisions", decision["next_baseline"])
    # Includes feedback links across collections, not just revision parentage.
    list(TopologicalSorter(edges).static_order())


def latest(entries):
    """Retain timestamp ties; JSON insertion order does not establish chronology."""
    times = {
        key: datetime.fromisoformat(value["created_at"].replace("Z", "+00:00"))  # noqa: FURB162 -- Python 3.9 CLI support
        for key, value in entries.items()
    }
    newest = max(times.values(), default=None)
    return {key: value for key, value in entries.items() if times[key] == newest}


def evidence_index(entries):
    fields = ("kind", "scope", "status", "subjects", "summary")
    return {
        key: {field: value[field] for field in fields if field in value}
        for key, value in entries.items()
    }


def show(record, target=None):
    """Project a validated record without reading artifacts or inferring acceptance."""
    if target is None:
        candidates = {}
        for key, candidate in record["candidates"].items():
            revisions = [
                rid for rid, revision in record["revisions"].items() if revision["candidate"] == key
            ]
            decisions = {
                did: decision
                for did, decision in record["decisions"].items()
                if decision["revision"] in revisions
            }
            selections = latest(
                {
                    did: decision
                    for did, decision in decisions.items()
                    if "next_baseline" in decision
                }
            )
            candidates[key] = {
                "hypothesis": candidate["hypothesis"],
                "baseline_revision": candidate["baseline_revision"],
                "revisions": revisions,
                "latest_decisions": latest(decisions),
                "latest_baseline_selections": {
                    did: decision["next_baseline"] for did, decision in selections.items()
                },
            }
        return {
            "investigation": {
                key: value for key, value in record.items() if key not in COLLECTIONS
            },
            "counts": {key: len(record[key]) for key in COLLECTIONS},
            "candidates": candidates,
            "running_evidence": evidence_index(
                {
                    key: value
                    for key, value in record["evidence"].items()
                    if value["status"] == "running"
                }
            ),
        }

    collection, _, key = target.partition("/")
    if collection not in COLLECTIONS or key not in record[collection]:
        raise ValueError(f"Unknown record target: {target}; expected collection/id")
    entry = record[collection][key]
    result = {collection: {key: entry}}
    if collection == "candidates":
        revisions = {
            rid: revision
            for rid, revision in record["revisions"].items()
            if revision["candidate"] == key
        }
        decisions = {
            did: decision
            for did, decision in record["decisions"].items()
            if decision["revision"] in revisions
        }
        evidence_ids = set(entry["motivation"])
        for revision in revisions.values():
            evidence_ids.update(revision.get("motivation", []))
        for decision in decisions.values():
            evidence_ids.update(decision["evidence"])
        evidence = {
            eid: value
            for eid, value in record["evidence"].items()
            if eid in evidence_ids
            or any(subject["revision"] in revisions for subject in value["subjects"].values())
        }
        result.update(revisions=revisions, decisions=decisions, evidence=evidence_index(evidence))
    elif collection == "evidence":
        context = {**record["context"], **entry.get("context", {})}
        exports = {
            subject["export"]: record["evidence"][subject["export"]]
            for subject in entry["subjects"].values()
            if "export" in subject
        }
        artifact_ids = set(entry["artifacts"]) | set(context.values())
        for exported in exports.values():
            artifact_ids.update(exported["artifacts"])
        result.update(
            effective_context=context,
            exports=exports,
            artifacts={aid: record["artifacts"][aid] for aid in sorted(artifact_ids)},
        )
    return result


def check_files(record, directory):
    counts = {"files_checked": 0, "hashes_checked": 0, "remote_files_unchecked": 0}

    def check_file(entry, base):
        uri = urlsplit(entry["uri"])
        if uri.scheme not in ("", "file") or uri.netloc not in ("", "localhost"):
            counts["remote_files_unchecked"] += 1
            return None
        path = (base / unquote(uri.path)).resolve()
        if not path.is_file():
            raise ValueError(f"Missing file: {path}")
        counts["files_checked"] += 1
        if "sha256" in entry:
            if digest(path) != entry["sha256"]:
                raise ValueError(f"Hash mismatch: {path}")
            counts["hashes_checked"] += 1
        if "pointer" in entry:
            value = read_json(path)
            for part in entry["pointer"].split("/")[1:]:
                part = part.replace("~1", "/").replace("~0", "~")
                try:
                    if isinstance(value, list):
                        if not part.isdigit() or str(int(part)) != part:
                            raise ValueError(part)
                        part = int(part)
                    value = value[part]
                except (KeyError, IndexError, TypeError, ValueError) as error:
                    raise ValueError(f"Invalid pointer {entry['pointer']!r}: {path}") from error
        return path

    for artifact in record["artifacts"].values():
        path = check_file(artifact, Path(directory))
        if path is not None and artifact["kind"] == "file_manifest":
            manifest = read_json(path)
            validate(manifest, "file-manifest")
            for entry in manifest["files"].values():
                check_file(entry, path.parent)
    return counts


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("bundle", help="Inventory finished files; print one artifact entry")
    pack.add_argument("output", type=Path, help="New manifest path in an existing directory")
    pack.add_argument("files", nargs="+", metavar="ROLE=PATH")
    check = commands.add_parser("check", help="Validate schema and links (requires jsonschema)")
    check.add_argument("record", type=Path)
    check.add_argument("--files", action="store_true", help="Also check local files and hashes")
    view = commands.add_parser(
        "show", help="Show an overview or linked record (requires jsonschema)"
    )
    view.add_argument("record", type=Path)
    view.add_argument(
        "target", nargs="?", metavar="COLLECTION/ID", help="Omit for the investigation overview"
    )
    return parser


def run(args):
    """Execute one record command and return the result for JSON output."""
    if args.command == "bundle":
        result = bundle(args.output, args.files)
    else:
        record = read_json(args.record)
        check_links(record)
        if args.command == "show":
            result = {"record_path": str(args.record.resolve()), **show(record, args.target)}
        else:
            result = {"schema_and_links": "valid", "file_check": "not_requested"}
            if args.files:
                result.update(check_files(record, args.record.resolve().parent))
                result["file_check"] = "partial" if result["remote_files_unchecked"] else "complete"
    return result


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
