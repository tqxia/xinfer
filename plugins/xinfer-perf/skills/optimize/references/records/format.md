# Linked experiment records

Keep one `investigation.json` with `schema_version: 2` in the investigation's
ignored `work/` or `artifacts/` directory. Link existing reports for detailed
results. Use `record.py check` routinely; consult the relevant schema only for
unclear fields or validation errors.

## Record relationships

The [investigation](schemas/investigation.schema.json) records its stable `id`,
start time, objective, budget, original `baseline_revision` and default `context`.
Context fields link to checkpoint, workload and environment artifacts, including
GPU UUID, Python/tool versions and native binary/library identities. Reuse existing
manifests. Preserve the original baseline and context; evidence inherits context
and overrides only supplied fields, e.g. `"context": {"workload": "kernel-fixture"}`.

Use stable IDs, unique within each collection; qualify cross-investigation nodes
as `<investigation-id>/<collection>/<id>`. Links point to inputs and form an
acyclic graph.

| Record | Links and meaning |
| --- | --- |
| [Candidate](schemas/candidate.schema.json) | `baseline_revision` → revision; `motivation` → evidence; optional `follows_decision` → decision. Record origin, initial readiness, `hypothesis` (mechanism and selection rationale), and `prediction` (expected effect and refuting evidence). |
| [Revision](schemas/revision.schema.json) | `candidate` → candidate; `parents` → revisions; `source.artifacts` → artifacts. The original baseline has `candidate: null` and no parents. Optional `motivation` → evidence / `follows_decision` → decision link inner-loop feedback; `prediction` overrides the candidate's when the local hypothesis changes. |
| [Evidence](schemas/evidence.schema.json) | `subjects` identify examined revisions/exports; `depends_on` → evidence; `artifacts` → artifact IDs. Covers captures, exports, validation, timing and command failures. |
| [Decision](schemas/decision.schema.json) | `revision` → revision; `evidence` → supporting evidence; optional `next_baseline` → revision. `assessment` combines predicted versus observed results, explanation, unknowns and outcome rationale; `next_action` names the follow-up. |

`parents` names the implementation(s) actually edited or combined, independently
of the candidate's comparison baseline. Later candidates may use a selected
`next_baseline` and link its decision.

Name subject roles explicitly (`implementation` for a single variant):

```json
{
  "subjects": {
    "baseline": {"revision": "r0", "export": "export-r0"},
    "candidate": {"revision": "r2", "export": "export-r2"}
  }
}
```

Each export is separate evidence with command/log and package identity, even for
the same revision. Native measurements, captures and validation name that exact
export, which must identify the subject's revision. Comparisons retain both
exports and link matching validation via `depends_on`. Omit `export` for the
export operation itself and for local/source checks; local evidence still names
its revision, fixture and environment.

## File and build identity

An [artifact](schemas/artifact.schema.json) has `kind`, `uri` and optional
`sha256` / JSON `pointer`. Relative URIs resolve against `investigation.json`;
use host-qualified URIs for remote files. Hashes identify whole files, including
when a pointer selects a report section. Reuse reported package/binary identities.

Bundle finished command logs, reports and telemetry into a
[file manifest](schemas/file-manifest.schema.json), linked as one `file_manifest`
artifact. Its `files` map contains role → `{uri, sha256}`, with URIs relative to
the manifest. Keep shared source/context/package artifacts and independently
referenced reports reusable. Individual file artifacts remain valid.

Preserve completed files; changed files get new manifests and artifact IDs.
Running evidence may link unhashed files until bundling on completion.
Command logs retain commands, working directory, exit statuses and execution host.

Revision `source` captures the Git commit, cumulative patch relative to it,
untracked sources and compiler/kernel settings. With no Git base, use
`commit: null` and a complete snapshot. Each code/configuration change creates a
new revision.

## Helper commands

Set `PERF_PLUGIN` to the plugin directory and `RUN` to the existing investigation
directory on the execution host. Use the [record helper](tools/record.py):

```bash
RECORD_TOOL="$PERF_PLUGIN/skills/optimize/references/records/tools/record.py"
python "$RECORD_TOOL" show "$RUN/investigation.json"
python "$RECORD_TOOL" show "$RUN/investigation.json" candidates/c2
python "$RECORD_TOOL" show "$RUN/investigation.json" evidence/paired-r2
python "$RECORD_TOOL" bundle "$RUN/paired.files.json" \
  command="$RUN/paired-status.json" log="$RUN/paired.log" \
  report="$RUN/comparison/summary.json" telemetry="$RUN/telemetry.csv"
python "$RECORD_TOOL" check "$RUN/investigation.json"
python "$RECORD_TOOL" check "$RUN/investigation.json" --files
```

`show` prints JSON for resumption and selective reading. The overview includes
the original baseline, each candidate's comparison baseline, latest decisions
and explicit baseline selections, plus running evidence. Latest is determined
per candidate by timestamp, retaining ties; it does not choose a global baseline.
An earlier baseline selection remains visible alongside a later rejection.
Candidate views include revisions, decisions and an evidence index. Evidence
views include exact export records, inherited context with overrides applied,
and artifact locations. Use `revisions/ID`, `decisions/ID` or `artifacts/ID` for
individual entries. Report bodies and manifest inventories remain in their files;
relative artifact URIs resolve against the directory of `record_path`. Records
are unchanged.

`bundle` prints an artifact to add and link from evidence; it does not edit the
record. Its URI is local to the execution host; qualify it for remote use.
`show` and `check` require `jsonschema` and check schema, links, cycles and
export/revision agreement. `check --files` also checks local files, hashes,
pointers and manifests; remote files remain unchecked. Run it where files are
available. Model acceptance comes from project validators; `show` displays
recorded status without rechecking results or live jobs.

## Lifecycle semantics

Follow the [experiment loop](../experiment-loop.md) for when to add records.

- Evidence `completed` means a usable result, regardless of correctness or speed;
  `failed` means command failure; `running` means unfinished; `not_run` records a
  gap. `summary` is required for `failed` / `not_run`, otherwise optional.
  `artifacts` may be empty for a gap; omit empty `depends_on` lists.
- Decision `keep` can mean a local winner awaiting integration. Record validator
  results separately in `acceptance.local` / `acceptance.model`; incomplete suites
  remain `unverified` unless a required check failed. Readiness is separate;
  diagnostics stay in reports. An exploratory baseline retains failed acceptance.
- Finish running entries; preserve completed entries and append new revisions,
  evidence and decisions. Resume from the latest decision and inspect running
  evidence and jobs before launching more work.

For version 1 records, preserve the original and migrate a working copy: merge
candidate `selection_reason` into `hypothesis`, `refutation` into `prediction`,
and decision `comparison` + `reason` into `assessment`. Remove merged fields and
set `schema_version: 2`, preserving text, IDs, links, timestamps and file identities.
