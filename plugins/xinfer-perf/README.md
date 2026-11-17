# xinfer-perf

A shared `optimize` skill for compiler and kernel performance investigations in
xinfer. It defaults to native decode latency per token and uses existing xinfer
commands, `nsys` and `ncu` directly. The coding agent chooses hypotheses and
edits; the existing validators and benchmarks supply evidence.

The [optimize skill](skills/optimize/SKILL.md) starts with baseline profiling,
selects candidates from measured opportunities, and evaluates them through
export, validation and paired native measurement. Kernel work uses a nested
optimization loop. An
[investigation record](skills/optimize/references/records/format.md) links
candidates, implementation revisions, profiles, validation, measurements and
decisions while keeping detailed results in their existing files.
The [record helper](skills/optimize/references/records/tools/record.py) shows
selective record views, bundles supporting files and checks links and identities;
see the record guide for its commands.

## Optimization loops

The [model loop](skills/optimize/references/experiment-loop.md) selects measured
opportunities and establishes native performance. Kernel work enters the nested
[design and implementation loops](skills/optimize/references/kernel-optimization-loop.md):

![Model loop containing kernel design and implementation loops, with design above implementation.](assets/optimization-loops.svg)

All loops share the investigation budget and existing records. Local kernel
improvements require full-model validation and native measurement before
model acceptance. Keep or reject the candidate using that evidence; stop when
the evidence or remaining budget does not support another trial.

## Load the plugin

Run these commands from a xinfer checkout. For Claude Code, load the plugin for
one session:

```bash
claude --plugin-dir "$PWD/plugins/xinfer-perf"
```

Invoke `/xinfer-perf:optimize`, or describe a compiler/kernel optimization task.

For Codex, the repository's `.agents/plugins/marketplace.json` exposes the
plugin through the `xinfer-local` marketplace without automatically enabling it.
Register this checkout once, then install the plugin:

```bash
codex plugin marketplace add "$PWD"
codex plugin list --marketplace xinfer-local --available --json
codex plugin add xinfer-perf@xinfer-local
```

Start a new Codex session in the checkout and select the plugin's `optimize`
skill. In the desktop app, open the plugin directory, choose the local
marketplace and install `xinfer-perf`. Refresh the local install after changing
plugin files; installed plugins may be loaded from a cached copy.

The plugin directory contains instructions and record bookkeeping tools.
The **xinfer checkout and GPU host** are separate inputs: run project commands
in that checkout with its matching
environment, even when the skill was loaded from a plugin cache. A local agent
can use its existing SSH tools to execute on the GPU host. Do not put
checkpoint, build or experiment files inside the installed plugin.

## Example requests

> Use xinfer-perf to investigate decode latency for this compiled model. Start
> by profiling the existing baseline, then try at most three compiler-side
> candidates supported by that profile in two hours. Keep numerical failures
> visible.

> Continue the experiment in work/perf/investigation.json. Read the linked
> results and finish the pending comparison before trying another change.

The default budget is three model-level candidate attempts or two hours,
whichever comes first. Select the model and hardware from the target xinfer
checkout.

Runtime optimization is deferred to a separate future skill. CodeBuddy packaging
and discovery are also future work; this version supports Codex and Claude Code.
A self-contained casebook for selected optimization examples is deferred.

## Check the package

From the xinfer checkout, validate the manifests:

```bash
claude plugin validate plugins/xinfer-perf
```

After registering the Codex marketplace,
`codex plugin list --marketplace xinfer-local --available --json` checks its
discovery.

Host packaging references:
[Codex](https://developers.openai.com/plugins/build/plugins) and
[Claude Code](https://code.claude.com/docs/en/plugins-reference).
