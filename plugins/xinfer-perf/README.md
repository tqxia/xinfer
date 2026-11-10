# xinfer-perf

A shared `optimize` skill for compiler and kernel performance investigations in
xinfer. It defaults to native decode latency per token and uses existing xinfer
commands, `nsys` and `ncu` directly. The coding agent chooses hypotheses and edits;
the existing validators and benchmarks supply evidence.

The default loop is baseline measurement → profiling → hypothesis →
compiler/kernel change → export/validation → paired measurement.
Profile before choosing candidates; a matching saved capture can be reused.
Each hypothesis cites an observed hotspot or execution behavior. If usable
profiling evidence is missing, the agent obtains it or reports the gap and
defers candidate selection.

A small [symptom index](skills/optimize/references/optimization-index.md) links
profiles to common technique notes and hardware-specific applicability guidance.
After each experiment, the agent compares its prediction with observed results
and records what that means for the next attempt. Extend the notes from actual
optimizations rather than preloading a large technique catalog.

## Load the plugin

Run these commands from a xinfer checkout. For Claude Code, load the plugin for
one session:

```bash
claude --plugin-dir "$PWD/plugins/xinfer-perf"
```

Invoke `/xinfer-perf:optimize`, or describe a compiler/kernel optimization task.

For Codex, the repository's `.agents/plugins/marketplace.json` exposes the plugin
through the `xinfer-local` marketplace without automatically enabling it.
Register this checkout once, then install the plugin:

```bash
codex plugin marketplace add "$PWD"
codex plugin list --marketplace xinfer-local --available --json
codex plugin add xinfer-perf@xinfer-local
```

Start a new Codex session in the checkout and select the plugin's `optimize` skill.
In the desktop app, open the plugin directory, choose the local marketplace and
install `xinfer-perf`. Refresh the local install after changing plugin files;
installed plugins may be loaded from a cached copy.

The plugin directory contains instructions. The **xinfer checkout and GPU host**
are separate inputs: run project commands in that checkout with its matching
environment, even when the skill was loaded from a plugin cache. A local agent
can use its existing SSH tools to execute on the GPU host. Do not put checkpoint,
build or experiment files inside the installed plugin.

## Example requests

> Use xinfer-perf to investigate decode latency for this compiled model. Start
> by profiling the existing baseline, then try at most three compiler-side
> candidates supported by that profile in two hours. Keep numerical failures visible.

> Continue the experiment described in work/perf/notes.md. Read the saved results
> and finish the pending comparison before trying another change.

The default budget is three candidate attempts or two hours, whichever comes
first. The first project demonstration targets Qwen3.5-0.8B on GB10; the skill
itself does not hardcode that model or hardware. Runtime optimization is deferred
to a separate future skill. CodeBuddy packaging and discovery are also future
work; this version supports Codex and Claude Code.

See the [GB10 Qwen3.5-0.8B experiment](../../docs/experiments/gb10-qwen35-perf-plugin-2026-11-10.md)
for baseline profiling, three compiler trials and correctness/performance feedback.

## Check the package

From the xinfer checkout, validate the manifests:

```bash
claude plugin validate plugins/xinfer-perf
```

After registering the Codex marketplace, `codex plugin list --marketplace
xinfer-local --available --json` checks its discovery.

Manifest validation and discovery do not exercise an LLM session or prove GPU
correctness. Use the project validation and benchmark commands for experiments.

Host packaging references: [Codex](https://developers.openai.com/plugins/build/plugins) and
[Claude Code](https://code.claude.com/docs/en/plugins-reference).
