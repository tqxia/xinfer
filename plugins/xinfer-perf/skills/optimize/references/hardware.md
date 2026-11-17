# GB10 hardware guidance

Consult gotchas before collecting performance measurements, recipes when
implementing a hardware feature, and kernel cases when choosing or revising a
design. Read only entries relevant to the current work.

## Gotchas

### Low clocks under load

- Check clocks under a representative load before measuring performance.
- If stuck at 611 MHz, resolve the clock fault before continuing. Full shutdown
  and PSU disconnection recovered the affected host; recheck clocks under load.
- After recovery, repeat affected measurements and revisit implementation choices
  based on them.

## Hardware recipes

Store one instruction/feature recipe in `hardware-recipes/<feature>.md`, including
the code example inline. Start from the
[recipe template](hardware-recipes/TEMPLATE.md).

## Kernel cases

Store one optimization account in `kernel-cases/<kernel-or-optimization>.md`.
Start from the [case template](kernel-cases/TEMPLATE.md).

## Adding entries

Keep both directories flat. Copy a template and replace its prompts when adding
an entry; index completed entries above by feature or kernel/workload. Keep
recipes and cases self-contained, with their essential code, explanation and
evidence included directly. Do not link to local experiment records or temporary
reports.
