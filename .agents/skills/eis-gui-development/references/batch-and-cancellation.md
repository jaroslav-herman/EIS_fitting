# Batch and cancellation pattern

## Ownership split

The service owns iteration and scientific work. It accepts immutable inputs or copied state plus a `threading.Event`, checks that event between units of work, and returns a report containing completed results, skipped targets, stopped state, and any failure. The worker must not touch Tk widgets.

The GUI captures current controls before submission, builds the ordered target list from the explorer, sets status/operation labels, and calls `_submit()`. A `_finish_*` callback applies completed results on the Tk thread, refreshes controls/plots/explorers, and reports completed, failed, and skipped counts.

## Required semantics

- `_submit()` rejects overlap through `busy`, clears the shared stop event for the new operation, disables normal actions, and keeps Stop enabled.
- Stop is cooperative: finish the current indivisible fit safely, retain completed results, and skip remaining targets.
- Check the event before the first target and between every target. Chained Up/Down or analysis-then-fit flows must also check it between stages.
- Do not roll back successful earlier targets because a later target fails or cancellation arrives.
- Preserve the explorer's visible order and explicit selection. For directional work, include the anchor exactly once and verify that it is selected.
- Never poll or update Tk from a worker. Use `root.after()` or the existing future poll on the main thread.
- Clear transient queue/template flags on success, failure, cancellation, and early validation returns.

## Focused verification

Extend `tests/test_batch_stop.py` or the nearest service test with: no stop, stop before first, stop after a completed item, failure after completed items, and equivalent-circuit parameter transfer when relevant. Manually verify the Stop button, status counts, retained fits, and a second operation after cancellation.
