<!-- Maintained by the `architecture-friction` agent skill. Humans welcome. -->

# Architecture friction

Places where this codebase's structure made a change harder than it should have
been, and the refactors that would relieve them.

This is **not** a bug tracker or a TODO list. Every entry answers one question:
*what would have made a change we actually made easy?*

- `## <a refactor direction>` — states a direction ("rework device locking so it
  can span threads"), not a specific design and not a place. Under it, one dated
  bullet per time that friction was felt.
- An occurrence may sit under several headings if any of those refactors would
  have relieved it. Its `(slug)` is identical in each, so all copies can be
  found and removed together when one lands.
- `## Unfiled` — raw notes captured mid-work, not yet sorted. An unsorted note
  still beats a forgotten one.
- `## Done` — refactors that landed, with dates, so the same friction doesn't
  get re-logged from scratch.

**Three bullets under one heading means the refactor is due.**

Adding an entry costs one line. Please add them.

## Let recorded-result tests assert scientific claims instead of whole output structs

- [2026-09-04] (vc-refractory-strength) Changing the vc refractory to keep the
  steepest event rather than the first-arriving one fixed the target defect but
  broke 19 unrelated baselines, purely on sub-millisecond shifts in which event
  got reported. Zero baselines were fixed. The comparison walks every field of
  every spike at rtol=0.01, so the recorded results pin the algorithm's exact
  output rather than the claim "one spike, here" — any behavior change, including
  a strict improvement, reads as a regression.

## Unfiled

- [2026-09-04] 95 of the recorded spike/psp baselines have `expected_result =
  None`, from before the detectors returned lists. They are indistinguishable in
  the harness from a real recorded answer, and only fail once the comparison
  reaches a type check.
