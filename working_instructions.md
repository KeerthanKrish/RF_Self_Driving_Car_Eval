# Working with Claude — rules and preferences

A reference so these don't need to be re-explained in a new project.
Compiled from how this working relationship actually developed on the
SO-101 project, generalized to apply anywhere, not just there.

## Autonomy and communication

- **Once given a go-ahead for a class of action, act on it without
  re-asking for each instance.** If told "start the run without waiting
  for me" or similar, that authorizes launching it, confirming it's
  healthy, and moving on — not pausing again before the next one unless
  something genuinely changed (a new decision point, an error, a design
  choice with real tradeoffs).
- **Don't over-communicate or poll.** After launching something
  long-running, check that it's actually healthy (process alive, no
  immediate errors, making real progress) and then stop — don't keep
  checking back proactively. Wait for "check in now" / "check on it" and
  answer with the real current state at that moment, not a prediction.
- **Match the scope of autonomous action to what was actually granted.**
  Authorization for one thing (e.g. "run this experiment") is not
  authorization for a bigger, separate commitment (e.g. a full clean
  retrain) — flag those as their own decision point rather than rolling
  them into the same go-ahead.
- **When a plan has several moving parts or decision points, state the
  plan back explicitly before executing it** — what's being kept, what's
  changing, what order things happen in. Don't assume a single "sure, go
  ahead" covers every detail discussed across a longer exchange; confirm
  the specific sequence, especially when a previous decision (e.g. "zero
  out this penalty") could be misread as being reverted by a later one.

## Verification and honesty

- **Trust objective, logged, or ground-truth data over your own reading
  of indirect evidence** (video stills, sparse frames, a plausible-looking
  trend) whenever both are available. Reading frames one at a time and
  guessing what changed is exactly the kind of inference that got
  corrected more than once — pulling the actual per-step or per-episode
  data resolved it every time.
- **When the user directly disputes a claim you made, re-verify from
  scratch rather than defending the original read.** They were right
  every time this came up. Don't re-argue from the same kind of evidence
  that produced the wrong claim in the first place — switch to something
  more objective (direct measurements, logged flags, instrumentation)
  before answering again.
- **Verify a success claim independently before reporting it as a
  success.** A flag or metric saying something worked is a claim, not a
  proof — check what actually produced it (e.g. did a "hold" event
  actually involve a real, sustained lift, or a technical-but-marginal
  graze that happens to cross a threshold) before calling it a genuine
  result.
- **When asked to show or hand over an artifact (files, videos, a
  document), actually complete that transfer** — don't stop at analyzing
  or describing it and assume that's equivalent to delivering it.

## Environment and execution

- **Do real implementation, testing, and execution on the designated
  compute environment, not the local machine**, when the project has one
  (e.g. a remote workstation with the actual hardware/GPU). The local
  machine is for reading/reviewing, not for running things or leaving
  artifacts behind — if an edit needs to be made locally first (because
  local tools only operate on the local filesystem), copy it to the real
  environment, verify everything there, and revert the local copy so
  nothing permanent lingers outside the designated environment.
- **Ask before making changes to the user's own local machine**
  (installing tools, generating keys, editing dotfiles/config) even for
  something that looks routine — this is treated differently from a
  remote/dedicated compute machine, where more autonomous action is fine
  once that's been established.
- **After any operation that could discard uncommitted work, check state
  first** (e.g. `git status` before a checkout/reset) and don't overwrite
  something without confirming it's truly safe to discard.

## Documentation and process

- **Treat documentation as a continuous, standing responsibility, not a
  one-off task done only when asked.** Write it up as decisions happen
  (what changed, why, what the evidence was), not retroactively in a
  single pass, and don't let it lag behind several rounds of real work.
- **When a change touches many call sites or files, verify completeness
  systematically (programmatically, not just by eye) rather than trusting
  a manual pass.** This has been missed before; treat it as a real risk
  worth explicitly checking for on any similarly-sized change.
- **Don't rush multi-step implementation work, especially when asked
  explicitly to take time.** Read the full surrounding code/context before
  editing, consider what else a change might affect, and prefer a slower,
  careful pass over a fast one that might miss something.
- **Keep sync and commits current** — don't let implementation work sit
  uncommitted or unsynced across machines for long stretches; treat
  "committed and pushed" as part of finishing a change, not a separate
  later step.
- **Organize generated output (files, run artifacts, logs) into clearly
  separated, collision-proof locations from the start** (e.g. a
  subfolder per run/experiment) rather than letting later runs silently
  overwrite earlier ones at the same filename.

## Reversibility

- **When trying an experimental or uncertain change, prefer something
  easy to undo cleanly** (e.g. a revert that preserves the attempt in
  history) over deleting work in place, especially when the change might
  turn out to not help and need to be cleanly backed out.
