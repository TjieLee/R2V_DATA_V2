# RA2VA Group Local Pipeline Integration

Approved scope: Native implementation, CPU fake-model tests and local commits only.
No server access, push, GPU/MiMo request, full-group run or changes to frozen
Multi/Single/T2VA semantics. Real two-node and GPU acceptance remains pending.

## Task 1: Native stage inputs and persistent stage adapters

Create `ra2va_group_pipeline.py` with Group-native audio/job metadata. Preserve
V3 reference order/ownership and existing 32 kHz sample clocks without input
media hashes. Reuse existing media operations and model backends, with one
backend context per stage, not per clip. Add tests before implementation for
canonical, SAM music-first, AuK speech, resolve, DiariZen and ASR.

Interfaces: GroupTask + stage payloads -> native frozen job and resolved paths.
Verify: focused stage tests, Ruff and py_compile; commit locally.

## Task 2: Durable Visual/Joint response boundary

Create `ra2va_group_mimo.py`: wrap the existing Two-step completion client,
persist request intent and real response separately for each turn, and replay
saved responses through the unchanged backend. Response-less request intents
remain `interrupted_request_unresolved`, never retry. Preserve real raw/usage
and distinguish historical calls from calls made during resume.

Interfaces: native frozen job + stems -> annotation, raw and actual call audit.
Verify interruption before Joint, after saved response, invalid response and
unresolved request tests; exact request/prompt equality; commit locally.

## Task 3: Group export and local CPU integration

Reuse existing Audio eligibility, H3 rendering, frame prompt projection,
four-field training export and donor segment cropping. Add only necessary
typed hooks for native metadata, leaving legacy defaults unchanged. Do not
invent media digests or rescan source media. Existing generated-file digest
fields may be populated once by their existing writers.

Expose an explicit local integration runner; retain Fake/local/HTTP modes.
CPU tests inject model backends and exercise actual media and H3 consumers.
Verify identity restrictions, all donor intervals, all twelve tasks, terminal
resume and zero real model calls. Run related Group/Two-step/Audio/H3 tests,
Ruff, py_compile and diff-check; review actual diff and commit locally.

## Review Focus

No implicit retries or fake media hashes; request intent ambiguity; saved raw
before validation; stage context lifetime; no protected prompt/default changes;
identity-restricted speech cannot become donor/identity-specific conditioning;
resume must not rerender terminal products. CPU evidence is not GPU or genuine
two-node evidence.
