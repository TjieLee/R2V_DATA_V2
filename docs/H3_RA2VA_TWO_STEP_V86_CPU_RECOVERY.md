# Two-step v86 CPU Recovery

## Scope

Complete, exact, ordered frozen ASR blocks identify dialogue slots directly:
`dialogue index -> frozen segment -> gN -> Sx`. Marker insertion no longer
requires an enumerated English speech verb. Existing single-speaker output is
preserved; explicit actor/marker conflicts still fail, including continuations
and Subject mentions that are only grammatical objects.

A stable single-source gN with acoustic `resolution=uncertain` can retain
identity-independent products. Its entity identity is not promoted: existing
Two-step publication restrictions exclude identity-specific voice, speech reuse
and donors. Unsafe embedded Caption attribution still blocks those products.

Visual v5, Joint v2, shared validation, Multi/Single, ASR, gN, AV evidence and
the two-call inference architecture are unchanged. Original inference provenance
is retained; `.86` is recorded separately as postprocessing provenance.

## Real Saved-eight Replay

Source: `ra2va-dualnode-pilot8-r11-18-20261010T072147Z/groups/group-000000`.
New output under the authorized R2VA root:
`ra2va-dualnode-pilot8-r11-18-cpu-v86-20261010T083451Z`.

| Metric | Original | CPU replay |
| --- | ---: | ---: |
| Annotation Ready | 6/8 | 8/8 |
| Effective training videos | 4/8 | 7/8 |
| Training tasks | 48 | 80 |
| Ready products | 12 | 20 |
| Failed products | 6 | 2 |
| Donors / segments | 6 / 18 | 6 / 18 |
| New model requests | -- | 0 |

Code: `a6b15ce`. Replay/export elapsed time: 18.674 seconds. All 16 responses were reused.
The four original successful videos remain successful. All eight Visual/Joint
raws, Audio observations, AV groundings and original inference provenance are
unchanged. All published products retain complete exact frozen ASR.

`cb8ebb` and `615ddb` each recover visual-only, speech-reuse and full-audio
products with one missing initial S1 marker added. `0afaaa` recovers visual-only
and full-audio products with neutral speech attribution, not identity products.
`c25ce1` remains unpublished: its mixed visual/speech attribution cannot be
safely neutralized by the existing conservative projection. Its two product
failures remain `two_step_identity_caption_publication_restricted`.

The four Visual types and four Full Audio types have seven samples per type;
the four Speech+BGM types have six samples per type (80 tasks total).

## Validation

- 537 focused and related tests passed, including Audio reuse, donor, Group
  replay, Single compact, T2VA backend and historical effective reconcile.
- Changed-file Ruff, py_compile and diff-check passed.
- Four stale Multi-call assertions reproduce on unchanged `de0445c`.
- Full pytest collection is blocked by seven missing-`httpx` import errors.
- Project-wide Ruff reports the same 89 errors on the unchanged baseline.

No GPU/model execution or historical output overwrite was performed.
