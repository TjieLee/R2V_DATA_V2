# RA2VA binding A/B review

This standalone, model-free export compares frozen legacy-LR-ASD and no-LR-ASD
MiMo records. It does not re-render H3, reconstruct bindings or alter existing QA.
Failed records retain their available annotation and original diagnostics.
Segments align only by exact ID; unmatched segments remain visible. Statistics
are descriptive, never an automatic quality verdict.

Set both explicit MiMo roots and a fresh independent `REVIEW_ROOT` outside
production/source trees. The validated random20 manifest supplies the clip order:

```bash
"$R2V_PYTHON" tools/build_h3_ra2va_binding_ab_review.py \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --visual-runs-root "$VISUAL_RUNS_ROOT" \
  --case-manifest /tmp/mimo-refgraph-random20.json \
  --legacy-mimo-root "$LEGACY_MIMO_ROOT" \
  --no-lrasd-mimo-root "$NO_LRASD_MIMO_ROOT" \
  --output-root "$REVIEW_ROOT"

"$R2V_PYTHON" -m http.server 8774 --bind 127.0.0.1 --directory "$REVIEW_ROOT"
```

Open `http://127.0.0.1:8774/review.html`. Export contains `review.html`, `data.json`,
`summary.json` and copied review-owned media only. SOURCE displays every frozen
Visual reference with its original numbering, not the actual MiMo numbering.
Separate OLD/NEW contexts display the frozen jobs' actual Pictures, Subjects,
selection/drop reasons and face promotions. Jobs are read from `source_contract.json`
(or a frozen `inventory.json` containing jobs), with request fingerprints checked
against each run record and reference media hashes verified. Selection is never
recomputed. For an externally stored exact OLD job inventory, optionally pass
`--legacy-source-contract PATH`. Without exact frozen input, OLD actual context
is explicitly unavailable; SOURCE is not substituted as actual context.
Grounding and captions always come from their respective frozen run.

Labels are exactly SAME, NEW_BETTER, NEW_WORSE, BOTH_WRONG and FORMAT_ONLY.
Keys 1-5 select them; notes and labels save immediately in localStorage, keyed by
the A/B data fingerprint. Unselected clips export with a null label. Export review
JSON before moving browsers or clearing browser storage; no POST service is used.
Judge speaker-to-Subject binding, not prose aesthetics, music or soundscape.

Input record fingerprints are checked without applying current model semantic
validators to failed annotations. No production artifact is written. The output
must not already exist; use another fresh directory for a changed comparison.
