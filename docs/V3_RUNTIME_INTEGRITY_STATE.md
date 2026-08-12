# V3 Runtime Integrity State

## Development Identity

```text
parent: 098a02f3b79625d7cb15accd6dcd4ba949280388
branch: feature/v3-runtime-integrity-v1
```

## Implemented Contracts

- `reference_dense_v1`, object phrase retry, type-aware geometry, 7/10
  coverage, conservative prefilter, final background guard, and disabled
  same-parent fallback remain unchanged.
- Dense annotation has explicit subject/object semantic boundaries and a
  conservative generic-object phrase sanitizer.
- Progressive SAM3 anchor search adds the five previously unprobed slots after
  the original fast path and publishes counters.
- The opt-in reference-integrity stage reviews synthetic, local, and
  topology-suspicious references against source evidence.
- Integrity rejection is entity-local and transactionally invalidates
  instruction/export.
- Audit output separates post-pair, post-reference-edit,
  post-reference-integrity, and final-export density.
- Object-Remover defaults to the validated existing LoRA at four steps. The
  adapter is mandatory and its active state is verified.
- Remove infrastructure failures remain retryable; completed quality-only
  rejection rounds are permanent.
- `cuda:N` remove configuration is validated against visible CUDA count before
  model loading.
- `staged_legacy` remains the default runtime.
- `streaming_v1` provides per-clip DAG scheduling, bounded CPU/stage workers,
  a global Qwen budget, deterministic output ordering, process-safe shared
  append/profiling writes, and one isolated persistent process for each heavy
  model stage.
- Streaming and same-parent fallback are mutually exclusive in V1.

## Frozen Evidence

```text
density120 annotation ready clips: 117
density120 annotation entities: 145
density120 accepted samples: 51
density120 final entity references: 63
density120 entity references/sample: 1.235
```

Contact-sheet failures motivating the integrity stage included disconnected
fragments, large missing surfaces, unnatural internal holes, destructive
partial scene structures, bad synthetic completion, and subject crops without
useful identity evidence. Legitimate discrete objects and recognizable subjects
must continue to pass.

## Historical Runtime Note

Stored historical production configs, including `prod5000`, used 40 remover
steps. They are evidence for the accidental 40-step runtime, not for the
restored `object_remover_4step_v1` profile.

## Validation Boundary

Local validation uses fake/CPU tests, Ruff, Python compilation, and diff
checks. No real Qwen, SAM3, Object-Remover, Boogu, CUDA, production data, or
pilot was run during implementation. Server smoke commands are recorded in
`V3_RUNTIME_INTEGRITY_V1.md` and remain pending.

## Next Review

Review the three commits independently:

1. semantic cleanup, progressive anchors, integrity routing, and audit counts;
2. remover profile, adapter/device checks, and retry state;
3. runtime config, process ownership, concurrency safety, profiling, and
   deterministic resume behavior.

After code review, create a fresh 20-clip local server config and execute the
documented smoke sequence. Do not overwrite completed runs.
