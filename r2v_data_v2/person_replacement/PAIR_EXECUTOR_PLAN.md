# Two-GPU pair executor implementation plan

Baseline: `fdca08616314562e637b622359b45f6ad6817ac7`. Separate text-only path;
the existing A/B runner and single-GPU PDD worker stay unchanged.

1. Test and implement case-level commit markers, atomic publication, deterministic
   pair shards, two-way preparation partition, append-only failure attempts and
   invocation-local retry budgets. No global cursor.
2. Test and implement independent Qwen preparation subprocesses, one per GPU,
   followed by a single persistent torchrun session per pair. Use pair flock and
   process-group cleanup; resume only from durable case markers.
3. Inspect Diffusers 0.40.0 and pinned Alibaba PDD, then test the distributed
   construction order and rank protocol with CPU doubles. Apply PDD before FSDP2;
   use no ComponentsManager offload on sharded modules. Preserve mixed dtypes.
4. Add thin pair/node CLIs, read-only status/dry-run, canonical operator paths,
   and explicit staged GPU acceptance instructions. No real GPU execution here.
5. Run focused and existing person-replacement tests, Ruff, compileall and diff
   checks. Review scope and commit small coherent units.

Source inspection: Diffusers `d035dcd7cc7c88e0a154609b62887d50bba9fdc2`,
Alibaba PDD `335001fb9e5455d68a0caa18ec2e319072150328`, and the existing inspected
ModelTC FSDP2 example `02e26d591f7a04d5d1a074c9566d5dd4f22f6225`.
FSDP2 preserves module forward signatures required by H3's denoiser. PDD adds
FP32 LoRA parameters around BF16 base linears: these need separate dtype-uniform
FSDP groups, not a dtype-changing cast. PDD heads remain the official modules.

Server-only acceptance remains mandatory: 311-frame case, three jobs in one
worker, prepare/generate kill-resume, bad-case isolation, then four-pair launch.
Local CPU tests are not evidence of H200 memory fit or production readiness.
