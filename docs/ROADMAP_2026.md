# Nedster roadmap — August 2026 and beyond

This roadmap separates plausible next-generation work from features already shipped. Priorities are ordered by safety and measurable throughput, not novelty.

## 1. Stateful inference fabric

Cold-loading weights is only half the latency problem. Nedster should manage three separately measurable resources: model weights, prefix/KV state, and tool-derived working memory.

- Add versioned prefix-cache snapshots for stable system/tool prompts.
- Reuse KV only when model, tokenizer, quantization, rope parameters, and prompt prefix match exactly.
- Build topology-aware placement for VRAM, unified memory, system RAM, and encrypted NVMe.
- Add NIXL-class transfer adapters for multi-GPU systems only after single-node profiling proves value.
- Track time-to-first-token, tokens/second, page-in time, cache hit rate, and energy per accepted task.

## 2. Eval-trained routing

Parameter count is a weak proxy for capability. Collect privacy-preserving local eval traces and choose workers using calibrated success/cost curves.

- Per-role benchmark packs for planning, code edits, shell diagnosis, and criticism.
- Contextual bandit routing with a deterministic fallback.
- Escalation only after local uncertainty or failed verification crosses a configured threshold.
- Privacy budgets that cap what evidence an optional cloud oracle may receive.

## 3. Linux-native containment

- Enforce Landlock file rules in the executing child, independent of model-visible paths.
- Add seccomp profiles generated from each tool manifest.
- Build reproducible Firecracker rootfs images and a vsock-only broker.
- Use immutable skill packages with signed manifests and content-addressed dependencies.
- Move capability signing to a separate minimal broker process; evaluate TPM-backed sealing where it improves the real threat model.

## 4. Transactional autonomy

- Workspace snapshots before mutation.
- File-level leases and optimistic conflict detection for parallel writers.
- Test-gated commit transactions with human-readable rollback plans.
- Policy-as-code supporting per-repository and per-skill authority.
- Deterministic replay of tool events without replaying model prose.

## 5. Multimodal and ambient agents

The router should treat vision, audio, and GUI control as scoped sensors and tools rather than granting a multimodal model unrestricted desktop access.

- Local visual diff worker for UI regression evidence.
- Screen regions as capabilities with expiry and redaction.
- On-device speech interface with push-to-authorize for mutations.
- Event-triggered agents that remain dormant until a signed local event wakes them.

## Success criteria

The architecture earns the name “ultimate local agent” when it improves verified task completion per joule and per minute while reducing the blast radius of a compromised prompt. Model leaderboard position alone is not a success metric.
