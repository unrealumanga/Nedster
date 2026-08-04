# Ironclad architecture

## Design invariants

1. Model output is untrusted data, including output from the router and critic.
2. Models propose typed actions; the Python runtime resolves scope, evaluates policy, grants authority, and executes.
3. The least capable model that can complete a role should receive that role.
4. Read-only independence may be parallelized; mutations are serialized until conflict-aware transactions exist.
5. Local-only is the default. Cloud escalation is explicit, visible, and separately configured.
6. Logs must describe state transitions and evidence, not hidden chain-of-thought.

## Control plane

The router receives the objective and tool descriptions, then emits a JSON-schema task plan. Each step declares a role, dependencies, allowed tools, and success criteria. A malformed plan falls back to a single bounded step rather than trying to recover intent with permissive regexes.

The scheduler scores the locally installed Ollama inventory against the detected hardware tier. Explicit configuration is deterministic; `auto` mode prefers smaller models for routing and criticism and larger suitable models for coding and reasoning. The router is held resident, while workers receive short keep-alive leases.

GPT-5.6 Sol can serve as an optional quality-first reasoner. The adapter uses the Responses API and structured outputs. It never activates solely because an API key happens to exist.

## Data plane

A worker receives one manifest, its permitted tool schemas, and prior evidence for that step. It returns a structured message containing zero or more proposed actions. `ToolRegistry` then:

1. validates the tool and argument shape;
2. resolves file targets under the project root;
3. assigns an intrinsic risk level;
4. evaluates autonomy mode, risk ceiling, denied fragments, and executable allowlist;
5. mints a short-lived capability bound to the canonical request;
6. consumes the capability once and dispatches the implementation.

This prevents a model from turning prose into authority. It also blocks replay and request substitution within the runtime. The signing key stays in the runtime process.

## Isolation ladder

The automatic Linux order is:

```text
Bubblewrap → gVisor/runsc → rootless Podman → Docker → disabled process fallback
```

Bubblewrap creates fresh namespaces, disables networking, binds the selected project only, and exposes system paths read-only. Container backends drop capabilities, set `no-new-privileges`, use a read-only root filesystem, deny networking, and mount only the project as writable.

The configured Firecracker wrapper is an integration seam, not a complete VM manager. A production Firecracker deployment still needs versioned rootfs images, KVM provisioning, vsock transport, resource limits, snapshot hygiene, and lifecycle cleanup. Native Landlock enforcement is similarly planned rather than claimed.

## State and verification

The ledger is local SQLite in WAL mode. Each event stores its canonical payload, the previous digest, and an HMAC digest. `verify-ledger` recomputes the chain from the beginning. This makes deletion, reordering, and mutation detectable to anyone who retains the key and chain head; it does not prevent a compromised host from replacing both the database and an unprotected local key.

Workers do not message one another directly. Their only shared state is router-approved manifests plus evidence emitted through the runtime. This limits feedback loops and makes the task graph inspectable.

## Concurrency rules

Independent steps may execute together only when their entire declared tool set is read-only (`read_file`, `list_files`, `search_text`, `git_diff`). Steps that can write or execute commands run one at a time. The current rule is deliberately conservative: a future version may add file-range leases and optimistic conflict detection.

## Inference topology

Ollama is the first-class runtime today because it exposes model inventory, structured outputs, thinking controls, tool-capable chat, and keep-alive behavior through a portable local API.

For users operating llama.cpp directly, `llama-command` generates a plan with:

- loopback binding;
- memory mapping;
- Flash Attention;
- tier-specific context and parallel slots;
- quantized K/V caches;
- prompt-lookup speculation by default, or draft-model speculation when supplied.

The command is printed, not executed. That preserves operator review and accommodates the fast-moving llama.cpp CLI.

## Explicit non-goals in 3.0 alpha

- No claim of silicon-enclave or hardware-rooted capability security.
- No direct worker-to-worker dialogue.
- No automatic internet access.
- No transparent cross-machine llama.cpp RPC, because upstream warns that RPC is insecure.
- No claim that KV caches switch between unrelated model architectures.
- No automatic destructive actions or external publication.
