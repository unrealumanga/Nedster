# Security policy

Nedster executes model-proposed actions. Treat every model response, retrieved document,
repository file, and tool result as untrusted input.

## Supported security posture

- Nedster starts in read-only policy mode.
- Every tool invocation requires a short-lived, request-bound, single-use capability.
- File tools resolve paths beneath the selected project root.
- Shell execution uses argv arrays with `shell=False` and an executable allowlist.
- Linux prefers Bubblewrap, gVisor, rootless Podman, or Docker with networking disabled.
- The event ledger is append-only and HMAC hash-chained so tampering can be detected.
- API keys are read from environment variables and must never be committed.

Capability tokens prevent confused-deputy and replay mistakes inside the agent runtime; they
are not a hardware security boundary. A compromised host can still inspect process memory.
Use full-disk encryption and a Linux sandbox or micro-VM when processing hostile code.

Windows and macOS currently fall back to a restricted child process unless a configured
container backend is available. That fallback is disabled for command execution by default.

## Reporting a vulnerability

Open a private GitHub security advisory for the repository. Do not include live credentials,
private source code, or exploit payloads that target systems you do not own.
