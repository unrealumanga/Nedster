# Security policy

Nedster executes model-proposed actions. Treat every model response, repository file and tool result as
untrusted input: a file in the project can contain instructions aimed at the model.

## Threat model (v4)

What Nedster enforces in code:

- **Command execution is off until consented.** The default `policy.max_auto_risk` is `write`. `execute` is set only
  by `nedster setup` (with your answer) or by editing the user config. A repository's `.nedster/config.toml` can
  only tighten security settings; endpoints, backends and looser policy in it are ignored with a warning.
- **No shell.** Commands are argv arrays. Programs are resolved from absolute PATH entries outside the project;
  a binary planted in the repository is never executed, and a path-qualified program inside the project is denied.
  The child environment is an allowlist (secrets are not passed).
- **Sandbox backends** (`bwrap` on Linux, `seatbelt` on macOS, `docker` anywhere) deny network and confine writes to
  the project and a private temp directory; `.git` and `.nedster` are read-only inside them. With no ready backend,
  execution is refused. The `host` backend (`sandbox.allow_host_execution = true`) has **no filesystem or network
  isolation**; on Windows it is contained only as a process tree (Job Object). Processes deliberately detached from
  the tree (WMI, Task Scheduler, launchd, `systemd-run`) are not tracked.
- **File tools are scoped to the project root.** Paths are resolved (symlinks and junctions included) and must stay
  inside it. `.git`, `.nedster` and a `.git` file's `gitdir` target cannot be written. Edits require the sha256
  from a read, so stale edits are refused; writes are atomic.
- **Git is hardened** (no external diff/textconv/fsmonitor; hooks directory outside the sandbox-writable area).
- **Outcomes come from tool records**, so model text cannot claim success; see README for the outcome list.

Known limits:

- Test code is project code. A hostile repository can print a fake passing summary; runner/last-line parsing
  stops trivial spoofing, not a determined repository. Use a sandbox backend for untrusted repositories.
- `policy.denied_fragments` is advisory substring matching, not a boundary.
- The project folder is writable inside every backend; `/undo` covers edit-tool changes only, not command effects.
- Seatbelt rules are verified by inspection and unit tests, not on a macOS runner in this repository.
- A compromised host or an attacker with your user account can read process memory and the session database
  under `~/.nedster/data`; use full-disk encryption.
- Inference goes only to the Ollama endpoint in `models.ollama_url`. A project config cannot change that URL.

## Reporting a vulnerability

Open a private GitHub security advisory for the repository. Do not include live credentials, private source code,
or exploit payloads that target systems you do not own.
