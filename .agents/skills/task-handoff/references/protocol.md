# Local handoff protocol v1

## Storage

Use `.agent-state/project.json` for local project identity. Keep each task under
`.agent-state/tasks/TASK_ID/`: `task.json` is authoritative; `handoff.md` is a derived
human-readable view; `history/NNNNNN.json` stores preceding revisions. Do not edit
the JSON directly during normal use. `context` builds its answer from task.json,
not the potentially stale Markdown view. A crash between the JSON and Markdown
writes does not make the Markdown authoritative.

Use `~/.local/share/agent-workspace/projects.json` as a local project registry.
Set `AW_HOME` to relocate it. Do not commit local state or the registry by default.
`.gitignore` is not encryption, not a backup, and does not untrack existing files.

## Concurrency and failure semantics

Mutation commands use a nonblocking OS advisory file lock. The operating system
releases it if a process exits. Handoff writes also require the exact revision read
before doing the work. The lock protects CLI metadata writes, not arbitrary manual
edits, model actions, source-code edits or remote filesystems. Use one code writer
per worktree. Do not initialize the same project concurrently or edit its rules
while init is applying.

Files are individually replaced atomically. Installation is not a single
transaction across all project paths and the global registry: a permissions error
can leave partial setup. Review output and backups before rerunning. Init is
idempotent for an unchanged installation; it is not an automatic upgrade manager.
Existing differing skills are refused, never overwritten.

## Workspace fingerprint

The checkpoint records branch, HEAD, Git status and a SHA-256 fingerprint of HEAD,
status, tracked staged/unstaged binary diffs, and bounded untracked file contents.
It never stores diff contents. External diff programs and text conversions are
disabled. At most 200 untracked entries, 1 MiB per regular file and 8 MiB total are
content-hashed; file name, size, mtime and mode are still considered. Submodules,
ignored files, large untracked files and external services need manual checking.
A matching fingerprint only indicates no detected change within this coverage.
It is not a consistent repository-wide snapshot if another process writes during
capture. It is not test verification, a code backup, or a security attestation.
No Git repository means no usable workspace fingerprint.

## Native sessions and privacy

Session references contain only the agent name, user-provided native ID, optional
label and registration time. There is no native-history parser, importer, watcher,
resume adapter, chat synchronization, API connection, telemetry or LLM call.
`search` searches current task handoffs and metadata, not native transcripts or
old handoff revisions. Run native agents or agent-deck separately.

Handoffs may still contain private summaries and file names. Inspect generated
context before sharing it. No automatic redaction or encryption is implemented.
A ChatGPT cloud conversation without authorized access to the same project must
receive an explicitly provided handoff; local symlinks do not give it access.

## Rules and skills

Init appends a marked block to AGENTS.md, prepends `@AGENTS.md` to CLAUDE.md when
needed, copies this skill once into `.agents/skills/task-handoff`, and links the
Claude skill entry to it. A preexisting CLAUDE.md -> AGENTS.md symlink is preserved.
Existing files modified by init are copied to `.agent-state/backups/TIMESTAMP/`.
Other existing symlinked configuration files or symlinked parent directories are
refused to avoid accidentally writing to a global/external configuration.

`link-skill` keeps an external skill directory as the source of truth, adds an
`.agents/skills/NAME` symlink and a `.claude/skills/NAME` symlink, and refuses name
collisions. Keep external targets accessible and stable. For team portability,
prefer actual skill files in `.agents/skills/NAME` and link only the Claude entry.
Verify the SKILL.md name matches the directory and check for per-agent overrides.
The CLI checks paths, not whether every agent has loaded or obeyed the skill.
