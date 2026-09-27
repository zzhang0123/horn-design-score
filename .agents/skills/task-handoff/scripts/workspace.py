#!/usr/bin/env python3
"""Local project/task handoffs. Python 3.9+, macOS/Linux/WSL, stdlib only.

Never reads or modifies native agent session stores. Never launches an agent,
executes a test, or sends network requests. Run --help for the command interface.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid
from typing import Any, Iterator

VERSION = "0.1.0"
SKILL_ROOT = Path(__file__).resolve().parent.parent
TASK_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
SKILL_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
CLI = "python3 .agents/skills/task-handoff/scripts/workspace.py"
BEGIN = "<!-- agent-workspace:begin -->"
END = "<!-- agent-workspace:end -->"
RULES = f"""{BEGIN}
## Cross-agent task handoff
- Keep shared project rules here. Keep portable skills in `.agents/skills`.
- When a user supplies a tracked task ID, read `.agents/skills/task-handoff/SKILL.md`
  and run `{CLI} context TASK_ID` before editing, even in a resumed conversation.
- Confirm the project/worktree, current code, task revision and latest handoff.
  Prior chat history is not the source of truth for the current workspace.
- At a milestone or before switching tools, checkpoint the goal, changes,
  decisions, actual test commands/results, blockers and next action. Use the
  task-handoff CLI and its expected-revision check; do not edit task.json directly.
- A recorded test result is a report, not independent verification. Never claim
  tests passed unless actually run. Do not auto-commit, push or reset for a handoff.
- Work sequentially in one worktree. Use separate worktrees for concurrent edits.
- Do not merge, rewrite or symlink native Codex/Claude/DSH session databases.
- Keep credentials and private transcripts out of handoffs. Local `.agent-state`
  is ignored by Git but is not encrypted or automatically redacted.
{END}
"""


class Error(Exception):
    """Expected user-facing error."""


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Error(f"Cannot read valid JSON from {path}: {exc}") from exc


def dump(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def atomic(path: Path, text: str) -> None:
    """Replace a complete file, preserving its existing permission bits."""
    if path.is_symlink():
        raise Error(f"Refusing to replace a symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600
    fd, temp = tempfile.mkstemp(prefix=".aw-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(temp, mode)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


@contextlib.contextmanager
def lock(path: Path) -> Iterator[None]:
    try:
        import fcntl
    except ImportError as exc:
        raise Error("Use macOS, Linux or WSL; native Windows locking is not supported.") from exc
    if path.is_symlink():
        raise Error(f"Refusing symlink lock: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as f:
        os.chmod(path, 0o600)
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Error(f"Another workspace operation is running: {path}. Retry after it finishes.") from exc
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def guard(root: Path, relative: str) -> Path:
    """Mutable project paths must not traverse symlinks."""
    out = root
    for part in Path(relative).parts:
        if part in ("..", "/"):
            raise Error("Unsafe relative path")
        out = out / part
        if out.is_symlink():
            raise Error(f"Refusing to write through a symlink: {out}")
    return out


def git(root: Path, *args: str) -> bytes | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), *args], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=30,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return proc.stdout if proc.returncode == 0 else None


def git_text(root: Path, *args: str) -> str | None:
    value = git(root, *args)
    return value.decode("utf-8", errors="replace").strip() if value is not None else None


def snapshot(root: Path) -> dict[str, Any]:
    """Hash tracked diffs plus bounded untracked content; never save source content."""
    if git_text(root, "rev-parse", "--is-inside-work-tree") != "true":
        return {"git": False, "captured_at": now(), "fingerprint": None,
                "complete": False, "note": "No Git fingerprint available"}
    head = git_text(root, "rev-parse", "HEAD")
    branch = git_text(root, "symbolic-ref", "--short", "HEAD") or "(detached/unborn)"
    status_bytes = git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    diff = git(root, "diff", "--no-ext-diff", "--no-textconv", "--binary")
    staged = git(root, "diff", "--cached", "--no-ext-diff", "--no-textconv", "--binary")
    untracked = git(root, "ls-files", "--others", "--exclude-standard", "-z")
    complete = all(x is not None for x in (status_bytes, diff, staged, untracked))
    h = hashlib.sha256()
    for content in ((head or "").encode(), status_bytes, diff, staged):
        h.update(content or b"")
        h.update(b"\0")
    budget = 8 * 1024 * 1024
    names = (untracked or b"").split(b"\0")
    count = 0
    for name in names:
        if not name:
            continue
        count += 1
        p = root / os.fsdecode(name)
        h.update(name + b"\0")
        try:
            meta = p.lstat()
            h.update(f"{meta.st_size}:{meta.st_mtime_ns}:{meta.st_mode}".encode())
            if p.is_symlink():
                h.update(os.fsencode(os.readlink(p)))
            elif count <= 200 and p.is_file() and meta.st_size <= min(budget, 1024 * 1024):
                content = p.read_bytes()
                h.update(content)
                budget -= len(content)
            else:
                complete = False
        except OSError:
            complete = False
        h.update(b"\0")
    # A dirty submodule may hide content changes behind an unchanged status marker.
    if (root / ".gitmodules").exists():
        complete = False
    status = (status_bytes or b"").decode("utf-8", errors="replace").replace("\0", "\n").rstrip()
    return {"git": True, "captured_at": now(), "head": head, "branch": branch,
            "status": status, "fingerprint": h.hexdigest(), "complete": complete,
            "note": "Read-only heuristic; ignored files and external state are excluded. "
                    "Untracked hashing is bounded; submodules require manual verification."}


def home() -> Path:
    return Path(os.environ.get("AW_HOME", "~/.local/share/agent-workspace")).expanduser().resolve()


def registry() -> dict[str, Any]:
    p = home() / "projects.json"
    return read_json(p) if p.exists() else {"schema": 1, "projects": {}}


def project_root(explicit: str | None = None) -> Path:
    current = Path(explicit or os.getcwd()).expanduser().resolve()
    candidates = [current] if explicit else [current, *current.parents]
    for p in candidates:
        guard(p, ".agent-state/project.json")
        if (p / ".agent-state/project.json").is_file():
            return p
        # Never silently select a different worktree/repository's parent project.
        if (p / ".git").exists():
            break
    raise Error("Project not initialized. Run: aw init /absolute/project/path --apply")


def require_task(root: Path, task_id: str) -> tuple[Path, dict[str, Any]]:
    if not TASK_RE.fullmatch(task_id):
        raise Error("Task ID: 1-64 ASCII letters/digits, underscores or hyphens; start with letter/digit.")
    p = guard(root, f".agent-state/tasks/{task_id}/task.json")
    if not p.is_file():
        raise Error(f"Unknown task {task_id}. Create it with: aw new {task_id} --title '...' ")
    data = read_json(p)
    if data.get("id") != task_id or data.get("schema") != 1:
        raise Error(f"Unsupported or mismatched task record: {p}")
    return p, data


def markdown(data: dict[str, Any]) -> str:
    c = data.get("checkpoint") or {}
    s = c.get("workspace") or {}
    lines = [f"# {data['id']}: {data['title']}", "", f"Revision: {data['revision']}",
             f"Status: {data['status']}", f"Updated: {data['updated_at']}",
             f"Project/worktree: {data['project_root']}", "", "## Goal", data['title'], ""]
    if not c:
        lines += ["## Handoff", "No checkpoint yet. Inspect the task and current code before starting.", ""]
    else:
        lines += ["## Latest handoff", f"Reported by: {c['agent']} at {c['at']}",
                  c['summary'], "", "## Decisions"]
        lines += c['decisions'] or ["None recorded."]
        lines += ["", "## Changed files (reported)"] + (c['files'] or ["See Git status; no explicit file list."])
        lines += ["", "## Tests (reported, not independently verified)", c['tests'],
                  "", "## Blockers"] + (c['blockers'] or ["None reported."])
        lines += ["", "## Next action", c['next'], "", "## Git checkpoint",
                  f"Branch: {s.get('branch', 'unavailable')}", f"HEAD: {s.get('head', 'unavailable')}",
                  f"Fingerprint: {s.get('fingerprint', 'unavailable')}",
                  f"Coverage complete within stated limits: {s.get('complete', False)}", ""]
    lines += ["## Native session references (metadata only)"]
    for item in data['sessions']:
        lines.append(f"- {item['agent']}: {item['session']} ({item.get('label', '')})")
    if not data['sessions']:
        lines.append("No native session IDs registered; this does not block handoffs.")
    lines += ["", "Generated from task.json. Do not edit this view; use the handoff command.", ""]
    return "\n".join(lines)


def save_task(root: Path, p: Path, data: dict[str, Any], previous: dict[str, Any] | None = None) -> None:
    if previous is not None:
        hist = guard(root, f".agent-state/tasks/{data['id']}/history/{previous['revision']:06d}.json")
        if not hist.exists():
            atomic(hist, dump(previous))
    atomic(p, dump(data))  # Authoritative; the Markdown view can always be regenerated.
    md = guard(root, f".agent-state/tasks/{data['id']}/handoff.md")
    atomic(md, markdown(data))


def existing_text(path: Path) -> str:
    if path.exists() and not path.is_file():
        raise Error(f"Expected a regular text file: {path}")
    return path.read_text(encoding="utf-8") if path.exists() else ""


def check_dirs(root: Path) -> None:
    for rel in (".agent-state", ".agents", ".agents/skills", ".claude", ".claude/skills"):
        p = guard(root, rel)
        if p.exists() and not p.is_dir():
            raise Error(f"Expected a directory: {p}")


def link_plan(path: Path, target: Path) -> bool:
    if path.is_symlink() and path.resolve() == target.resolve():
        return False
    if path.exists() or path.is_symlink():
        raise Error(f"Refusing to replace existing skill entry: {path}. Reconcile it manually first.")
    return True


def install_link(path: Path, target: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(os.path.relpath(target, path.parent), target_is_directory=True)


def same_tree(a: Path, b: Path) -> bool:
    def listing(p: Path) -> dict[str, bytes]:
        return {str(x.relative_to(p)): x.read_bytes() for x in p.rglob("*")
                if x.is_file() and "__pycache__" not in x.parts and x.suffix != ".pyc"}
    return listing(a) == listing(b)


def cmd_init(args: argparse.Namespace) -> None:
    root = Path(args.path).expanduser().resolve()
    if not root.is_dir():
        raise Error("Project directory must already exist.")
    top = git_text(root, "rev-parse", "--show-toplevel")
    if top and Path(top).resolve() != root:
        raise Error(f"Initialize the Git/worktree root instead: {top}")
    check_dirs(root)
    conf = guard(root, ".agent-state/project.json")
    if not conf.exists() and (root / ".agent-state/tasks").exists():
        raise Error("Tasks exist without project.json; recover the project metadata before init.")
    proj = read_json(conf) if conf.exists() else {
        "schema": 1, "id": str(uuid.uuid4()), "name": args.name or root.name,
        "created_at": now(),
    }
    if args.name and proj['name'] != args.name:
        raise Error("Existing project name differs; init does not rename it.")
    agents = guard(root, "AGENTS.md")
    old_agents = existing_text(agents)
    if (BEGIN in old_agents) != (END in old_agents):
        raise Error("Incomplete managed block in AGENTS.md; repair it before init.")
    new_agents = old_agents if BEGIN in old_agents else old_agents.rstrip() + ("\n\n" if old_agents else "") + RULES
    claude = root / "CLAUDE.md"
    changes = [(agents, new_agents)]
    if claude.is_symlink() and claude.resolve() == agents.resolve():
        pass  # Existing supported CLAUDE.md -> AGENTS.md; never add a self-import.
    else:
        guard(root, "CLAUDE.md")
        old = existing_text(claude)
        imported = any(x.strip() in ("@AGENTS.md", "@./AGENTS.md") for x in old.splitlines())
        changes.append((claude, old if imported else "@AGENTS.md\n\n" + old))
    ignore = guard(root, ".gitignore")
    old = existing_text(ignore)
    changes.insert(0, (ignore, old if "/.agent-state/" in old.splitlines() else
                       old.rstrip() + ("\n" if old else "") + "\n# Local agent handoffs and installation backups\n/.agent-state/\n"))
    dest = root / ".agents/skills/task-handoff"
    need_copy = not dest.exists()
    if dest.is_symlink():
        raise Error("task-handoff is already a symlink; preserve it and configure this project manually.")
    if not need_copy and (not dest.is_dir() or not same_tree(dest, SKILL_ROOT)):
        raise Error(f"Existing {dest} differs. Init will not overwrite a modified skill.")
    clink = root / ".claude/skills/task-handoff"
    need_link = link_plan(clink, dest)
    changes = [(p, text) for p, text in changes if not p.exists() or existing_text(p) != text]
    # Read the registry before any mutation so corruption cannot be silently overwritten.
    registry()
    print(f"{'APPLY' if args.apply else 'PLAN (no writes)'}: {root}")
    for p, _ in changes:
        print(f"  {'Back up and update' if p.exists() else 'Create'} {p.relative_to(root)}")
    if need_copy:
        print("  Copy portable task-handoff skill into .agents/skills/task-handoff")
    if need_link:
        print("  Link .claude/skills/task-handoff -> ../../.agents/skills/task-handoff")
    print(f"  Keep local state in .agent-state; register project in {home() / 'projects.json'}")
    if not args.apply:
        print("Review the plan, then rerun with --apply.")
        return
    with lock(guard(root, ".agent-state/.lock")):
        backup = root / ".agent-state/backups" / stamp()
        for p, text in changes:
            if p.exists():
                b = backup / p.name
                b.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, b)
            atomic(p, text)
        if need_copy:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(SKILL_ROOT, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        if need_link:
            install_link(clink, dest)
        if not conf.exists():
            atomic(conf, dump(proj))
        with lock(home() / ".registry.lock"):
            reg = registry()
            reg['projects'][proj['id']] = {**proj, "path": str(root)}
            atomic(home() / "projects.json", dump(reg))
    print("Initialized. Review the Git diff and run: aw doctor")


def cmd_new(args: argparse.Namespace) -> None:
    root = project_root(args.project)
    if not TASK_RE.fullmatch(args.task):
        raise Error("Invalid task ID; use e.g. T-001 or login-refactor.")
    if not args.title.strip():
        raise Error("A non-empty title is required.")
    with lock(guard(root, ".agent-state/.lock")):
        p = guard(root, f".agent-state/tasks/{args.task}/task.json")
        if p.exists():
            raise Error(f"Task already exists: {args.task}")
        proj = read_json(root / ".agent-state/project.json")
        data = {"schema": 1, "id": args.task, "title": args.title,
                "project_id": proj['id'], "project_root": str(root), "status": "todo",
                "revision": 0, "created_at": now(), "updated_at": now(),
                "sessions": [], "checkpoint": None}
        save_task(root, p, data)
    print(f"Created {args.task} at revision 0. Run: aw context {args.task}")


def cmd_handoff(args: argparse.Namespace) -> None:
    root = project_root(args.project)
    if not all(x.strip() for x in (args.summary, args.next, args.tests)):
        raise Error("summary, next and tests must be non-empty; use 'not run' when appropriate.")
    with lock(guard(root, ".agent-state/.lock")):
        p, old = require_task(root, args.task)
        if old['revision'] != args.expect:
            raise Error(f"Revision conflict: expected {args.expect}, current {old['revision']}. "
                        f"Run 'aw context {args.task}', reconcile changes, then retry.")
        data = {**old, "revision": old['revision'] + 1, "updated_at": now(), "status": args.status}
        data['checkpoint'] = {
            "at": now(), "agent": args.agent, "summary": args.summary, "next": args.next,
            "tests": args.tests, "decisions": args.decision or [], "files": args.file or [],
            "blockers": args.blocker or [], "workspace": snapshot(root),
        }
        save_task(root, p, data, old)
    print(f"Saved {args.task} revision {data['revision']}; prior revision retained in history/.")
    print("This records a handoff. It does not stop an agent or verify reported tests.")


def cmd_context(args: argparse.Namespace) -> None:
    root = project_root(args.project)
    _, data = require_task(root, args.task)
    current = snapshot(root)
    if args.json:
        print(dump({"task": data, "current_workspace": current}), end="")
        return
    print(f"Continue task {args.task} in {root}. Read AGENTS.md and the task-handoff skill.")
    print("Use the current files as truth; reconcile old conversation history against this handoff.")
    print(f"Before recording the next handoff, use --expect {data['revision']}.")
    print("\n" + markdown(data))
    previous = (data.get('checkpoint') or {}).get('workspace', {})
    if data['project_root'] != str(root):
        print("WARNING: The recorded project path differs. Confirm this moved/duplicated workspace.")
    if not previous.get('fingerprint') or not current.get('fingerprint'):
        print("Workspace comparison: unavailable or no previous checkpoint.")
    elif previous['fingerprint'] != current['fingerprint']:
        print("WARNING: Workspace differs from the last checkpoint. Inspect diff and tests before editing.")
    else:
        print("Workspace fingerprint matches within its stated coverage. This is not a test result.")
    if not current.get('complete') or not previous.get('complete'):
        print("Coverage is limited: manually check ignored files, large untracked files, submodules and external state.")
    print(f"Current branch: {current.get('branch', 'unavailable')}; HEAD: {current.get('head', 'unavailable')}")
    if current.get('status'):
        print("Current Git status:\n" + current['status'])


def cmd_bind(args: argparse.Namespace) -> None:
    root = project_root(args.project)
    if not args.session.strip() or any(ord(c) < 32 for c in args.session):
        raise Error("Use a non-empty, single-line native session ID.")
    with lock(guard(root, ".agent-state/.lock")):
        p, old = require_task(root, args.task)
        if any(x['agent'] == args.agent and x['session'] == args.session for x in old['sessions']):
            print("This native session is already registered; no change.")
            return
        data = {**old, "revision": old['revision'] + 1, "updated_at": now(),
                "sessions": [*old['sessions'], {"agent": args.agent, "session": args.session,
                                               "label": args.label, "at": now()}]}
        save_task(root, p, data, old)
    print(f"Registered native session reference; task revision is now {data['revision']}.")


def cmd_projects(args: argparse.Namespace) -> None:
    rows = list(registry()['projects'].values())
    if args.json:
        print(dump(rows), end="")
    else:
        for p in rows:
            found = Path(p['path']).is_dir()
            print(f"{p['name']}\t{p['path']}" + ("" if found else " [missing]"))
        if not rows:
            print("No registered projects. Run: aw init /path/to/project --apply")


def cmd_tasks(args: argparse.Namespace) -> None:
    root = project_root(args.project)
    task_root = guard(root, ".agent-state/tasks")
    rows = [require_task(root, p.name)[1] for p in sorted(task_root.iterdir()) if p.is_dir()] if task_root.exists() else []
    if args.json:
        print(dump(rows), end="")
    else:
        for d in rows:
            print(f"{d['id']}\t{d['status']}\tr{d['revision']}\t{d['title']}")
        if not rows:
            print("No tasks. Run: aw new T-001 --title 'Your goal'")


def cmd_search(args: argparse.Namespace) -> None:
    needle = args.query.casefold()
    matches = []
    for proj in registry()['projects'].values():
        root = Path(proj['path'])
        try:
            tasks = guard(root, ".agent-state/tasks")
            for p in sorted(tasks.iterdir()) if tasks.is_dir() else []:
                if not p.is_dir():
                    continue
                _, data = require_task(root, p.name)
                if needle in markdown(data).casefold():
                    matches.append({"project": proj['name'], "path": str(root),
                                    "task": data['id'], "title": data['title'], "status": data['status']})
        except Error as exc:
            print(f"WARNING: {proj['name']}: {exc}", file=sys.stderr)
    if args.json:
        print(dump(matches), end="")
    else:
        for r in matches:
            print(f"{r['project']} / {r['task']} / {r['status']}: {r['title']}\n  {r['path']}")
        if not matches:
            print("No matches in current task handoffs. Native chat transcripts are not indexed.")


def cmd_link_skill(args: argparse.Namespace) -> None:
    root = project_root(args.project)
    check_dirs(root)
    source = Path(args.source).expanduser().resolve()
    if not source.is_dir() or not (source / 'SKILL.md').is_file():
        raise Error("Source must be a skill directory containing SKILL.md.")
    name = source.name
    if not SKILL_RE.fullmatch(name) or name in ('synced', 'task-handoff'):
        raise Error("Use a kebab-case directory name; synced and task-handoff are reserved here.")
    canonical = root / '.agents/skills' / name
    clink = root / '.claude/skills' / name
    new_canonical = False if canonical == source else link_plan(canonical, source)
    new_claude = link_plan(clink, canonical)
    print(f"{'APPLY' if args.apply else 'PLAN (no writes)'}: {name}")
    if new_canonical:
        print(f"  {canonical} -> {source}")
    if new_claude:
        print(f"  {clink} -> {canonical}")
    if not args.apply:
        print("Review the skill and then rerun with --apply. No existing skills are replaced.")
        return
    with lock(guard(root, '.agent-state/.lock')):
        if new_canonical:
            install_link(canonical, source)
        if new_claude:
            install_link(clink, canonical)
    print("Linked. Reopen affected agent sessions and verify actual skill discovery.")


def cmd_doctor(args: argparse.Namespace) -> None:
    root = project_root(args.project)
    print(f"Project: {root}")
    errors = 0
    for rel in ('AGENTS.md', 'CLAUDE.md', '.agents/skills/task-handoff/SKILL.md',
                '.claude/skills/task-handoff/SKILL.md'):
        ok = (root / rel).is_file()
        print(f"{'OK' if ok else 'MISSING'}: {rel}")
        errors += int(not ok)
    if (root / 'AGENTS.override.md').exists():
        print("WARNING: AGENTS.override.md can take precedence over AGENTS.md in Codex.")
    override = root / '.dsh/skills/task-handoff'
    if override.exists():
        print("WARNING: .dsh/skills/task-handoff may shadow the shared DSH skill.")
    for command in ('python3', 'git', 'codex', 'claude', 'dsh', 'agent-deck'):
        print(f"PATH {command}: {shutil.which(command) or '(not found; not required for state management)'}")
    print("The dsh PATH entry is NOT identity verification. Check agent-deck deepseek status or inspect dsh --help.")
    ignored = git(root, 'check-ignore', '-q', '.agent-state/project.json')
    if git_text(root, 'rev-parse', '--is-inside-work-tree') == 'true':
        print('OK: .agent-state is Git-ignored' if ignored is not None else 'WARNING: .agent-state is not Git-ignored')
        tracked = git(root, 'ls-files', '--', '.agent-state')
        if tracked:
            print('WARNING: .agent-state already has tracked files; .gitignore does not untrack them.')
    print("Filesystem checks only. Actual discovery, permissions and session freshness must be verified in each agent.")
    if errors:
        raise Error(f"{errors} required paths missing.")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version', action='version', version=VERSION)
    p.add_argument('--project', '-p', help='Initialized project/worktree root; default: discover from cwd')
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('init', help='Preview or apply non-destructive project setup')
    q.add_argument('path', nargs='?', default='.')
    q.add_argument('--name')
    q.add_argument('--apply', action='store_true')
    q.set_defaults(func=cmd_init)
    q = sub.add_parser('new', help='Create a task (does not create a native agent session)')
    q.add_argument('task'); q.add_argument('--title', required=True); q.set_defaults(func=cmd_new)
    q = sub.add_parser('handoff', help='Save a checked, versioned handoff and a Git fingerprint')
    q.add_argument('task'); q.add_argument('--agent', choices=['claude', 'codex', 'dsh', 'human'], required=True)
    q.add_argument('--expect', type=int, required=True, help='Revision read before doing the work')
    for field in ('summary', 'next', 'tests'):
        q.add_argument('--' + field, required=True)
    for field in ('decision', 'file', 'blocker'):
        q.add_argument('--' + field, action='append')
    q.add_argument('--status', choices=['ready', 'blocked', 'done'], default='ready')
    q.set_defaults(func=cmd_handoff)
    q = sub.add_parser('context', help='Generate a handoff prompt; compare current worktree')
    q.add_argument('task'); q.add_argument('--json', action='store_true'); q.set_defaults(func=cmd_context)
    q = sub.add_parser('bind', help='Register a native session ID; no import or resume')
    q.add_argument('task'); q.add_argument('--agent', choices=['claude', 'codex', 'dsh'], required=True)
    q.add_argument('--session', required=True); q.add_argument('--label', default='')
    q.set_defaults(func=cmd_bind)
    for name, func in [('projects', cmd_projects), ('tasks', cmd_tasks)]:
        q = sub.add_parser(name); q.add_argument('--json', action='store_true'); q.set_defaults(func=func)
    q = sub.add_parser('search', help='Search current handoffs across registered projects; not native chat logs')
    q.add_argument('query'); q.add_argument('--json', action='store_true'); q.set_defaults(func=cmd_search)
    q = sub.add_parser('link-skill', help='Preview/link a shared skill without replacing existing entries')
    q.add_argument('source'); q.add_argument('--apply', action='store_true'); q.set_defaults(func=cmd_link_skill)
    q = sub.add_parser('doctor', help='Read-only local filesystem checks'); q.set_defaults(func=cmd_doctor)
    return p


def main() -> int:
    try:
        args = parser().parse_args()
        args.func(args)
        return 0
    except (Error, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == '__main__':
    sys.exit(main())
