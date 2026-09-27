<!-- agent-workspace:begin -->
## Cross-agent task handoff
- Keep shared project rules here. Keep portable skills in `.agents/skills`.
- When a user supplies a tracked task ID, read `.agents/skills/task-handoff/SKILL.md`
  and run `python3 .agents/skills/task-handoff/scripts/workspace.py context TASK_ID` before editing, even in a resumed conversation.
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
<!-- agent-workspace:end -->
