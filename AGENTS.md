## Learned User Preferences

- On Windows PowerShell, chain shell commands with `;` rather than `&&`, and pass git commit messages with `-m` rather than bash HEREDOC.
- When committing this repo, exclude the hundreds of untracked `*(2)*` / `*(3)*` Windows copy-duplicates and other unrelated untracked noise; stage only the files from the current work.

## Learned Workspace Facts

- This Cursor workspace is `C:\hyp\ghx\echo\echoself` (slug `c-hyp-ghx-echo-echoself`); sibling EchoSelf checkouts also exist at `C:\hyp\ghx\echoself` and `C:\hyp\ghx\echoself-1`.
- Agent transcripts for this workspace live under `c-hyp-ghx-echo-echoself/agent-transcripts`; older shared transcripts remain under `c-hyp-ghx-echoself-1`; the `c-hyp-ghx-echoself` slug has no `agent-transcripts/` folder.
- The working tree often contains hundreds of untracked `*(2)*` / `*(3)*` Explorer copy-duplicates of docs, agents, workflows, and training-progress files; do not treat them as source of truth or commit them.
- NanEcho cached-CI grounding lives in `.training-progress/nanecho-cached-ci/` (`training_summary.json`, `cache/metadata.json`, and related introspection/cache files).
- Arena/Task runners here only accept `inherit`, `claude-opus-5-thinking-high`, `composer-2.5-fast`, `cursor-grok-4.5-high-fast`, `cursor-grok-4.6-high-fast`, and `gpt-5.6-sol-medium` — not pstack `*-max` / `*-xhigh` defaults.
- Autognosis is configured via `autognosis.json` and observed with `python -m echoself.autognosis` (or `python -m echoself autognosis`); `--remember` writes local mech0 `autognosic` facts only when asked.
