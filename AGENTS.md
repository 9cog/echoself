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

## Cursor Cloud specific instructions

This repo is a dual stack: a Remix/Vite web app ("Deep Tree Echo") plus a large Python ML/training toolkit (NanEcho, `train_*.py`, `test_*.py`). The Cloud update script already runs `npm install`, `pip install -r requirements.txt`, and `pip install matplotlib`, so those do not need rerunning.

- Web app (dev): `npm run dev` serves `http://localhost:3000` (classic Remix compiler + `remix-serve`, per `package.json`). It runs with NO secrets — `SUPABASE_URL`/`SUPABASE_ANON_KEY`/`OPENAI_API_KEY` are optional; loaders fall back to empty/empty-data when unset. `.env` is optional (`cp .env.example .env` only holds placeholders).
- Lint/typecheck/build: use the `package.json` scripts (`npm run lint`, `npm run typecheck`, `npm run build`). Lint runs through `scripts/run-eslint.cjs` (legacy, non-flat config) and currently reports 0 errors / ~100 warnings — warnings are expected, not failures.
- Python tests: `python3 -m pytest --ignore=test_data_validation.py` (that file needs a CUDA GPU). Two extra caveats not in the README: `matplotlib` (installed by the update script, not in `requirements.txt`) is required or `test_120cell.py` / `test_matula_transformer.py` / `test_recursive_transformer.py` fail at collection; and `NanEcho/test_production_runtime.py` has a pre-existing collection error (imports `_controlled_samples`, which no longer exists in `prepare_nanecho.py`) — ignore it with `--ignore=NanEcho/test_production_runtime.py`. With those ignores, expect ~269 passed / ~30 pre-existing failures (missing fixture/data files), which are unrelated to environment setup.
- Known pre-existing client-side bugs (NOT environment problems, documented in the testing skill / README): `/editor` (Monaco) throws `CallbackIterable is not a constructor` and never renders; `/terminal`'s route component (`app/components/TerminalComponent.client.tsx`) does not import `xterm/css/xterm.css`, so the xterm viewport can render with collapsed/near-invisible styling. The terminal BACKEND works fine regardless — `POST /terminal` with form field `command` executes real commands (`help`, `echo`, `version`, `ls`, `pwd`, `node`, `python`) and returns output in the Remix `actionData`.
- All routes SSR to HTTP 200 (`/`, `/editor`, `/terminal`, `/chat`, `/memory`, `/layla`, `/map`); `/memory`, `/chat`, `/layla` need Supabase/OpenAI for full functionality but still load.
