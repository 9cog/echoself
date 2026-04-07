# Testing the EchoSelf Web App

## Local Dev Setup

```bash
# Install dependencies
npm install
pip install -r requirements.txt

# Create env file (uses placeholder values, Supabase optional for dev)
cp -n .env.example .env || true

# Start dev server
npm run dev
# App runs at http://localhost:3000
```

## Build & Lint Commands

```bash
npm run build        # Production build (Remix + Vite)
npm run typecheck    # TypeScript type checking (tsc)
npm run lint         # ESLint (uses legacy config, not flat config)
```

## Python Tests

```bash
# Run Python tests (skip GPU-dependent test file)
python3 -m pytest --ignore=test_data_validation.py
```

**Known issue**: `test_data_validation.py` requires CUDA GPU and will fail on CI/dev machines without GPU. Skip it with `--ignore`.

## Key Routes to Test

| Route | Feature | Navigation |
|-------|---------|------------|
| `/` | Homepage | Direct URL |
| `/editor` | Monaco code editor | Click "Code Editor" link on homepage |
| `/terminal` | xterm terminal emulator | Direct URL (no homepage link) |
| `/chat` | AI Chat | Click "AI Chat" on homepage |
| `/memory` | Memory System | Click "Memory System" on homepage |
| `/layla` | EchoLayla AI | Click "EchoLayla AI" on homepage |
| `/map` | Echo Home Map | Click "Echo Home Map" on homepage |

## Browser-Only Components

The app uses browser-only packages that **cannot** be imported during SSR:

- **`monaco-editor`** — No Node.js entry point; must use `.client.tsx` suffix
- **`xterm`** — CJS module with named exports incompatible with Node ESM

These components use the Remix `.client.tsx` convention + `React.lazy()` + `ClientOnly` wrapper pattern. When testing:

1. Verify the dev server starts without crashing (`npm run dev`)
2. Verify all routes return HTTP 200 (server-side rendering works)
3. Verify client-side components actually render (not stuck on "Loading..." fallback)

## Testing NanEcho Config Rewriter (apply_adaptation.py)

The config-rewriter is a Python CLI script with no UI. Test it by crafting JSON input files and verifying output files.

### CLI Invocation

```bash
python3 NanEcho/apply_adaptation.py \
  --analysis automation_analysis.json \
  --trigger next_cycle_trigger.json \
  --config nanecho_config.json \
  --eval-report eval_history/ \
  --output adapted_config.json \
  --history-file .training-progress/adaptation_history.jsonl
```

### Required Input Files

- **`automation_analysis.json`**: Must have `overall_fidelity` (float) and `recommendations` (dict) at top level
- **`next_cycle_trigger.json`**: Can be `{}` for most tests
- **`nanecho_config.json`**: Must have `model`, `training`, `echo_self`, `data` sections
- **`eval_history/eval_*.json`** (optional): For dimension_coverage, keyword_coverage, spectral radius tests. Must have `dimension_coverage`, `keyword_coverage`, `avg_latency_ms`, and optionally `esn_telemetry.spectral_radius`
- **`adaptation_history.jsonl`** (optional): For regression detection and coherence halt tests. Each line is JSON with `fidelity_after`, `total_changes`, `config_snapshot`, `delta_snapshot`

### Key Safety Features to Test

| Feature | Trigger Condition | Key Assertion |
|---------|------------------|---------------|
| Delta Clamp | Any parameter change | Change ≤ 20% of original value |
| n_embd/n_head Divisibility | n_embd and n_head both change | `n_embd % n_head == 0` after adaptation |
| Weight Normalization | `dimension_coverage < 0.50` | `sum(weights.values()) == 1.0` |
| Regression Detection | Current fidelity < prior fidelity (with prior changes > 0) | Both adapted_config.json AND nanecho_config.json reverted |
| Coherence Halt | Fidelity < 0.20 for 2+ consecutive cycles | `halt_adaptation.flag` file created with `consecutive_crisis_cycles >= 2` |
| Spectral Radius | `esn_telemetry.spectral_radius` outside [0.85, 0.95] | Adjusted toward 0.95 but clamped at 20% |
| Compounding Prevention | Both fidelity + eval rules would modify same param | Only 1 change per parameter per cycle |

### Testing Tips

- **Use isolated directories**: Create a fresh temp dir for each test to avoid cross-contamination
- **Delta-only output**: `adapted_config.json` only contains changed keys, not the full config
- **Verify both files on revert**: Regression detection must restore both `adapted_config.json` AND `nanecho_config.json` — the second one uses `config_snapshot` from `adaptation_history.jsonl`
- **Coherence halt prevents all changes**: When halted, no `adapted_config.json` is written at all
- **The `_CLAMP_EXEMPT` set**: `persona_weight` and `dimension_weights` are exempt from delta clamping
- **Pre-existing CI failure**: The `test` job (deno lint) may fail due to `no-var` in `app/types/global.d.ts` — this is pre-existing on `main` and unrelated to any changes

## Known Issues

- **monaco-editor client-side error**: `monaco-editor@^0.52.2` might throw `TypeError: CallbackIterable is not a constructor` when loaded through Vite bundling. This is a compatibility issue between the monaco-editor version and Vite's ESM bundling. Potential fixes: pin to an older compatible version, use `vite-plugin-monaco-editor`, or use `@monaco-editor/react` wrapper.
- **xterm works correctly** with the `.client.tsx` + lazy loading pattern.
- **deno lint CI failure**: Pre-existing `deno lint` errors in `app/types/global.d.ts` (no-var) and `src/services/cognitiveEngineBridge.ts` (ban-unused-ignore) are not related to app functionality.
- **pytest failures**: 17 pre-existing test failures from missing fixture files and CUDA requirements.

## No Auth Required

The local dev server has no authentication. Supabase credentials in `.env` are optional — the app runs without them but some features (like persistent storage) might not work.

## Devin Secrets Needed

None required for local testing. Optional:
- `SUPABASE_URL` and `SUPABASE_ANON_KEY` for full Supabase functionality
