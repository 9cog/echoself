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
