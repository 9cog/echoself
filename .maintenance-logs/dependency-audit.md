# Dependency Audit Report - 2026-03-27 03:00:53 UTC

## Dependency Analysis Summary
```json
{
  "timestamp": "2026-03-27T03:00:51.313Z",
  "summary": {
    "totalDependencies": 27,
    "totalDevDependencies": 27,
    "unusedDependencies": 20,
    "securityVulnerabilities": 0
  },
  "details": {
    "unusedDependencies": [
      "@remix-run/serve",
      "@types/node",
      "autoprefixer",
      "prettier",
      "@remix-run/dev",
      "@tailwindcss/cli",
      "@tailwindcss/postcss",
      "@types/localforage",
      "@types/mermaid",
      "@types/python-shell",
      "@types/react",
      "@types/react-dom",
      "@typescript-eslint/eslint-plugin",
      "@typescript-eslint/parser",
      "eslint-import-resolver-typescript",
      "eslint-plugin-import",
      "eslint-plugin-jsx-a11y",
      "eslint-plugin-react",
      "eslint-plugin-react-hooks",
      "vite-tsconfig-paths"
    ],
    "dependencyUsage": {
      "@huggingface/inference": 1,
      "@remix-run/node": 12,
      "@remix-run/react": 13,
      "@remix-run/serve": 0,
      "@stackblitz/sdk": 2,
      "@supabase/supabase-js": 7,
      "@types/node": 0,
      "autoprefixer": 0,
      "framer-motion": 5,
      "hnswlib-node": 2,
      "isbot": 1,
      "mermaid": 1,
      "ml-distance": 1,
      "ml-matrix": 2,
      "monaco-editor": 5,
      "openai": 17,
      "prettier": 0,
      "python-shell": 1,
      "react": 46,
      "react-dom": 4,
      "react-icons": 23,
      "react-markdown": 2,
      "tailwindcss": 1,
      "xterm": 2,
      "xterm-addon-fit": 2,
      "xterm-addon-web-links": 2,
      "zustand": 1,
      "@codemirror/lang-css": 1,
      "@codemirror/lang-html": 1,
      "@codemirror/lang-javascript": 1,
      "@codemirror/lang-json": 1,
      "@codemirror/lang-markdown": 1,
      "@codemirror/theme-one-dark": 1,
      "@remix-run/dev": 0,
      "@tailwindcss/cli": 0,
      "@tailwindcss/postcss": 0,
      "@types/localforage": 0,
      "@types/mermaid": 0,
      "@types/python-shell": 0,
      "@types/react": 0,
      "@types/react-dom": 0,
      "@typescript-eslint/eslint-plugin": 0,
      "@typescript-eslint/parser": 0,
      "@uiw/react-codemirror": 1,
      "@uiw/react-split": 2,
      "eslint": 7,
      "eslint-import-resolver-typescript": 0,
      "eslint-plugin-import": 0,
      "eslint-plugin-jsx-a11y": 0,
      "eslint-plugin-react": 0,
      "eslint-plugin-react-hooks": 0,
      "typescript": 8,
      "vite": 2,
      "vite-tsconfig-paths": 0
    },
    "securityVulnerabilities": {},
    "securitySummary": {}
  }
}```

## Security Audit
```
# npm audit report

brace-expansion  <5.0.5
Severity: moderate
brace-expansion: Zero-step sequence causes process hang and memory exhaustion - https://github.com/advisories/GHSA-f886-m6hf-6m8v
fix available via `npm audit fix --force`
Will install eslint@4.0.0, which is a breaking change
node_modules/@eslint/eslintrc/node_modules/brace-expansion
node_modules/@humanwhocodes/config-array/node_modules/brace-expansion
node_modules/brace-expansion
node_modules/eslint-plugin-import/node_modules/brace-expansion
node_modules/eslint-plugin-jsx-a11y/node_modules/brace-expansion
node_modules/eslint-plugin-react/node_modules/brace-expansion
node_modules/eslint/node_modules/brace-expansion
node_modules/rimraf/node_modules/brace-expansion
  minimatch  2.0.0 - 10.0.2
  Depends on vulnerable versions of brace-expansion
  node_modules/@eslint/eslintrc/node_modules/minimatch
  node_modules/@humanwhocodes/config-array/node_modules/minimatch
  node_modules/@typescript-eslint/typescript-estree/node_modules/minimatch
  node_modules/eslint-plugin-import/node_modules/minimatch
  node_modules/eslint-plugin-jsx-a11y/node_modules/minimatch
  node_modules/eslint-plugin-react/node_modules/minimatch
  node_modules/eslint/node_modules/minimatch
  node_modules/minimatch
  node_modules/rimraf/node_modules/minimatch
    @eslint/eslintrc  0.0.1 || >=0.1.1
    Depends on vulnerable versions of minimatch
    node_modules/@eslint/eslintrc
      eslint  0.12.0 - 2.0.0-rc.1 || 4.1.0 - 10.0.0-rc.2
      Depends on vulnerable versions of @eslint/eslintrc
      Depends on vulnerable versions of @humanwhocodes/config-array
      Depends on vulnerable versions of file-entry-cache
      Depends on vulnerable versions of minimatch
      node_modules/eslint
        @typescript-eslint/eslint-plugin  <=8.55.1-alpha.3
        Depends on vulnerable versions of @typescript-eslint/type-utils
        Depends on vulnerable versions of @typescript-eslint/utils
        Depends on vulnerable versions of eslint
        node_modules/@typescript-eslint/eslint-plugin
        @typescript-eslint/parser  1.1.1-alpha.0 - 8.55.1-alpha.3
        Depends on vulnerable versions of @typescript-eslint/typescript-estree
        Depends on vulnerable versions of eslint
        node_modules/@typescript-eslint/parser
        @typescript-eslint/type-utils  5.62.1-alpha.0 - 8.0.0-alpha.62 || 8.14.1-alpha.0 - 8.55.1-alpha.3
        Depends on vulnerable versions of @typescript-eslint/typescript-estree
        Depends on vulnerable versions of @typescript-eslint/utils
        Depends on vulnerable versions of eslint
        node_modules/@typescript-eslint/type-utils
        @typescript-eslint/utils  <=8.55.1-alpha.3
        Depends on vulnerable versions of @typescript-eslint/typescript-estree
        Depends on vulnerable versions of eslint
        node_modules/@typescript-eslint/utils
    @humanwhocodes/config-array  *
    Depends on vulnerable versions of minimatch
    node_modules/@humanwhocodes/config-array
    @remix-run/dev  *
    Depends on vulnerable versions of @npmcli/package-json
    Depends on vulnerable versions of @vanilla-extract/integration
    Depends on vulnerable versions of cacache
    Depends on vulnerable versions of esbuild
    Depends on vulnerable versions of minimatch
    Depends on vulnerable versions of remark-mdx-frontmatter
    node_modules/@remix-run/dev
    @typescript-eslint/typescript-estree  6.16.0 - 7.5.0
    Depends on vulnerable versions of minimatch
    node_modules/@typescript-eslint/typescript-estree
    eslint-plugin-import  >=1.15.0
    Depends on vulnerable versions of minimatch
    node_modules/eslint-plugin-import
    eslint-plugin-jsx-a11y  >=6.5.0
    Depends on vulnerable versions of minimatch
    node_modules/eslint-plugin-jsx-a11y
    eslint-plugin-react  >=7.23.0
    Depends on vulnerable versions of minimatch
    node_modules/eslint-plugin-react
    glob  4.3.0 - 10.5.0
    Depends on vulnerable versions of minimatch
    node_modules/glob
    node_modules/rimraf/node_modules/glob
      @npmcli/package-json  3.1.0 - 6.2.0
      Depends on vulnerable versions of glob
      node_modules/@npmcli/package-json
      cacache  6.1.1 - 19.0.1
      Depends on vulnerable versions of glob
      Depends on vulnerable versions of tar
      node_modules/cacache
      rimraf  2.3.0 - 3.0.2 || 4.2.0 - 5.0.10
      Depends on vulnerable versions of glob
      node_modules/rimraf
        flat-cache  1.3.4 - 4.0.0
        Depends on vulnerable versions of rimraf
        node_modules/flat-cache
          file-entry-cache  4.0.0 - 7.0.2
          Depends on vulnerable versions of flat-cache
          node_modules/file-entry-cache

dompurify  3.1.3 - 3.3.1
Severity: moderate
DOMPurify contains a Cross-site Scripting vulnerability - https://github.com/advisories/GHSA-v8jm-5vwx-cfxm
DOMPurify contains a Cross-site Scripting vulnerability - https://github.com/advisories/GHSA-v2wj-7wpq-c8vv
fix available via `npm audit fix`
node_modules/dompurify

esbuild  <=0.24.2
Severity: moderate
esbuild enables any website to send any requests to the development server and read the response - https://github.com/advisories/GHSA-67mh-4wv8-2f99
No fix available
node_modules/esbuild
node_modules/vite/node_modules/esbuild
  @vanilla-extract/integration  *
  Depends on vulnerable versions of esbuild
  Depends on vulnerable versions of vite
  Depends on vulnerable versions of vite-node
  node_modules/@vanilla-extract/integration
  vite  0.11.0 - 6.1.6
  Depends on vulnerable versions of esbuild
  node_modules/vite
    vite-node  <=2.2.0-beta.2
    Depends on vulnerable versions of vite
    node_modules/@vanilla-extract/integration/node_modules/vite-node

estree-util-value-to-estree  <3.3.3
Severity: moderate
estree-util-value-to-estree allows prototype pollution in generated ESTree - https://github.com/advisories/GHSA-f7f6-9jq7-3rqj
fix available via `npm audit fix`
node_modules/estree-util-value-to-estree
  remark-mdx-frontmatter  <=2.1.1
  Depends on vulnerable versions of estree-util-value-to-estree
  node_modules/remark-mdx-frontmatter

flatted  <=3.4.1
Severity: high
flatted vulnerable to unbounded recursion DoS in parse() revive phase - https://github.com/advisories/GHSA-25h7-pfq9-p65f
Prototype Pollution via parse() in NodeJS flatted - https://github.com/advisories/GHSA-rf6f-7fwh-wjgh
fix available via `npm audit fix`
node_modules/flatted


picomatch  <=2.3.1 || 4.0.0 - 4.0.3
Severity: high
Picomatch has a ReDoS vulnerability via extglob quantifiers - https://github.com/advisories/GHSA-c2c7-rcm5-vvqj
Picomatch has a ReDoS vulnerability via extglob quantifiers - https://github.com/advisories/GHSA-c2c7-rcm5-vvqj
Picomatch: Method Injection in POSIX Character Classes causes incorrect Glob Matching - https://github.com/advisories/GHSA-3v7f-55p6-f55p
Picomatch: Method Injection in POSIX Character Classes causes incorrect Glob Matching - https://github.com/advisories/GHSA-3v7f-55p6-f55p
fix available via `npm audit fix`
node_modules/picomatch
node_modules/tinyglobby/node_modules/picomatch

tar  <=7.5.10
Severity: high
node-tar Vulnerable to Arbitrary File Creation/Overwrite via Hardlink Path Traversal - https://github.com/advisories/GHSA-34x7-hfp2-rc4v
node-tar is Vulnerable to Arbitrary File Overwrite and Symlink Poisoning via Insufficient Path Sanitization - https://github.com/advisories/GHSA-8qq5-rm4j-mr97
Arbitrary File Read/Write via Hardlink Target Escape Through Symlink Chain in node-tar Extraction - https://github.com/advisories/GHSA-83g3-92jg-28cx
tar has Hardlink Path Traversal via Drive-Relative Linkpath - https://github.com/advisories/GHSA-qffp-2rhf-9h96
node-tar Symlink Path Traversal via Drive-Relative Linkpath - https://github.com/advisories/GHSA-9ppj-qmqm-q256
Race Condition in node-tar Path Reservations via Unicode Ligature Collisions on macOS APFS - https://github.com/advisories/GHSA-r6q2-hw4h-h46w
fix available via `npm audit fix`
node_modules/@tailwindcss/oxide/node_modules/tar
node_modules/@tailwindcss/postcss/node_modules/tar
node_modules/tar

undici  <=6.23.0
Severity: high
Undici: Malicious WebSocket 64-bit length overflows parser and crashes the client - https://github.com/advisories/GHSA-f269-vfmq-vjvj
Undici has an HTTP Request/Response Smuggling issue - https://github.com/advisories/GHSA-2mjp-6q6p-2qxm
Undici has Unbounded Memory Consumption in WebSocket permessage-deflate Decompression - https://github.com/advisories/GHSA-vrm6-8vpv-qv8q
Undici has Unhandled Exception in WebSocket Client Due to Invalid server_max_window_bits Validation - https://github.com/advisories/GHSA-v9p9-hfj2-hcw8
Undici has CRLF Injection in undici via `upgrade` option - https://github.com/advisories/GHSA-4992-7rv2-5pvq
fix available via `npm audit fix`
node_modules/undici

yaml  2.0.0 - 2.8.2
Severity: moderate
yaml is vulnerable to Stack Overflow via deeply nested YAML collections - https://github.com/advisories/GHSA-48c2-rrv3-qjmp
fix available via `npm audit fix`
node_modules/yaml

32 vulnerabilities (20 moderate, 12 high)

To address issues that do not require attention, run:
  npm audit fix

To address all issues possible (including breaking changes), run:
  npm audit fix --force

Some issues need review, and may require choosing
a different dependency.
No security issues found
```
