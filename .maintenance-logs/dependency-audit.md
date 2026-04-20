# Dependency Audit Report - 2026-04-20 03:01:29 UTC

## Dependency Analysis Summary
```json
{
  "timestamp": "2026-04-20T03:01:27.011Z",
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
      "openai": 19,
      "prettier": 0,
      "python-shell": 1,
      "react": 47,
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

brace-expansion  <1.1.13 || >=2.0.0 <2.0.3
Severity: moderate
brace-expansion: Zero-step sequence causes process hang and memory exhaustion - https://github.com/advisories/GHSA-f886-m6hf-6m8v
brace-expansion: Zero-step sequence causes process hang and memory exhaustion - https://github.com/advisories/GHSA-f886-m6hf-6m8v
fix available via `npm audit fix`
node_modules/@eslint/eslintrc/node_modules/brace-expansion
node_modules/@humanwhocodes/config-array/node_modules/brace-expansion
node_modules/brace-expansion
node_modules/eslint-plugin-import/node_modules/brace-expansion
node_modules/eslint-plugin-jsx-a11y/node_modules/brace-expansion
node_modules/eslint-plugin-react/node_modules/brace-expansion
node_modules/eslint/node_modules/brace-expansion
node_modules/rimraf/node_modules/brace-expansion

dompurify  <=3.3.3
Severity: moderate
DOMPurify contains a Cross-site Scripting vulnerability - https://github.com/advisories/GHSA-v8jm-5vwx-cfxm
DOMPurify is vulnerable to mutation-XSS via Re-Contextualization  - https://github.com/advisories/GHSA-h8r8-wccr-v5f2
DOMPurify contains a Cross-site Scripting vulnerability - https://github.com/advisories/GHSA-v2wj-7wpq-c8vv
DOMPurify ADD_ATTR predicate skips URI validation - https://github.com/advisories/GHSA-cjmm-f4jc-qw8r
DOMPurify USE_PROFILES prototype pollution allows event handlers - https://github.com/advisories/GHSA-cj63-jhhr-wcxv
DOMPurify's ADD_TAGS function form bypasses FORBID_TAGS due to short-circuit evaluation - https://github.com/advisories/GHSA-39q2-94rc-95cp
fix available via `npm audit fix`
node_modules/dompurify

esbuild  <=0.24.2
Severity: moderate
esbuild enables any website to send any requests to the development server and read the response - https://github.com/advisories/GHSA-67mh-4wv8-2f99
No fix available
node_modules/esbuild
node_modules/vite/node_modules/esbuild
  @remix-run/dev  *
  Depends on vulnerable versions of @vanilla-extract/integration
  Depends on vulnerable versions of cacache
  Depends on vulnerable versions of esbuild
  Depends on vulnerable versions of remark-mdx-frontmatter
  node_modules/@remix-run/dev
  @vanilla-extract/integration  *
  Depends on vulnerable versions of esbuild
  Depends on vulnerable versions of vite
  Depends on vulnerable versions of vite-node
  node_modules/@vanilla-extract/integration
  vite  <=6.4.1
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

lodash  <=4.17.23
Severity: high
lodash vulnerable to Code Injection via `_.template` imports key names - https://github.com/advisories/GHSA-r5fr-rjxr-66jc
lodash vulnerable to Prototype Pollution via array path bypass in `_.unset` and `_.omit` - https://github.com/advisories/GHSA-f23m-r3pf-42rh
fix available via `npm audit fix`
node_modules/lodash

lodash-es  <=4.17.23
Severity: high
lodash vulnerable to Code Injection via `_.template` imports key names - https://github.com/advisories/GHSA-r5fr-rjxr-66jc
lodash vulnerable to Prototype Pollution via array path bypass in `_.unset` and `_.omit` - https://github.com/advisories/GHSA-f23m-r3pf-42rh
fix available via `npm audit fix`
node_modules/lodash-es
  @chevrotain/cst-dts-gen  11.0.0 - 11.2.0
  Depends on vulnerable versions of @chevrotain/gast
  Depends on vulnerable versions of lodash-es
  node_modules/@chevrotain/cst-dts-gen
  @chevrotain/gast  11.0.0 - 11.2.0
  Depends on vulnerable versions of lodash-es
  node_modules/@chevrotain/gast
  chevrotain  11.0.0 - 11.2.0
  Depends on vulnerable versions of @chevrotain/cst-dts-gen
  Depends on vulnerable versions of @chevrotain/gast
  Depends on vulnerable versions of lodash-es
  node_modules/chevrotain
    chevrotain-allstar  0.3.0 - 0.3.1
    Depends on vulnerable versions of chevrotain
    node_modules/chevrotain-allstar
    langium  2.0.0-next.239179f - 4.2.1
    Depends on vulnerable versions of chevrotain
    Depends on vulnerable versions of chevrotain-allstar
    node_modules/langium

minimatch  9.0.0 - 9.0.6
Severity: high
minimatch has a ReDoS via repeated wildcards with non-matching literal in pattern - https://github.com/advisories/GHSA-3ppc-4f35-3m26
minimatch has ReDoS: matchOne() combinatorial backtracking via multiple non-adjacent GLOBSTAR segments - https://github.com/advisories/GHSA-7r86-cg39-jmmj
minimatch ReDoS: nested *() extglobs generate catastrophically backtracking regular expressions - https://github.com/advisories/GHSA-23c5-xmqv-rm74
fix available via `npm audit fix`
node_modules/@typescript-eslint/typescript-estree/node_modules/minimatch
  @typescript-eslint/typescript-estree  6.16.0 - 7.5.0
  Depends on vulnerable versions of minimatch
  node_modules/@typescript-eslint/typescript-estree
    @typescript-eslint/parser  6.16.0 - 7.5.0
    Depends on vulnerable versions of @typescript-eslint/typescript-estree
    node_modules/@typescript-eslint/parser
    @typescript-eslint/type-utils  6.16.0 - 7.5.0
    Depends on vulnerable versions of @typescript-eslint/typescript-estree
    Depends on vulnerable versions of @typescript-eslint/utils
    node_modules/@typescript-eslint/type-utils
      @typescript-eslint/eslint-plugin  6.16.0 - 7.5.0
      Depends on vulnerable versions of @typescript-eslint/type-utils
      Depends on vulnerable versions of @typescript-eslint/utils
      node_modules/@typescript-eslint/eslint-plugin
    @typescript-eslint/utils  6.16.0 - 7.5.0
    Depends on vulnerable versions of @typescript-eslint/typescript-estree
    node_modules/@typescript-eslint/utils

path-to-regexp  <0.1.13
Severity: high
path-to-regexp vulnerable to Regular Expression Denial of Service via multiple route parameters - https://github.com/advisories/GHSA-37ch-88jc-xwx2
fix available via `npm audit fix`
node_modules/path-to-regexp

picomatch  <=2.3.1 || 4.0.0 - 4.0.3
Severity: high
Picomatch: Method Injection in POSIX Character Classes causes incorrect Glob Matching - https://github.com/advisories/GHSA-3v7f-55p6-f55p
Picomatch: Method Injection in POSIX Character Classes causes incorrect Glob Matching - https://github.com/advisories/GHSA-3v7f-55p6-f55p
Picomatch has a ReDoS vulnerability via extglob quantifiers - https://github.com/advisories/GHSA-c2c7-rcm5-vvqj
Picomatch has a ReDoS vulnerability via extglob quantifiers - https://github.com/advisories/GHSA-c2c7-rcm5-vvqj
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
  cacache  14.0.0 - 18.0.4
  Depends on vulnerable versions of tar
  node_modules/cacache

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

29 vulnerabilities (9 moderate, 20 high)

To address issues that do not require attention, run:
  npm audit fix

Some issues need review, and may require choosing
a different dependency.
No security issues found
```
