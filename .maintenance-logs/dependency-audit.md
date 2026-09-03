# Dependency Audit Report - 2026-09-03 02:09:55 UTC

## Dependency Analysis Summary
```json
{
  "timestamp": "2026-09-03T02:09:53.936Z",
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
      "@remix-run/node": 13,
      "@remix-run/react": 14,
      "@remix-run/serve": 0,
      "@stackblitz/sdk": 2,
      "@supabase/supabase-js": 8,
      "@types/node": 0,
      "autoprefixer": 0,
      "framer-motion": 5,
      "hnswlib-node": 2,
      "isbot": 1,
      "mermaid": 1,
      "ml-distance": 1,
      "ml-matrix": 2,
      "monaco-editor": 6,
      "openai": 20,
      "prettier": 0,
      "python-shell": 1,
      "react": 50,
      "react-dom": 4,
      "react-icons": 23,
      "react-markdown": 2,
      "tailwindcss": 1,
      "xterm": 3,
      "xterm-addon-fit": 3,
      "xterm-addon-web-links": 3,
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
      "typescript": 9,
      "vite": 3,
      "vite-tsconfig-paths": 0
    },
    "securityVulnerabilities": {},
    "securitySummary": {}
  }
}```

## Security Audit
```
# npm audit report

@babel/core  <=7.29.0
@babel/core: Arbitrary File Read via sourceMappingURL Comment - https://github.com/advisories/GHSA-4x5r-pxfx-6jf8
fix available via `npm audit fix`
node_modules/@babel/core

@remix-run/router  1.3.0 - 1.23.2
Severity: moderate
React Router's same-origin redirect with path starting // causes open redirect via protocol-relative URL reinterpretation - https://github.com/advisories/GHSA-2j2x-hqr9-3h42
fix available via `npm audit fix`
node_modules/@remix-run/router
  @remix-run/dev  *
  Depends on vulnerable versions of @remix-run/node
  Depends on vulnerable versions of @remix-run/router
  Depends on vulnerable versions of @remix-run/server-runtime
  Depends on vulnerable versions of @vanilla-extract/integration
  Depends on vulnerable versions of cacache
  Depends on vulnerable versions of esbuild
  Depends on vulnerable versions of remark-mdx-frontmatter
  node_modules/@remix-run/dev
  @remix-run/react  <=0.0.0-nightly-ff40409-20230514 || >=1.11.0-pre.0
  Depends on vulnerable versions of @remix-run/router
  Depends on vulnerable versions of @remix-run/server-runtime
  Depends on vulnerable versions of react-router
  Depends on vulnerable versions of react-router-dom
  Depends on vulnerable versions of turbo-stream
  node_modules/@remix-run/react
  @remix-run/server-runtime  <=0.0.0-nightly-ff40409-20230514 || >=1.11.0-pre.0
  Depends on vulnerable versions of @remix-run/router
  Depends on vulnerable versions of turbo-stream
  node_modules/@remix-run/server-runtime
    @remix-run/node  2.10.0-pre.0 - 2.17.4
    Depends on vulnerable versions of @remix-run/server-runtime
    node_modules/@remix-run/node
      @remix-run/express  2.10.0-pre.0 - 2.17.4
      Depends on vulnerable versions of @remix-run/node
      node_modules/@remix-run/express
      @remix-run/serve  2.10.0-pre.0 - 2.17.4
      Depends on vulnerable versions of @remix-run/express
      Depends on vulnerable versions of @remix-run/node
      node_modules/@remix-run/serve
  react-router  6.0.0 - 7.17.0
  Depends on vulnerable versions of @remix-run/router
  node_modules/react-router
    react-router-dom  6.0.0-alpha.0 - 7.17.0
    Depends on vulnerable versions of @remix-run/router
    Depends on vulnerable versions of react-router
    node_modules/react-router-dom


@supabase/auth-js  <=2.69.1
auth-js Vulnerable to Insecure Path Routing from Malformed User Input - https://github.com/advisories/GHSA-8r88-6cj9-9fh5
fix available via `npm audit fix`
node_modules/@supabase/auth-js
  @supabase/supabase-js  2.41.1 - 2.49.10 || 2.58.1-canary.0
  Depends on vulnerable versions of @supabase/auth-js
  node_modules/@supabase/supabase-js

body-parser  <1.20.6
body-parser vulnerable to denial of service when invalid limit value silently disables size enforcement - https://github.com/advisories/GHSA-v422-hmwv-36x6
fix available via `npm audit fix`
node_modules/body-parser

brace-expansion  <=1.1.17 || 2.0.0 - 2.1.3
Severity: high
brace-expansion: Zero-step sequence causes process hang and memory exhaustion - https://github.com/advisories/GHSA-f886-m6hf-6m8v
brace-expansion: Zero-step sequence causes process hang and memory exhaustion - https://github.com/advisories/GHSA-f886-m6hf-6m8v
brace-expansion: DoS via exponential-time expansion of consecutive non-expanding {} groups - https://github.com/advisories/GHSA-3jxr-9vmj-r5cp
brace-expansion: DoS via exponential-time expansion of consecutive non-expanding {} groups - https://github.com/advisories/GHSA-3jxr-9vmj-r5cp
brace-expansion: DoS via unbounded expansion length causing an out-of-memory process crash - https://github.com/advisories/GHSA-mh99-v99m-4gvg
brace-expansion: DoS via unbounded expansion length causing an out-of-memory process crash - https://github.com/advisories/GHSA-mh99-v99m-4gvg
brace-expansion: DoS via unbounded intermediate arrays, bypassing the CVE-2026-14257 mitigation - https://github.com/advisories/GHSA-rgw5-rvv9-x895
brace-expansion: DoS via unbounded intermediate arrays, bypassing the CVE-2026-14257 mitigation - https://github.com/advisories/GHSA-rgw5-rvv9-x895
fix available via `npm audit fix`
node_modules/@eslint/eslintrc/node_modules/brace-expansion
node_modules/@humanwhocodes/config-array/node_modules/brace-expansion
node_modules/brace-expansion
node_modules/eslint-plugin-import/node_modules/brace-expansion
node_modules/eslint-plugin-jsx-a11y/node_modules/brace-expansion
node_modules/eslint-plugin-react/node_modules/brace-expansion
node_modules/eslint/node_modules/brace-expansion
node_modules/rimraf/node_modules/brace-expansion

browserslist  <=4.28.6
Severity: high
Browserslist: Unbounded memory growth (no cache eviction) via distinct query results, leading to eventual OOM - https://github.com/advisories/GHSA-c83g-rgw3-j3cx
Browserslist: Uncaught crash / prototype write via untrusted browserslist-stats.json custom stats (normalizeStats) - https://github.com/advisories/GHSA-73wf-gq98-2v4g
fix available via `npm audit fix`
node_modules/browserslist

dompurify  <=3.4.12
Severity: moderate
DOMPurify contains a Cross-site Scripting vulnerability - https://github.com/advisories/GHSA-v8jm-5vwx-cfxm
DOMPurify contains a Cross-site Scripting vulnerability - https://github.com/advisories/GHSA-v2wj-7wpq-c8vv
DOMPurify: FORBID_TAGS bypassed by function-based ADD_TAGS predicate (asymmetry with FORBID_ATTR fix) - https://github.com/advisories/GHSA-h7mw-gpvr-xq4m
DOMPurify has a SAFE_FOR_TEMPLATES bypass in RETURN_DOM mode - https://github.com/advisories/GHSA-crv5-9vww-q3g8
DOMPurify: Prototype Pollution to XSS Bypass via CUSTOM_ELEMENT_HANDLING Fallback - https://github.com/advisories/GHSA-v9jr-rg53-9pgp
DOMPurify: Cross-realm IN_PLACE sanitization leaves executable markup intact via realm-bound `instanceof` checks - https://github.com/advisories/GHSA-hpcv-96wg-7vj8
DOMPurify: IN_PLACE mode preserves attributes of a clobbered root element, allowing XSS via attacker-controlled root DOM - https://github.com/advisories/GHSA-r47g-fvhr-h676
DOMPurify IN_PLACE Sanitization Bypass via Attached Shadow Root Inside <template>.content - https://github.com/advisories/GHSA-rp9w-3fw7-7cwq
DOMPurify: `CUSTOM_ELEMENT_HANDLING` bypasses `afterSanitizeElements` for allowed custom elements. - https://github.com/advisories/GHSA-c2j3-45gr-mqc4
DOMPurify: Permanent `ALLOWED_ATTR` pollution via `setConfig()` bypassing the hook clone-guard (incomplete fix of the 3.4.7 hook-pollution patch) - https://github.com/advisories/GHSA-cmwh-pvxp-8882
DOMPurify: Trusted Types policy survives `clearConfig()` and can poison later `RETURN_TRUSTED_TYPE` output - https://github.com/advisories/GHSA-vxr8-fq34-vvx9
DOMPurify: SAFE_FOR_TEMPLATES bypass - template expressions survive sanitization inside <template> content when using DOM output modes - https://github.com/advisories/GHSA-gvmj-g25r-r7wr
DOMPurify: `IN_PLACE` mode trusts attacker-controlled `nodeName` on live non-form nodes, allowing script retention and XSS via attacker-supplied DOM objects - https://github.com/advisories/GHSA-x4vx-rjvf-j5p4
DOMPurify: Hook mutation of `data.allowedTags` / `data.allowedAttributes` permanently pollutes `DEFAULT_ALLOWED_TAGS` / `DEFAULT_ALLOWED_ATTR` - https://github.com/advisories/GHSA-76mc-f452-cxcm
DOMPurify's ADD_TAGS function form bypasses FORBID_TAGS due to short-circuit evaluation - https://github.com/advisories/GHSA-39q2-94rc-95cp
DOMPurify ADD_ATTR predicate skips URI validation - https://github.com/advisories/GHSA-cjmm-f4jc-qw8r
DOMPurify USE_PROFILES prototype pollution allows event handlers - https://github.com/advisories/GHSA-cj63-jhhr-wcxv
DOMPurify is vulnerable to mutation-XSS via Re-Contextualization  - https://github.com/advisories/GHSA-h8r8-wccr-v5f2
DOMPurify: IN_PLACE hook removal leaves a detached subtree executable, causing XSS - https://github.com/advisories/GHSA-55q2-fjhq-7xh7
fix available via `npm audit fix`
node_modules/dompurify

esbuild  <=0.24.2
Severity: moderate
esbuild enables any website to send any requests to the development server and read the response - https://github.com/advisories/GHSA-67mh-4wv8-2f99
fix available via `npm audit fix --force`
Will install vite@8.2.2, which is a breaking change
node_modules/esbuild
node_modules/vite/node_modules/esbuild
  @vanilla-extract/integration  *
  Depends on vulnerable versions of esbuild
  Depends on vulnerable versions of vite
  Depends on vulnerable versions of vite-node
  node_modules/@vanilla-extract/integration
  vite  <=6.4.2
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

js-yaml  4.0.0 - 4.3.0
Severity: high
JS-YAML: Quadratic-complexity DoS in merge key handling via repeated aliases - https://github.com/advisories/GHSA-h67p-54hq-rp68
js-yaml: YAML merge-key chains can force quadratic CPU consumption - https://github.com/advisories/GHSA-52cp-r559-cp3m
JS-YAML: Quadratic CPU consumption in !!omap resolution (3.x and 4.x) — CVE-2026-59870 fix not backported - https://github.com/advisories/GHSA-5p4m-2wfm-xmqj
fix available via `npm audit fix`
node_modules/js-yaml

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

mermaid  11.0.0-alpha.1 - 11.16.0
Severity: moderate
Mermaid: Improper sanitization of `classDef` in state diagrams leads to HTML injection - https://github.com/advisories/GHSA-ghcm-xqfw-q4vr
Mermaid: Improper sanitization of `classDefs` in diagrams leads to CSS injection - https://github.com/advisories/GHSA-xcj9-5m2h-648r
Mermaid Gantt Charts are vulnerable to an Infinite Loop DoS - https://github.com/advisories/GHSA-6m6c-36f7-fhxh
Mermaid: Improper sanitization of configuration leads to CSS injection - https://github.com/advisories/GHSA-87f9-hvmw-gh4p
Mermaid configuration APIs allow prototype pollution - https://github.com/advisories/GHSA-c4c3-pg64-4m4v
Mermaid allows CSS injection applying to sibling elements of the diagram - https://github.com/advisories/GHSA-6x64-9x62-f2gx
Mermaid Architecture diagrams are vulnerable to prototype pollution - https://github.com/advisories/GHSA-3rrr-jr9j-h3q3
Mermaid XY Charts are vulnerable to an infinite loop DoS - https://github.com/advisories/GHSA-2v8p-3f2j-5mp7
Mermaid radar diagrams are vulnerable to DoS - https://github.com/advisories/GHSA-rhh3-jpg6-66xh
fix available via `npm audit fix`
node_modules/mermaid

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

morgan  1.2.0 - 1.10.1
Severity: moderate
morgan vulnerable to Log Forging via unneutralized control characters in :remote-user - https://github.com/advisories/GHSA-4vj7-5mj6-jm8m
fix available via `npm audit fix`
node_modules/morgan

nanoid  <=3.3.17
Severity: high
nanoid: non-secure generators can loop indefinitely with negative size - https://github.com/advisories/GHSA-28wg-ghj8-5hjv
nanoid: custom generators can loop indefinitely when size is zero - https://github.com/advisories/GHSA-2v37-7h3g-55p8
nanoid: Integer Overflow or Wraparound - https://github.com/advisories/GHSA-xwg4-73v4-xw9w
fix available via `npm audit fix`
node_modules/nanoid

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

postcss  <=8.5.22
Severity: high
PostCSS has XSS via Unescaped </style> in its CSS Stringify Output - https://github.com/advisories/GHSA-qx2v-qp2m-jg93
PostCSS: Arbitrary file read and information disclosure via attacker-controlled sourceMappingURL in CSS comments - https://github.com/advisories/GHSA-6g55-p6wh-862q
PostCSS: incomplete fix of GHSA-6g55-p6wh-862q — attacker-controlled sourceMappingURL reads arbitrary .map files when `from` is unset - https://github.com/advisories/GHSA-fxqj-rqcc-2cmp
PostCSS: Path Traversal in Previous Source Map Auto-Loading (sourceMappingURL) leads to Arbitrary .map File Disclosure - https://github.com/advisories/GHSA-r28c-9q8g-f849
fix available via `npm audit fix`
node_modules/postcss

postcss-selector-parser  7.1.0 - 7.1.2
postcss-selector-parser allows denial of service through uncontrolled AST recursion - https://github.com/advisories/GHSA-w9m9-85wc-3x92
fix available via `npm audit fix`
node_modules/postcss-selector-parser

qs  2.2.5 - 6.15.3
Severity: moderate
qs has a remotely triggerable DoS: qs.stringify crashes with TypeError on null/undefined entries in comma-format arrays when encodeValuesOnly is set - https://github.com/advisories/GHSA-q8mj-m7cp-5q26
qs array-limit bypass via bracket-key comma parsing - https://github.com/advisories/GHSA-x5fp-wj9c-mxmx
qs: Denial of Service via Attacker Controlled isBuffer - https://github.com/advisories/GHSA-4mjr-xmp4-gh2g
fix available via `npm audit fix`
node_modules/qs



tar  <=7.5.20
Severity: critical
node-tar Vulnerable to Arbitrary File Creation/Overwrite via Hardlink Path Traversal - https://github.com/advisories/GHSA-34x7-hfp2-rc4v
node-tar is Vulnerable to Arbitrary File Overwrite and Symlink Poisoning via Insufficient Path Sanitization - https://github.com/advisories/GHSA-8qq5-rm4j-mr97
Arbitrary File Read/Write via Hardlink Target Escape Through Symlink Chain in node-tar Extraction - https://github.com/advisories/GHSA-83g3-92jg-28cx
tar has Hardlink Path Traversal via Drive-Relative Linkpath - https://github.com/advisories/GHSA-qffp-2rhf-9h96
node-tar Symlink Path Traversal via Drive-Relative Linkpath - https://github.com/advisories/GHSA-9ppj-qmqm-q256
Race Condition in node-tar Path Reservations via Unicode Ligature Collisions on macOS APFS - https://github.com/advisories/GHSA-r6q2-hw4h-h46w
node-tar applies PAX size override to intermediary GNU long-name/long-link headers, causing tar parser interpretation differential (file smuggling) - https://github.com/advisories/GHSA-vmf3-w455-68vh
node-tar: Process crash via PAX numeric path type confusion - https://github.com/advisories/GHSA-w8wr-v893-vjvp
node-tar: Decompression/parse DoS via unlimited input - https://github.com/advisories/GHSA-23hp-3jrh-7fpw
node-tar: Negative tar entry size causes infinite loop in archive replace - https://github.com/advisories/GHSA-8x88-c5mf-7j5w
node-tar: Uncaught Exception DoS via NUL byte in PAX path/linkpath records - https://github.com/advisories/GHSA-gvwx-54wh-qm9j
node-tar: Uncontrolled recursion in mapHas/filesFilter allows uncatchable stack-overflow DoS via crafted long-path tar with member selection - https://github.com/advisories/GHSA-r292-9mhp-454m
fix available via `npm audit fix`
node_modules/@tailwindcss/oxide/node_modules/tar
node_modules/@tailwindcss/postcss/node_modules/tar
node_modules/tar
  cacache  14.0.0 - 18.0.4
  Depends on vulnerable versions of tar
  node_modules/cacache

turbo-stream  <3.0.0
Severity: high
React Router vulnerable to Denial of Service via reflected user input in single-fetch - https://github.com/advisories/GHSA-rxv8-25v2-qmq8
fix available via `npm audit fix`
node_modules/turbo-stream

undici  <=6.27.0
Severity: high
Undici: Malicious WebSocket 64-bit length overflows parser and crashes the client - https://github.com/advisories/GHSA-f269-vfmq-vjvj
Undici has an HTTP Request/Response Smuggling issue - https://github.com/advisories/GHSA-2mjp-6q6p-2qxm
Undici has Unbounded Memory Consumption in WebSocket permessage-deflate Decompression - https://github.com/advisories/GHSA-vrm6-8vpv-qv8q
Undici has Unhandled Exception in WebSocket Client Due to Invalid server_max_window_bits Validation - https://github.com/advisories/GHSA-v9p9-hfj2-hcw8
Undici has CRLF Injection in undici via `upgrade` option - https://github.com/advisories/GHSA-4992-7rv2-5pvq
undici vulnerable to HTTP header injection via Set-Cookie percent-decoding - https://github.com/advisories/GHSA-p88m-4jfj-68fv
undici WebSocket client vulnerable to denial of service via fragment count bypass - https://github.com/advisories/GHSA-vxpw-j846-p89q
undici vulnerable to Set-Cookie SameSite attribute downgrade via permissive substring matching - https://github.com/advisories/GHSA-g8m3-5g58-fq7m
undici vulnerable to downstream response desynchronization via retry interceptor - https://github.com/advisories/GHSA-8xcm-r25x-g524
undici vulnerable to CRLF Injection via blob-like body 'type' property - https://github.com/advisories/GHSA-m8rv-5g2x-5cg5
undici vulnerable to cookie attribute injection via unsanitized domain and unparsed setCookie fields - https://github.com/advisories/GHSA-v3r7-h72x-cjcm
undici vulnerable to HTTP response queue poisoning via keep-alive socket reuse - https://github.com/advisories/GHSA-35p6-xmwp-9g52
fix available via `npm audit fix`
node_modules/undici

uuid  <11.1.1
Severity: moderate
uuid: Missing buffer bounds check in v3/v5/v6 when buf is provided - https://github.com/advisories/GHSA-w5hq-g745-h8pq
fix available via `npm audit fix`
node_modules/uuid

valibot  <=1.4.1
Severity: moderate
Valibot: record() issue paths can make flatten() throw for inherited Object property names - https://github.com/advisories/GHSA-5qjj-4xww-7phc
fix available via `npm audit fix`
node_modules/valibot


ws  7.0.0 - 7.5.10 || 8.0.0 - 8.20.1
Severity: high
ws: Uninitialized memory disclosure - https://github.com/advisories/GHSA-58qx-3vcg-4xpx
ws: Memory exhaustion DoS from tiny fragments and data chunks - https://github.com/advisories/GHSA-96hv-2xvq-fx4p
ws: Memory exhaustion DoS from tiny fragments and data chunks - https://github.com/advisories/GHSA-96hv-2xvq-fx4p
fix available via `npm audit fix`
node_modules/@remix-run/dev/node_modules/ws
node_modules/ws

yaml  2.0.0 - 2.8.2
Severity: moderate
yaml is vulnerable to Stack Overflow via deeply nested YAML collections - https://github.com/advisories/GHSA-48c2-rrv3-qjmp
fix available via `npm audit fix`
node_modules/yaml

53 vulnerabilities (5 low, 15 moderate, 32 high, 1 critical)

To address issues that do not require attention, run:
  npm audit fix

To address all issues (including breaking changes), run:
  npm audit fix --force
No security issues found
```
