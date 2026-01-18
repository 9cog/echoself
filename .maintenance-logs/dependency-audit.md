# Dependency Audit Report - 2026-01-18 02:44:25 UTC

## Dependency Analysis Summary

````json
{
  "timestamp": "2026-01-18T02:44:22.998Z",
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
      "eslint": 8,
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
````

# npm audit report

@remix-run/node <=2.17.2
Severity: critical
Depends on vulnerable versions of @remix-run/server-runtime
React Router has Path Traversal in File Session Storage - https://github.com/advisories/GHSA-9583-h5hc-x8cw
fix available via `npm audit fix`
node_modules/@remix-run/dev/node_modules/@remix-run/node
node_modules/@remix-run/node
@remix-run/dev \*
Depends on vulnerable versions of @remix-run/node
Depends on vulnerable versions of @remix-run/router
Depends on vulnerable versions of @remix-run/server-runtime
Depends on vulnerable versions of @vanilla-extract/integration
Depends on vulnerable versions of cacache
Depends on vulnerable versions of esbuild
Depends on vulnerable versions of remark-mdx-frontmatter
Depends on vulnerable versions of valibot
node_modules/@remix-run/dev
@remix-run/express <=2.17.1
Depends on vulnerable versions of @remix-run/node
node_modules/@remix-run/express
@remix-run/serve <=2.17.1
Depends on vulnerable versions of @remix-run/express
Depends on vulnerable versions of @remix-run/node
node_modules/@remix-run/serve

@remix-run/react <=2.17.3
Severity: high
Depends on vulnerable versions of @remix-run/router
Depends on vulnerable versions of @remix-run/server-runtime
React Router SSR XSS in ScrollRestoration - https://github.com/advisories/GHSA-8v8x-cx79-35w7
React Router has XSS Vulnerability - https://github.com/advisories/GHSA-3cgp-3xvw-98x8
Depends on vulnerable versions of react-router
Depends on vulnerable versions of react-router-dom
fix available via `npm audit fix`
node_modules/@remix-run/react

@remix-run/router <=1.23.1
Severity: high
React Router vulnerable to XSS via Open Redirects - https://github.com/advisories/GHSA-2w69-qvjg-hvjx
fix available via `npm audit fix`
node_modules/@remix-run/router
@remix-run/server-runtime <=2.17.3
Depends on vulnerable versions of @remix-run/router
node_modules/@remix-run/dev/node_modules/@remix-run/server-runtime
node_modules/@remix-run/server-runtime
react-router 6.0.0 - 6.30.2
Depends on vulnerable versions of @remix-run/router
node_modules/react-router
react-router-dom 6.0.0-alpha.0 - 6.30.2
Depends on vulnerable versions of @remix-run/router
Depends on vulnerable versions of react-router
node_modules/react-router-dom

diff <8.0.3
jsdiff has a Denial of Service vulnerability in parsePatch and applyPatch - https://github.com/advisories/GHSA-73rr-hh4g-fpgx
fix available via `npm audit fix`
node_modules/diff
uvu >=0.0.4
Depends on vulnerable versions of diff
node_modules/uvu
mdast-util-from-markdown 1.0.4 - 1.3.1
Depends on vulnerable versions of uvu
node_modules/mdast-util-from-markdown
mdast-util-mdx-jsx 2.1.1 - 2.1.4
Depends on vulnerable versions of mdast-util-from-markdown
node_modules/mdast-util-mdx-jsx
micromark 3.0.7 - 3.2.0
Depends on vulnerable versions of uvu
node_modules/micromark
micromark-core-commonmark 1.0.3 - 1.1.0
Depends on vulnerable versions of uvu
node_modules/micromark-core-commonmark
micromark-extension-mdx-expression 1.0.2 - 1.0.8
Depends on vulnerable versions of uvu
node_modules/micromark-extension-mdx-expression
micromark-extension-mdx-jsx 1.0.2 - 1.0.5
Depends on vulnerable versions of uvu
node_modules/micromark-extension-mdx-jsx
micromark-extension-mdxjs-esm 1.0.2 - 1.0.5
Depends on vulnerable versions of uvu
node_modules/micromark-extension-mdxjs-esm
micromark-factory-label 1.0.2 - 1.1.0
Depends on vulnerable versions of uvu
node_modules/micromark-factory-label
micromark-factory-mdx-expression 1.0.4 - 1.0.9
Depends on vulnerable versions of uvu
node_modules/micromark-factory-mdx-expression
micromark-util-events-to-acorn 1.0.4 - 1.2.3
Depends on vulnerable versions of uvu
node_modules/micromark-util-events-to-acorn
micromark-util-subtokenize 1.0.2 - 1.1.0
Depends on vulnerable versions of uvu
node_modules/micromark-util-subtokenize

esbuild <=0.24.2
Severity: moderate
esbuild enables any website to send any requests to the development server and read the response - https://github.com/advisories/GHSA-67mh-4wv8-2f99
fix available via `npm audit fix --force`
Will install vite@6.4.1, which is a breaking change
node_modules/esbuild
node_modules/vite/node_modules/esbuild
@vanilla-extract/integration \*
Depends on vulnerable versions of esbuild
Depends on vulnerable versions of vite
Depends on vulnerable versions of vite-node
node_modules/@vanilla-extract/integration
vite 0.11.0 - 6.1.6
Depends on vulnerable versions of esbuild
node_modules/vite
vite-node <=2.2.0-beta.2
Depends on vulnerable versions of vite
node_modules/@vanilla-extract/integration/node_modules/vite-node

estree-util-value-to-estree <3.3.3
Severity: moderate
estree-util-value-to-estree allows prototype pollution in generated ESTree - https://github.com/advisories/GHSA-f7f6-9jq7-3rqj
fix available via `npm audit fix`
node_modules/estree-util-value-to-estree
remark-mdx-frontmatter <=2.1.1
Depends on vulnerable versions of estree-util-value-to-estree
node_modules/remark-mdx-frontmatter

qs <6.14.1
Severity: high
qs's arrayLimit bypass in its bracket notation allows DoS via memory exhaustion - https://github.com/advisories/GHSA-6rw7-vpxm-498p
fix available via `npm audit fix`
node_modules/qs
body-parser <=1.20.3 || 2.0.0-beta.1 - 2.0.2
Depends on vulnerable versions of qs
node_modules/body-parser
express 2.5.8 - 2.5.11 || 3.2.1 - 3.2.3 || 4.0.0-rc1 - 4.21.2 || 5.0.0-alpha.1 - 5.0.1
Depends on vulnerable versions of body-parser
Depends on vulnerable versions of qs
node_modules/express

tar <=7.5.2
Severity: high
node-tar is Vulnerable to Arbitrary File Overwrite and Symlink Poisoning via Insufficient Path Sanitization - https://github.com/advisories/GHSA-8qq5-rm4j-mr97
fix available via `npm audit fix`
node_modules/@tailwindcss/oxide/node_modules/tar
node_modules/@tailwindcss/postcss/node_modules/tar
node_modules/tar
cacache 14.0.0 - 18.0.4
Depends on vulnerable versions of tar
node_modules/cacache

undici <6.23.0
Undici has an unbounded decompression chain in HTTP responses on Node.js Fetch API via Content-Encoding leads to resource exhaustion - https://github.com/advisories/GHSA-g9mf-h72j-4rw9
fix available via `npm audit fix`
node_modules/undici

valibot 0.31.0 - 1.1.0
Severity: high
Valibot has a ReDoS vulnerability in `EMOJI_REGEX` - https://github.com/advisories/GHSA-vqpr-j7v3-hqw9
fix available via `npm audit fix`
node_modules/valibot

35 vulnerabilities (14 low, 6 moderate, 12 high, 3 critical)

To address issues that do not require attention, run:
npm audit fix

To address all issues (including breaking changes), run:
npm audit fix --force
No security issues found

```

```
