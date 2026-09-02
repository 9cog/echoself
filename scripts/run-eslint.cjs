#!/usr/bin/env node
// deno-lint-ignore-file no-node-globals
// CommonJS script: without the directive above, `deno lint --fix` (run by the
// quality workflow) inserts an ESM `import process` here and Node can no longer
// load the file, which makes `npm run lint` fail before ESLint even starts.
"use strict";

/**
 * Cross-platform ESLint runner. package.json cannot use Unix
 * `ESLINT_USE_FLAT_CONFIG=false cmd` syntax on Windows cmd.exe.
 */
process.env.ESLINT_USE_FLAT_CONFIG = "false";

const { spawnSync } = require("node:child_process");
const path = require("node:path");

const eslintBin = path.join(
  __dirname,
  "..",
  "node_modules",
  "eslint",
  "bin",
  "eslint.js"
);

const args = [
  "--ignore-path",
  ".gitignore",
  "--cache",
  "--cache-location",
  "./node_modules/.cache/eslint",
  ".",
  ...process.argv.slice(2),
];

const result = spawnSync(process.execPath, [eslintBin, ...args], {
  stdio: "inherit",
  env: process.env,
});

process.exit(result.status ?? 1);
