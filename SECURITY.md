# Security Policy

## Supported versions

Security fixes are applied to the latest released version and the `main`
branch.

## Reporting a vulnerability

Please report security vulnerabilities privately via
[GitHub security advisories](https://github.com/enioarz/polars-gdx/security/advisories/new)
so we can assess and fix the issue before public disclosure.

Please include a description of the issue, steps to reproduce it, and its
impact. We aim to respond within a few days.

Do **not** open a public GitHub issue for security problems.

## Scope notes

polars-gdx reads data files that may come from untrusted sources. Parsing
malformed or adversarial GDX files is in scope for security reports, as is the
Rust FFI layer that wraps the vendored GDX C library.
