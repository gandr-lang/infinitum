# AGENTS.md

Read this file, then `docs/agents/baseline.md`, then every row below whose trigger fires. The trigger column **binds**: a matching row is read before acting, never after.

Cascade: every `AGENTS.md` on the path applies simultaneously, nothing replaced wholesale; the deepest nearest parent wins on overlap or conflict.

| About to | Read | Never |
| -------- | ---- | ----- |
| write or emit anything | `docs/agents/baseline.md` | a register, reference, or tool route the page rules out |
| write, change, or review Rust here | `docs/agents/rust.md` | a partial function, recursion, or an undocumented item |
| write a test, or credit a specification as covered | `docs/agents/testing-contracts.md` | coverage cited where adequacy is the metric |
| build, format, gate, test, or commit | `docs/agents/source-workflow.md` | a raw tool invocation where a task owns the gate |
| change hosted CI, its pins, or its shared pattern | `docs/agents/ci-local.md` | hosted-run trial and error, or a bare-runner-only shape |

## workflow-tooling

No shell sources and no Python sources enter this tree. Bespoke tooling is a Rust binary in the workspace; a mise task is a launch line for one, never a program in its own right. Python is admitted later for model integration alone, run through `uv`, and it arrives with the work that needs it rather than ahead of it.

## vendored-components

A component this engine needs from another engine is vendored at build or setup time, or rewritten here. It is never stored in this tree: the licensing of a copied component travels with the copy, and this repository is published.

## vendored-page-bindings

`docs/agents/rust.md`, `testing-contracts.md`, `source-workflow.md`, and `ci-local.md` are byte-identical copies of the shared shape, taken verbatim and never edited here. `docs/agents/baseline.md` and its pin `baseline.sha256` are one coupled artifact, checked by `mise run check:baseline-hash`. Reconcile a source change upstream and adopt the resulting artifacts together; a copied page is never patched to conceal drift.

Each row below binds a shared page's sites to this tree. A site a row calls a source example still carries its rule; a binding governs how the shared page applies here rather than relaxing it.

| Shared site | Binding here | Reversal |
| ----------- | ------------ | -------- |
| `rust.md` crate shape, the `<category>-<name>` directory schema and its package prefix | A workspace whose members are `crates/*`. One member exists: `crates/cli`, the published-name holder, so its package is the reserved name `infinitum` rather than a derivation of its directory — the page's own exception for that case, as the source repository takes it for its driver. The category axis stays unspent while it would carry one value; the second member opens the schema with its first category. | A second member lands. |
| `rust.md` toolchain and `mise run toolchain:bump` | The channel stays the shared nightly: the vendored `rustfmt.toml` selects options that only nightly rustfmt reads, so a stable pin would silently drop the formatting profile the gate enforces. Components are `clippy` and `rustfmt` alone, the page's own allowance for a workspace running no Dylint. No bump task is adopted: with no `clippy_utils` tag to move in step, a channel bump is one line in `rust-toolchain.toml`. | A Dylint policy or a second pin couples to the channel. |
| `rust.md` publication | `publish = false`, as the page requires of every crate including a published-name holder. Flipping it is the manual publish act, taken by the owner, never by CI or an agent. | The page's posture is revisited. |
| `rust.md` optional specification instrumentation | `quenchant` supplies `#[quenchant::spec(...)]`. The consumer-side `anodized` feature is declared here and off by default, so the default build removes the attribute and its supported nested markers and ordinary code remains. | A verification lane needs the enforcing build by default. |
| `source-workflow.md` gates table | The wall here is `mise run check` (conflict markers, shared baseline, private-item rustdoc), `mise run treefmt:check`, `cargo clippy --workspace --all-targets -- --deny warnings`, and `cargo nextest run --profile ci --workspace`. The external-Dylint, contracts, witness, and publication-surface gates belong to a workspace that consumes that policy; this tree consumes none of it yet, so those tasks are absent rather than stubbed. | This tree adopts the external workflow policy. |
| `source-workflow.md` tooling posture and the shell-lint formatter entry | No shell or Python sources exist here, so the `shellcheck` pin and its `treefmt.toml` entry are both absent — a formatter entry and its pin move together, and treefmt refuses to start when a configured command is missing. | A shell source is admitted, which the workflow-tooling rule above refuses. |
| `source-workflow.md` project changelog | Adopted: `cliff.toml` generates the root `CHANGELOG.md` through three `treefmt` formatters — generate, normalize with the Markdown formatter, install only changed bytes — so the existing `treefmt:check` wall owns changelog currency and no separate gate exists. Both format tasks run with `treefmt`'s content cache off, because commit history changes without any watched file changing. Until the first `v<version>` tag, every entry sits under Unreleased. | The tag namespace or the generator changes. |
| `source-workflow.md` commits and hooks | `prek.toml` carries the shared floor plus the shared-baseline check; `commitlint.config.mjs` carries the type and scope vocabulary and the three rules that check an agent commit's provenance block — the role line, the opaque session line, and the owner co-author line. The block is typed by the author here rather than appended by a tracked hook, because tracked configuration cannot carry the mechanism that would install it. | The commit rules move. |
| `ci-local.md` container image | Adopted whole: `.github/workflows/ci-image.yml` publishes `ghcr.io/gandr-lang/infinitum-ci` under the pin-file tag, and `INFINITUM_CI_IMAGE` — a repository variable — relays the switch through `ci-image-ref` into each Rust job's `container`. Empty is the bootstrap state until the first image build is green and its measurement is recorded. | A measured rollback empties the switch permanently. |
| `ci-local.md` path filter and the `scripts/ci` examples | The `changes` job is the Git/JQ pipeline, not a script: a failed fetch, a failed diff, or a merge-group entry forces the Rust category on. The page's `scripts/ci/*.sh` residents are the source repository's own; none is adopted here. | A filter needs logic a pipeline cannot express, which is then a Rust binary. |
| `ci-local.md` pin-drift and private-boundary gates | Neither is adopted: each is a program, and this tree has no gate binary to hold it. `CARGO_NEXTEST_VERSION` in the workflow and the nextest pin in `mise.toml` are therefore checked by review until one exists. | A gate binary opens here. |
| `ci-local.md` platform lanes | The three cross-OS nextest lanes are kept: platform independence is this engine's thesis, and a host assumption in the driver surfaces only on the host that exposes it. The big-endian and Miri lanes are absent — there is no wire format and no unsafe code to interpret. | Either condition changes. |

Revisit a binding when its named package, task, artifact, or owner changes. A changed interpretation and its callers land together.
