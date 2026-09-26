# infinitum

The infinitum inference engine.

## Layout

- `crates/cli` — the `infinitum` crate: the engine driver, installs the `infinitum` binary.
- `AGENTS.md` and `docs/agents/` — guidance for coding agents working in this repository.

## Install

```sh
cargo install infinitum
```

## Compiler policy

`mise run check:dylint` loads the quenchant plugin at the Git revision in `Cargo.toml` and checks every workspace target. It is also part of `mise run check`. The nightly in `rust-toolchain.toml` must match that revision; the CI image bakes its compiler driver and changes tag when the pin files change.

Authored `unsafe` functions, traits, implementations, and extern blocks require a `# Safety` rustdoc section with a nonempty `- unsafe invariants:` bullet. For an `unsafe extern "C++"` block inside `#[cxx::bridge]`, put that section on the bridge module, because the attribute macro consumes the inner block before ordinary Rust lint passes see it.

## License

Apache-2.0 WITH LLVM-exception.
