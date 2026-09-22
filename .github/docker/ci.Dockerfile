# The Linux CI image carries the pinned toolchain and the CI subset of mise
# tools. Pin files are copied, never retyped. The base digest is a rebuild
# trigger, not a tag input: rotating it republishes the same pin-file tag, so
# the workflow switch needs no independent update.
ARG UBUNTU_BASE=docker.io/library/ubuntu:24.04@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517
FROM ${UBUNTU_BASE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# rustup needs curl and certificates; Rust linking needs build-essential.
# Cargo Git sources need git.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Build tools as UID 1000. Job steps run as root to write runner-owned mounts.
RUN userdel --remove ubuntu && useradd --create-home --home-dir /opt/ci --uid 1000 ci
USER 1000

ENV RUSTUP_HOME=/opt/ci/.rustup \
    CARGO_HOME=/opt/ci/.cargo \
    PATH=/opt/ci/.local/bin:/opt/ci/.cargo/bin:$PATH

# This must match ci.yml: local-only tools never enter the image or CI installs.
ENV MISE_DISABLE_TOOLS=cargo:silvanshade/aifix,cargo:tirith,github:colbymchenry/codegraph,github:max-sixty/worktrunk,github:nektos/act,github:j178/prek,npm:@commitlint/cli,weave-mcp \
    MISE_CARGO_BINSTALL_QUICKINSTALL=true \
    MISE_CARGO_BINSTALL_NATIVE=true

# mise may rewrite its lockfile while installing, so the build user owns pins.
# The version-2 lockfile is two parts: mise.lock and the per-npm-tool sidecars
# under .mise/locks, which the npm backend reads to install from lock.
COPY --chown=1000:1000 rust-toolchain.toml mise.toml mise.lock .github/workflows/ci.yml /opt/pins/
COPY --chown=1000:1000 .mise/locks /opt/pins/.mise/locks
WORKDIR /opt/pins

# The pin supplies channel, profile and components.
RUN set -eux; \
    curl -fsSL https://sh.rustup.rs -o /tmp/rustup-init.sh; \
    sh /tmp/rustup-init.sh -y --no-modify-path --default-toolchain none; \
    rm /tmp/rustup-init.sh; \
    rustup toolchain install; \
    rustup default "$(sed -n 's/^channel = "\(.*\)"/\1/p' rust-toolchain.toml)"

# ci.yml selects mise's version. A BuildKit secret authenticates release
# attestations without persisting a credential in any layer.
RUN --mount=type=secret,id=github_token,uid=1000,required=false set -eux; \
    if [ -s /run/secrets/github_token ]; then GITHUB_TOKEN="$(cat /run/secrets/github_token)"; export GITHUB_TOKEN; fi; \
    expected="$(sed -n 's/^  MISE_DISABLE_TOOLS: "\(.*\)"$/\1/p' /opt/pins/ci.yml)"; \
    [ "$MISE_DISABLE_TOOLS" = "$expected" ] || { echo "MISE_DISABLE_TOOLS drifts from ci.yml: image='$MISE_DISABLE_TOOLS' ci='$expected'" >&2; exit 1; }; \
    MISE_VERSION="$(grep -A2 'uses: jdx/mise-action' /opt/pins/ci.yml | sed -n 's/^ *version: *//p' | head -n1 | tr -d '"')"; \
    [ -n "$MISE_VERSION" ]; \
    case "$(uname -m)" in \
        x86_64) ARCH=x64 ;; \
        aarch64) ARCH=arm64 ;; \
        *) echo "unsupported architecture $(uname -m)" >&2; exit 1 ;; \
    esac; \
    mkdir -p /opt/ci/.local/bin; \
    curl -fsSL "https://github.com/jdx/mise/releases/download/v${MISE_VERSION}/mise-v${MISE_VERSION}-linux-${ARCH}" -o /opt/ci/.local/bin/mise; \
    chmod +x /opt/ci/.local/bin/mise; \
    mise install

# Cargo resolves its subcommands by name; expose pinned binaries without shims.
RUN ln -sf "$(mise which cargo-nextest)" /opt/ci/.cargo/bin/cargo-nextest

# act invokes JavaScript actions through PATH rather than the hosted runner's
# bundled Node. Expose the pinned interpreter.
RUN ln -sf "$(mise which node)" /opt/ci/.local/bin/node

# The runner overrides HOME. Explicit homes retain the image's populated
# tools, while root can write the workspace and runner file-command mounts.
ENV MISE_DATA_DIR=/opt/ci/.local/share/mise \
    MISE_CACHE_DIR=/opt/ci/.cache/mise
USER root
WORKDIR /opt/ci
