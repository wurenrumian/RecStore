# Manual Release Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a manual GitHub Actions release workflow that builds CPU release artifacts for `ps_server` and `recstore_ops`, verifies them minimally, and publishes them to GitHub Releases with a user-provided version.

**Architecture:** Keep release publishing isolated in a new workflow instead of mixing it into CI. Reuse the existing build and runtime validation flow, but harden `ci/pack/pack_artifact.sh` so release bundles do not ship host-specific loader and glibc libraries that break zero-config usage.

**Tech Stack:** GitHub Actions, Bash, Python `unittest`, existing RecStore CI packaging helpers

---

### Task 1: Lock packaging behavior with a regression test

**Files:**
- Create: `ci/pack/test_pack_artifact.py`
- Modify: `ci/pack/pack_artifact.sh`

- [ ] **Step 1: Write the failing test**

Add a focused Python `unittest` that invokes `ci/pack/pack_artifact.sh` against a fake ELF target with fake `lddtree` output. Assert that core host libraries like `libc.so.6` and `ld-linux-x86-64.so.2` are excluded, while a non-core dependency is still packaged.

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest ci.pack.test_pack_artifact -v`
Expected: FAIL because the current packer still copies host glibc / loader dependencies.

- [ ] **Step 3: Write minimal implementation**

Add dependency filtering to `ci/pack/pack_artifact.sh` based on basename patterns for host-provided runtime libraries, and record excluded entries in the manifest for debugging.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest ci.pack.test_pack_artifact -v`
Expected: PASS

### Task 2: Add a manual GitHub release workflow

**Files:**
- Create: `.github/workflows/release.yml`
- Modify: `.github/workflows/ci-build.yml`

- [ ] **Step 1: Add the workflow definition**

Create a dedicated `workflow_dispatch` workflow that accepts at least `version` and optional `release_name`.

- [ ] **Step 2: Reuse the CPU build pipeline**

Mirror the existing CPU build flow: checkout, submodules, host prereqs, docker env init, torch wheel build/install, CMake configure/build, and package artifacts.

- [ ] **Step 3: Add release naming and verification**

Rename packaged outputs to stable release asset names, run the existing package smoke path (`ps_server` + OP test), and fail before publishing if verification breaks.

- [ ] **Step 4: Publish the GitHub Release**

Use a release action with `contents: write` permission to create or update a Release for the requested version and upload the built assets.

- [ ] **Step 5: Run workflow syntax validation**

Run: `python3 - <<'PY' ...`
Expected: YAML parses cleanly and required keys exist.

### Task 3: Update operator naming and docs for downloaders

**Files:**
- Modify: `.github/workflows/ci-build.yml`
- Modify: `docs/dev/zero_config_run.md`

- [ ] **Step 1: Normalize the packaged ops filename**

Ensure workflow packaging resolves whichever output exists for the torch ops shared library and publishes it under a stable `recstore_ops` release asset name.

- [ ] **Step 2: Update zero-config guidance**

Document that release artifacts intentionally exclude host core runtime libraries and still require a compatible Linux userspace.

- [ ] **Step 3: Run targeted verification**

Run: `python3 -m unittest ci.pack.test_pack_artifact -v`
Expected: PASS
