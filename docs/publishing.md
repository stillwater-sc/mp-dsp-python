# Publishing `mpdsp` to PyPI

Assessment and step-by-step guide for getting the first (and subsequent)
releases of this package onto the Python Package Index.

## TL;DR

Most of the wheel-build and publishing infrastructure **is already in
the repo**. What's missing is a small one-time PyPI-side setup, a
reproducibility fix to the peer-dependency pins, and a dry run on
TestPyPI. After that, publishing a release is `git tag vX.Y.Z && git
push --tags`.

## What's already in place

| Asset | Where | What it does |
|-------|-------|--------------|
| `.github/workflows/publish.yml` | repo | Builds sdist + cibuildwheel matrix across Linux / Windows / macOS; supports TestPyPI and PyPI targets; OIDC trusted publishing (no API tokens) |
| `.github/workflows/release.yml` | repo | Tag-triggered; runs tests; creates GitHub release with auto-generated notes |
| `.github/workflows/regression.yml` | repo | Nightly cron catches dependency breakage |
| `pyproject.toml` | repo | Valid PEP 621 metadata, classifiers, optional extras (`plot`, `notebook`, `dashboard`, `dev`, `all`) |
| `README.md` | repo | 397 lines — renders cleanly as the PyPI landing page |
| `LICENSE` | repo | MIT |
| Version sourced from `CMakeLists.txt` + runtime `__dsp_version__` | PR #35 | Single source of truth; lockstep with `mixed-precision-dsp` |
| PyPI package name `mpdsp` | — | **Available** (as of the last check) |

## What needs to happen before the first release

### 1. PyPI platform setup (one-time, manual)

Nothing in the repo can publish until this is done:

1. **Log in to the PyPI account** that will own the `mpdsp` project. If
   the `stillwater-sc` org doesn't have a PyPI account yet, create one.
2. **Register `mpdsp` as a "pending publisher"** on PyPI
   (Account → *Publishing* → *Add a new pending publisher*):
   - **PyPI Project Name**: `mpdsp`
   - **Owner**: `stillwater-sc`
   - **Repository name**: `mp-dsp-python`
   - **Workflow name**: `publish.yml`
   - **Environment name**: `pypi`
3. **Do the same on TestPyPI** (<https://test.pypi.org>) with
   Environment name `testpypi`. TestPyPI is the staging registry;
   publishing there first lets us catch packaging problems without
   burning a release number on real PyPI.
4. **Create two GitHub Environments** in the repo's *Settings →
   Environments*: one named `pypi`, one named `testpypi`. Both
   workflows already reference these names. Optionally add *required
   reviewers* on the `pypi` environment so real PyPI pushes need manual
   approval.

**Why trusted publishing matters**: OIDC means the workflow
authenticates via GitHub's short-lived identity token, not a long-lived
API token stored as a repository secret. Nothing secret lives in the
repo.

### 2. Reproducibility: pin the peer-dependency `GIT_TAG`s

Current `CMakeLists.txt` has three lines like:

```cmake
FetchContent_Declare(dsp
    GIT_REPOSITORY https://github.com/stillwater-sc/mixed-precision-dsp.git
    GIT_TAG main
    GIT_SHALLOW TRUE)
```

`main` is a moving target. For a published wheel this is untenable —
the same `mpdsp` version could bind against different upstream
snapshots depending on when the wheel was built. Fix:

- `dsp` → `v0.4.1` (matches the lockstep version on our side)
- `universal` → a specific release tag
- `mtl5` → `v5.2.0`

Keep it parameterized (one CMake variable per peer) so bumping the peer
version at release time is a one-line change.

### 3. Review the README for PyPI rendering

PyPI renders `README.md` as Markdown on the project page. Most things
work fine, but:

- Internal links like `[docs](./docs/thing.md)` render as broken on
  PyPI — either absolutize them to the GitHub URL or remove.
- GitHub-flavored alerts (`> [!NOTE]`) don't render on PyPI — plain
  blockquotes do.
- Images need absolute URLs if they're in the README.

A quick pre-release review of the 397-line README is enough.

### 4. Decide: supported Python versions

Currently `cp39..cp312` in `publish.yml` via
`CIBW_BUILD: "cp39-* cp310-* cp311-* cp312-*"`. Python 3.13 is
released; adding `cp313-*` broadens the audience with no code risk
(cibuildwheel handles it).

Also: current macOS matrix is `arm64` only. Intel Mac users would need
an `x86_64` row (`runs-on: macos-13`) added to the matrix — optional,
depending on whether any users have Intel Macs.

### 5. Dry run to TestPyPI

Before cutting the real release:

1. Merge PR #35 (version sync) to `main`.
2. Merge the reproducibility-pin PR.
3. Trigger `publish.yml` manually via `workflow_dispatch`, target
   `testpypi`:
   ```bash
   gh workflow run publish.yml -f target=testpypi --ref main
   # Capture the run ID from the dispatch rather than relying on "most
   # recent" — another run could race in between these commands.
   RUN_ID=$(gh run list --workflow=publish.yml --limit 1 \
       --json databaseId --jq '.[0].databaseId')
   gh run watch "$RUN_ID"  # block until the matrix finishes (~20–40 min)
   ```
4. In a clean virtualenv:
   ```bash
   pip install -i https://test.pypi.org/simple/ \
               --extra-index-url https://pypi.org/simple/ \
               mpdsp
   python -c "import mpdsp; print(mpdsp.__version__, mpdsp.__dsp_version__)"
   ```
   Both should print `0.4.1`.
5. Visit <https://test.pypi.org/project/mpdsp/> and verify the README
   renders correctly and the metadata looks right.

If something's wrong, fix, bump to `0.4.1.post1` (or similar), and
retry. TestPyPI allows unlimited retries with post-release suffixes.

### 6. Cut the real release

There are **two paths** here — a plain semver tag (`vX.Y.Z`) and a
post-release tag (`vX.Y.Z.postN`) — and they need different follow-up
steps because `release.yml` only fires for one of them. See the table
in "Ongoing release cadence" below for the quick version; the reasoning
follows here.

#### Path A — plain semver (`vX.Y.Z`)

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

The tag triggers `release.yml`:

1. Tests run on Linux / Windows / macOS.
2. GitHub Release is created with auto-generated notes via
   `softprops/action-gh-release` running under `GITHUB_TOKEN`.

That release **does not auto-trigger `publish.yml`.** GitHub
deliberately suppresses workflow chaining when events originate from
`GITHUB_TOKEN` — a loop-prevention safeguard. So the `release:
published` event fires, but the listener in `publish.yml` skips it.

Dispatch `publish.yml` manually after the release is cut:

```bash
gh workflow run publish.yml -f target=pypi --ref main
RUN_ID=$(gh run list --workflow=publish.yml --limit 1 \
    --json databaseId --jq '.[0].databaseId')
gh run watch "$RUN_ID"
```

That runs the sdist + cibuildwheel matrix and uploads to real PyPI via
the OIDC trusted publisher. Users can then:

```bash
pip install mpdsp
```

**Fixing the auto-chain would require a PAT** (swap `GITHUB_TOKEN` for a
personal access token in `release.yml`'s `create-release` step — events
from a PAT do chain). We chose the manual dispatch instead to avoid
managing another secret.

#### Path B — post-release (`vX.Y.Z.postN`)

The flow is **inverted** compared to semver:

```bash
git tag vX.Y.Z.postN
git push origin vX.Y.Z.postN
```

`release.yml` does **not** fire on this tag — its glob only matches
`vX.Y.Z` and `vX.Y.Z-*` (see issue #73). So no automatic Release page
**and no automatic gated test matrix.** That's the same test-gate gap
the semver path has, relocated — there, `release.yml`'s matrix runs
*after* the tag and before the Release page appears; here, the tag
shortcuts straight to the Release page without any matrix in between.

Before creating the Release, manually confirm `ci.yml` on `main` is
green. `ci.yml` runs the same Linux/Windows/macOS build+test matrix
`release.yml` would have; it already ran when the release-bump PR
(the one that flipped `result = "{value}.postN"`) was merged, so in
practice this is a one-line verification rather than a new workflow
invocation:

```bash
gh run list --branch main --workflow=ci.yml --limit 1 \
    --json status,conclusion,headSha \
    --jq '.[0] | "\(.status) \(.conclusion)  \(.headSha[:7])"'
# Expect: "COMPLETED SUCCESS <sha of the merged release bump>"
```

If that doesn't show `SUCCESS`, re-run CI on `main` via the Actions UI
or stop and diagnose before tagging. The tag-and-publish sequence
itself has no test gate.

With CI green, create the Release page:

```bash
gh release create vX.Y.Z.postN --title "vX.Y.Z.postN" --notes "..."
```

That command is authenticated as **you** (a user), not `GITHUB_TOKEN`,
so the `release: published` event it emits **does** chain into
`publish.yml` automatically. No manual dispatch required — in fact, a
manual dispatch on top of this would start a second parallel publish
run that races the release-event run, and whichever finishes uploading
first wins while the other hits PyPI's file-name-reuse policy and fails
(cosmetically — the files still land).

Capture the run ID and attach to it explicitly rather than relying on
bare `gh run watch`, which can attach to any "most recent" run
(including unrelated ones if someone pushes to another workflow
between your commands):

```bash
RUN_ID=$(gh run list --workflow=publish.yml --limit 1 \
    --json databaseId --jq '.[0].databaseId')
gh run watch "$RUN_ID"
```

So for the post-release path, the full sequence is:

```bash
# 1. Verify CI on main is green (no release.yml gate on postN, see #73):
gh run list --branch main --workflow=ci.yml --limit 1 \
    --json status,conclusion --jq '.[0] | "\(.status) \(.conclusion)"'

# 2. Tag, push, create the Release page (user-auth → chains into publish.yml):
git tag vX.Y.Z.postN && git push origin vX.Y.Z.postN
gh release create vX.Y.Z.postN --title "..." --notes "..."

# 3. Watch the publish run that the release event just kicked off:
RUN_ID=$(gh run list --workflow=publish.yml --limit 1 \
    --json databaseId --jq '.[0].databaseId')
gh run watch "$RUN_ID"
```

When #73 is resolved, the semver and post paths will converge back into
a single "tag → release.yml (gated test matrix) creates Release under
PAT → publish.yml chains automatically" flow, and the manual CI-green
pre-check collapses into the workflow.

## Considerations worth flagging

| Concern | Reality |
|---------|---------|
| **Wheel size** | 7 dtype instantiations × ~50 image functions × template expansion → each Linux wheel will be in the 15–25 MB range. Not prohibitive, but notable. The Universal and MTL5 template families are what dominate the final binary size. |
| **Build time per release** | cibuildwheel clones peer repos and compiles heavy template code. Expect 20–40 min per Python × platform cell, in parallel via the matrix. Total wall-clock is generally under an hour. |
| **manylinux ABI** | cibuildwheel defaults to `manylinux_2_28` (GCC 12). Our C++20 codebase is fine with that toolchain. |
| **macOS coverage** | Matrix currently uses `arm64` only. `x86_64` Mac support requires an extra matrix row and roughly doubles macOS build time. |
| **Windows** | C++20 + nanobind on MSVC has been working throughout CI. No specific surprises expected for wheel builds. |
| **First-time upload** | PyPI runs naming-conflict and policy checks on first upload of a name. Plan for 1–2 TestPyPI iterations before going to real PyPI. |

## Ongoing release cadence

The two paths at a glance:

| Tag shape | `release.yml` fires? | GitHub Release page | `publish.yml` to PyPI |
|-----------|---------------------|---------------------|-----------------------|
| `vX.Y.Z` | Yes (glob matches) | Auto-created (under `GITHUB_TOKEN` → doesn't chain) | **Manual dispatch** |
| `vX.Y.Z.postN` | No (glob doesn't match — #73) | **Manual `gh release create`** (user-auth → **does** chain) | Automatic, don't dispatch |

### Path A — plain semver (`vX.Y.Z`)

Wraps an upstream `mixed-precision-dsp X.Y.Z` tag in lockstep.

1. Update `project(mp-dsp-python VERSION X.Y.Z ...)` in `CMakeLists.txt`
   to match the `mixed-precision-dsp` version being wrapped.
2. Bump the peer `GIT_TAG` pins accordingly (`dsp` → `vX.Y.Z`, plus
   `universal` / `mtl5` if their versions changed).
3. Reset `pyproject.toml`'s `[tool.scikit-build.metadata.version]
   result` template back to `"{value}"` if it's currently on a
   `.postN` suffix.
4. Merge to `main`.
5. Tag and push:
   ```bash
   git tag vX.Y.Z && git push origin vX.Y.Z
   ```
6. After `release.yml` publishes the GitHub Release, manually dispatch
   `publish.yml` — the auto-chain is disabled (see §6 Path A for why):
   ```bash
   gh workflow run publish.yml -f target=pypi --ref main
   RUN_ID=$(gh run list --workflow=publish.yml --limit 1 \
       --json databaseId --jq '.[0].databaseId')
   gh run watch "$RUN_ID"
   ```

### Path B — Python-only bindings fix (`vX.Y.Z.postN`)

When the C++ library is unchanged (e.g. a wrapper-layer bug, a dashboard
addition, a pure-Python accessor over data upstream already exposes).

1. Bump `pyproject.toml`'s `[tool.scikit-build.metadata.version]
   result` template to `"{value}.postN"` (PEP 440 post-release).
   `CMakeLists.txt` stays on `X.Y.Z` — CMake's `project(VERSION)`
   doesn't accept `.postN`.
2. Regenerate `docs/api_reference.md` (picks up the new version string)
   and merge to `main`.
3. Verify `ci.yml` on `main` is green (no release.yml test gate on the
   postN path — see §6 Path B):
   ```bash
   gh run list --branch main --workflow=ci.yml --limit 1 \
       --json status,conclusion --jq '.[0] | "\(.status) \(.conclusion)"'
   ```
4. Tag, push, manually create the Release page:
   ```bash
   git tag vX.Y.Z.postN && git push origin vX.Y.Z.postN
   gh release create vX.Y.Z.postN --title "..." --notes "..."
   # publish.yml is already running from the release-event chain:
   RUN_ID=$(gh run list --workflow=publish.yml --limit 1 \
       --json databaseId --jq '.[0].databaseId')
   gh run watch "$RUN_ID"
   ```
5. Do **not** also `gh workflow run publish.yml` — that starts a
   second parallel run that races the release-event run and whichever
   finishes second fails on file-name-reuse (harmless, but noisy).

The `X.Y.Z` prefix still identifies the C++ version being wrapped. The
`.postN` suffix tells users "same C++ underneath, Python bindings
repackaged".

### gh CLI reference

```bash
# Dry-run to TestPyPI (useful when validating a release candidate)
gh workflow run publish.yml -f target=testpypi --ref main

# Real release to PyPI — only needed on the semver path. On the postN
# path, `gh release create` already chained publish.yml automatically.
gh workflow run publish.yml -f target=pypi --ref main

# Watch a specific publish run finish — capture the ID first rather
# than relying on "most recent", which can race with unrelated runs
# (e.g. a CI run triggered by a concurrent push to another branch).
RUN_ID=$(gh run list --workflow=publish.yml --limit 1 \
    --json databaseId --jq '.[0].databaseId')
gh run watch "$RUN_ID"

# List recent publish runs
gh run list --workflow=publish.yml --limit 5
```

`--ref main` tells the dispatch which branch/tag to check out. Always
use `main` for a published release — by the time you're publishing, the
version bump has already been merged there. Dispatching against a
feature branch would produce a wheel with that branch's `CMakeLists.txt`
version, which is almost never what you want.

## Minimum path to first release

A focused sequence of what to do when ready:

1. **Merge PR #35** (version lockstep + `__dsp_version__`) to `main`.
2. **Open a small PR** pinning the three `GIT_TAG main` references to
   specific versions. One-file change.
3. **Configure trusted publishers on PyPI + TestPyPI** (step 1 of this
   guide). Outside the repo, ~15 min.
4. **Dry-run to TestPyPI** via `workflow_dispatch`; verify install in
   a clean venv.
5. **Tag `vX.Y.Z` and push**, then manually dispatch
   `gh workflow run publish.yml -f target=pypi --ref main` →
   published to PyPI.

After that, the package is live. Later releases follow the cadence in
the previous section.

## References

- [scikit-build-core documentation](https://scikit-build-core.readthedocs.io/)
- [cibuildwheel documentation](https://cibuildwheel.readthedocs.io/)
- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
- [PEP 440 — Version Identification](https://peps.python.org/pep-0440/)
- `mtl5-python` — the peer precedent for the versioning convention
