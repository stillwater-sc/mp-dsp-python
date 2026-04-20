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
   gh run watch           # block until the matrix finishes (~20–40 min)
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

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

The tag triggers `release.yml`:

1. Tests run on Linux / Windows / macOS.
2. GitHub Release is created with auto-generated notes via
   `softprops/action-gh-release`.

The release **does not auto-trigger `publish.yml`.** `release.yml`
creates the GitHub Release under the default `GITHUB_TOKEN`, and GitHub
deliberately suppresses workflow chaining when events originate from
`GITHUB_TOKEN` — a loop-prevention safeguard. So the `release:
published` event fires, but the listener in `publish.yml` skips it.

Dispatch `publish.yml` manually after the release is cut:

```bash
gh workflow run publish.yml -f target=pypi --ref main
gh run watch
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

Once the machinery is proven, the flow for future releases is:

1. Update `project(mp-dsp-python VERSION X.Y.Z ...)` in `CMakeLists.txt`
   to match the `mixed-precision-dsp` version being wrapped.
2. Bump the peer `GIT_TAG` pins accordingly (`dsp` → `vX.Y.Z`, plus
   `universal` / `mtl5` if their versions changed).
3. Merge to `main`.
4. Tag and push:
   ```bash
   git tag vX.Y.Z && git push origin vX.Y.Z
   ```
5. After `release.yml` publishes the GitHub Release, manually dispatch
   `publish.yml` — the auto-chain is disabled (see §6 above for why):
   ```bash
   gh workflow run publish.yml -f target=pypi --ref main
   gh run watch
   ```

For a Python-only bindings fix (e.g. a wrapper-layer bug that doesn't
correspond to any C++ change):

1. Bump `pyproject.toml`'s `[tool.scikit-build.metadata.version]
   result` template from `"{value}"` to `"{value}.postN"` (PEP 440
   post-release). `CMakeLists.txt` stays on `X.Y.Z` — CMake's
   `project(VERSION)` doesn't accept `.postN`.
2. Tag and push `vX.Y.Z.postN`, then manually dispatch `publish.yml`
   as in the release flow above.

The `X.Y.Z` prefix still identifies the C++ version being wrapped. The
`.postN` suffix tells users "same C++ underneath, Python bindings
repackaged".

### gh CLI reference

```bash
# Dry-run to TestPyPI (useful when validating a release candidate)
gh workflow run publish.yml -f target=testpypi --ref main

# Real release to PyPI (after tagging and release.yml completes)
gh workflow run publish.yml -f target=pypi --ref main

# Watch the most recent dispatch finish
gh run watch

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
