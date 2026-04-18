"""Version-sync tests: pin the lockstep convention between the Python
package version, the C++ library version, and the single source of
truth in CMakeLists.txt.

Convention (see pyproject.toml): __version__ is sourced dynamically
from CMakeLists.txt by scikit-build-core's regex provider. This test
file guards three invariants:

  1. mpdsp.__version__ is readable and well-formed
  2. mpdsp.__dsp_version__ reports the upstream C++ library version
     the wheel was built against
  3. The two match at X.Y.Z (i.e. lockstep held at build time). If a
     developer is intentionally running a Python-only post-release
     like 0.4.1.post1, the X.Y.Z prefix still matches.
"""

from __future__ import annotations

import os
import re

import mpdsp


# In CI wheel-test runs we set MPDSP_REQUIRE_CORE=1 so a silently-failed
# _core import (broken install path, missing symbol, ABI mismatch) fails
# the test instead of skipping it. Local source-checkout runs leave the
# var unset — those may legitimately have no extension built yet.
_REQUIRE_CORE = os.environ.get("MPDSP_REQUIRE_CORE") == "1"


def _skip_or_fail_if_no_core():
    if mpdsp.HAS_CORE:
        return
    msg = (
        f"mpdsp._core unavailable (import error: "
        f"{mpdsp.__core_import_error__!r}); "
        f"running against {mpdsp.__file__}"
    )
    if _REQUIRE_CORE:
        raise AssertionError(msg)
    import pytest
    pytest.skip(msg)


def test_version_attribute_is_well_formed():
    """__version__ is a PEP 440 version string."""
    assert isinstance(mpdsp.__version__, str)
    # Must start with X.Y.Z; suffix (post-release, dev, etc.) is optional.
    assert re.match(r"^\d+\.\d+\.\d+", mpdsp.__version__), (
        f"unexpected version: {mpdsp.__version__}")


def test_dsp_version_attribute_is_well_formed():
    """__dsp_version__ comes from the upstream C++ library.

    We accept any trailing suffix (e.g. `-dev`, `+git-abc123`) after the
    numeric prefix so development builds of the peer library don't make
    this test a liability. The strict invariant — that the numeric
    triple reconstructs to the same string — is pinned in
    `test_dsp_version_info_matches_string`.
    """
    _skip_or_fail_if_no_core()
    assert isinstance(mpdsp.__dsp_version__, str)
    assert re.match(r"^\d+\.\d+\.\d+", mpdsp.__dsp_version__)


def test_dsp_version_info_triple():
    """__dsp_version_info__ is a (major, minor, patch) tuple of ints."""
    _skip_or_fail_if_no_core()
    info = mpdsp.__dsp_version_info__
    assert isinstance(info, tuple)
    assert len(info) == 3
    assert all(isinstance(x, int) and x >= 0 for x in info)


def test_dsp_version_info_matches_string():
    """The tuple reconstructs to the string form."""
    _skip_or_fail_if_no_core()
    major, minor, patch = mpdsp.__dsp_version_info__
    assert f"{major}.{minor}.{patch}" == mpdsp.__dsp_version__


def test_lockstep_prefix():
    """The Python package's X.Y.Z prefix matches the C++ library's
    version string. A Python-only post-release (`0.4.1.post1`) still
    starts with the same X.Y.Z — that's the whole point of post-
    releases in PEP 440 — so a prefix check is what we want here.

    Failure means either the C++ peer was upgraded without a
    corresponding Python bump, or a developer is running against a
    non-matching local peer checkout. Both are worth surfacing.
    """
    _skip_or_fail_if_no_core()
    # Strip any pre-release or post-release marker in the third segment
    # (e.g. "0.4.1rc1" or "0.4.1.post1" -> "0.4.1"). Regex grabs the
    # numeric X.Y.Z prefix, which is what we pin against dsp_version.
    py_prefix = re.match(r"^(\d+\.\d+\.\d+)", mpdsp.__version__).group(1)
    assert py_prefix == mpdsp.__dsp_version__, (
        f"Version lockstep broken: mpdsp.__version__={mpdsp.__version__} "
        f"prefix={py_prefix} but __dsp_version__={mpdsp.__dsp_version__}"
    )
