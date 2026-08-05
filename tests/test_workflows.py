"""Tests for the GitHub Actions workflows (Issue #73).

A tag filter that silently matches nothing is invisible: the workflow simply
does not run, there is no failing job to notice, and the gap is only found
when someone wonders where the release page went. That is exactly how
`.postN` tags went unreleased. These tests make the filters and the
pre-release classification assertable from the test suite instead.

Nothing here talks to GitHub. The filter patterns are translated to regexes
using GitHub's documented filter-pattern semantics and matched locally.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

_WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"
_RELEASE_YML = _WORKFLOWS / "release.yml"


def _tag_patterns(path: Path) -> list[str]:
    """The quoted globs under the workflow's `tags:` key.

    Parsed by hand rather than with PyYAML, which is not a dependency of
    this project and would make these tests skip in exactly the CI where
    they matter.
    """
    lines = path.read_text().splitlines()
    patterns: list[str] = []
    in_tags = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("tags:"):
            in_tags = True
            continue
        if in_tags:
            if stripped.startswith("#") or not stripped:
                continue
            if stripped.startswith("- "):
                patterns.append(stripped[2:].strip().strip('"').strip("'"))
                continue
            break  # dedented out of the list
    return patterns


def _github_filter_to_regex(pattern: str) -> re.Pattern:
    """Translate a GitHub filter pattern to an anchored regex.

    Per GitHub's filter-pattern cheat sheet the special characters are
    `*` (any run of non-`/`), `**` (any run including `/`), `?` (one
    character), `+` (one or more of the *preceding* character), and `[]`
    character ranges. Everything else — notably `.` — is a literal.

    The `+`-applies-to-what-precedes rule is the subtle one, and it is why
    `v[0-9]+.[0-9]+.[0-9]+` reads like a regex but is not: the dots are
    literal dots, not any-character.
    """
    out: list[str] = []
    i = 0
    while i < len(pattern):
        char = pattern[i]
        if char == "*":
            if pattern.startswith("**", i):
                out.append(".*")
                i += 2
            else:
                out.append("[^/]*")
                i += 1
        elif char == "?":
            out.append("[^/]")
            i += 1
        elif char == "+":
            out.append("+")          # quantifies whatever was emitted last
            i += 1
        elif char == "[":
            end = pattern.index("]", i)
            out.append(pattern[i:end + 1])
            i = end + 1
        else:
            out.append(re.escape(char))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def _matches_any(tag: str, patterns: list[str]) -> bool:
    return any(_github_filter_to_regex(p).match(tag) for p in patterns)


# (tag, should_trigger_release_yml)
_TAGS = [
    # Full releases.
    ("v0.8.0", True),
    ("v1.2.34", True),
    ("v10.20.30", True),
    # Post-releases — the case this issue was filed for.
    ("v0.5.0.post1", True),
    ("v0.8.0.post12", True),
    # Pre-releases, with and without a separator.
    ("v0.8.0-rc1", True),
    ("v0.8.0rc1", True),
    ("v0.8.0a1", True),
    ("v0.8.0b2", True),
    ("v0.8.0.dev0", True),
    # Not releases.
    ("v0.8", False),
    ("0.8.0", False),
    ("nightly", False),
    ("v0.8.0/extra", False),
]


class TestReleaseTagFilter:
    def test_post_release_tags_trigger(self):
        """The regression this issue reports: `.postN` must fire release.yml.

        Cutting v0.5.0.post1 produced no workflow run at all, because
        `.post1` has no `-` separator and so matched neither the exact-triple
        pattern nor the `-*` one.
        """
        patterns = _tag_patterns(_RELEASE_YML)
        assert _matches_any("v0.5.0.post1", patterns)
        assert _matches_any("v0.8.0.post12", patterns)

    @pytest.mark.parametrize("tag,should_match", _TAGS)
    def test_tag_matching(self, tag, should_match):
        patterns = _tag_patterns(_RELEASE_YML)
        assert _matches_any(tag, patterns) is should_match, (
            f"{tag!r} should {'' if should_match else 'not '}trigger "
            f"release.yml (patterns: {patterns})")

    def test_patterns_are_actually_parsed(self):
        """Guard the parser itself — an empty pattern list would make every
        assertion above vacuously wrong in the permissive direction."""
        patterns = _tag_patterns(_RELEASE_YML)
        assert patterns, "no tag patterns found in release.yml"
        assert all(p.startswith("v") for p in patterns)

    def test_filter_translation_respects_github_semantics(self):
        """`.` is literal and `+` quantifies the preceding element.

        If this translation were wrong the tests above would be measuring
        something other than what GitHub does.
        """
        pattern = _github_filter_to_regex("v[0-9]+.[0-9]+.[0-9]+")
        assert pattern.match("v0.8.0")
        assert pattern.match("v10.20.30")
        # A literal dot, so a different separator must not match.
        assert not pattern.match("v0x8x0")
        # No trailing wildcard, so a suffix must not match — this is the
        # precise reason the original pattern missed `.postN`.
        assert not pattern.match("v0.8.0.post1")


class TestReleasePrereleaseClassification:
    """The prerelease flag routes publish.yml to TestPyPI or PyPI.

    `github.event.release.prerelease` is what publish.yml keys off, so
    misclassifying a release candidate pushes it to real PyPI, which cannot
    be undone. The old rule was `contains(github.ref, '-')`, which calls
    `v1.2.3rc1` a full release.
    """

    # The rule the workflow implements, extracted from it below.
    _EXPECTED = [
        ("0.8.0", False),
        ("1.2.34", False),
        ("0.8.0.post1", False),
        ("0.8.0.post12", False),
        ("0.8.0-rc1", True),
        ("0.8.0rc1", True),
        ("0.8.0a1", True),
        ("0.8.0b2", True),
        ("0.8.0.dev0", True),
        ("0.8.0-alpha", True),
    ]

    @staticmethod
    def _workflow_regex() -> str:
        """Pull the classification regex out of release.yml.

        Read from the workflow rather than restated here, so the test cannot
        pass against a rule the workflow no longer implements.
        """
        text = _RELEASE_YML.read_text()
        match = re.search(r'VERSION"\s*=~\s*(\^\S+\$)', text)
        assert match, "classification regex not found in release.yml"
        return match.group(1)

    @pytest.mark.parametrize("version,is_prerelease", _EXPECTED)
    def test_classification(self, version, is_prerelease):
        pattern = re.compile(self._workflow_regex())
        assert bool(pattern.match(version)) is (not is_prerelease)

    def test_old_rule_would_have_been_wrong(self):
        """Pin why the rule changed: a hyphen test misroutes `rc` tags."""
        old_rule_says_prerelease = "-" in "0.8.0rc1"
        assert old_rule_says_prerelease is False
        # The new rule catches it.
        assert not re.compile(self._workflow_regex()).match("0.8.0rc1")

    @pytest.mark.parametrize("version,is_prerelease", _EXPECTED)
    def test_shell_implementation_agrees(self, version, is_prerelease):
        """Run the actual bash conditional, not a Python transliteration.

        Python's `re` and bash ERE agree on this pattern, but the workflow
        runs bash — so the authoritative check is bash.
        """
        script = (
            f'if [[ "{version}" =~ {self._workflow_regex()} ]]; then\n'
            f'  echo false\nelse\n  echo true\nfi\n')
        result = subprocess.run(["bash", "-c", script],
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == str(is_prerelease).lower()


class TestReleaseChain:
    """Where the prerelease flag actually lands.

    `publish.yml` listens for `release: published`, but a Release created by
    `release.yml` under `GITHUB_TOKEN` does *not* trigger it — GitHub
    suppresses workflow chaining from `GITHUB_TOKEN` events as loop
    prevention, so publishing is dispatched by hand (docs/publishing.md).

    The flag is still load-bearing in two live paths: a Release created by a
    human token (`gh release create`, the documented workaround) does chain,
    and it decides the Release page's own pre-release badge. Getting it
    wrong routes a release candidate to real PyPI, which cannot be undone.
    """

    def test_publish_listens_for_published_releases(self):
        text = (_WORKFLOWS / "publish.yml").read_text()
        assert "release:" in text and "types: [ published ]" in text

    def test_publish_routes_on_the_prerelease_flag(self):
        text = (_WORKFLOWS / "publish.yml").read_text()
        assert "github.event.release.prerelease" in text
        assert "!github.event.release.prerelease" in text

    def test_release_sets_the_flag_from_the_classification_step(self):
        text = _RELEASE_YML.read_text()
        assert "prerelease: ${{ steps.version.outputs.PRERELEASE }}" in text

        # Only executable lines — the comment above the classification step
        # names the old heuristic on purpose, to explain why it went.
        executable = [ln for ln in text.splitlines()
                      if not ln.strip().startswith("#")]
        assert not any("contains(github.ref, '-')" in ln for ln in executable), \
            "the hyphen heuristic is back in live YAML; see issue #73"
