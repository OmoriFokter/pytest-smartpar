"""pytest-randomly policy: reordering off for --smartpar/--smartpar-profile, required for detect."""

from __future__ import annotations

import pytest

from smartpar import plugin

N_TESTS = 30
SOURCE_ORDER = [f"test_{i:02d}" for i in range(N_TESTS)]


@pytest.fixture
def many_tests(pytester: pytest.Pytester) -> None:
    pytester.makepyfile("\n".join(f"def {name}(): pass" for name in SOURCE_ORDER))


@pytest.fixture
def modes_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    # The modes themselves arrive in later phases; until then they abort in
    # pytest_configure. Lift that guard so the policy can be observed in a real
    # in-process session.
    monkeypatch.setattr(plugin, "_UNIMPLEMENTED_MODES", {})


def run_order(pytester: pytest.Pytester, *args: str) -> list[str]:
    reprec = pytester.inline_run("-p", "randomly", "-p", "no:cacheprovider", *args)
    reports = reprec.getreports("pytest_runtest_logreport")
    return [r.nodeid.split("::")[-1] for r in reports if r.when == "call"]


@pytest.mark.parametrize("flag", ["--smartpar", "--smartpar-profile"])
def test_policy_disables_only_reordering(pytester: pytest.Pytester, flag: str) -> None:
    config = pytester.parseconfig(flag)
    plugin.apply_randomly_policy(config)
    assert config.option.randomly_reorganize is False
    assert config.stash[plugin.RANDOMLY_REORDER_DISABLED] is True
    # The plugin itself (and its seeding) stays registered.
    assert config.pluginmanager.hasplugin("randomly")


def test_policy_untouched_without_smartpar_mode(pytester: pytest.Pytester) -> None:
    config = pytester.parseconfig()
    plugin.apply_randomly_policy(config)
    assert config.option.randomly_reorganize is True
    assert config.stash[plugin.RANDOMLY_REORDER_DISABLED] is False


def test_policy_noop_when_randomly_absent(pytester: pytest.Pytester) -> None:
    config = pytester.parseconfig("--smartpar", "-p", "no:randomly")
    plugin.apply_randomly_policy(config)
    assert config.stash[plugin.RANDOMLY_REORDER_DISABLED] is False


def test_randomly_shuffles_without_smartpar(pytester: pytest.Pytester, many_tests: None) -> None:
    # Control experiment: proves the seed below really shuffles, so the next
    # test is sensitive to the policy.
    assert run_order(pytester, "--randomly-seed=12345") != SOURCE_ORDER


@pytest.mark.parametrize("flag", ["--smartpar", "--smartpar-profile"])
def test_order_kept_and_warning_printed(
    pytester: pytest.Pytester, many_tests: None, modes_enabled: None, flag: str
) -> None:
    assert run_order(pytester, flag, "--randomly-seed=12345") == SOURCE_ORDER

    result = pytester.runpytest(flag, "-p", "randomly", "--randomly-seed=12345")
    result.assert_outcomes(passed=N_TESTS)
    warnings = [line for line in result.outlines if line == plugin.RANDOMLY_REORDER_WARNING]
    assert len(warnings) == 1


def test_detect_requires_randomly(pytester: pytest.Pytester) -> None:
    pytester.makepyfile("def test_ok(): pass")
    result = pytester.runpytest("--smartpar-detect", "-p", "no:randomly")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*--smartpar-detect requires pytest-randomly*"])
    result.stderr.fnmatch_lines(["*pip install pytest-randomly*"])


def test_detect_with_randomly_passes_policy(pytester: pytest.Pytester) -> None:
    config = pytester.parseconfig("--smartpar-detect", "-p", "randomly")
    plugin.apply_randomly_policy(config)
    assert config.option.randomly_reorganize is True
    assert config.stash[plugin.RANDOMLY_REORDER_DISABLED] is False
