"""Usage examples are opt-in because they need data, weights and optional extras."""

from pathlib import Path
import shutil

import pytest

from .profiles import MODULES


def pytest_addoption(parser):
    group = parser.getgroup("usage examples")
    group.addoption("--run-usage-examples", action="store_true",
                    help="Run real usage scripts with temporary fast configurations.")
    group.addoption("--usage-include", nargs="+", choices=tuple(MODULES),
                    help="Only run these example IDs, e.g. 01 06 08A.")
    group.addoption("--usage-exclude", nargs="+", choices=tuple(MODULES), default=[],
                    help="Exclude these example IDs.")
    group.addoption("--usage-device", choices=("cpu", "auto", "gpu"), default="auto",
                    help="Device for usage examples (default: auto).")
    group.addoption("--usage-skip-xgboost", action="store_true",
                    help="Omit only XGBoost probes from temporary configs (macOS CI workaround).")
    group.addoption("--usage-data-dir", type=Path,
                    help="Shared dataset cache (default: repository/data).")
    group.addoption("--usage-output", type=Path,
                    help="New directory for logs/artifacts (default: pytest temporary directory).")
    group.addoption("--usage-timeout", type=float, default=1800,
                    help="Maximum seconds per example, including downloads (default: 1800).")
    group.addoption("--usage-clean-on-success", action="store_true", default=True,
                    help="Delete this session's usage outputs only if the entire pytest session succeeds (default).")
    group.addoption("--no-usage-clean-on-success", action="store_false",
                    dest="usage_clean_on_success",
                    help="Keep this session's usage outputs even when the entire pytest session succeeds.")
    group.addoption("--usage-strict-warnings", action="store_true",
                    help="Also fail on recorded workflow warnings; warnings are always reported.")


@pytest.fixture(scope="session")
def usage_root(request, tmp_path_factory):
    if not request.config.getoption("--run-usage-examples"):
        return None
    root = request.config.getoption("--usage-output")
    if root is None:
        root = tmp_path_factory.mktemp("usage-examples")
    else:
        root = root.resolve()
        # Refuse reuse so stale artifacts cannot satisfy this run's checks.
        root.mkdir(parents=True, exist_ok=False)
    request.config._usage_output_root = root
    return root


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    root = getattr(session.config, "_usage_output_root", None)
    if (exitstatus == pytest.ExitCode.OK and root is not None
            and session.config.getoption("--usage-clean-on-success")):
        shutil.rmtree(root)
        reporter = session.config.pluginmanager.getplugin("terminalreporter")
        if reporter:
            reporter.write_line(f"Removed successful usage-test outputs: {root}")
