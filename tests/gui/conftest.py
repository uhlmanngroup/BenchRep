"""Keep Playwright traces for failed GUI tests, including fixture failures.

Locally, traces go to outputs/gui-failures; CI sets BENCHREP_GUI_TRACE_DIR.
Open a retained trace with: python -m playwright show-trace PATH/trace.zip
"""

from contextlib import contextmanager
import os
from pathlib import Path
import re
import shutil
import uuid
import warnings

import pytest


_TRACES = pytest.StashKey[list[Path]]()
_FAILURES = pytest.StashKey[list[str]]()
_SAVED = pytest.StashKey[list[Path]]()


def _diagnostic_warning(message):
    # Diagnostics must not replace the original test failure, even under -W error.
    with warnings.catch_warnings():
        warnings.simplefilter("always", pytest.PytestWarning)
        warnings.warn(message, pytest.PytestWarning, stacklevel=2)


@pytest.fixture
def traced_page(request, tmp_path):
    """Create a page whose trace is finalized before its browser context closes."""
    @contextmanager
    def open_page(browser, **options):
        page = browser.new_page(**options)
        context = page.context
        trace = tmp_path / f"browser-{uuid.uuid4().hex}.zip"
        started = False
        try:
            try:
                # DOM snapshots and network events suffice; no screenshot files.
                context.tracing.start(screenshots=False, snapshots=True, sources=True)
                started = True
            except Exception as exc:
                _diagnostic_warning(f"Could not start GUI trace: {exc}")
            yield page
        finally:
            try:
                if started:
                    try:
                        context.tracing.stop(path=trace)
                    except Exception as exc:
                        _diagnostic_warning(f"Could not save GUI trace: {exc}")
                    else:
                        request.node.stash.setdefault(_TRACES, []).append(trace)
            finally:
                context.close()

    return open_page


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.failed:
        item.stash.setdefault(_FAILURES, []).append(
            f"{report.when}:\n{report.longreprtext}"
        )
    # Wait for all teardown: a later fixture may fail after the browser closes.
    if report.when != "teardown":
        return
    traces = item.stash.get(_TRACES, [])
    failures = item.stash.get(_FAILURES, [])
    for index, trace in enumerate(traces):
        try:
            if not failures:
                trace.unlink(missing_ok=True)
                continue
            root = Path(os.environ.get("BENCHREP_GUI_TRACE_DIR", "outputs/gui-failures"))
            if not root.is_absolute():
                root = item.config.rootpath / root
            name = re.sub(r"[^A-Za-z0-9_.-]+", "-", item.name)[:80]
            folder = root / f"{name}-{uuid.uuid4().hex[:12]}-{index}"
            folder.mkdir(parents=True, exist_ok=False)
            destination = folder / "trace.zip"
            shutil.move(str(trace), destination)
            (folder / "failure.txt").write_text(
                item.nodeid + "\n\n" + "\n\n".join(failures), encoding="utf-8"
            )
            item.config.stash.setdefault(_SAVED, []).append(destination)
        except OSError as exc:
            _diagnostic_warning(f"Could not retain GUI trace {trace}: {exc}")


def pytest_terminal_summary(terminalreporter):
    traces = terminalreporter.config.stash.get(_SAVED, [])
    if traces:
        terminalreporter.section("GUI failure traces")
        for path in traces:
            terminalreporter.write_line(str(path))
