"""Execute each example in an isolated Python process, using real workflows."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from .profiles import MODULES

ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).with_name("runner.py")


@pytest.mark.parametrize("example", MODULES)
def test_usage_example(example, request, usage_root, record_property):
    config = request.config
    if not config.getoption("--run-usage-examples"):
        pytest.skip("Pass --run-usage-examples to execute the usage scripts.")
    included = config.getoption("--usage-include")
    if (included is not None and example not in included) or example in config.getoption("--usage-exclude"):
        pytest.skip("Excluded by usage-example selection.")
    work = usage_root / example
    work.mkdir()
    data = (config.getoption("--usage-data-dir") or ROOT / "data").resolve()
    command = [sys.executable, str(RUNNER), example,
               "--data-dir", str(data), "--device", config.getoption("--usage-device")]
    if config.getoption("--usage-strict-warnings"):
        command.append("--strict-warnings")
    env = os.environ.copy()
    env.update(MPLBACKEND="Agg", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2", NUMBA_NUM_THREADS="2", PYTHONUNBUFFERED="1")
    # A real installation is required: do not make uninstalled code importable.
    env.pop("PYTHONPATH", None)
    log = work / "run.log"
    record_property("usage_log", str(log))
    start = time.monotonic()
    try:
        with log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(command, cwd=work, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT,
                                       timeout=config.getoption("--usage-timeout"))
    except subprocess.TimeoutExpired:
        pytest.fail(f"Example {example} timed out. Log: {log}", pytrace=False)
    finally:
        elapsed = time.monotonic() - start
        record_property("usage_seconds", round(elapsed, 2))
    report_path = work / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    warnings = report.get("warnings", [])
    reporter = config.pluginmanager.getplugin("terminalreporter")
    if reporter:
        reporter.write_line(f"Example {example}: {elapsed:.1f}s; {len(warnings)} workflow warning(s); {log}")
        for issue in report.get("expected_issues", []):
            reporter.write_line(f"  Expected: {issue}")
        for warning in warnings:
            reporter.write_line(f"  {warning}")
    if completed.returncode:
        tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-70:])
        pytest.fail(f"Example {example} failed (exit {completed.returncode}).\nLog: {log}\n{tail}", pytrace=False)
    assert report.get("passed") is True, f"Missing success report: {report_path}"
