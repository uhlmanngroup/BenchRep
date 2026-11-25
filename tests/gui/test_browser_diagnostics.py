"""Exercise trace retention through real pytest phases without a browser install."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "scenario,failed,retained",
    [
        ("pass", False, False),
        ("skip", False, False),
        ("setup", True, True),
        ("call", True, True),
        ("teardown", True, True),
        ("later_teardown", True, True),
        ("start_error", True, False),
        ("stop_error", True, False),
    ],
)
def test_browser_trace_lifecycle(tmp_path, scenario, failed, retained):
    suite = tmp_path / "suite"
    suite.mkdir()
    shutil.copyfile(Path(__file__).with_name("conftest.py"), suite / "conftest.py")
    (suite / "test_sample.py").write_text(
        '''
import os
from pathlib import Path
from types import SimpleNamespace
import pytest

scenario = os.environ["TRACE_TEST_SCENARIO"]

class Context:
    def __init__(self):
        self.tracing = self
        self.started = False
        self.stopped = False

    def start(self, **kwargs):
        assert kwargs == dict(screenshots=False, snapshots=True, sources=True)
        if scenario == "start_error":
            raise RuntimeError("trace start error")
        self.started = True

    def stop(self, *, path):
        assert self.started
        self.stopped = True
        if scenario == "stop_error":
            raise RuntimeError("trace save error")
        Path(path).write_bytes(b"fake trace")

    def close(self):
        assert not self.started or self.stopped
        Path(os.environ["TRACE_TEST_CLOSED"]).write_text("closed")

class Browser:
    def new_page(self, **options):
        assert options == {"viewport": {"width": 800, "height": 600}}
        return SimpleNamespace(context=Context())

@pytest.fixture
def later_failure():
    yield
    if scenario == "later_teardown":
        pytest.fail("original later teardown failure")

@pytest.fixture
def page(later_failure, traced_page):
    with traced_page(Browser(), viewport={"width": 800, "height": 600}) as page:
        if scenario == "setup":
            pytest.fail("original setup failure")
        yield page
        if scenario == "teardown":
            pytest.fail("original teardown failure")

def test_sample(page):
    if scenario == "skip":
        pytest.skip("intentional skip")
    if scenario in {"call", "start_error", "stop_error"}:
        pytest.fail("original call failure")
''', encoding="utf-8",
    )
    output = tmp_path / "retained"
    closed = tmp_path / "closed"
    env = os.environ.copy()
    env.update(
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTEST_ADDOPTS="",
        BENCHREP_GUI_TRACE_DIR=str(output),
        TRACE_TEST_SCENARIO=scenario,
        TRACE_TEST_CLOSED=str(closed),
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-W", "error",
         "--confcutdir", str(suite), "--basetemp", str(tmp_path / "pytest-temp"),
         str(suite)],
        cwd=suite, env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=60,
    )
    assert result.returncode == (1 if failed else 0), result.stdout
    assert closed.read_text() == "closed"
    traces = list(output.rglob("trace.zip"))
    assert len(traces) == int(retained), result.stdout
    assert not list((tmp_path / "pytest-temp").rglob("browser-*.zip"))
    if retained:
        assert traces[0].read_bytes() == b"fake trace"
        description = traces[0].with_name("failure.txt").read_text()
        assert "test_sample.py::test_sample" in description
        assert "original" in description
        assert str(traces[0]) in result.stdout
    if scenario in {"start_error", "stop_error"}:
        assert "original call failure" in result.stdout
        assert "Could not" in result.stdout
