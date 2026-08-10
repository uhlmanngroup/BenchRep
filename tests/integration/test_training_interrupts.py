from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
import yaml

import pytest


CONFIG_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "configs"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(
    os.name != "posix",
    reason="Signal subprocess behavior is POSIX-specific.",
)
@pytest.mark.parametrize(
    "sent_signal",
    [
        pytest.param(signal.SIGINT, id="sigint"),
        pytest.param(signal.SIGTERM, id="sigterm"),
    ],
)
def test_signal_during_training_returns_control_to_benchrep(
    tmp_path: Path,
    sent_signal: signal.Signals,
) -> None:
    """SIGINT and SIGTERM should stop fitting without skipping finalization."""
    training_started_path = tmp_path / "training_started"
    workflow_returned_path = tmp_path / "workflow_returned"
    release_path = tmp_path / "release_training"

    env = os.environ.copy()
    python_paths = [
        str(REPOSITORY_ROOT),
        str(REPOSITORY_ROOT / "src"),
    ]
    if existing_python_path := env.get("PYTHONPATH"):
        python_paths.append(existing_python_path)
    env["PYTHONPATH"] = os.pathsep.join(python_paths)

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tests.fixtures.interruptible_training",
            "--config",
            str(CONFIG_DIR / "training_tiny_synthetic_ae.yaml"),
            "--output-root",
            str(tmp_path / "outputs"),
            "--training-started-path",
            str(training_started_path),
            "--release-path",
            str(release_path),
            "--workflow-returned-path",
            str(workflow_returned_path),
        ],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )

    try:
        deadline = time.monotonic() + 30
        while not training_started_path.exists():
            if process.poll() is not None:
                output, _ = process.communicate()
                pytest.fail(
                    "Training subprocess exited before fitting began.\n"
                    f"Subprocess output:\n{output}"
                )
            if time.monotonic() >= deadline:
                pytest.fail("Training subprocess did not begin within 30 seconds.")
            time.sleep(0.05)

        process.send_signal(sent_signal)
        release_path.touch()
        output, _ = process.communicate(timeout=30)

    except BaseException:
        if process.poll() is None:
            process.kill()
            process.communicate()
        raise

    assert process.returncode == 0, output
    assert workflow_returned_path.is_file(), output

    manifest_path = Path(
        workflow_returned_path.read_text(encoding="utf-8")
    )
    assert manifest_path.is_file(), output

    with manifest_path.open(encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)

    last_checkpoint_path = manifest["checkpoints"]["last_checkpoint_path"]

    assert last_checkpoint_path is not None, output
    assert Path(last_checkpoint_path).is_file(), output