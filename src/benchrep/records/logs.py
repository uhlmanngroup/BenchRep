from __future__ import annotations

import sys
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO

from benchrep.runtime import RunContext

RUN_LOGGER_NAME = "benchrep.run"
RUN_LOG_FILENAME = "benchrep.run.log"
STDOUT_LOG_FILENAME = "stdout.log"
STDERR_LOG_FILENAME = "stderr.log"

class TeeStream:
    """Write stream output to multiple streams."""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@contextmanager
def capture_console_streams(
    *,
    log_out_dir: RunContext | Path | str,
    capture_stdout: bool = False,
) -> Iterator[None]:
    """
    Tee stderr, and optionally stdout, into the run log directory.

    The console still receives the original output. Stderr is always written
    to `stderr.log`. Stdout is written to `stdout.log` only when
    `capture_stdout=True`.

    Lightning progress bars update in place in the terminal, but raw stdout
    capture records each update separately. This can make `stdout.log` very
    large. Enable `capture_stdout` only when debugging console output.
    """

    if isinstance(log_out_dir, RunContext):
        log_out_dir = log_out_dir.log_dir
    else:
        log_out_dir = Path(log_out_dir).expanduser().resolve()

    stderr_path = log_out_dir / STDERR_LOG_FILENAME
    stdout_path = log_out_dir / STDOUT_LOG_FILENAME

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    stderr_file = None
    stdout_file = None

    try:
        stderr_file = stderr_path.open("a", encoding="utf-8")
        sys.stderr = TeeStream(original_stderr, stderr_file)

        if capture_stdout:
            stdout_file = stdout_path.open("a", encoding="utf-8")
            sys.stdout = TeeStream(original_stdout, stdout_file)

        yield

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        if stdout_file is not None:
            stdout_file.close()

        if stderr_file is not None:
            stderr_file.close()


def setup_run_logger(
    *,
    log_out_dir: RunContext | Path | str,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create a run-level logger for BenchRep status messages.

       The run logger records BenchRep-controlled lifecycle/status messages and
       saves the output locally. This is separate from the Lightning logger,
       which handles training metrics.

       The logger writes each message to both the console and a local log file
       under the run log directory. Existing handlers on the same named logger are
       removed before new handlers are added, which prevents duplicated log lines
       if this setup function is called more than once.

       Parameters
       ----------
       log_out_dir:
           Either a `RunContext` or an explicit log output directory. If a
           `RunContext` is provided, logs are written under `RunContext.log_dir`.
           If a path/string is provided, it is interpreted directly as the
           directory where the run log file should be written.
       filename:
           Name of the local log file created inside the resolved log directory.
       level:
           Minimum logging level recorded by both the file and console handlers.
           The default, `logging.INFO`, records info, warning, error, and critical
           messages, but skips debug messages.

       Returns
       -------
       logging.Logger
           Configured logger that writes to both the console and the local run log
           file.
       """
    if isinstance(log_out_dir, RunContext):
        log_out_dir = log_out_dir.log_dir
    else:
        log_out_dir = Path(log_out_dir).expanduser().resolve()

    log_path = log_out_dir / RUN_LOG_FILENAME

    run_logger = get_run_logger()
    run_logger.setLevel(level)
    run_logger.propagate = False

    # Avoid duplicate messages if setup is called more than once.
    for handler in list(run_logger.handlers):
        run_logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Output to local file
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Output to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    run_logger.addHandler(file_handler)
    run_logger.addHandler(console_handler)

    return run_logger


def get_run_logger() -> logging.Logger:
    return logging.getLogger(RUN_LOGGER_NAME)