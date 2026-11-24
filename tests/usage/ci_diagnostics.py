"""Temporary macOS/XGBoost diagnostics; remove after the investigation."""

from __future__ import annotations

import atexit
import faulthandler
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def log(message):
    print(f"[diagnostic {time.strftime('%Y-%m-%dT%H:%M:%S')} pid={os.getpid()}] {message}",
          file=sys.stderr, flush=True)


def sample_native(pid, directory):
    """Run outside the target process so a held GIL cannot block sampling."""
    directory = Path(directory)
    for index in range(1, 16):
        time.sleep(60)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        path = directory / f"native-{index:02d}.txt"
        log(f"Sampling native stacks of pid={pid} to {path}")
        try:
            result = subprocess.run(
                ["/usr/bin/sample", str(pid), "5", "-file", str(path)],
                capture_output=True, text=True, timeout=20,
            )
            log(f"sample exit={result.returncode}: {result.stdout} {result.stderr}")
        except (OSError, subprocess.TimeoutExpired) as error:
            log(f"Native sampling unavailable: {error}")


def enable(directory):
    """Observe calls without importing XGBoost early or changing its settings."""
    from threadpoolctl import threadpool_info

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    # Keep independent streams: workflow stdout/stderr capture must not hide traces.
    stacks = (directory / "python-stacks.log").open("w", encoding="utf-8")
    events = (directory / "events.log").open("w", encoding="utf-8", buffering=1)
    faulthandler.enable(file=stacks)
    faulthandler.dump_traceback_later(60, repeat=True, file=stacks)
    sampler = None
    sampler_log = None
    if sys.platform == "darwin":
        sampler_log = (directory / "native-sampler.log").open("w", encoding="utf-8")
        sampler = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "sample", str(os.getpid()), str(directory)],
            stdout=sampler_log, stderr=subprocess.STDOUT,
        )

    def record(message):
        events.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}\n")

    active = {}
    pools_recorded = False

    def profile(frame, event, arg):
        nonlocal pools_recorded
        if event not in ("call", "return"):
            return
        module = frame.f_globals.get("__name__", "")
        name = frame.f_code.co_name
        selected = (
            module.startswith("xgboost.")
            and name in {"fit", "set_label", "_meta_from_numpy", "_load_lib", "_create_dmatrix"}
        ) or (
            module.startswith("benchrep.")
            and name in {"build_xgboost_predictability_probe", "_train", "predict", "evaluate"}
        )
        if not selected:
            return
        key = id(frame)
        label = f"{module}.{name}"
        if event == "call":
            active[key] = time.monotonic()
            record(f"ENTER {label}")
            for field in ("X", "y", "label", "data"):
                value = frame.f_locals.get(field)
                if hasattr(value, "shape"):
                    record(f"  {field}: shape={value.shape} dtype={getattr(value, 'dtype', None)} "
                           f"strides={getattr(value, 'strides', None)}")
            if module.startswith("xgboost.") and name == "fit" and not pools_recorded:
                pools_recorded = True
                record("Collecting loaded native thread pools")
                try:
                    record("THREADPOOLS " + json.dumps(threadpool_info()))
                except Exception as error:
                    record(f"Thread-pool inspection failed: {error}")
        else:
            started = active.pop(key, None)
            if started is not None:
                # 'return' can also mean exception unwinding; it is not a success assertion.
                record(f"LEAVE {label} elapsed={time.monotonic() - started:.3f}s")

    previous_profile = sys.getprofile()
    sys.setprofile(profile)
    record(f"Diagnostics enabled; pid={os.getpid()}; python={sys.version}")
    record("THREAD_ENV " + json.dumps({key: os.environ.get(key) for key in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS",
    )}))

    def close():
        sys.setprofile(previous_profile)
        faulthandler.cancel_dump_traceback_later()
        faulthandler.disable()
        if sampler is not None:
            sampler.terminate()
            try:
                sampler.wait(timeout=5)
            except subprocess.TimeoutExpired:
                sampler.kill()
                sampler.wait()
            sampler_log.close()
        stacks.close()
        events.close()

    atexit.register(close)


def standalone(directory):
    enable(directory)
    # A hung smoke check must finish before the workflow's four-minute step limit.
    faulthandler.dump_traceback_later(180, exit=True)
    log("Importing NumPy and XGBoost (no BenchRep imports)")
    import numpy as np
    import xgboost as xgb

    log(f"NumPy {np.__version__}; XGBoost {xgb.__version__}; build={xgb.build_info()}")
    rng = np.random.default_rng(137)
    X = rng.normal(size=(512, 32)).astype(np.float32)
    y = rng.normal(size=512)
    # Approximate one two-fold CV training split from the fast 08A profile.
    model = xgb.XGBRegressor(n_estimators=10, max_depth=6, learning_rate=0.05,
                            n_jobs=1, random_state=137)
    log("Starting standalone regression fit")
    model.fit(X[:256], y[:256])
    prediction = model.predict(X[256:])
    if prediction.shape != (256,) or not np.isfinite(prediction).all():
        raise AssertionError("Invalid standalone predictions")
    log("Standalone fit and prediction passed")


if __name__ == "__main__":
    if sys.argv[1] == "sample":
        sample_native(int(sys.argv[2]), sys.argv[3])
    elif sys.argv[1] == "standalone":
        standalone(sys.argv[2])
    else:
        raise SystemExit("Expected sample or standalone")
