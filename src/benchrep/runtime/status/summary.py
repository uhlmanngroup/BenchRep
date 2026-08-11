from __future__ import annotations

from collections.abc import Iterable, Mapping
import logging


_STATUS_BUCKETS = {
    "completed": "ok",
    "completed_with_warnings": "warnings",
    "completed_after_interruption": "warnings",
    "failed": "errors",
    "skipped": "skipped",
}


def build_outcome_summary(
    statuses: Iterable[str],
) -> dict[str, int]:
    summary = {
        "total": 0,
        "ok": 0,
        "warnings": 0,
        "errors": 0,
        "skipped": 0,
    }

    for status in statuses:
        if status == "disabled":
            continue

        bucket = _STATUS_BUCKETS.get(status)

        if bucket is None:
            raise ValueError(
                f"Cannot summarize outcome status {status!r}."
            )

        summary["total"] += 1
        summary[bucket] += 1

    return summary


def log_outcome_summary(
    *,
    run_log: logging.Logger,
    workflow_name: str,
    workflow_status: str,
    summary: Mapping[str, int],
) -> None:
    log = (
        run_log.info
        if workflow_status == "completed"
        else run_log.warning
    )

    log(
        "%s workflow status: %s; outcomes: total=%d, ok=%d, "
        "warnings=%d, errors=%d, skipped=%d",
        workflow_name,
        workflow_status,
        summary["total"],
        summary["ok"],
        summary["warnings"],
        summary["errors"],
        summary["skipped"],
    )