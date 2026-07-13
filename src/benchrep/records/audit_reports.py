from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml


if TYPE_CHECKING:
    from benchrep.runtime.utils import AuditItem


AuditStage = Literal["training", "prediction", "evaluation"]
AuditReportStatus = Literal["ok", "warning", "error"]


def write_audit_report(
    *,
    stage: AuditStage,
    audit_items: Sequence[AuditItem],
    output_path: Path | str,
    audited_at: str,
) -> Path:
    """Write a structured runtime output-audit report."""
    output_path = Path(output_path)

    n_ok = sum(item.status == "ok" for item in audit_items)
    n_warnings = sum(item.status == "warning" for item in audit_items)
    n_errors = sum(item.status == "error" for item in audit_items)
    n_skipped = sum(item.status == "skipped" for item in audit_items)

    if n_errors:
        status: AuditReportStatus = "error"
    elif n_warnings:
        status = "warning"
    else:
        status = "ok"

    report = {
        "stage": stage,
        "status": status,
        "audited_at": audited_at,
        "summary": {
            "total": len(audit_items),
            "ok": n_ok,
            "warnings": n_warnings,
            "errors": n_errors,
            "skipped": n_skipped,
        },
        "items": [
            {
                "name": item.name,
                "status": item.status,
                "message": item.message,
            }
            for item in audit_items
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(report, handle, sort_keys=False)

    return output_path