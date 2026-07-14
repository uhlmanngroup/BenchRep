from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_type_hints, Literal

from benchrep.records import get_run_logger
from benchrep.interfaces.contracts import ContractKind
from benchrep.assembly.config import ConfigCompositionResult, load_yaml


CompatibilityPolicy = Literal["error", "warn"]
AuditStatus = Literal["ok", "warning", "error", "skipped"]


@dataclass(frozen=True, slots=True)
class PreconditionResult:
    """Metadata used to improve error messages after stage-level runtime failures."""

    should_wrap_batch_contract_errors: bool = False
    expected_batch_type: type[Any] | None = None
    expected_batch_contract_kind: ContractKind | None = None
    model_family_name: str | None = None


@dataclass(frozen=True, slots=True)
class AuditItem:
    """One structured check result emitted by a runtime output audit."""

    name: str
    status: AuditStatus
    message: str


def run_compatibility_check(
    *,
    check: Callable[[], None],
    compatibility_policy: CompatibilityPolicy,
    success_message: str,
    error_prefix: str,
    warning_prefix: str,
) -> None:
    """Run one compatibility check and apply the configured error/warning policy."""
    run_log = get_run_logger()

    try:
        check()
        run_log.info(success_message)

    except TypeError as exc:
        if compatibility_policy == "error":
            raise TypeError(
                f"{error_prefix} "
                f"Original reason: {exc}"
            ) from exc

        run_log.warning(
            "%s Original reason: %s",
            warning_prefix,
            exc,
        )


def format_external_datamodule_failure_message(
    *,
    stage: Literal["training", "prediction"],
    precondition_result: PreconditionResult,
    original_error: BaseException,
) -> str:
    expected_batch_type = precondition_result.expected_batch_type
    expected_batch_contract_kind = precondition_result.expected_batch_contract_kind
    model_family_name = precondition_result.model_family_name

    if expected_batch_type is None or expected_batch_contract_kind is None:
        return (
            f"{stage.capitalize()} failed. "
            f"Original error ({type(original_error).__name__}): {original_error}"
        )

    return (
        f"{stage.capitalize()} failed while using an external datamodule with an internal model. "
        "This may indicate that the datamodule does not produce the expected BenchRep "
        "batch contract, although the original error may also be unrelated.\n\n"
        f"Expected batch contract for model family `{model_family_name}`:\n"
        f"{format_expected_contract(expected_batch_type, expected_batch_contract_kind)}\n\n"
        f"Original error ({type(original_error).__name__}): {original_error}"
    )


def format_expected_contract(
    expected_type: type[Any],
    expected_contract_kind: ContractKind,
) -> str:
    """Format an expected batch/output contract for user-facing diagnostics."""
    type_name = getattr(expected_type, "__name__", repr(expected_type))
    lines = [
        f"- contract: `{type_name}`",
        f"- kind: `{expected_contract_kind}`",
    ]

    if expected_contract_kind == "typeddict":
        type_hints = get_type_hints(expected_type)
        required_keys = getattr(expected_type, "__required_keys__", frozenset())
        optional_keys = getattr(expected_type, "__optional_keys__", frozenset())

        if required_keys:
            lines.append("- required fields:")
            for key in sorted(required_keys):
                annotation = type_hints.get(key, Any)
                lines.append(f"  - `{key}`: `{format_annotation(annotation)}`")

        if optional_keys:
            lines.append("- optional fields:")
            for key in sorted(optional_keys):
                annotation = type_hints.get(key, Any)
                lines.append(f"  - `{key}`: `{format_annotation(annotation)}`")

    return "\n".join(lines)


def format_annotation(annotation: Any) -> str:
    """Return a compact display name for a type annotation."""
    return getattr(annotation, "__name__", repr(annotation))


def audit_existing_file(
    *,
    audit_items: list[AuditItem],
    name: str,
    path: Path | str,
    expected_suffixes: set[str] | None = None,
) -> bool:
    """Record whether an expected output file exists and has an expected suffix."""
    path = Path(path)

    if not path.exists():
        audit_items.append(
            AuditItem(
                name=name,
                status="error",
                message=f"not found at '{path}'",
            )
        )
        return False

    if not path.is_file():
        audit_items.append(
            AuditItem(
                name=name,
                status="error",
                message=f"path exists but is not a file: '{path}'",
            )
        )
        return False

    if expected_suffixes is not None:
        normalized_suffixes = {suffix.lower() for suffix in expected_suffixes}
        actual_suffix = path.suffix.lower()

        if actual_suffix not in normalized_suffixes:
            expected = ", ".join(sorted(normalized_suffixes))
            audit_items.append(
                AuditItem(
                    name=name,
                    status="error",
                    message=(
                        f"path has suffix {actual_suffix!r}, "
                        f"expected one of {{{expected}}}: '{path}'"
                    ),
                )
            )
            return False

    audit_items.append(
        AuditItem(
            name=name,
            status="ok",
            message=f"found at '{path}'",
        )
    )
    return True


def audit_existing_dir(
    *,
    audit_items: list[AuditItem],
    name: str,
    path: Path | str,
) -> bool:
    """Record whether an expected output directory exists and is a directory."""
    path = Path(path)

    if not path.exists():
        audit_items.append(
            AuditItem(
                name=name,
                status="error",
                message=f"not found at '{path}'",
            )
        )
        return False

    if not path.is_dir():
        audit_items.append(
            AuditItem(
                name=name,
                status="error",
                message=f"path exists but is not a directory: '{path}'",
            )
        )
        return False

    audit_items.append(
        AuditItem(
            name=name,
            status="ok",
            message=f"found at '{path}'",
        )
    )
    return True


def log_audit_summary(
    stage: Literal["training", "prediction", "evaluation"],
    audit_items: list[AuditItem],
) -> None:
    run_log = get_run_logger()

    run_log.info("")
    run_log.info("=" * 53)
    run_log.info(f"{stage.capitalize()} output audit summary")
    run_log.info("=" * 53)

    n_errors = sum(item.status == "error" for item in audit_items)
    n_warnings = sum(item.status == "warning" for item in audit_items)

    run_log.info(
        f"{stage.capitalize()} output audit summary: %s error(s), %s warning(s).",
        n_errors,
        n_warnings,
    )

    run_log.info("")

    for item in audit_items:
        message = f"{stage.capitalize()} output audit: %s: %s" % (item.name, item.message)

        if item.status == "ok":
            run_log.info(message)
        elif item.status == "warning":
            run_log.warning(message)
        elif item.status == "error":
            run_log.error(message)
        elif item.status == "skipped":
            run_log.info(message)


def audit_config_records(
    *,
    audit_items: list[AuditItem],
    config_composition_result: ConfigCompositionResult[Any],
    resolved_config_path: Path | str,
) -> None:
    """Audit config files expected from the config composition result."""
    resolved_config_path = Path(resolved_config_path)
    original_config_path = config_composition_result.original_config_path

    # Always written.
    audit_existing_file(
        audit_items=audit_items,
        name="resolved config",
        path=resolved_config_path,
        expected_suffixes={".yaml", ".yml"},
    )

    # Written only when an original YAML path was supplied.
    if original_config_path is not None:
        audit_existing_file(
            audit_items=audit_items,
            name="original config source",
            path=original_config_path,
            expected_suffixes={".yaml", ".yml"},
        )

        audit_existing_file(
            audit_items=audit_items,
            name="original config record",
            path=resolved_config_path.parent / "original_config.yaml",
            expected_suffixes={".yaml", ".yml"},
        )
    else:
        audit_items.append(
            AuditItem(
                name="original config record",
                status="skipped",
                message=(
                    "no original config record was expected because the run "
                    "was not given an original YAML config"
                ),
            )
        )


def audit_resolved_config_reconstructability(
    *,
    audit_items: list[AuditItem],
    config_composition_result: ConfigCompositionResult[Any],
    resolved_config_path: Path | str,
    run_reconstructable_from_resolved_config: bool | None,
) -> None:
    """Verify the manifest's resolved-config reconstructability claim."""
    name = "run reconstructability from resolved config"

    if run_reconstructable_from_resolved_config is None:
        audit_items.append(
            AuditItem(
                name=name,
                status="error",
                message=(
                    "manifest does not report "
                    "`run_reconstructable_from_resolved_config`"
                ),
            )
        )
        return

    if not run_reconstructable_from_resolved_config:
        audit_items.append(
            AuditItem(
                name=name,
                status="skipped",
                message=(
                    "manifest reports that the run is not reconstructable "
                    "from the resolved config"
                ),
            )
        )
        return

    resolved_config_path = Path(resolved_config_path)
    config_schema = type(config_composition_result.effective_config)

    try:
        resolved_raw = load_yaml(resolved_config_path)
        config_schema.model_validate(resolved_raw)
    except Exception as exc:
        audit_items.append(
            AuditItem(
                name=name,
                status="error",
                message=(
                    "manifest reports that the run is reconstructable from "
                    f"the resolved config, but validation as "
                    f"{config_schema.__name__} failed: {exc}"
                ),
            )
        )
        return

    audit_items.append(
        AuditItem(
            name=name,
            status="ok",
            message=(
                "manifest reports that the run is reconstructable from the "
                f"resolved config, which successfully validates as "
                f"{config_schema.__name__}"
            ),
        )
    )