from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, get_type_hints, Literal

from benchrep.records import get_run_logger
from benchrep.interfaces.contracts import ContractKind


CompatibilityPolicy = Literal["error", "warn"]


@dataclass(frozen=True, slots=True)
class PreconditionResult:
    """Compatibility precondition metadata and non-fatal warnings."""

    should_wrap_batch_contract_errors: bool = False
    expected_batch_type: type[Any] | None = None
    expected_batch_contract_kind: ContractKind | None = None
    model_family_name: str | None = None
    warnings: tuple[str, ...] = ()


def run_compatibility_check(
    *,
    check: Callable[[], None],
    compatibility_policy: CompatibilityPolicy,
    success_message: str,
    error_prefix: str,
    warning_prefix: str,
) -> str | None:
    """Run one compatibility check and apply the configured error/warning policy."""
    run_log = get_run_logger()

    try:
        check()
        run_log.info(success_message)
        return None

    except TypeError as exc:
        if compatibility_policy == "error":
            raise TypeError(
                f"{error_prefix} "
                f"Original reason: {exc}"
            ) from exc

        warning = (
            f"{warning_prefix} "
            f"Original reason: {exc}"
        )
        run_log.warning(warning)

        return warning


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
