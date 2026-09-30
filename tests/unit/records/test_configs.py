from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from pydantic import BaseModel

from benchrep.records.configs import (
    config_to_serializable_dict,
    save_resolved_config,
)


class _RecordingConfig(BaseModel):
    payload: dict[str, Any]


def _identity_collate(samples):
    return samples


@pytest.mark.parametrize("input_kind", ["model", "dict"])
def test_config_recording_preserves_supported_values(
    tmp_path: Path,
    input_kind: str,
) -> None:
    payload = {
        "path": tmp_path,
        "options": {
            "enabled": True,
            "count": 3,
            "weight": 0.5,
            "optional": None,
            "names": ["a", "b"],
            "shape": (2, 3),
        },
    }
    config = (
        _RecordingConfig(payload=payload)
        if input_kind == "model"
        else {"payload": payload}
    )

    recorded = config_to_serializable_dict(config)

    assert recorded == {
        "payload": {
            "path": str(tmp_path),
            "options": {
                "enabled": True,
                "count": 3,
                "weight": 0.5,
                "optional": None,
                "names": ["a", "b"],
                "shape": [2, 3],
            },
        },
    }
    assert payload["path"] is tmp_path
    assert payload["options"]["shape"] == (2, 3)


@pytest.mark.parametrize("input_kind", ["model", "dict"])
def test_config_recording_marks_runtime_objects_without_mutating_them(
    tmp_path: Path,
    input_kind: str,
) -> None:
    generator = torch.Generator().manual_seed(137)
    original_state = generator.get_state().clone()
    payload = {
        "collate_fn": _identity_collate,
        "nested": [{"generator": generator}],
    }
    config = (
        _RecordingConfig(payload=payload)
        if input_kind == "model"
        else {"payload": payload}
    )

    recorded = config_to_serializable_dict(config)
    recorded_payload = recorded["payload"]

    assert recorded_payload["collate_fn"] == {
        "__benchrep_unserializable__": {
            "type": "builtins.function",
            "name": f"{__name__}._identity_collate",
        },
    }
    generator_description = recorded_payload["nested"][0]["generator"][
        "__benchrep_unserializable__"
    ]
    assert generator_description["type"].endswith(".Generator")

    # Exercise the actual YAML writer, not just the conversion helper.
    path = save_resolved_config(
        resolved_config=config,
        out_dir=tmp_path,
    )
    with path.open(encoding="utf-8") as handle:
        assert yaml.safe_load(handle) == recorded

    live_payload = (
        config.payload
        if isinstance(config, _RecordingConfig)
        else config["payload"]
    )
    assert live_payload["collate_fn"] is _identity_collate
    assert live_payload["nested"][0]["generator"] is generator
    assert torch.equal(generator.get_state(), original_state)
