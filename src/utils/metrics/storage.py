from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

EpochRecord = Dict[str, Any]


def utc_timestamp() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def load_metrics_history(path: Path) -> List[EpochRecord]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to decode metrics history at {path}") from exc
    if isinstance(data, list):
        return data
    raise ValueError(f"Metrics history at {path} must be a JSON list.")


def save_metrics_history(path: Path, history: List[EpochRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2))


def find_epoch_record(history: List[EpochRecord], epoch: int) -> Optional[EpochRecord]:
    for entry in history:
        try:
            if int(entry.get("epoch")) == int(epoch):
                return entry
        except (TypeError, ValueError):
            continue
    return None


def ensure_epoch_record(history: List[EpochRecord], epoch: int) -> EpochRecord:
    existing = find_epoch_record(history, epoch)
    if existing is not None:
        return existing
    record: EpochRecord = {"epoch": int(epoch), "timestamp": utc_timestamp()}
    history.append(record)
    history.sort(key=lambda item: item.get("epoch", 0))
    return record


def remove_epoch_record(history: List[EpochRecord], epoch: int) -> None:
    history[:] = [entry for entry in history if int(entry.get("epoch", -1)) != int(epoch)]


