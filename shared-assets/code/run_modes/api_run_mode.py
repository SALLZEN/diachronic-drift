from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable
import json
import os


class RunMode(str, Enum):
    LIVE = "live"
    RESUME = "resume"
    REPLAY = "replay"


@dataclass(frozen=True)
class RunModeState:
    mode: RunMode
    cache_path: Path
    cache_exists: bool
    completed_ids: set[str]
    pending_ids: list[str]
    expected_n: int
    completed_n: int
    pending_n: int

    @property
    def is_complete(self) -> bool:
        return self.expected_n > 0 and self.pending_n == 0


def parse_run_mode(value: str) -> RunMode:
    try:
        return RunMode(value.strip().lower())
    except Exception as exc:
        raise ValueError(
            f"Invalid RUN_MODE={value!r}. Use 'live', 'resume', or 'replay'."
        ) from exc


def load_jsonl_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_run_mode(
    *,
    run_mode: str,
    cache_path: Path,
    expected_ids: Iterable[str],
    completed_id_fn: Callable[[list[dict]], set[str]],
) -> RunModeState:
    mode = parse_run_mode(run_mode)
    expected_ids = [str(x) for x in expected_ids]

    cache_rows = load_jsonl_rows(cache_path)
    cache_exists = bool(cache_rows)
    completed_ids = {str(x) for x in completed_id_fn(cache_rows)}
    pending_ids = [item_id for item_id in expected_ids if item_id not in completed_ids]

    if mode is RunMode.LIVE and cache_exists:
        raise FileExistsError(
            f"Cache already exists at {cache_path}. "
            "Use RUN_MODE='resume' or RUN_MODE='replay'."
        )

    if mode is RunMode.REPLAY and not cache_exists:
        raise FileNotFoundError(
            f"No cache found at {cache_path}. "
            "RUN_MODE='replay' requires an existing cached run."
        )

    return RunModeState(
        mode=mode,
        cache_path=cache_path,
        cache_exists=cache_exists,
        completed_ids=completed_ids,
        pending_ids=pending_ids,
        expected_n=len(expected_ids),
        completed_n=len(completed_ids),
        pending_n=len(pending_ids),
    )


def should_create_client(state: RunModeState) -> bool:
    return state.mode in {RunMode.LIVE, RunMode.RESUME} and state.pending_n > 0


def require_env_var(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise EnvironmentError(f"{name} is not set in the environment.")
    return value


def rebuild_or_resume_message(state: RunModeState) -> str:
    if state.mode is RunMode.REPLAY:
        return (
            f"[REPLAY] Using cached results only from {state.cache_path} "
            f"({state.completed_n} cached rows, {state.pending_n} pending ignored)."
        )
    if state.mode is RunMode.RESUME:
        return (
            f"[RESUME] Found {state.completed_n} cached rows in {state.cache_path}; "
            f"{state.pending_n} rows still need live processing."
        )
    return (
        f"[LIVE] Starting a fresh run at {state.cache_path}; "
        f"{state.expected_n} rows scheduled for live processing."
    )
