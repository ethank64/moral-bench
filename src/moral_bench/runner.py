"""Run items through a model, caching every prediction so reruns cost nothing."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .ethics import Item
from .models import SystemOne


def prompt_hash(item: Item) -> str:
    return hashlib.sha256(f"{item.state}\x00{item.question}".encode()).hexdigest()[:16]


def predictions_path(results_dir: Path, model_name: str) -> Path:
    return results_dir / "predictions" / f"{model_name}.jsonl"


def read_predictions(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run(
    items: list[Item],
    model: SystemOne,
    path: Path,
    workers: int = 4,
    on_done: Callable[[], None] | None = None,
) -> int:
    """Predict every item not already in the cache at `path`. Returns the number of new calls."""
    done = {(r["id"], r["prompt_hash"]) for r in read_predictions(path)}
    todo = [it for it in items if (it.id, prompt_hash(it)) not in done]
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def one(item: Item) -> dict:
        p = model.p_yes(item.state, item.question)
        return {
            "id": item.id,
            "framework": item.framework,
            "prompt_hash": prompt_hash(item),
            "label": item.label,
            "p_yes": p,
        }

    with path.open("a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as pool:
        for fut in as_completed([pool.submit(one, it) for it in todo]):
            with lock:
                f.write(json.dumps(fut.result()) + "\n")
                f.flush()
            if on_done:
                on_done()
    return len(todo)
