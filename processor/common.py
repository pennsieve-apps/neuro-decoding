"""Small helpers shared by neuro-embedding and neuro-decoding: JSON, CSV, versions, memory, logging."""
from __future__ import annotations

import json
import logging
import math
import platform
import resource
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

log = logging.getLogger(__name__)


def setup_logging(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout,
                        force=True)
    lg = logging.getLogger(name)
    lg.setLevel(logging.INFO)
    for noisy in ("matplotlib", "PIL", "fontTools", "numba", "h5py", "cebra", "urllib3", "kaleido"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return lg


def clean_json(obj):
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist") and not isinstance(obj, (str, bytes)):
        return clean_json(obj.tolist())
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return clean_json(obj.item())
        except (ValueError, AttributeError):
            return str(obj)
    return obj


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(clean_json(data), fh, indent=2)
        fh.write("\n")


def app_version() -> str:
    try:
        return (REPO_ROOT / "VERSION").read_text().strip()
    except OSError:
        return "unknown"


def tool_versions(pkgs) -> dict:
    from importlib.metadata import PackageNotFoundError, version

    out = {"python": platform.python_version()}
    for pkg in pkgs:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    return out


def peak_rss_mb() -> float | None:
    try:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)
    except (OSError, ValueError):
        return None


def rel(path, root: Path):
    if path is None:
        return None
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return str(path)


def fmt(v):
    if isinstance(v, bool) or type(v).__name__ == "bool_":
        return "true" if v else "false"
    if isinstance(v, float):
        return "" if math.isnan(v) or math.isinf(v) else f"{v:.6g}"
    if v is None:
        return ""
    return v


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    import csv

    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(columns)
        for r in rows:
            w.writerow([fmt(r.get(c)) for c in columns])


class Heartbeat:
    """Logs progress at most every `every` seconds, so a long step never looks hung."""

    def __init__(self, lg: logging.Logger, what: str, every: float = 30.0):
        self.lg, self.what, self.every, self.last = lg, what, every, time.monotonic()

    def __call__(self, msg: str, force: bool = False) -> None:
        now = time.monotonic()
        if force or now - self.last >= self.every:
            self.lg.info("%s: %s", self.what, msg)
            self.last = now
