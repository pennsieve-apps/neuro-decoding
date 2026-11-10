"""
Reads neuro-align's `aligned` output (workflow-systems-neuroscience docs/conventions.md, "Decoding"):

  neuro-align/<session_id>/aligned.npz     t, X (bins x neurons), Y (bins x variables), neuron_ids,
                                           neuron_meta, behavior_names, epoch_mask
  neuro-align/<session_id>/events.parquet  table, start_time, stop_time, <columns>   (optional)
  neuro-align/<session_id>/spikes.npz      raw spike times per unit                  (optional)
  neuro-align/<session_id>/manifest.json   bin size, variables, neural source, ...    (optional)

Files are found by name, recursively, following symlinks (INPUT_DIR entries are symlinks), so the
port may be linked in its own folder or merged at the root. The reader is tolerant about the
details conventions.md leaves open (how neuron_meta and spikes.npz are stored) and never needs
pickle.

This module is shared, identical, by neuro-embedding and neuro-decoding.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SESSION_RE = re.compile(r"[^A-Za-z0-9_-]")
SHEET_KEYS = ("subject_id", "condition", "group")
REGION_KEYS = ("region", "brain_area", "location", "brain_region", "acronym", "area")
LABEL_KEYS = ("label", "quality", "kslabel", "bombcell_label")


@dataclass
class Session:
    session_id: str
    folder: Path
    t: np.ndarray                       # bin centres, s (session clock)
    X: np.ndarray                       # bins x neurons, float32
    Y: np.ndarray                       # bins x variables, float32 (NaN = missing)
    neuron_ids: list[str]
    behavior_names: list[str]
    epoch_mask: np.ndarray              # bool, bins
    bin_size: float
    neural_source: str = "units"        # units | roi
    regions: list[str] | None = None    # per neuron, None when unknown
    neuron_labels: list[str] | None = None
    variables: dict = field(default_factory=dict)   # name -> {unit, source, kind}
    manifest: dict = field(default_factory=dict)
    sheet: dict = field(default_factory=dict)       # subject_id, condition, group, ...
    events_path: Path | None = None
    spikes_path: Path | None = None

    @property
    def is_counts(self) -> bool:
        return self.neural_source == "units"

    def unit(self, name: str) -> str:
        return (self.variables.get(name) or {}).get("unit") or ""


def walk(root: Path):
    """Every file under root, following symlinks (guarding against loops)."""
    seen = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        try:
            real = os.path.realpath(dirpath)
        except OSError:
            continue
        if real in seen:
            dirnames[:] = []
            continue
        seen.add(real)
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for f in sorted(filenames):
            if not f.startswith("."):
                yield Path(dirpath) / f


def sanitise(s: str) -> str:
    return SESSION_RE.sub("-", s).strip("-") or "session"


def read_sheet(files: list[Path]) -> dict[str, dict]:
    """session_id -> row of any sessions.csv found (neuro-align's or the lab's)."""
    rows: dict[str, dict] = {}
    for p in files:
        if p.name != "sessions.csv":
            continue
        try:
            with open(p, newline="") as fh:
                for r in csv.DictReader(fh):
                    sid = (r.get("session_id") or "").strip()
                    if sid:
                        rows.setdefault(sid, {}).update({k: v for k, v in r.items() if k and v not in (None, "")})
        except (OSError, csv.Error, UnicodeDecodeError) as exc:
            log.warning("%s: unreadable sessions.csv (%s)", p, exc)
    return rows


@dataclass
class Found:
    sessions: list[tuple[str, Path]]    # (session_id, folder holding aligned.npz)
    sheet: dict[str, dict]
    skipped: dict[str, str]


def discover(input_dir: Path, port: str | None = "aligned") -> Found:
    """Every aligned.npz under INPUT_DIR (or its `port` subfolder when that exists)."""
    root = input_dir / port if port and (input_dir / port).is_dir() else input_dir
    files = list(walk(root))
    if root != input_dir:   # sessions.csv may also sit at the root
        files += [p for p in walk(input_dir) if p.name == "sessions.csv" and p not in files]
    sheet = read_sheet(files)
    out, skipped, seen = [], {}, set()
    for p in files:
        if p.name != "aligned.npz" and not p.name.endswith("_aligned.npz"):
            continue
        folder = p.parent
        sid = None
        mf = folder / "manifest.json"
        if mf.is_file():
            try:
                sid = json.loads(mf.read_text()).get("session_id")
            except (OSError, ValueError):
                pass
        if not sid:
            try:
                with np.load(p, allow_pickle=False) as z:
                    if "session_id" in z.files:
                        sid = str(z["session_id"])
            except Exception:
                pass
        if not sid:
            sid = p.name[: -len("_aligned.npz")] if p.name.endswith("_aligned.npz") else folder.name
        sid = sanitise(str(sid))
        if sid in seen:
            skipped[str(p)] = f"duplicate session_id {sid}"
            continue
        seen.add(sid)
        out.append((sid, p))
    return Found(out, sheet, skipped)


def _str_list(a) -> list[str]:
    a = np.asarray(a)
    if a.ndim == 0:
        a = a.reshape(1)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a.tolist()]


def _meta_column(meta, keys, n) -> list[str] | None:
    """A per-neuron string column from neuron_meta: a structured array, a JSON string (list of
    dicts or dict of lists), or a 2-D string array with a header row."""
    if meta is None:
        return None
    m = np.asarray(meta)
    if m.dtype.names:
        names = {k.lower(): k for k in m.dtype.names}
        for k in keys:
            if k in names:
                return _str_list(m[names[k]])
        return None
    if m.dtype.kind in "US" and m.ndim == 0:
        try:
            j = json.loads(str(m))
        except ValueError:
            return None
        if isinstance(j, dict):
            low = {k.lower(): v for k, v in j.items()}
            for k in keys:
                if k in low and len(low[k]) == n:
                    return [str(v) for v in low[k]]
        elif isinstance(j, list) and len(j) == n and all(isinstance(r, dict) for r in j):
            for k in keys:
                vals = [{kk.lower(): vv for kk, vv in r.items()}.get(k) for r in j]
                if any(v not in (None, "") for v in vals):
                    return ["" if v is None else str(v) for v in vals]
    if m.dtype.kind in "US" and m.ndim == 2 and m.shape[0] == n + 1:
        header = [str(h).lower() for h in m[0]]
        for k in keys:
            if k in header:
                return _str_list(m[1:, header.index(k)])
    return None


def load(sid: str, path: Path, sheet: dict | None = None) -> Session:
    folder = path.parent
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
        missing = {"t", "X"} - keys
        if missing:
            raise ValueError(f"aligned.npz lacks {sorted(missing)}")
        t = np.asarray(z["t"], dtype=np.float64).ravel()
        X = np.asarray(z["X"], dtype=np.float32)
        if X.ndim != 2 or X.shape[0] != t.size:
            if X.ndim == 2 and X.shape[1] == t.size:
                X = X.T.copy()
            else:
                raise ValueError(f"X has shape {X.shape}, t has {t.size} bins")
        Y = np.asarray(z["Y"], dtype=np.float32) if "Y" in keys else np.zeros((t.size, 0), np.float32)
        if Y.ndim == 1:
            Y = Y[:, None]
        if Y.shape[0] != t.size and Y.ndim == 2 and Y.shape[1] == t.size:
            Y = Y.T.copy()
        names = _str_list(z["behavior_names"]) if "behavior_names" in keys else [f"var{i}" for i in range(Y.shape[1])]
        if len(names) != Y.shape[1]:
            raise ValueError(f"{len(names)} behavior_names for {Y.shape[1]} columns of Y")
        ids = _str_list(z["neuron_ids"]) if "neuron_ids" in keys else [str(i) for i in range(X.shape[1])]
        meta = z["neuron_meta"] if "neuron_meta" in keys else None
        epoch = np.asarray(z["epoch_mask"], dtype=bool).ravel() if "epoch_mask" in keys else np.ones(t.size, bool)
        if epoch.size != t.size:
            epoch = np.ones(t.size, bool)
        bin_from_npz = float(z["bin_size"]) if "bin_size" in keys else None
        units_npz = _str_list(z["behavior_units"]) if "behavior_units" in keys else None
        kinds_npz = _str_list(z["behavior_kinds"]) if "behavior_kinds" in keys else None
        kind_npz = str(z["neural_kind"]) if "neural_kind" in keys else ""
    manifest = {}
    mf = folder / "manifest.json"
    if mf.is_file():
        try:
            manifest = json.loads(mf.read_text())
        except (OSError, ValueError) as exc:
            log.warning("%s: unreadable manifest.json (%s)", sid, exc)
    bin_size = manifest.get("bin_size_s") or manifest.get("bin_size") or bin_from_npz
    if not bin_size:
        bin_size = float(np.median(np.diff(t))) if t.size > 1 else 1.0
    bin_size = float(bin_size)
    neural = manifest.get("neural") if isinstance(manifest.get("neural"), dict) else {}
    src = str(kind_npz or neural.get("kind") or manifest.get("neural_source") or manifest.get("neural_kind") or "").lower()
    if src not in ("units", "roi"):
        integral = X.size > 0 and np.all(np.mod(X[: min(len(X), 2000)], 1) == 0) and X.min() >= 0
        src = "units" if integral else "roi"
    variables = {}
    beh = manifest.get("behavior") if isinstance(manifest.get("behavior"), dict) else {}
    for v in (beh.get("variables") or manifest.get("variables") or []):
        if isinstance(v, dict) and v.get("name"):
            variables[str(v["name"])] = v
        elif isinstance(v, str):
            variables[v] = {"name": v}
    if isinstance(manifest.get("variables"), dict):
        variables = {k: (v if isinstance(v, dict) else {"unit": v}) for k, v in manifest["variables"].items()}
    for i, n in enumerate(names):   # aligned.npz's own units and kinds win
        v = variables.setdefault(n, {"name": n})
        if units_npz and i < len(units_npz) and units_npz[i]:
            v["unit"] = units_npz[i]
        if kinds_npz and i < len(kinds_npz) and kinds_npz[i]:
            v["kind"] = kinds_npz[i]
    # circular variables (angles) become their sine and cosine, which linear methods can use
    circ = [i for i, n in enumerate(names) if variables.get(n, {}).get("circular")
            or str(variables.get(n, {}).get("kind", "")).lower() == "angle"]
    if circ:
        cols, new_names = [], []
        for i, n in enumerate(names):
            if i in circ:
                a = Y[:, i].astype(np.float64)
                cols += [np.sin(a), np.cos(a)]
                for suf in ("sin", "cos"):
                    new_names.append(f"{n}_{suf}")
                    variables[f"{n}_{suf}"] = {"name": f"{n}_{suf}", "unit": "", "kind": f"{variables[n].get('kind', 'angle')}_{suf}",
                                               "source": f"{suf} of {n}"}
            else:
                cols.append(Y[:, i].astype(np.float64))
                new_names.append(n)
        Y = np.column_stack(cols).astype(np.float32) if cols else Y
        names = new_names
    regions = _meta_column(meta, REGION_KEYS, X.shape[1])
    if regions is None and isinstance(manifest.get("neuron_meta"), (list, dict)):
        regions = _meta_column(np.array(json.dumps(manifest["neuron_meta"])), REGION_KEYS, X.shape[1])
    if regions is not None:
        regions = [r if r and r.lower() not in ("nan", "none", "null") else "unknown" for r in regions]
    labels = _meta_column(meta, LABEL_KEYS, X.shape[1])
    sheet_row = dict(sheet or {})
    if isinstance(manifest.get("sheet"), dict):
        for k, v in manifest["sheet"].items():
            if v not in (None, "") and k not in sheet_row and k not in ("session_id", "path"):
                sheet_row[k] = str(v)
    for k in SHEET_KEYS:
        if manifest.get(k) not in (None, "") and k not in sheet_row:
            sheet_row[k] = str(manifest[k])
    finite = np.isfinite(X)
    if not finite.all():
        X = np.where(finite, X, 0).astype(np.float32)
    ev = folder / "events.parquet"
    sp = folder / "spikes.npz"
    return Session(sid, folder, t, X, Y, ids, names, epoch, bin_size, src, regions, labels, variables, manifest,
                   sheet_row, ev if ev.is_file() else None, sp if sp.is_file() else None)


def rebin(s: Session, target: float) -> Session:
    """Sum (counts) or average (ROI rates) X over k consecutive bins, average Y; k = round(target / bin)."""
    k = int(round(target / s.bin_size)) if target and target > s.bin_size else 1
    if k <= 1:
        return s
    n = s.t.size // k
    if n < 2:
        return s
    sl = slice(0, n * k)
    Xr = s.X[sl].reshape(n, k, -1)
    X = Xr.sum(axis=1) if s.is_counts else Xr.mean(axis=1)
    with np.errstate(all="ignore"):
        Yr = s.Y[sl].reshape(n, k, -1)
        cnt = np.isfinite(Yr).sum(axis=1)
        Y = np.where(cnt > 0, np.nansum(Yr, axis=1) / np.maximum(cnt, 1), np.nan)
    t = s.t[sl].reshape(n, k).mean(axis=1)
    epoch = s.epoch_mask[sl].reshape(n, k).mean(axis=1) >= 0.5
    out = Session(**{**s.__dict__})
    out.t, out.X, out.Y, out.epoch_mask, out.bin_size = t, X.astype(np.float32), Y.astype(np.float32), epoch, s.bin_size * k
    return out


def read_events(s: Session):
    """events.parquet as a DataFrame (empty when absent or unreadable)."""
    import pandas as pd

    if s.events_path is None:
        return pd.DataFrame(columns=["table", "start_time", "stop_time"])
    try:
        df = pd.read_parquet(s.events_path)
    except Exception as exc:   # pyarrow errors vary
        log.warning("%s: unreadable events.parquet (%s)", s.session_id, exc)
        return pd.DataFrame(columns=["table", "start_time", "stop_time"])
    if "table" not in df.columns:
        df["table"] = "events"
    return df


def read_spikes(s: Session) -> list[np.ndarray] | None:
    """Spike times per neuron (in neuron_ids order) from spikes.npz: either spike_times + unit_index
    (+ neuron_ids), or one array per unit id (keys = neuron ids, or `unit_<id>`)."""
    if s.spikes_path is None:
        return None
    try:
        with np.load(s.spikes_path, allow_pickle=False) as z:
            keys = set(z.files)
            if {"times", "offsets"} <= keys:
                times = np.asarray(z["times"], float)
                off = np.asarray(z["offsets"], np.int64)
                ids = _str_list(z["neuron_ids"]) if "neuron_ids" in keys else list(s.neuron_ids)
                per = {u: np.sort(times[off[i]:off[i + 1]]) for i, u in enumerate(ids) if i + 1 < off.size}
                if not all(u in per for u in s.neuron_ids):
                    return None
                return [per[u] for u in s.neuron_ids]
            if {"spike_times", "unit_index"} <= keys or {"spike_times", "spike_units"} <= keys:
                times = np.asarray(z["spike_times"], float)
                unit = np.asarray(z["unit_index"] if "unit_index" in keys else z["spike_units"])
                ids = _str_list(z["neuron_ids"]) if "neuron_ids" in keys else None
                pos = {u: i for i, u in enumerate(s.neuron_ids)}
                out = [[] for _ in s.neuron_ids]
                if unit.dtype.kind in "iu" and ids is not None:
                    unit_ids = [ids[i] if 0 <= i < len(ids) else None for i in range(len(ids))]
                    order = np.argsort(unit, kind="stable")
                    bounds = np.searchsorted(unit[order], np.arange(len(ids) + 1))
                    for i, uid in enumerate(unit_ids):
                        if uid in pos:
                            out[pos[uid]] = times[order[bounds[i]:bounds[i + 1]]]
                elif unit.dtype.kind in "iu":
                    for i in range(len(s.neuron_ids)):
                        out[i] = times[unit == i]
                else:
                    us = np.array(_str_list(unit))
                    for uid, i in pos.items():
                        out[i] = times[us == uid]
                return [np.sort(np.asarray(x, float)) for x in out]
            out = []
            for uid in s.neuron_ids:
                for k in (uid, f"unit_{uid}", f"u{uid}"):
                    if k in keys:
                        out.append(np.sort(np.asarray(z[k], float)))
                        break
                else:
                    return None
            return out
    except Exception as exc:
        log.warning("%s: unreadable spikes.npz (%s)", s.session_id, exc)
        return None


# --- trial (categorical) variables -------------------------------------------------------------

ALIGN_CANDIDATES = ("gabor_stimulus_onset_time", "stimulus_onset_time", "stim_on_time", "stimOn_times", "stimulus_on_time", "stim_on_times", "stimulus_onset",
                    "go_cue_time", "goCue_times", "go_cue_times", "first_movement_time", "firstMovement_times")
TIME_LIKE = re.compile(r"(^|_)(start|stop|time|times|onset|offset|id|index|trial|duration)$|_times?$|Times?$", re.I)


def trial_table(events):
    """The trials table (TimeIntervals named trials), or None."""
    if events is None or events.empty:
        return None
    tables = events["table"].astype(str)
    for name in ("trials", "trial"):
        m = tables.str.lower() == name
        if m.any():
            return events[m].dropna(axis=1, how="all").reset_index(drop=True)
    return None


def _pick(cols, *names):
    low = {c.lower(): c for c in cols}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def trial_variables(trials, wanted: list[str] | None = None, max_levels: int = 6) -> dict[str, np.ndarray]:
    """name -> per-trial label (object array, None = missing). Auto: every non-time column with 2 to
    max_levels distinct values, plus `stimulus_side` from contrast_left/right."""
    out: dict[str, np.ndarray] = {}
    if trials is None or trials.empty:
        return out
    cl = _pick(trials.columns, "contrast_left", "contrastLeft", "contrast_l")
    cr = _pick(trials.columns, "contrast_right", "contrastRight", "contrast_r")
    if cl and cr:
        a = trials[cl].to_numpy(float)
        b = trials[cr].to_numpy(float)
        side = np.full(len(trials), None, dtype=object)
        only_l, only_r = np.isfinite(a) & ~np.isfinite(b), np.isfinite(b) & ~np.isfinite(a)
        side[only_l], side[only_r] = "left", "right"
        both = np.isfinite(a) & np.isfinite(b)
        side[both & (a > b)], side[both & (b > a)] = "left", "right"
        if (side != None).sum() >= 10:   # noqa: E711
            out["stimulus_side"] = side
    for c in trials.columns:
        if c in ("table", "start_time", "stop_time") or TIME_LIKE.search(c):
            continue
        if c in (cl, cr) and "stimulus_side" in out:
            continue
        col = trials[c]
        vals = col.dropna()
        if vals.empty:
            continue
        uniq = vals.unique()
        if not 2 <= len(uniq) <= max_levels:
            continue
        lab = np.array([None if (v is None or (isinstance(v, float) and np.isnan(v))) else
                        (f"{v:g}" if isinstance(v, (float, np.floating)) else str(v)) for v in col], dtype=object)
        out[c] = lab
    if wanted:
        out = {k: v for k, v in out.items() if k in wanted}
    return out


def align_column(trials, name: str = "auto") -> str:
    if name and name != "auto":
        if name not in trials.columns:
            raise ValueError(f"TRIAL_ALIGN column {name!r} not in the trials table ({', '.join(trials.columns)})")
        return name
    return _pick(trials.columns, *ALIGN_CANDIDATES) or "start_time"


def trial_counts(s: Session, trials, align: str, window: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Per-trial activity: X summed (counts) or averaged (rates) over bins whose centre falls in
    [align + window[0], align + window[1]). Returns (trials x neurons, valid trial mask)."""
    t_al = trials[align].to_numpy(float)
    feats = np.zeros((len(trials), s.X.shape[1]), np.float32)
    ok = np.isfinite(t_al)
    lo = np.searchsorted(s.t, t_al + window[0])
    hi = np.searchsorted(s.t, t_al + window[1])
    csum = np.vstack([np.zeros((1, s.X.shape[1])), np.cumsum(s.X, axis=0, dtype=np.float64)])
    for i in np.flatnonzero(ok):
        a, b = lo[i], hi[i]
        if b <= a:
            ok[i] = False
            continue
        v = csum[b] - csum[a]
        feats[i] = v if s.is_counts else v / (b - a)
    return feats, ok


# --- behaviour variables -----------------------------------------------------------------------

POSITION_RE = re.compile(r"(^|[_\-\s])(pos|position|linpos|linear_?position|location|x|y|lin)([_\-\s]|$)|position", re.I)


def is_position(name: str, meta: dict | None = None) -> bool:
    """Spatial position (1-D linearised, x or y). neuro-align's kinds: position, position_x, position_y."""
    kind = str((meta or {}).get("kind", "")).lower()
    if kind:
        if kind in ("position", "position_x", "position_y", "linear_position"):
            return True
        if kind not in ("continuous", "other", ""):
            return False
    if name.lower().startswith("wheel"):
        return False
    if re.search(r"speed|velocity|vel|accel|head|angle|direction", name, re.I):
        return False
    return bool(POSITION_RE.search(name))


# left out unless asked for by name
AUTO_SKIP_KINDS = {"wheel_position": "cumulative wheel angle, not a behavioural state (drifts over the session); "
                                     "wheel_velocity is decoded instead"}


def continuous_variables(s: Session, wanted: list[str] | None, min_valid: float = 0.5) -> tuple[list[int], dict[str, str]]:
    """Columns of Y to use and the reasons others were left out."""
    keep, why = [], {}
    for j, n in enumerate(s.behavior_names):
        if wanted and n not in wanted:
            continue
        kind = str(s.variables.get(n, {}).get("kind", "continuous")).lower()
        if kind in ("categorical", "discrete", "binary", "event", "direction"):
            why[n] = f"kind {kind}"
            continue
        if not wanted and kind in AUTO_SKIP_KINDS:
            why[n] = AUTO_SKIP_KINDS[kind]
            continue
        y = s.Y[:, j]
        m = np.isfinite(y) & s.epoch_mask
        if m.mean() < min_valid * s.epoch_mask.mean():
            why[n] = f"only {m.mean():.0%} of bins valid"
            continue
        if np.nanstd(y[m]) <= 0 or np.unique(y[m][: 5000]).size < 3:
            why[n] = "fewer than 3 distinct values"
            continue
        keep.append(j)
    if wanted:
        for n in wanted:
            if n not in s.behavior_names and n not in why:
                why[n] = "not in this session"
    return keep, why
