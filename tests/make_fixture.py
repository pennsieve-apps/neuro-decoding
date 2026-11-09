"""
Synthetic neuro-align `aligned` output with known structure (numpy, pandas and pyarrow only, so the
platform's review generator image can run it).

Three sessions, written exactly as neuro-align writes its `aligned` port
(workflow-systems-neuroscience docs/conventions.md, "Decoding"):

  neuro-align/<session_id>/aligned.npz    t, X, Y, neuron_ids, neuron_meta, behavior_names, behavior_units,
                                          behavior_kinds, epoch_mask, bin_size, session_id, neural_kind
  neuro-align/<session_id>/spikes.npz     times, offsets, neuron_ids
  neuro-align/<session_id>/events.parquet table, start_time, stop_time, <columns>
  neuro-align/<session_id>/manifest.json  bin size, variables (unit, source, kind), neural source, ...
  neuro-align/sessions.csv                session_id, path, subject_id, condition, group, status, n_*

- `sub-rat1_ses-track1`, `sub-rat2_ses-track1`: CA1-like place cells on a 160 cm linear track,
  back-and-forth laps; direction-selective Gaussian place fields; Y = position (cm), speed (cm/s).
  The position is decodable (R^2 > 0.8) and the population lies on a ring/line manifold.
- `sub-mouse1_ses-ibl1`: an IBL-like task, 140 trials; VISp units respond to the contralateral
  (here: right) stimulus, MOs units to the choice and the wheel velocity, CA1 units are unrelated.
  Y = wheel_velocity (rad/s); events.parquet has the trials table (choice, contrast_left/right,
  stim_on_time, feedback_type, probability_left).

Layouts: `linked/aligned/neuro-align/...` (the port in its own folder, placement: subfolder) and
`flat/neuro-align/...` (merged at the root). With --embedding, a neuro-embedding-style
`embedding/neuro-embedding/<session_id>/<session_id>_embeddings.npz` is added (a noisy planted
3-D embedding), for neuro-decoding's optional `embedding` port.

    python -m tests.make_fixture OUT [--layout linked|flat|both] [--embedding] [--short]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

BIN = 0.025


def _poisson_spikes(rng, rate, t0, dt):
    """Spike times from a piecewise-constant rate (Hz) per bin starting at t0."""
    counts = rng.poisson(rate * dt)
    idx = np.repeat(np.arange(rate.size), counts)
    return t0 + (idx + rng.random(idx.size)) * dt, counts


def _bin_edges(t_start, t_stop):
    n = int(round((t_stop - t_start) / BIN))
    edges = t_start + np.arange(n + 1) * BIN
    return edges, edges[:-1] + BIN / 2


def track_session(seed: int, n_place: int, n_noise: int, duration: float, t_start: float, run_epoch: bool):
    rng = np.random.default_rng(seed)
    edges, t = _bin_edges(t_start, t_start + duration)
    L = 160.0
    # position: back and forth, each lap at its own speed (15-40 cm/s) with a pause of 0.5-4 s at the
    # ends, so laps are irregular as in real data (a periodic trace would defeat the circular-shift null)
    pos = np.empty(t.size)
    p, direction, wait = 0.0, 1, 0.0
    v_lap = rng.uniform(15, 40)
    speed_noise = 1 + 0.15 * np.sin(np.arange(t.size) * BIN * 0.7 + rng.random() * 6)
    for i in range(t.size):
        if wait > 0:
            wait -= BIN
        else:
            p += direction * v_lap * speed_noise[i] * BIN
            if p >= L or p <= 0:
                p = min(max(p, 0.0), L)
                direction *= -1
                wait = rng.uniform(0.5, 4.0)
                v_lap = rng.uniform(15, 40)
        pos[i] = p
    vel = np.gradient(pos, BIN)
    speed = np.abs(vel)
    moving = speed > 2.0
    rightward = (vel > 2.0).astype(float)
    leftward = (vel < -2.0).astype(float)
    centers = rng.uniform(5, L - 5, n_place)
    widths = rng.uniform(8, 14, n_place)
    peaks = rng.uniform(8, 20, n_place)
    dir_pref = rng.integers(0, 2, n_place)   # 0 rightward, 1 leftward
    rates = np.empty((t.size, n_place + n_noise))
    for j in range(n_place):
        field = np.exp(-0.5 * ((pos - centers[j]) / widths[j]) ** 2)
        gate = rightward if dir_pref[j] == 0 else leftward
        rates[:, j] = 0.2 + peaks[j] * field * (0.15 + 0.85 * gate)
    for j in range(n_noise):
        rates[:, n_place + j] = rng.uniform(0.5, 4) * (1 + 0.3 * np.sin(t * rng.uniform(0.05, 0.3)))
    ids, spikes, X = [], [], np.empty_like(rates, dtype=np.float32)
    for j in range(rates.shape[1]):
        st, c = _poisson_spikes(rng, rates[:, j], t_start, BIN)
        spikes.append(st)
        X[:, j] = c
        ids.append(f"u{j:03d}")
    Y = np.column_stack([pos, speed]).astype(np.float32)
    meta = _meta(["CA1"] * rates.shape[1], rng)
    epoch = moving if run_epoch else np.ones(t.size, bool)
    # laps as TimeIntervals (a run in one direction)
    starts = np.flatnonzero(np.diff(np.r_[0, moving.astype(int)]) == 1)
    stops = np.flatnonzero(np.diff(np.r_[moving.astype(int), 0]) == -1)
    laps = pd.DataFrame({"table": "laps", "start_time": edges[starts], "stop_time": edges[stops + 1],
                         "direction": np.where(vel[starts + 2] > 0, "right", "left")})
    variables = [{"name": "position", "unit": "cm", "source": "processing/behavior/Position/linear", "kind": "continuous"},
                 {"name": "speed", "unit": "cm/s", "source": "derived from position", "kind": "continuous"}]
    truth = {"place_cells": n_place, "noise_cells": n_noise, "place_field_centers": centers.round(2).tolist(),
             "direction_preference": dir_pref.tolist()}
    return dict(t=t, X=X, Y=Y, ids=ids, meta=meta, names=["position", "speed"], epoch=epoch, events=laps,
                spikes=spikes, variables=variables, truth=truth, epoch_name="run" if run_epoch else "all")


def ibl_session(seed: int, n_trials: int, t_start: float):
    rng = np.random.default_rng(seed)
    trial_len = rng.uniform(2.0, 3.0, n_trials)
    starts = t_start + 1.0 + np.r_[0, np.cumsum(trial_len + 0.5)[:-1]]
    stops = starts + trial_len
    duration = stops[-1] + 1.0 - t_start
    edges, t = _bin_edges(t_start, t_start + duration)
    stim_on = starts + 0.4
    side = rng.choice([-1, 1], n_trials)               # -1 left, +1 right
    contrast = rng.choice([0.0, 0.0625, 0.125, 0.25, 1.0], n_trials)
    correct = rng.random(n_trials) < np.where(contrast > 0.1, 0.9, 0.6)
    choice = np.where(correct, side, -side)            # IBL: -1 / +1 (we keep the side convention)
    prob_left = np.repeat(rng.choice([0.2, 0.8], n_trials // 20 + 1), 20)[:n_trials]
    contrast_left = np.where(side < 0, contrast, np.nan)
    contrast_right = np.where(side > 0, contrast, np.nan)
    move_on = stim_on + rng.uniform(0.15, 0.35, n_trials)
    wheel = np.zeros(t.size)
    stim_drive = np.zeros(t.size)
    choice_drive = np.zeros(t.size)
    for k in range(n_trials):
        m = (t >= move_on[k]) & (t < move_on[k] + 0.6)
        tau = (t[m] - move_on[k]) / 0.6
        wheel[m] += choice[k] * 4.0 * np.sin(np.pi * tau)
        s = (t >= stim_on[k] + 0.05) & (t < stim_on[k] + 0.35)
        stim_drive[s] += side[k] * (0.3 + contrast[k])
        c = (t >= move_on[k] - 0.1) & (t < move_on[k] + 0.6)
        choice_drive[c] += choice[k]
    wheel += 0.2 * rng.standard_normal(t.size)
    regions = ["VISp"] * 30 + ["MOs"] * 30 + ["CA1"] * 20
    # a slow latent shared by VISp and MOs (not CA1): a one-dimensional "communication subspace"
    shared = np.zeros(t.size)
    for i in range(1, t.size):
        shared[i] = shared[i - 1] * np.exp(-BIN / 0.5) + np.sqrt(1 - np.exp(-2 * BIN / 0.5)) * rng.standard_normal()
    rates = np.empty((t.size, len(regions)))
    for j, reg in enumerate(regions):
        base = rng.uniform(1, 6) * (np.exp(0.6 * shared * (1 if j % 2 else -1)) if reg in ("VISp", "MOs") else 1.0)
        if reg == "VISp":
            pref = 1 if j % 3 else -1                  # mostly right-stimulus (contralateral) units
            rates[:, j] = base * np.exp(0.9 * pref * stim_drive)
        elif reg == "MOs":
            w = rng.uniform(0.15, 0.35) * (1 if j % 2 else -1)
            pref = 1 if j % 4 < 2 else -1
            rates[:, j] = base * np.exp(np.clip(w * wheel + 0.5 * pref * choice_drive, -3, 3))
        else:
            rates[:, j] = base * (1 + 0.3 * np.sin(t * rng.uniform(0.05, 0.3)))
    ids, spikes, X = [], [], np.empty(rates.shape, np.float32)
    for j in range(rates.shape[1]):
        st, c = _poisson_spikes(rng, rates[:, j], t_start, BIN)
        spikes.append(st)
        X[:, j] = c
        ids.append(f"{j}")
    trials = pd.DataFrame({
        "table": "trials", "start_time": starts, "stop_time": stops, "stim_on_time": stim_on,
        "first_movement_time": move_on, "choice": choice.astype(float), "contrast_left": contrast_left,
        "contrast_right": contrast_right, "feedback_type": np.where(correct, 1.0, -1.0),
        "probability_left": prob_left})
    variables = [{"name": "wheel_velocity", "unit": "rad/s", "source": "processing/behavior/WheelVelocity", "kind": "continuous"}]
    truth = {"stimulus_units": "VISp", "choice_and_wheel_units": "MOs", "unrelated_units": "CA1",
             "shared_latent": "VISp and MOs share one slow latent (tau 0.5 s)"}
    return dict(t=t, X=X, Y=wheel[:, None].astype(np.float32), ids=ids, meta=_meta(regions, rng),
                names=["wheel_velocity"], epoch=np.ones(t.size, bool), events=trials, spikes=spikes,
                variables=variables, truth=truth, epoch_name="all")


def _meta(regions, rng):
    n = len(regions)
    meta = np.zeros(n, dtype=[("id", "U64"), ("region", "U64"), ("label", "U32"), ("depth", "f8"), ("x", "f8"),
                              ("y", "f8"), ("rate_hz", "f8")])
    meta["region"] = regions
    meta["label"] = "good"
    meta["depth"] = np.sort(rng.uniform(200, 3800, n))
    meta["x"] = rng.choice([11.0, 27.0, 43.0, 59.0], n)
    meta["y"] = meta["depth"]
    return meta


KINDS = {"position": "position", "speed": "speed", "wheel_velocity": "wheel_velocity"}


def write_session(root: Path, sid: str, s: dict, subject: str, condition: str, group: str):
    """As neuro-align 0.1.0 writes it (processor/main.py, "write")."""
    d = root / "neuro-align" / sid
    d.mkdir(parents=True, exist_ok=True)
    meta = s["meta"]
    meta["id"] = s["ids"]
    dur = s["t"].size * BIN
    meta["rate_hz"] = s["X"].sum(axis=0) / dur
    units = [v["unit"] for v in s["variables"]]
    kinds = [KINDS.get(n, "other") for n in s["names"]]
    np.savez_compressed(d / "aligned.npz", t=s["t"], X=s["X"], Y=s["Y"].astype(np.float32),
                        neuron_ids=np.array(s["ids"], dtype="U64"), neuron_meta=meta,
                        behavior_names=np.array(s["names"], dtype="U64"), behavior_units=np.array(units, dtype="U64"),
                        behavior_kinds=np.array(kinds, dtype="U32"), epoch_mask=s["epoch"],
                        bin_size=np.float64(BIN), session_id=np.array(sid), neural_kind=np.array("units"))
    lens = np.array([len(x) for x in s["spikes"]], dtype=np.int64)
    np.savez_compressed(d / "spikes.npz", times=np.concatenate(s["spikes"]), offsets=np.r_[0, np.cumsum(lens)].astype(np.int64),
                        neuron_ids=np.array(s["ids"], dtype="U64"))
    s["events"].to_parquet(d / "events.parquet", index=False)
    regions: dict = {}
    for r in meta["region"]:
        regions[str(r)] = regions.get(str(r), 0) + 1
    variables = [{**v, "kind": k, "circular": False, "note": "",
                  "range": [float(np.nanmin(s["Y"][:, i])), float(np.nanmax(s["Y"][:, i]))], "fraction_valid": 1.0}
                 for i, (v, k) in enumerate(zip(s["variables"], kinds))]
    manifest = {
        "session_id": sid, "app": "neuro-align", "version": "0.1.0", "bin_size_s": BIN, "n_bins": int(s["t"].size),
        "t_start": float(s["t"][0] - BIN / 2), "t_stop": float(s["t"][-1] + BIN / 2),
        "time": "seconds on the session clock of the neural file; t = bin centres",
        "neural": {"kind": "units", "source": "units", "signal": "spike_times", "X": "spike counts per bin",
                   "files": [f"{sid}.nwb"], "n_neurons": len(s["ids"]), "regions": regions,
                   "filters": {"min_rate_hz": 0.1, "brain_areas": None}, "dropped": {}, "frame_period_s": None},
        "behavior": {"files": [f"{sid}.nwb"], "variables": variables,
                     "linear_track": {"source": "1D position in the file"} if "position" in s["names"] else None},
        "epoch": {"name": s["epoch_name"], "n_bins": int(s["epoch"].sum())},
        "events": {"tables": sorted(s["events"]["table"].unique().tolist()), "n_rows": int(len(s["events"]))},
        "sheet": {"session_id": sid, "subject_id": subject, "condition": condition, "group": group},
        "files": {"aligned": "aligned.npz", "spikes": "spikes.npz", "events": "events.parquet"},
        "synthetic_truth": s["truth"], "warnings": [],
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"session_id": sid, "path": f"{sid}.nwb", "subject_id": subject, "condition": condition,
            "group": group, "status": "ok", "n_neurons": len(s["ids"]), "n_bins": int(s["t"].size)}


def planted_embedding(root: Path, sid: str, s: dict, seed: int):
    """A neuro-embedding-style embeddings file: a 3-D ring of position plus noise (fixture only)."""
    rng = np.random.default_rng(seed)
    y = s["Y"][:, 0].astype(float)
    ang = np.pi * (y - y.min()) / max(np.ptp(y), 1e-9)
    vel = np.gradient(y)
    z = np.sign(vel) * 0.5
    emb = np.column_stack([np.cos(ang), np.sin(ang), z]) + 0.15 * rng.standard_normal((y.size, 3))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    n = y.size
    train = np.zeros(n, bool)
    train[: int(0.8 * n)] = True
    d = root / "neuro-embedding" / sid
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"{sid}_embeddings.npz", session_id=np.array(sid), t=s["t"],
                        cebra_time=emb.astype(np.float32) + 0.05 * rng.standard_normal(emb.shape).astype(np.float32),
                        cebra_behavior=emb.astype(np.float32), cebra_behavior_train_mask=train,
                        label_names=np.array(["position"]), epoch_mask=s["epoch"])


def write(out: Path, layout: str = "both", embedding: bool = False, short: bool = False) -> None:
    out = Path(out)
    dur = 240.0 if short else 400.0
    sessions = [
        ("sub-rat1_ses-track1", track_session(1, 50, 15, dur, 12.0, run_epoch=False), "rat1", "track", "hippocampus"),
        ("sub-rat2_ses-track1", track_session(2, 45, 15, dur * 0.8, 3.0, run_epoch=True), "rat2", "track", "hippocampus"),
        ("sub-mouse1_ses-ibl1", ibl_session(3, 100 if short else 140, 30.0), "mouse1", "task", "ibl"),
    ]
    layouts = ["linked", "flat"] if layout == "both" else [layout]
    for lay in layouts:
        base = out / lay
        if base.exists():
            shutil.rmtree(base)
        root = base / "aligned" if lay == "linked" else base
        rows = [write_session(root, sid, s, subj, cond, grp) for sid, s, subj, cond, grp in sessions]
        pd.DataFrame(rows).to_csv(root / "neuro-align" / "sessions.csv", index=False)
        if embedding:
            eroot = base / "embedding" if lay == "linked" else base
            for k, (sid, s, *_r) in enumerate(sessions[:2]):
                planted_embedding(eroot, sid, s, 10 + k)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out")
    ap.add_argument("--layout", default="both", choices=["linked", "flat", "both"])
    ap.add_argument("--embedding", action="store_true", help="also write a planted neuro-embedding output")
    ap.add_argument("--short", action="store_true", help="shorter sessions (tests)")
    a = ap.parse_args()
    write(Path(a.out), a.layout, a.embedding, a.short)
    print(f"wrote {a.out}")
