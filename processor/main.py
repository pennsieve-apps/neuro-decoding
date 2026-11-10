"""
neuro-decoding: cross-validated decoding of behaviour from population activity.

Reads neuro-align's `aligned` output (and optionally neuro-embedding's `embedding` output) and
writes, under $OUTPUT_DIR/neuro-decoding/:

  decoding.csv                       one row per session x variable x decoder x region (+ nulls)
  dropping.csv                       neuron-dropping curves
  heldout.csv, heldout_<var>.png     held-out-session decoding (>= 2 sessions sharing a variable)
  <session_id>/<session_id>_decoded_traces.png   decoded vs true (the headline figure)
  <session_id>/<session_id>_null.png             score vs null, per variable and region
  <session_id>/<session_id>_dropping.png         neuron-dropping curve
  <session_id>/summary.json, summary.json, provenance.json
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path

from processor.common import Heartbeat, app_version, peak_rss_mb, rel, setup_logging, tool_versions, write_csv, write_json

log = setup_logging("neuro-decoding")

APP = "neuro-decoding"
DECODERS = ("ridge", "mlp", "bayes", "logistic", "knn", "rrr")
DEFAULTS = {
    "VARIABLES": "auto",
    "TRIAL_VARIABLES": "auto",
    "TRIAL_ALIGN": "auto",
    "TRIAL_WINDOW_S": "-0.1,0.5",
    "TRIAL_WINDOWS": "auto",
    "DECODE_BIN_S": "0.1",
    "SMOOTH_S": "0.05",
    "HISTORY_BINS": "10",
    "CV_FOLDS": "5",
    "CV_GAP_S": "2.0",
    "N_SHUFFLES": "200",
    "MIN_SHIFT_S": "20",
    "DECODERS": "ridge,mlp,bayes,logistic,knn,rrr",
    "N_POS_BINS": "40",
    "DROP_REPEATS": "10",
    "MIN_REGION_NEURONS": "10",
    "MIN_UNITS_PER_REGION": "10",
    "RRR_MAX_RANK": "10",
    "HELDOUT_DIMS": "10",
    "KNN_K": "25",
    "TRACE_WINDOW_S": "60",
    "SEED": "42",
    "THREADS": "0",
}
PARAMETERS = tuple(DEFAULTS)
CSV_COLUMNS = ["session_id", "subject_id", "condition", "group", "variable", "kind", "unit", "decoder", "features",
               "region", "n_neurons", "n_samples", "cv", "metric", "score", "score_sd", "r", "median_abs_error",
               "chance", "null", "score_corrected", "null_mean", "null_sd", "null_p95", "p_value", "z", "n_shuffles"]
RRR_DETREND_S = 60.0   # communication subspaces: running mean removed before the regression
BAYES_WINDOW_S = 0.5   # spike counts of the Bayesian decoder are summed over this centred window
CITATIONS = ["10.1523/ENEURO.0506-19.2020", "10.1152/jn.1998.79.2.1017", "10.1038/s41593-019-0555-4",
             "10.1038/s41586-025-09226-1", "10.1038/s41586-025-09235-0", "10.1016/j.neuron.2019.01.026",
             "10.1038/s41586-023-06031-6"]

# IBL's per-variable windows (Findling et al. 2025; IBL 2025): (role, name pattern, align candidates, window)
IBL_WINDOWS = (
    ("prior", r"probability_?left|block|prior", ("gabor_stimulus_onset_time", "stim_on_time", "stimOn_times",
                                                  "stimulus_on_time", "go_cue_time", "start_time"), (-0.6, -0.1)),
    ("choice", r"choice", ("first_movement_time", "firstMovement_times", "wheel_movement_onset_time"), (-0.1, 0.0)),
    ("feedback", r"feedback|reward|correct", ("feedback_time", "feedback_times", "feedbackTime"), (0.0, 0.2)),
    ("stimulus", r"side|contrast", ("gabor_stimulus_onset_time", "stim_on_time", "stimOn_times",
                                    "stimulus_on_time"), (0.0, 0.1)),
)


class ParameterError(ValueError):
    pass


def _list(v: str):
    v = v.strip()
    if v.lower() == "auto":
        return "auto"
    if v.lower() in ("none", "off", "false"):
        return []
    return [x.strip() for x in v.split(",") if x.strip()]


def read_parameters(env=os.environ) -> dict:
    raw = {k: (env.get(k) if env.get(k) not in (None, "") else v) for k, v in DEFAULTS.items()}
    p: dict = {"VARIABLES": _list(raw["VARIABLES"]), "TRIAL_VARIABLES": _list(raw["TRIAL_VARIABLES"]),
               "TRIAL_ALIGN": raw["TRIAL_ALIGN"].strip()}
    dec = [d.strip().lower() for d in raw["DECODERS"].split(",") if d.strip()]
    bad = [d for d in dec if d not in DECODERS]
    if bad:
        raise ParameterError(f"DECODERS must be a subset of {', '.join(DECODERS)}, got {', '.join(bad)}")
    p["DECODERS"] = sorted(set(dec) | {"ridge"}, key=DECODERS.index)
    try:
        for k in ("CV_FOLDS", "HISTORY_BINS", "N_SHUFFLES", "N_POS_BINS", "DROP_REPEATS", "MIN_REGION_NEURONS", "MIN_UNITS_PER_REGION", "RRR_MAX_RANK", "HELDOUT_DIMS",
                  "KNN_K", "SEED", "THREADS"):
            p[k] = int(raw[k])
        for k in ("DECODE_BIN_S", "SMOOTH_S", "CV_GAP_S", "MIN_SHIFT_S", "TRACE_WINDOW_S"):
            p[k] = float(raw[k])
        w = [float(x) for x in raw["TRIAL_WINDOW_S"].split(",")]
    except ValueError as exc:
        raise ParameterError(f"bad numeric parameter: {exc}") from exc
    if len(w) != 2 or w[1] <= w[0]:
        raise ParameterError("TRIAL_WINDOW_S must be 'start,stop' (s, relative to TRIAL_ALIGN), stop > start")
    p["TRIAL_WINDOW_S"] = w
    tw = raw["TRIAL_WINDOWS"].strip()
    if tw.lower() in ("auto", "off", "none"):
        p["TRIAL_WINDOWS"] = tw.lower()
    else:
        spec = {}
        for item in tw.split(";"):
            if not item.strip():
                continue
            try:
                var, rest = item.split("=", 1)
                col, a, b = rest.rsplit(":", 2)
                spec[var.strip()] = (col.strip(), float(a), float(b))
            except ValueError as exc:
                raise ParameterError(f"TRIAL_WINDOWS: 'auto', 'off' or 'variable=column:start:stop;...', got {item!r}") from exc
            if spec[var.strip()][2] <= spec[var.strip()][1]:
                raise ParameterError(f"TRIAL_WINDOWS: stop must be after start in {item!r}")
        p["TRIAL_WINDOWS"] = spec
    if p["MIN_UNITS_PER_REGION"] < 4 or p["RRR_MAX_RANK"] < 1:
        raise ParameterError("MIN_UNITS_PER_REGION >= 4 and RRR_MAX_RANK >= 1 required")
    if p["HISTORY_BINS"] < 1 or p["HISTORY_BINS"] > 100:
        raise ParameterError("HISTORY_BINS must be in 1..100")
    if p["CV_FOLDS"] < 2 or p["N_SHUFFLES"] < 0 or p["N_POS_BINS"] < 5 or p["DROP_REPEATS"] < 1 \
            or p["MIN_REGION_NEURONS"] < 2 or p["HELDOUT_DIMS"] < 1 or p["KNN_K"] < 1 or p["THREADS"] < 0:
        raise ParameterError("CV_FOLDS >= 2, N_SHUFFLES >= 0, N_POS_BINS >= 5, DROP_REPEATS >= 1, "
                             "MIN_REGION_NEURONS >= 2, HELDOUT_DIMS >= 1, KNN_K >= 1, THREADS >= 0 required")
    if min(p["DECODE_BIN_S"], p["SMOOTH_S"], p["CV_GAP_S"], p["MIN_SHIFT_S"]) < 0 or p["TRACE_WINDOW_S"] <= 0:
        raise ParameterError("DECODE_BIN_S, SMOOTH_S, CV_GAP_S, MIN_SHIFT_S must be >= 0 and TRACE_WINDOW_S > 0")
    return p


def features(s, smooth_s: float):
    """Smoothed activity (Gaussian, sigma SMOOTH_S); sqrt of counts for spiking data."""
    import numpy as np
    from scipy.ndimage import gaussian_filter1d

    F = np.sqrt(s.X) if s.is_counts else s.X
    sig = smooth_s / s.bin_size
    return gaussian_filter1d(F.astype(np.float64), sig, axis=0, mode="nearest") if sig > 0 else F.astype(np.float64)


def find_embeddings(input_dir: Path, A) -> dict:
    """session_id -> path of neuro-embedding's <session_id>_embeddings.npz (optional `embedding` port)."""
    import numpy as np

    root = input_dir / "embedding" if (input_dir / "embedding").is_dir() else input_dir
    out = {}
    for pth in A.walk(root):
        if pth.name.endswith("_embeddings.npz"):
            sid = pth.name[: -len("_embeddings.npz")]
            try:
                with np.load(pth, allow_pickle=False) as z:
                    if "session_id" in z.files:
                        sid = str(z["session_id"])
            except Exception:
                continue
            out[A.sanitise(sid)] = pth
    return out


def direction_of(y, dt):
    import numpy as np
    from scipy.ndimage import gaussian_filter1d

    v = np.gradient(gaussian_filter1d(y, max(1.0, 0.2 / dt)), dt)
    thr = 0.1 * np.percentile(np.abs(v), 95)
    return np.where(v > thr, 1, np.where(v < -thr, -1, 0))


def trial_window(var: str, trials, p: dict, default_align: str):
    """(align column, (start, stop), role) for a trial variable. TRIAL_WINDOWS=auto uses IBL's
    windows by role when the trials table has the event column (prior: -0.6 to -0.1 s before the
    stimulus; stimulus: 0 to 0.1 s after it; choice: -0.1 to 0 s before the first movement;
    feedback: 0 to 0.2 s after it), else TRIAL_ALIGN + TRIAL_WINDOW_S; 'off' always uses the
    latter; a mapping 'var=column:start:stop;...' overrides per variable."""
    import re

    default = (default_align, tuple(p["TRIAL_WINDOW_S"]), "default")
    tw = p["TRIAL_WINDOWS"]
    if isinstance(tw, dict):
        if var in tw:
            col, a, b = tw[var]
            if col not in trials.columns:
                raise ValueError(f"TRIAL_WINDOWS: column {col!r} not in the trials table")
            return col, (a, b), "TRIAL_WINDOWS"
        return default
    if tw == "off":
        return default
    for role, pattern, cols, win in IBL_WINDOWS:
        if re.search(pattern, var, re.I):
            col = next((c for c in cols if c in trials.columns and trials[c].notna().any()), None)
            if col:
                return col, win, role
            return default
    return default


def drop_duplicate_labels(tvars: dict) -> tuple[dict, dict]:
    """Trial variables whose labels are a one-to-one relabelling of an earlier one (IBL
    reward_volume_uL vs is_mouse_rewarded) are dropped: decoding them twice adds nothing."""
    import numpy as np

    keep, dups = {}, {}
    for name, lab in tvars.items():
        twin = None
        for other, olab in keep.items():
            m = np.array([a is not None and b is not None for a, b in zip(lab, olab)])
            if m.sum() < 10 or m.sum() < 0.9 * sum(a is not None for a in lab):
                continue
            pairs = set(zip(np.asarray(lab)[m], np.asarray(olab)[m]))
            if len(pairs) == len({a for a, _ in pairs}) == len({b for _, b in pairs}):
                twin = other
                break
        if twin:
            dups[name] = twin
        else:
            keep[name] = lab
    return keep, dups


def lag_groups(H: int) -> list[list[int]]:
    """History lags 0..H-1 grouped on a log scale: [0], [1], [2, 3], [4..7], [8..15], ... (the last group
    cut at H-1). Recent bins keep their own weight, older ones are averaged, which keeps the number
    of features (neurons x groups) small, like a raised-cosine history basis."""
    groups, lo = [], 0
    while lo < H:
        hi = max(lo + 1, 2 * lo) if lo else 1
        groups.append(list(range(lo, min(hi, H))))
        lo = hi
    return groups


def lagged(F, H: int):
    """For every history group, the mean of F over its lags (past bins; the first bins repeat row 0)."""
    import numpy as np

    F = F.astype(np.float32)
    if H <= 1:
        return F
    groups = lag_groups(H)
    out = np.empty((F.shape[0], F.shape[1] * len(groups)), np.float32)
    shifted = {}
    for lag in range(H):
        sh = np.empty_like(F)
        sh[lag:] = F[: F.shape[0] - lag] if lag else F
        sh[:lag] = F[0]
        shifted[lag] = sh
    for g, lags in enumerate(groups):
        out[:, g * F.shape[1]:(g + 1) * F.shape[1]] = np.mean([shifted[l] for l in lags], axis=0)
    return out


def run_session(s, p, emb_path, out_dir: Path, A, D, F, hb):
    from processor import pseudo as P
    from processor import rrr as R

    rrr_fig = None
    import numpy as np

    sid = s.session_id
    rng = np.random.default_rng(p["SEED"])
    sheet = {k: s.sheet.get(k) for k in A.SHEET_KEYS}
    raw = s
    s = A.rebin(s, p["DECODE_BIN_S"]) if p["DECODE_BIN_S"] > 0 else s
    dt = s.bin_size
    rows, drops, figs, nulls, trace_panels = [], [], [], [], []
    heldout_data = {}
    res = {"session_id": sid, "n_neurons": int(s.X.shape[1]), "decode_bin_s": dt, "neural_source": s.neural_source,
           **{k: v for k, v in sheet.items() if v is not None}, "variables": {}}
    regions = None
    if s.regions:
        names, counts = np.unique(np.array(s.regions), return_counts=True)
        regions = [(r, np.flatnonzero(np.array(s.regions) == r)) for r, c in sorted(zip(names, counts), key=lambda x: -x[1])
                   if c >= p["MIN_REGION_NEURONS"] and r.lower() not in ("unknown", "root", "void", "")]
        res["regions"] = {r: int(len(i)) for r, i in regions}
        if len(regions) < 2 and not (len(regions) == 1 and len(names) > 1):
            regions = regions if len(names) > 1 else None
    Fbase = features(s, p["SMOOTH_S"])
    # the gap between training and test blocks must cover the history window, or the features of
    # the first test bins overlap the last training bins
    hist_s = p["HISTORY_BINS"] * s.bin_size
    gap_s = max(p["CV_GAP_S"], hist_s)
    if gap_s > p["CV_GAP_S"]:
        log.warning("%s: CV_GAP_S %.2f s is shorter than the history window (%d bins x %.3g s); using %.2f s",
                    sid, p["CV_GAP_S"], p["HISTORY_BINS"], s.bin_size, gap_s)
    gap_bins = int(np.ceil(gap_s / s.bin_size))
    res["cv_gap_s"] = gap_s
    H = p["HISTORY_BINS"]
    Fall = lagged(Fbase, H)
    n_neu = Fbase.shape[1]

    n_groups = len(lag_groups(H)) if H > 1 else 1

    def lag_cols(neurons):   # the history columns of a set of neurons
        neurons = np.asarray(neurons)
        return np.concatenate([g * n_neu + neurons for g in range(n_groups)])
    ep = s.epoch_mask

    def row(var, kind, unit, decoder, region, n_neu, n_samp, cv, metric, res_, chance=None, null=None, nst=None,
            feats="activity"):
        r = {"session_id": sid, **sheet, "variable": var, "kind": kind, "unit": unit, "decoder": decoder,
             "features": feats, "region": region, "n_neurons": n_neu, "n_samples": n_samp, "cv": cv, "metric": metric,
             "score": res_.get("score"), "score_sd": res_.get("score_sd"), "r": res_.get("r"),
             "median_abs_error": res_.get("median_abs_error"), "chance": chance, "null": null}
        r.update({k: v for k, v in (nst or {}).items() if k in CSV_COLUMNS})
        if r.get("score") is not None and r.get("null_mean") is not None:
            # IBL's corrected score (Findling et al. 2025): the score minus the MEDIAN of the null. Our
            # own addition for R²: a negative null median (shifted behaviour predicts worse than its
            # mean) is not credited, i.e. R² minus max(null median, 0).
            med = (nst or {}).get("null_median", r["null_mean"])
            base = max(med, 0.0) if metric == "r2" else med
            r["score_corrected"] = r["score"] - base
        rows.append(r)
        return r

    # --- continuous variables ---
    cols, why = A.continuous_variables(s, None if p["VARIABLES"] == "auto" else p["VARIABLES"])
    if why:
        res["variables_skipped"] = why
    emb = None
    if emb_path is not None:
        emb = np.load(emb_path, allow_pickle=False)
    for j in cols:
        var = s.behavior_names[j]
        unit = s.unit(var)
        y_all = s.Y[:, j].astype(np.float64)
        m = ep & np.isfinite(y_all)
        idx = np.flatnonzero(m)
        if idx.size < 50:
            res["variables"][var] = {"status": "skipped", "reason": f"{idx.size} valid bins"}
            continue
        y, Fv, tv = y_all[idx], Fall[idx], s.t[idx]
        folds = D.blocked_folds(tv, p["CV_FOLDS"], gap_s)
        if len(folds) < 2:
            res["variables"][var] = {"status": "skipped", "reason": "too short for the folds"}
            continue
        t0 = time.monotonic()
        rd = D.ridge_cv(Fv, y, folds, gap=gap_bins)
        real_fixed, null = (D.circular_test(Fv, y, tv, folds, rd["alpha"], p["N_SHUFFLES"], p["MIN_SHIFT_S"], rng)
                            if p["N_SHUFFLES"] else (rd["score"], np.array([])))
        nst = D.null_stats(real_fixed, null)
        cv = f"blocked {len(folds)}-fold, gap {gap_s:g} s"
        row(var, "continuous", unit, "ridge", "all", s.X.shape[1], idx.size, cv, "r2", rd, None, "circular shift", nst,
            feats=f"activity, {H} bins of history")
        vres = {"ridge_r2": rd["score"], "ridge_r": rd["r"], "null_p95": nst["null_p95"], "p_value": nst["p_value"],
                "z": nst["z"], "median_abs_error": rd["median_abs_error"], "unit": unit}
        nulls.append({"var": var, "metric": "R²", "chance": None, "items": [{"label": "ridge, all", "score": rd["score"], **nst}]})
        preds = {"ridge": rd}
        if "mlp" in p["DECODERS"]:
            ml = D.mlp_cv(Fv, y, folds, p["SEED"])
            preds["mlp"] = ml
            row(var, "continuous", unit, "mlp", "all", s.X.shape[1], idx.size, cv, "r2", ml, feats=f"activity, {H} bins of history")
            vres["mlp_r2"] = ml["score"]
        if "bayes" in p["DECODERS"] and s.is_counts and A.is_position(var, s.variables.get(var)):
            k = max(1, int(round(BAYES_WINDOW_S / dt)))
            Cs = np.apply_along_axis(lambda c: np.convolve(c, np.ones(k), mode="same"), 0, s.X.astype(np.float64))
            by = D.bayes_cv(Cs[idx], y, folds, p["N_POS_BINS"], dt * k)
            preds["bayes"] = by
            row(var, "continuous", unit, "bayes", "all", s.X.shape[1], idx.size, cv, "r2", by)
            vres["bayes_r2"] = by["score"]
            vres["bayes_median_abs_error"] = by["median_abs_error"]
        # kNN on CEBRA embeddings (neuro-embedding output)
        if emb is not None and "knn" in p["DECODERS"]:
            et = emb["t"]
            near = np.clip(np.searchsorted(et, tv), 0, et.size - 1)
            if "cebra_time" in emb.files:
                kn = D.knn_cv(emb["cebra_time"][near], y, folds, p["KNN_K"])
                row(var, "continuous", unit, "knn", "all", s.X.shape[1], idx.size, cv, "r2", kn, feats="cebra_time")
                vres["knn_cebra_time_r2"] = kn["score"]
                preds["knn_cebra_time"] = kn
            if "cebra_behavior" in emb.files and "cebra_behavior_train_mask" in emb.files:
                trm = emb["cebra_behavior_train_mask"][near]
                tr, te = np.flatnonzero(trm), np.flatnonzero(~trm)
                if tr.size > p["KNN_K"] and te.size > 10:
                    kb = D.knn_cv(emb["cebra_behavior"][near], y, [(tr, te)], p["KNN_K"])
                    labs = ",".join(str(x) for x in emb["label_names"]) if "label_names" in emb.files else ""
                    row(var, "continuous", unit, "knn", "all", s.X.shape[1], idx.size,
                        "train/test split of neuro-embedding (CEBRA-Behavior trained on the train bins)", "r2", kb,
                        feats=f"cebra_behavior[{labs}]")
                    vres["knn_cebra_behavior_r2"] = kb["score"]
        # regions
        if regions:
            items = nulls[-1]["items"]
            for reg, ii in regions:
                ci = lag_cols(ii)
                rr = D.ridge_cv(Fv[:, ci], y, folds, gap=gap_bins)
                rf, nn = (D.circular_test(Fv[:, ci], y, tv, folds, rr["alpha"], min(p["N_SHUFFLES"], 100), p["MIN_SHIFT_S"],
                                          rng) if p["N_SHUFFLES"] else (rr["score"], np.array([])))
                ns = D.null_stats(rf, nn)
                row(var, "continuous", unit, "ridge", reg, int(len(ii)), idx.size, cv, "r2", rr, None, "circular shift", ns,
                    feats=f"activity, {H} bins of history")
                items.append({"label": f"ridge, {reg} ({len(ii)})", "score": rr["score"], **ns})
                vres.setdefault("regions", {})[reg] = {"r2": rr["score"], "p_value": ns["p_value"], "n_neurons": int(len(ii))}
        # neuron dropping
        drops.append({"name": var, "metric": "R²", "rows": D.dropping_curve(Fv, y, folds, "continuous", rd["alpha"],
                                                                             p["DROP_REPEATS"], rng, lag_cols)})
        # trace window: start of the middle fold
        tr_f = folds[len(folds) // 2][1]
        tt = tv[tr_f]
        w = np.arange(tt.size) < max(10, int(round(p["TRACE_WINDOW_S"] / dt)))   # TRACE_WINDOW_S of epoch time
        pl = [("ridge", preds["ridge"]["pred_folds"][len(folds) // 2][w], rd["score"])]
        best = max((k for k in preds if k != "ridge"), key=lambda k: preds[k]["score"], default=None)
        if best:
            pl.append((best.replace("_", " "), preds[best]["pred_folds"][len(folds) // 2][w], preds[best]["score"]))
        trace_panels.append({"name": var, "unit": unit, "t": tt[w], "true": y[tr_f][w], "preds": pl, "dt": dt,
                             "score": rd["score"], "position": A.is_position(var, s.variables.get(var))})
        decodable = (nst["p_value"] is not None and nst["p_value"] < 0.05) if p["N_SHUFFLES"] else rd["score"] > 0.1
        if decodable:   # held-out-session decoding only for variables this session decodes
            heldout_data[var] = (Fbase[idx], y, direction_of(y, dt) if A.is_position(var, s.variables.get(var)) else None)
        vres["seconds"] = round(time.monotonic() - t0, 1)
        res["variables"][var] = vres
        hb(f"{sid} {var}: ridge R2 {rd['score']:.3f} (null p95 {nst['null_p95'] if nst['null_p95'] is None else round(nst['null_p95'], 3)}), "
           f"{vres['seconds']}s", force=True)

    # --- communication subspaces between regions (reduced-rank regression) ---
    res["rrr_rows"] = []
    if "rrr" in p["DECODERS"]:
        rrr_regions = {}
        if s.regions:
            arr = np.array(s.regions)
            for r in sorted(set(s.regions)):
                ii = np.flatnonzero(arr == r)
                if ii.size >= p["MIN_UNITS_PER_REGION"] and r.lower() not in ("unknown", "root", "void", ""):
                    rrr_regions[r] = ii
        if len(rrr_regions) < 2:
            why_rrr = ("units carry no brain region" if not s.regions else
                       f"{len(rrr_regions)} region(s) with >= {p['MIN_UNITS_PER_REGION']} units")
            log.info("%s: communication subspaces skipped (%s)", sid, why_rrr)
            res["rrr"] = {"status": "skipped", "reason": why_rrr}
        else:
            t0 = time.monotonic()
            ep_idx = np.flatnonzero(ep)
            # slow drifts (arousal, units lost or found) are removed, as Semedo et al. used trial
            # residuals: each unit minus its running mean over RRR_DETREND_S, computed within each
            # fold's training and test segments separately (no smoothing across a fold boundary)
            w = max(1, int(round(RRR_DETREND_S / dt)))
            folds_r = D.blocked_folds(s.t[ep_idx], p["CV_FOLDS"], gap_s)
            rr_rows = R.communication(Fbase[ep_idx], rrr_regions, folds_r, np.random.default_rng(p["SEED"]),
                                      p["RRR_MAX_RANK"], detrend_bins=w)
            for r in rr_rows:
                r.update(session_id=sid, n_bins=int(ep_idx.size))
            res["rrr_rows"] = rr_rows
            res["rrr"] = {"regions": {k: int(v.size) for k, v in rrr_regions.items()},
                          "seconds": round(time.monotonic() - t0, 1),
                          "pairs": [{k: r[k] for k in ("source", "target", "kind", "full_r2", "optimal_rank", "within_r2")}
                                    for r in rr_rows]}
            rrr_fig = (rr_rows, list(rrr_regions))
            hb(f"{sid}: communication subspaces between {len(rrr_regions)} regions in {res['rrr']['seconds']}s", force=True)

    # --- categorical trial variables ---
    if p["TRIAL_VARIABLES"] != [] and "logistic" in p["DECODERS"]:
        trials = A.trial_table(A.read_events(raw))
        if trials is not None and len(trials) >= 2 * p["CV_FOLDS"]:
            tvars = A.trial_variables(trials, None if p["TRIAL_VARIABLES"] == "auto" else p["TRIAL_VARIABLES"])
            tvars, dups = drop_duplicate_labels(tvars)
            for dvar, other in dups.items():
                res["variables"][dvar] = {"status": "skipped", "reason": f"same trial labels as {other}"}
                log.info("%s: %s skipped (same trial labels as %s)", sid, dvar, other)
            al = A.align_column(trials, p["TRIAL_ALIGN"])
            feats_cache: dict = {}

            def trial_features(col, win):
                if (col, win) not in feats_cache:
                    Xt, ok_ = A.trial_counts(raw, trials, col, win)
                    feats_cache[(col, win)] = (np.sqrt(Xt) if raw.is_counts else Xt, ok_)
                return feats_cache[(col, win)]

            pseudo = P.ibl_task(trials, tvars)
            res["trials"] = {"n_trials": int(len(trials)), "default_align": al, "default_window_s": p["TRIAL_WINDOW_S"],
                             "null": "pseudo-sessions (IBL task model, action-kernel choices)" if pseudo is not None
                             else "trial-label shuffles", "windows": {}}
            for var, lab in tvars.items():
                wcol, win, role = trial_window(var, trials, p, al)
                res["trials"]["windows"][var] = {"align": wcol, "window_s": list(win), "role": role}
                Ft, ok = trial_features(wcol, win)
                keep = ok & np.array([v is not None for v in lab])
                levels, counts = np.unique(lab[keep].astype(str), return_counts=True)
                good = levels[counts >= max(5, p["CV_FOLDS"])]
                keep &= np.isin(lab.astype(str), good)
                if good.size < 2 or keep.sum() < 4 * p["CV_FOLDS"]:
                    res["variables"][var] = {"status": "skipped", "reason": f"{good.size} classes with enough trials"}
                    continue
                yk = lab[keep].astype(str)
                Fk = Ft[keep]
                folds = D.trial_folds(yk.size, p["CV_FOLDS"])
                if any(np.unique(yk[tr]).size < 2 for tr, _ in folds):
                    res["variables"][var] = {"status": "skipped", "reason": "a training fold has one class only"}
                    continue
                t0 = time.monotonic()
                lg = D.logistic_cv(Fk, yk, folds, p["SEED"])
                chance = 1.0 / good.size
                gen = P.pseudo_labels(trials, var, tvars, keep) if pseudo is not None else None
                null_kind = "pseudo-session" if gen is not None else "trial-label shuffle"
                if p["N_SHUFFLES"] == 0:
                    null = np.array([])
                elif gen is not None:
                    null = D.pseudo_null(Fk, yk, folds, lg["C"], [gen(rng) for _ in range(p["N_SHUFFLES"])])
                else:
                    null = D.shuffle_null(Fk, yk, folds, lg["C"], p["N_SHUFFLES"], rng)
                nst = D.null_stats(lg["score"], null)
                cv = f"blocked {len(folds)}-fold over trials"
                row(var, "categorical", "", "logistic", "all", raw.X.shape[1], int(yk.size), cv, "balanced_accuracy", lg,
                    chance, null_kind, nst, feats=f"counts in [{win[0]:g}, {win[1]:g}] s of {wcol}")
                vres = {"balanced_accuracy": lg["score"], "chance": chance, "null": null_kind, "null_p95": nst["null_p95"],
                        "p_value": nst["p_value"], "z": nst["z"], "classes": good.tolist(), "n_trials": int(yk.size)}
                items = [{"label": "logistic, all", "score": lg["score"], **nst}]
                if regions:
                    for reg, ii in regions:
                        lr = D.logistic_cv(Fk[:, ii], yk, folds, p["SEED"])
                        n_reg = min(p["N_SHUFFLES"], 100)
                        if n_reg == 0:
                            nn = np.array([])
                        elif gen is not None:
                            nn = D.pseudo_null(Fk[:, ii], yk, folds, lr["C"], [gen(rng) for _ in range(n_reg)])
                        else:
                            nn = D.shuffle_null(Fk[:, ii], yk, folds, lr["C"], n_reg, rng)
                        ns = D.null_stats(lr["score"], nn)
                        row(var, "categorical", "", "logistic", reg, int(len(ii)), int(yk.size), cv, "balanced_accuracy",
                            lr, chance, null_kind, ns, feats=f"counts in [{win[0]:g}, {win[1]:g}] s of {wcol}")
                        items.append({"label": f"logistic, {reg} ({len(ii)})", "score": lr["score"], **ns})
                        vres.setdefault("regions", {})[reg] = {"balanced_accuracy": lr["score"], "p_value": ns["p_value"],
                                                               "n_neurons": int(len(ii))}
                nulls.append({"var": var, "metric": "balanced accuracy", "chance": chance, "items": items})
                drops.append({"name": var, "metric": "bal. acc.", "rows": D.dropping_curve(
                    Fk, yk, folds, "categorical", lg["C"], p["DROP_REPEATS"], rng)})
                if len([pp for pp in trace_panels if pp.get("categorical")]) < 2:
                    order = np.argsort(np.concatenate([te for _, te in folds]))
                    n_show = min(80, yk.size)
                    trace_panels.append({"name": var, "score": lg["score"], "chance": chance, "categorical": (
                        np.arange(n_show), lg["true"][order][:n_show], lg["pred"][order][:n_show])})
                vres["seconds"] = round(time.monotonic() - t0, 1)
                res["variables"][var] = vres
                hb(f"{sid} {var}: balanced accuracy {lg['score']:.3f} (chance {chance:.2f}, null p95 "
                   f"{nst['null_p95'] if nst['null_p95'] is None else round(nst['null_p95'], 3)})", force=True)
    if emb is not None:
        emb.close()
    if not rows:
        raise ValueError("no variable could be decoded" + (f" (skipped: {why})" if why else ""))

    d = out_dir / sid
    d.mkdir(parents=True, exist_ok=True)
    if trace_panels:
        # at most three continuous panels: a 1-D position first, then the best decoded variables
        cont = sorted([x for x in trace_panels if not x.get("categorical")],
                      key=lambda x: (not (x["position"] and x["name"] == "position"), -x["score"]))[:3]
        cat = [x for x in trace_panels if x.get("categorical")][:2]
        figs.append({"path": F.traces(sid, cont + cat, d / f"{sid}_decoded_traces.png"), "caption": trace_caption(sid, res)})
    if nulls:
        groups = [{"title": g["var"], "metric": g["metric"], "chance": g["chance"], "items": g["items"]} for g in nulls]
        figs.append({"path": F.vs_null(sid, groups, d / f"{sid}_null.png"), "caption": (
            f"{sid}: decoding score of each variable (dots; blue when p < 0.05) next to its null distribution "
            "(grey bar: 5-95% of scores after circularly shifting the behaviour in time by at least "
            f"{p['MIN_SHIFT_S']:g} s, or shuffling trial labels; tick: median). "
            + ("Further columns decode from one brain region at a time (number of neurons in brackets). " if regions else "")
            + "A score is meaningful only when it clears its null; the dotted line is chance (0 for R²).")})
    if rrr_fig:
        figs.append({"path": F.rrr_matrix(sid, *rrr_fig, d / f"{sid}_rrr.png"), "caption": (
            f"{sid}: communication between brain regions (reduced-rank regression, Semedo et al. 2019). Left: how well "
            "each source region's activity (rows) predicts held-out units of each target region (columns), cross-"
            "validated R² of a ridge model; the diagonal is the target region predicted from its own other units, "
            "with the same number of predictors. Right: the number of dimensions needed (optimal rank). An "
            "off-diagonal rank lower than the diagonal one means the regions talk through a low-dimensional "
            "'communication subspace' rather than sharing all of their activity.")})
    if drops:
        figs.append({"path": F.dropping(sid, drops, d / f"{sid}_dropping.png"), "caption": (
            f"{sid}: neuron-dropping curve, the decoding score with random subsets of neurons (mean ± SD over "
            f"{p['DROP_REPEATS']} subsets). A curve still rising at the right end means more neurons would decode "
            "better; a plateau means the information is redundant across neurons.")})
    res["figures"] = [{"path": f"{sid}/{Path(f['path']).name}", "caption": f["caption"]} for f in figs]
    return rows, drops, res, heldout_data


def trace_caption(sid, res) -> str:
    parts = []
    for v, r in res["variables"].items():
        if "ridge_r2" in r:
            extra = "".join(f", {k[:-3]} {r[k]:.2f}" for k in ("mlp_r2", "bayes_r2", "knn_cebra_time_r2") if r.get(k) is not None)
            parts.append(f"{v}: ridge R² {r['ridge_r2']:.2f}{extra} (null 95th percentile "
                         f"{r['null_p95'] if r['null_p95'] is None else format(r['null_p95'], '.2f')})")
        elif "balanced_accuracy" in r:
            parts.append(f"{v}: balanced accuracy {r['balanced_accuracy']:.2f} (chance {r['chance']:.2f})")
    return (f"{sid}: behaviour decoded from population activity on held-out data (black: true; colours: decoders "
            "trained on the other, non-overlapping blocks of the session). For trial variables, dots are the "
            "decoded class per held-out trial, circled when wrong. " + "; ".join(parts) + ".")


def main() -> int:
    started = datetime.now(timezone.utc)
    t_start = time.monotonic()
    input_dir, output_dir = os.environ.get("INPUT_DIR"), os.environ.get("OUTPUT_DIR")
    if not input_dir or not output_dir:
        log.error("INPUT_DIR and OUTPUT_DIR must be set")
        return 2
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    try:
        p = read_parameters()
    except ParameterError as exc:
        log.error("%s", exc)
        return 2
    if not input_dir.is_dir():
        log.error("INPUT_DIR %s is not a folder", input_dir)
        return 1
    import tempfile

    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl"))
    from processor.resources import resolve_threads, set_thread_env

    threads, thread_src = resolve_threads(p["THREADS"])
    set_thread_env(threads)
    import numpy as np

    from processor import aligned as A
    from processor import decoders as D
    from processor import figures as F
    from processor import pseudo as P

    found = A.discover(input_dir, "aligned")
    for name, reason in found.skipped.items():
        log.warning("%s: skipped, %s", name, reason)
    if not found.sessions:
        log.error("no neuro-align aligned.npz found under %s", input_dir)
        return 1
    embs = find_embeddings(input_dir, A)
    log.info("%d sessions, %d with a neuro-embedding output; decoders %s; %d-fold blocked CV, gap %g s; %d shuffles; "
             "%d threads (%s)", len(found.sessions), sum(s in embs for s, _ in found.sessions), ",".join(p["DECODERS"]),
             p["CV_FOLDS"], p["CV_GAP_S"], p["N_SHUFFLES"], threads, thread_src)
    out = output_dir / APP
    out.mkdir(parents=True, exist_ok=True)
    hb = Heartbeat(log, "progress", 30)
    all_rows, drop_rows, results, statuses, figures_all, warnings = [], [], {}, {}, [], []
    ho_inputs: dict[str, dict] = {}
    rrr_all: list[dict] = []
    for sid, path in found.sessions:
        t0 = time.monotonic()
        try:
            s = A.load(sid, path, found.sheet.get(sid))
            rows, drops, res, ho = run_session(s, p, embs.get(sid), out, A, D, F, hb)
            all_rows += rows
            for c in drops:
                for r in c["rows"]:
                    drop_rows.append({"session_id": sid, "variable": c["name"], **r})
            rrr_all += res.pop("rrr_rows", [])
            for var, v in ho.items():
                ho_inputs.setdefault(var, {})[sid] = v
            write_json(out / sid / "summary.json", {"app": APP, "version": app_version(), **res})
            results[sid] = res
            figures_all += res["figures"]
            statuses[sid] = {"status": "ok", "seconds": round(time.monotonic() - t0, 1)}
            log.info("%s: %d neurons, %.3g s bins; %s; %.1fs", sid, res["n_neurons"], res["decode_bin_s"],
                     "; ".join(f"{v} " + (f"R2 {r['ridge_r2']:.2f}" if "ridge_r2" in r else
                                           f"acc {r['balanced_accuracy']:.2f}" if "balanced_accuracy" in r else r.get("status", ""))
                               for v, r in res["variables"].items()), time.monotonic() - t0)
        except Exception as exc:
            log.exception("%s: failed: %s", sid, exc)
            statuses[sid] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "seconds": round(time.monotonic() - t0, 1)}
            warnings.append(f"{sid}: failed ({type(exc).__name__}: {exc})")
    if not results:
        log.error("every session failed")
        return 1

    # held-out-session decoding
    ho_rows, ho_summary = [], {}
    for var, data in ho_inputs.items():
        if len(data) < 2:
            continue
        try:
            rr = D.heldout_session(data, p["HELDOUT_DIMS"])
        except Exception as exc:
            log.warning("held-out session decoding of %s failed: %s", var, exc)
            continue
        for r in rr:
            ho_rows.append({"variable": var, **{k: (",".join(v) if isinstance(v, list) else v) for k, v in r.items()
                                                if k not in ("pred", "true")}})
            all_rows.append({"session_id": r["session_id"], **{k: results[r["session_id"]].get(k) for k in A.SHEET_KEYS},
                             "variable": var, "kind": "continuous", "decoder": "ridge", "features": f"PCA {r['n_dims']} dims, aligned",
                             "region": "all", "n_samples": r["n_test"], "metric": "r2", "score": r["aligned"],
                             "cv": f"held-out session (trained on {', '.join(r['train_sessions'])}; aligned on the first half, tested on the second)"})
        ho_summary[var] = {r["session_id"]: {"aligned": r["aligned"], "unaligned": r["unaligned"],
                                             "within_session": r["within_session"]} for r in rr}
        fn = f"heldout_{A.sanitise(var)}.png"
        F.heldout(rr, var, out / fn)
        figures_all.append({"path": fn, "caption": (
            f"Decoding {var} in a session the decoder never saw. Each session's activity is reduced to "
            f"{rr[0]['n_dims']} principal components; the other sessions' components are rotated onto a common space "
            f"(orthogonal Procrustes on {var}-binned averages) and a ridge decoder is trained there. The held-out "
            "session is rotated using the first half of its own data and decoded on the second half. Compare with its "
            "own decoder (within session) and with no alignment: an aligned score close to within-session means "
            "different animals share the same population code.")})
        log.info("held-out session %s: %s", var, ", ".join(f"{r['session_id']} {r['aligned']:.2f} (within "
                                                           f"{r['within_session']:.2f}, unaligned {r['unaligned']:.2f})" for r in rr))
    if ho_rows:
        write_csv(out / "heldout.csv", ho_rows, ["variable", "session_id", "train_sessions", "reference", "n_dims",
                                                  "n_conditions_matched", "n_test", "within_session", "aligned", "unaligned"])
    write_csv(out / "decoding.csv", all_rows, CSV_COLUMNS)
    # per variable x region (x decoder) across sessions: mean score, mean null-corrected score and the
    # sessions' p-values combined with Fisher's method (as the IBL brain-wide map does)
    from scipy.stats import combine_pvalues

    groups: dict = {}
    for r in all_rows:
        if r.get("p_value") is None or str(r.get("cv", "")).startswith("held-out"):
            continue
        groups.setdefault((r["variable"], r["region"], r["decoder"], r["metric"]), []).append(r)
    reg_rows = []
    for (var, region, dec, metric), rs in sorted(groups.items()):
        ps = [max(float(r["p_value"]), 1e-12) for r in rs]
        fisher = float(combine_pvalues(ps, method="fisher").pvalue) if len(ps) > 1 else ps[0]
        reg_rows.append({"variable": var, "region": region, "decoder": dec, "metric": metric, "n_sessions": len(rs),
                         "n_neurons": sum(int(r.get("n_neurons") or 0) for r in rs),
                         "mean_score": float(np.mean([r["score"] for r in rs])),
                         "mean_score_corrected": float(np.mean([r["score_corrected"] for r in rs if r.get("score_corrected") is not None] or [np.nan])),
                         "fisher_p": fisher, "null": rs[0].get("null")})
    if reg_rows:
        write_csv(out / "regions.csv", reg_rows, ["variable", "region", "decoder", "metric", "n_sessions", "n_neurons",
                                                  "mean_score", "mean_score_corrected", "fisher_p", "null"])
    if rrr_all:
        write_csv(out / "rrr.csv", rrr_all, ["session_id", "source", "target", "kind", "n_source", "n_target", "n_bins",
                                            "optimal_rank", "max_rank", "rank_at_cap", "rrr_r2", "full_r2", "within_r2",
                                            "within_rank"])
    write_csv(out / "dropping.csv", drop_rows, ["session_id", "variable", "n_neurons", "mean", "sd", "repeats"])
    headline = {sid: {k: r.get(k) for k in ("n_neurons", "decode_bin_s", "neural_source", "regions", *A.SHEET_KEYS)
                      if r.get(k) is not None} | {"variables": r["variables"]} for sid, r in results.items()}
    write_json(out / "summary.json", {
        "app": APP, "version": app_version(), "sessions": headline, "heldout_session": ho_summary,
        "figures": figures_all, "tables": ["decoding.csv", "dropping.csv"] + (["regions.csv"] if reg_rows else []) + (["rrr.csv"] if rrr_all else []) + (["heldout.csv"] if ho_rows else []),
        "warnings": warnings, "n_ok": len(results), "n_failed": len(statuses) - len(results)})
    write_json(out / "provenance.json", {
        "app": APP, "version": app_version(), "variant": "default",
        "tools": tool_versions(["scikit-learn", "numpy", "scipy", "pandas", "pyarrow", "matplotlib"]),
        "parameters": p, "inputs": [rel(pth, input_dir) for _, pth in found.sessions] + [rel(x, input_dir) for x in embs.values()],
        "started": started.isoformat().replace("+00:00", "Z"),
        "finished": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "wall_seconds": round(time.monotonic() - t_start, 1), "peak_rss_mb": peak_rss_mb(), "gpu": None,
        "threads": threads, "sessions": statuses, "citations": CITATIONS})
    log.info("done: %d ok, %d failed, %.1fs, peak RSS %s MB", len(results), len(statuses) - len(results),
             time.monotonic() - t_start, peak_rss_mb())
    return 0
