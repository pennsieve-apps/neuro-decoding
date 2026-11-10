"""
Communication subspaces between brain regions: reduced-rank regression (RRR) of one region's
activity on another's (Semedo et al. 2019, Neuron; MacDowell et al. 2025, Nat Commun).

For every target region T, its units are split at random into two halves: T_src and T_tgt.
- within: T_tgt predicted from T_src (how well a region predicts itself; Semedo's V1-V1).
- between: T_tgt predicted from a source region S, subsampled to the size of T_src so that the
  comparison with "within" is not driven by the number of predictors.
Each prediction is a ridge regression (full rank; alpha by inner blocked CV), and a reduced-rank
version of it: the ridge coefficients projected onto the top r principal axes of the ridge
predictions (the RRR solution with a ridge penalty). The rank is cross-validated with contiguous
blocks: the optimal rank is the smallest r whose mean R^2 is within one SEM (across folds) of the
peak of the cross-validated RRR curve, Semedo et al.'s rule ("the smallest number of dimensions for
which predictive performance was within one S.E.M. of the peak performance"). A rank equal to the
largest tried is flagged (`rank_at_cap`): the true dimensionality may be higher. R^2 is over all
target units (1 - SSE / SST, per fold, averaged). Slow drifts are removed per contiguous segment of
each fold (training segments and the test block separately), so no smoothing crosses a fold edge.
"""
from __future__ import annotations

import numpy as np

ALPHAS = np.logspace(-1, 5, 13)


def _z(a, b):
    mu, sd = a.mean(axis=0), a.std(axis=0)
    sd = np.maximum(sd, max(1e-3, 0.1 * float(np.median(sd))))
    return (a - mu) / sd, (b - mu) / sd


def _r2(Y, P):
    ss = np.sum((Y - Y.mean(axis=0)) ** 2)
    return float(1 - np.sum((Y - P) ** 2) / ss) if ss > 0 else float("nan")


def _ridge(X, Y, alpha):
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + alpha * np.eye(d), X.T @ Y)


def _alpha(X, Y, inner=3):
    blocks = np.array_split(np.arange(len(X)), inner)
    best, best_a = -np.inf, ALPHAS[len(ALPHAS) // 2]
    for a in ALPHAS:
        s = []
        for b in blocks:
            tr = np.setdiff1d(np.arange(len(X)), b)
            B = _ridge(X[tr], Y[tr] - Y[tr].mean(axis=0), a)
            s.append(_r2(Y[b], X[b] @ B + Y[tr].mean(axis=0)))
        if np.mean(s) > best:
            best, best_a = np.mean(s), a
    return best_a


def _segments(idx: np.ndarray) -> list[np.ndarray]:
    cut = np.flatnonzero(np.diff(idx) != 1) + 1
    return np.split(idx, cut)


def detrend(F: np.ndarray, idx: np.ndarray, w: int) -> np.ndarray:
    """F[idx] minus its running mean over w bins, computed within each contiguous run of idx."""
    from scipy.ndimage import uniform_filter1d

    out = []
    for seg in _segments(idx):
        X = F[seg]
        out.append(X - uniform_filter1d(X, min(w, len(seg)), axis=0, mode="nearest") if w > 1 else X)
    return np.vstack(out)


def rrr_cv(X, Y, folds, max_rank: int, detrend_bins: int = 0) -> dict:
    """R^2 per rank (1..max_rank) and of the full ridge model, per fold."""
    ranks = np.arange(1, max_rank + 1)
    per_rank = np.zeros((len(folds), ranks.size))
    full = np.zeros(len(folds))
    for f, (tr, te) in enumerate(folds):
        Xtr, Xte = detrend(X, tr, detrend_bins), detrend(X, te, detrend_bins)
        Ytr, Yte = detrend(Y, tr, detrend_bins), detrend(Y, te, detrend_bins)
        a, b = _z(Xtr, Xte)
        ym = Ytr.mean(axis=0)
        alpha = _alpha(a, Ytr)
        B = _ridge(a, Ytr - ym, alpha)
        full[f] = _r2(Yte, b @ B + ym)
        _, _, Vt = np.linalg.svd(a @ B, full_matrices=False)
        for k, r in enumerate(ranks):
            V = Vt[:r].T
            per_rank[f, k] = _r2(Yte, b @ (B @ V @ V.T) + ym)
    m = per_rank.mean(axis=0)
    peak = int(np.argmax(m))
    sem = per_rank[:, peak].std(ddof=1) / np.sqrt(len(folds)) if len(folds) > 1 else 0.0
    opt = int(ranks[np.flatnonzero(m >= m[peak] - sem)[0]])
    return {"ranks": ranks.tolist(), "r2_by_rank": m.tolist(), "full_r2": float(full.mean()),
            "optimal_rank": opt, "rrr_r2": float(m[opt - 1]), "peak_rank": int(ranks[peak]),
            "rank_at_cap": bool(opt == ranks[-1])}


def communication(F, regions: dict, folds, rng, max_rank: int = 10, detrend_bins: int = 0) -> list[dict]:
    """F: bins x neurons (z-scorable activity); regions: name -> unit column indices."""
    rows = []
    names = list(regions)
    for tgt in names:
        idx = rng.permutation(regions[tgt])
        half = idx.size // 2
        t_src, t_tgt = idx[:half], idx[half:]
        Y = F[:, t_tgt]
        mr = int(min(max_rank, t_tgt.size, half))
        w = rrr_cv(F[:, t_src], Y, folds, mr, detrend_bins)
        rows.append({"source": tgt, "target": tgt, "kind": "within", "n_source": int(half), "n_target": int(t_tgt.size),
                     "optimal_rank": w["optimal_rank"], "rrr_r2": w["rrr_r2"], "full_r2": w["full_r2"],
                     "r2_by_rank": w["r2_by_rank"], "rank_at_cap": w["rank_at_cap"], "max_rank": mr})
        for src in names:
            if src == tgt:
                continue
            s_idx = regions[src]
            if s_idx.size > half:
                s_idx = rng.choice(s_idx, half, replace=False)
            mr_b = int(min(max_rank, t_tgt.size, s_idx.size))
            b = rrr_cv(F[:, s_idx], Y, folds, mr_b, detrend_bins)
            rows.append({"source": src, "target": tgt, "kind": "between", "n_source": int(s_idx.size),
                         "n_target": int(t_tgt.size), "optimal_rank": b["optimal_rank"], "rrr_r2": b["rrr_r2"],
                         "full_r2": b["full_r2"], "r2_by_rank": b["r2_by_rank"], "rank_at_cap": b["rank_at_cap"],
                         "max_rank": mr_b})
    for r in rows:
        w = next(x for x in rows if x["target"] == r["target"] and x["kind"] == "within")
        r["within_r2"] = w["full_r2"]
        r["within_rank"] = w["optimal_rank"]
    return rows
