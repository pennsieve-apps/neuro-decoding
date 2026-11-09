"""
Decoders, blocked cross-validation and null distributions of neuro-decoding.

Continuous variables: ridge regression (alpha chosen by inner blocked CV), a small MLP, and for
position a Bayesian (Poisson naive-Bayes) population decoder; categorical trial variables:
logistic regression. Folds are contiguous blocks of time (or of trials), never shuffled bins, with
a gap of CV_GAP_S removed from the training set on both sides of every test block, so that the
autocorrelation of behaviour and activity does not inflate the scores (Glaser et al. 2020).
"""
from __future__ import annotations

import numpy as np

ALPHAS = np.logspace(-2, 5, 15)
CS = (0.001, 0.01, 0.1, 1.0, 10.0)


# --- folds ------------------------------------------------------------------------------------

def blocked_folds(t: np.ndarray, k: int, gap_s: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """k contiguous test blocks over the time-ordered samples; training = the rest minus a gap."""
    n = t.size
    blocks = np.array_split(np.arange(n), k)
    out = []
    for b in blocks:
        if b.size == 0:
            continue
        lo, hi = t[b[0]] - gap_s, t[b[-1]] + gap_s
        train = np.flatnonzero((t < lo) | (t > hi))
        if train.size >= 10:
            out.append((train, b))
    return out


def trial_folds(n: int, k: int) -> list[tuple[np.ndarray, np.ndarray]]:
    blocks = np.array_split(np.arange(n), k)
    return [(np.setdiff1d(np.arange(n), b), b) for b in blocks if b.size]


def r2(y, p) -> float:
    ss = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - p) ** 2) / ss) if ss > 0 else float("nan")


def summarize_continuous(y_true_folds, y_pred_folds) -> dict:
    scores = [r2(a, b) for a, b in zip(y_true_folds, y_pred_folds)]
    yt, yp = np.concatenate(y_true_folds), np.concatenate(y_pred_folds)
    r = float(np.corrcoef(yt, yp)[0, 1]) if np.std(yp) > 0 and np.std(yt) > 0 else float("nan")
    return {"score": float(np.nanmean(scores)), "score_sd": float(np.nanstd(scores)), "r": r,
            "median_abs_error": float(np.median(np.abs(yt - yp))), "fold_scores": scores}


def balanced_accuracy(y, p) -> float:
    from sklearn.metrics import balanced_accuracy_score

    return float(balanced_accuracy_score(y, p))


# --- ridge ------------------------------------------------------------------------------------

def _standardise(Xtr, Xte):
    """z-score with the training fold's mean and SD. A neuron nearly silent in the training blocks
    (drift, a unit lost or found mid-session) would get a tiny SD and explode on the test block, so
    the SD is floored at 10 % of the median SD and z is clipped to +-10."""
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    floor = max(1e-3, 0.1 * float(np.median(sd))) if sd.size else 1e-3
    sd = np.maximum(sd, floor)
    return np.clip((Xtr - mu) / sd, -10, 10), np.clip((Xte - mu) / sd, -10, 10)


def ridge_cv(F: np.ndarray, y: np.ndarray, folds, inner: int = 3) -> dict:
    """Ridge with alpha chosen per outer fold by inner blocked CV; returns predictions per fold."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold

    preds, trues, alphas = [], [], []
    for tr, te in folds:
        a, b = _standardise(F[tr], F[te])
        m = RidgeCV(alphas=ALPHAS, cv=KFold(inner, shuffle=False)).fit(a, y[tr])
        preds.append(m.predict(b))
        trues.append(y[te])
        alphas.append(float(m.alpha_))
    out = summarize_continuous(trues, preds)
    out.update(pred_folds=preds, alpha=float(np.median(alphas)))
    return out


class FastRidge:
    """Ridge with a fixed alpha on fixed folds, solved once per fold, so that many targets (the
    circularly shifted nulls) cost one matrix product each."""

    def __init__(self, F: np.ndarray, folds, alpha: float):
        self.folds = folds
        self.mats = []
        for tr, te in folds:
            a, b = _standardise(F[tr], F[te])
            d = a.shape[1]
            P = np.linalg.solve(a.T @ a + alpha * np.eye(d), a.T)   # d x n_train
            self.mats.append((P, b))

    def score(self, y: np.ndarray) -> float:
        s = []
        for (tr, te), (P, b) in zip(self.folds, self.mats):
            mu = y[tr].mean()
            s.append(r2(y[te], b @ (P @ (y[tr] - mu)) + mu))
        return float(np.nanmean(s))


def circular_null(F, y, t, folds, alpha, n_shuffles, min_shift_s, rng) -> np.ndarray:
    """Ridge R^2 with the behaviour circularly shifted against the activity by a random offset of at
    least min_shift_s (and at most the duration minus that)."""
    n = y.size
    dt = np.median(np.diff(t)) if n > 1 else 1.0
    k = int(np.ceil(min_shift_s / dt))
    if n <= 2 * k + 1:
        k = max(1, n // 4)
    fr = FastRidge(F, folds, alpha)
    shifts = rng.integers(k, n - k, n_shuffles)
    return np.array([fr.score(np.roll(y, int(s))) for s in shifts])


# --- MLP --------------------------------------------------------------------------------------

def mlp_cv(F, y, folds, seed: int) -> dict:
    from sklearn.neural_network import MLPRegressor

    preds, trues = [], []
    for tr, te in folds:
        a, b = _standardise(F[tr], F[te])
        mu, sd = y[tr].mean(), y[tr].std() or 1.0
        m = MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3, max_iter=300,
                         early_stopping=True, validation_fraction=0.15, n_iter_no_change=15, random_state=seed)
        m.fit(a, (y[tr] - mu) / sd)
        preds.append(m.predict(b) * sd + mu)
        trues.append(y[te])
    out = summarize_continuous(trues, preds)
    out["pred_folds"] = preds
    return out


# --- Bayesian (Poisson naive Bayes) position decoder ------------------------------------------

def bayes_cv(C: np.ndarray, y: np.ndarray, folds, n_bins: int, dt: float) -> dict:
    """Poisson naive-Bayes population decoder (Zhang et al. 1998; as pynapple's decode_1d):
    occupancy-normalised tuning curves from the training fold (smoothed by one bin), the training
    occupancy as prior, and the posterior mean position per test bin (more robust to the occasional
    far jump than the MAP). C: spike counts in the decoding window of each bin."""
    from scipy.ndimage import gaussian_filter1d

    preds, trues = [], []
    for tr, te in folds:
        lo, hi = np.percentile(y[tr], [0.5, 99.5])
        edges = np.linspace(lo, hi, n_bins + 1)
        centres = (edges[:-1] + edges[1:]) / 2
        b = np.clip(np.digitize(y[tr], edges) - 1, 0, n_bins - 1)
        occ = np.bincount(b, minlength=n_bins).astype(float)
        tc = np.zeros((n_bins, C.shape[1]))
        np.add.at(tc, b, C[tr])
        tc = tc / np.maximum(occ, 1)[:, None]
        tc = gaussian_filter1d(tc, 1.0, axis=0, mode="nearest")
        tc = np.maximum(tc, 1e-4)
        ok = occ > 0
        logp = C[te] @ np.log(tc[ok]).T - tc[ok].sum(axis=1)[None, :] + np.log(occ[ok] / occ.sum())[None, :]
        w = np.exp(logp - logp.max(axis=1, keepdims=True))
        w /= w.sum(axis=1, keepdims=True)
        preds.append(w @ centres[ok])
        trues.append(y[te])
    out = summarize_continuous(trues, preds)
    out["pred_folds"] = preds
    return out


# --- categorical (trials) ---------------------------------------------------------------------

def logistic_cv(F, y, folds, seed: int) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    preds, trues, cs = [], [], []
    for tr, te in folds:
        a, b = _standardise(F[tr], F[te])
        best, best_c = -1, CS[2]
        _, counts = np.unique(y[tr], return_counts=True)
        n_inner = int(min(3, counts.min()))
        if n_inner >= 2:
            for c in CS:
                s = []
                for itr, ite in StratifiedKFold(n_inner, shuffle=False).split(a, y[tr]):
                    m = LogisticRegression(C=c, class_weight="balanced", max_iter=2000).fit(a[itr], y[tr][itr])
                    s.append(balanced_accuracy(y[tr][ite], m.predict(a[ite])))
                if np.mean(s) > best:
                    best, best_c = np.mean(s), c
        m = LogisticRegression(C=best_c, class_weight="balanced", max_iter=2000).fit(a, y[tr])
        preds.append(m.predict(b))
        trues.append(y[te])
        cs.append(best_c)
    scores = [balanced_accuracy(a, b) for a, b in zip(trues, preds)]
    yt, yp = np.concatenate(trues), np.concatenate(preds)
    return {"score": float(balanced_accuracy(yt, yp)), "score_sd": float(np.std(scores)), "fold_scores": scores,
            "C": float(np.median(cs)), "pred": yp, "true": yt}


def logistic_score_fixed(F, y, folds, C) -> float:
    from sklearn.linear_model import LogisticRegression

    preds, trues = [], []
    for tr, te in folds:
        if np.unique(y[tr]).size < 2:
            continue
        a, b = _standardise(F[tr], F[te])
        m = LogisticRegression(C=C, class_weight="balanced", max_iter=1000).fit(a, y[tr])
        preds.append(m.predict(b))
        trues.append(y[te])
    if not preds:
        return float("nan")
    return balanced_accuracy(np.concatenate(trues), np.concatenate(preds))


def shuffle_null(F, y, folds, C, n_shuffles, rng) -> np.ndarray:
    return np.array([logistic_score_fixed(F, rng.permutation(y), folds, C) for _ in range(n_shuffles)])


# --- null summaries, neuron dropping ----------------------------------------------------------

def null_stats(real: float, null: np.ndarray) -> dict:
    null = null[np.isfinite(null)]
    if null.size == 0 or not np.isfinite(real):
        return {"null_mean": None, "null_sd": None, "null_p95": None, "p_value": None, "z": None, "n_shuffles": 0}
    sd = float(null.std())
    return {"null_mean": float(null.mean()), "null_sd": sd, "null_p95": float(np.percentile(null, 95)),
            "p_value": float((1 + np.sum(null >= real)) / (1 + null.size)),
            "z": float((real - null.mean()) / sd) if sd > 0 else None, "n_shuffles": int(null.size),
            "null_p05": float(np.percentile(null, 5)), "null_median": float(np.median(null))}


def neuron_sizes(n: int, k: int = 8) -> list[int]:
    return sorted({int(x) for x in np.unique(np.round(np.geomspace(1, n, k)))})


def dropping_curve(F, y, folds, kind: str, param: float, repeats: int, rng) -> list[dict]:
    """Score vs number of neurons, random subsets (repeats per size) with the full model's alpha / C."""
    n = F.shape[1]
    rows = []
    for size in neuron_sizes(n):
        reps = 1 if size == n else repeats
        s = []
        for _ in range(reps):
            idx = rng.choice(n, size, replace=False)
            if kind == "continuous":
                s.append(FastRidge(F[:, idx], folds, param).score(y))
            else:
                s.append(logistic_score_fixed(F[:, idx], y, folds, param))
        rows.append({"n_neurons": size, "mean": float(np.nanmean(s)), "sd": float(np.nanstd(s)), "repeats": reps})
    return rows


# --- held-out session (aligned latent dynamics) -----------------------------------------------

def _latents(F, fit_rows, k):
    from sklearn.decomposition import PCA

    mu, sd = F[fit_rows].mean(axis=0), F[fit_rows].std(axis=0)
    sd[sd <= 0] = 1
    Z = (F - mu) / sd
    p = PCA(n_components=k, random_state=0).fit(Z[fit_rows])
    return p.transform(Z)


def _conditions(y, edges, direction):
    b = np.clip(np.digitize(y, edges) - 1, 0, len(edges) - 2)
    return b * 2 + (direction > 0) if direction is not None else b


def _cond_means(Z, cond, n_cond):
    M = np.full((n_cond, Z.shape[1]), np.nan)
    for c in range(n_cond):
        m = cond == c
        if m.sum() >= 3:
            M[c] = Z[m].mean(axis=0)
    return M


def _procrustes(src, dst):
    """Orthogonal R minimising |src_c R - dst_c| on the rows both have; returns (R, mu_src, mu_dst)."""
    ok = np.all(np.isfinite(src), axis=1) & np.all(np.isfinite(dst), axis=1)
    a, b = src[ok], dst[ok]
    ma, mb = a.mean(axis=0), b.mean(axis=0)
    U, _, Vt = np.linalg.svd((a - ma).T @ (b - mb))
    return U @ Vt, ma, mb, int(ok.sum())


def heldout_session(data: dict, k_dims: int, n_bins: int = 20) -> list[dict]:
    """data: sid -> (F features, y target, direction or None). For each session in turn, train a
    ridge decoder on the other sessions' PCA latents (each aligned to a reference session by
    orthogonal Procrustes on behaviour-binned mean latents), align the held-out session with the
    first half of its own data (calibration), and test on its second half."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold

    sids = list(data)
    k = int(min(k_dims, *[data[s][0].shape[1] for s in sids]))
    pooled = np.concatenate([data[s][1] for s in sids])
    edges = np.quantile(pooled, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    has_dir = all(data[s][2] is not None for s in sids)
    n_cond = n_bins * (2 if has_dir else 1)
    rows = []
    for h in sids:
        train = [s for s in sids if s != h]
        ref = max(train, key=lambda s: data[s][1].size)
        lat, means = {}, {}
        for s in train:
            F, y, d = data[s]
            lat[s] = _latents(F, np.arange(y.size), k)
            means[s] = _cond_means(lat[s], _conditions(y, edges, d if has_dir else None), n_cond)
        Zs, ys = [], []
        for s in train:
            if s == ref:
                Zs.append(lat[s] - np.nanmean(means[s], axis=0))
            else:
                R, ma, mb, _ = _procrustes(means[s], means[ref])
                Zs.append((lat[s] - ma) @ R + mb - np.nanmean(means[ref], axis=0))
            ys.append(data[s][1])
        Ztr, ytr = np.concatenate(Zs), np.concatenate(ys)
        dec = RidgeCV(alphas=ALPHAS, cv=KFold(3, shuffle=False)).fit(Ztr, ytr)
        F, y, d = data[h]
        half = y.size // 2
        cal, test = np.arange(half), np.arange(half, y.size)
        Zh = _latents(F, cal, k)
        Mh = _cond_means(Zh[cal], _conditions(y[cal], edges, d[cal] if has_dir else None), n_cond)
        R, ma, mb, n_used = _procrustes(Mh, means[ref])
        aligned = (Zh[test] - ma) @ R + mb - np.nanmean(means[ref], axis=0)
        within = RidgeCV(alphas=ALPHAS, cv=KFold(3, shuffle=False)).fit(Zh[cal], y[cal])
        rows.append({"session_id": h, "train_sessions": train, "reference": ref, "n_dims": k,
                     "n_conditions_matched": n_used, "n_test": int(test.size),
                     "aligned": r2(y[test], dec.predict(aligned)),
                     "unaligned": r2(y[test], dec.predict(Zh[test] - Zh[cal].mean(axis=0) + 0)),
                     "within_session": r2(y[test], within.predict(Zh[test])),
                     "pred": dec.predict(aligned), "true": y[test]})
    return rows


# --- kNN on an embedding ----------------------------------------------------------------------

def knn_cv(E, y, folds, k: int) -> dict:
    from sklearn.neighbors import KNeighborsRegressor

    preds, trues = [], []
    for tr, te in folds:
        m = KNeighborsRegressor(n_neighbors=min(k, tr.size - 1), metric="cosine", weights="distance").fit(E[tr], y[tr])
        preds.append(m.predict(E[te]))
        trues.append(y[te])
    out = summarize_continuous(trues, preds)
    out["pred_folds"] = preds
    return out
