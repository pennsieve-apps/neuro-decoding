"""
Decoders, blocked cross-validation and null distributions of neuro-decoding.

Continuous variables: ridge regression (alpha chosen by inner blocked CV), a small MLP, and for
position a Bayesian (Poisson naive-Bayes) population decoder; categorical trial variables:
logistic regression. Folds are contiguous blocks of time (or of trials), never shuffled bins, with
a gap of CV_GAP_S removed from the training set on both sides of every test block, so that the
autocorrelation of behaviour and activity does not inflate the scores (Glaser et al. 2020).
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

# lbfgs on label-permuted / pseudo-session labels often stops early on separable folds; harmless
warnings.filterwarnings("ignore", category=ConvergenceWarning)

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


def pooled_r2(y_true_folds, y_pred_folds) -> float:
    """R² over every held-out bin of every fold together (1 - SSE / SST around the mean of all
    held-out bins). A per-fold R² is measured against that block's own variance, so a block in
    which the animal barely moves gives a huge negative R² even when its errors are the smallest of
    the session; pooling avoids that."""
    return r2(np.concatenate(y_true_folds), np.concatenate(y_pred_folds))


def summarize_continuous(y_true_folds, y_pred_folds) -> dict:
    """score = pooled R² (see pooled_r2); fold_scores and score_sd describe the per-fold R² only."""
    scores = [r2(a, b) for a, b in zip(y_true_folds, y_pred_folds)]
    yt, yp = np.concatenate(y_true_folds), np.concatenate(y_pred_folds)
    r = float(np.corrcoef(yt, yp)[0, 1]) if np.std(yp) > 0 and np.std(yt) > 0 else float("nan")
    return {"score": pooled_r2(y_true_folds, y_pred_folds), "score_sd": float(np.nanstd(scores)), "r": r,
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


def _svd(a):
    """Eigen-decomposition of the Gram matrix a'a (d x d): ridge for every alpha from one
    decomposition, and cheaper than an SVD of the n x d design when n >> d."""
    lam, V = np.linalg.eigh(a.T @ a)
    return a, np.maximum(lam, 0.0), V


def _ridge_preds(dec, y, b, alphas) -> np.ndarray:
    """Predictions on b for every alpha: coef = V diag(1 / (lam + alpha)) V' a' (y - mean)."""
    a, lam, V = dec
    mu = y.mean()
    z = V.T @ (a.T @ (y - mu))
    W = 1.0 / (lam[None, :] + np.asarray(alphas)[:, None])               # alphas x d
    coefs = (W * z[None, :]) @ V.T                                        # alphas x d
    return b @ coefs.T + mu                                               # n_test x alphas


def inner_blocks(n: int, inner: int, gap: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Contiguous inner validation blocks over the training rows, with `gap` rows dropped from the
    inner training set on each side of the block."""
    out = []
    for blk in np.array_split(np.arange(n), inner):
        if blk.size == 0:
            continue
        keep = np.ones(n, bool)
        keep[max(0, blk[0] - gap): blk[-1] + gap + 1] = False
        tr = np.flatnonzero(keep)
        if tr.size > 10:
            out.append((tr, blk))
    return out


def _inner_alpha(a, y, inner: int, gap: int = 0) -> float:
    """Alpha with the best pooled R² over contiguous inner blocks of the training rows."""
    P_all, Y_all = [], []
    for tr, blk in inner_blocks(len(y), inner, gap):
        P_all.append(_ridge_preds(_svd(a[tr]), y[tr], a[blk], ALPHAS))
        Y_all.append(y[blk])
    P, Y = np.vstack(P_all), np.concatenate(Y_all)
    scores = [r2(Y, P[:, i]) for i in range(len(ALPHAS))]
    return float(ALPHAS[int(np.nanargmax(scores))])


def ridge_cv(F: np.ndarray, y: np.ndarray, folds, inner: int = 3, gap: int = 0) -> dict:
    """Ridge with alpha chosen per outer fold by inner blocked CV (contiguous blocks of the training
    data); one SVD per fit evaluates every alpha. Returns predictions per fold."""
    preds, trues, alphas = [], [], []
    for tr, te in folds:
        a, b = _standardise(F[tr], F[te])
        alpha = _inner_alpha(a, y[tr], inner, gap)
        preds.append(_ridge_preds(_svd(a), y[tr], b, [alpha])[:, 0])
        trues.append(y[te])
        alphas.append(alpha)
    out = summarize_continuous(trues, preds)
    out.update(pred_folds=preds, alpha=float(np.median(alphas)))
    return out


class FastRidge:
    """Ridge with a fixed alpha on fixed folds, decomposed once per fold, so that many targets (the
    circularly shifted nulls) cost two matrix-vector products each."""

    def __init__(self, F: np.ndarray, folds, alpha: float):
        self.folds = folds
        self.mats = []
        for tr, te in folds:
            a, b = _standardise(F[tr], F[te])
            _, lam, V = _svd(a)
            self.mats.append(((a @ V).T, (b @ V) / (lam + alpha)[None, :]))

    def score(self, y: np.ndarray) -> float:
        """Pooled R² over the held-out bins of all folds (as summarize_continuous)."""
        P, T = [], []
        for (tr, te), (AVt, B) in zip(self.folds, self.mats):
            mu = y[tr].mean()
            P.append(B @ (AVt @ (y[tr] - mu)) + mu)
            T.append(y[te])
        return pooled_r2(T, P)


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


def circular_test(F, y, t, folds, alpha, n_shuffles, min_shift_s, rng) -> tuple[float, np.ndarray]:
    """(R² of the unshifted behaviour, null R²) with the SAME estimator: ridge with the fixed alpha
    of the real fit on the same folds. The p-value compares like with like; the reported score
    stays the nested-CV ridge (re-running the inner alpha search for every shift would multiply the
    cost by the number of shuffles, a shortcut documented in the README)."""
    real = FastRidge(F, folds, alpha).score(y)
    return real, circular_null(F, y, t, folds, alpha, n_shuffles, min_shift_s, rng)


# --- MLP --------------------------------------------------------------------------------------

MLP_ALPHAS = (0.01, 0.1, 1.0, 10.0)


def _mlp_fit(a, z, seed: int, alpha: float, val_frac: float = 0.15, max_epochs: int = 200, patience: int = 10):
    """MLP (64-32) trained epoch by epoch, early-stopped on the last `val_frac` of the (time-ordered)
    training rows, a contiguous block, not a random sample of autocorrelated bins. Returns the
    network at its best validation epoch and that validation R²."""
    import copy
    import warnings

    from sklearn.neural_network import MLPRegressor

    n = len(z)
    nv = max(10, int(val_frac * n))
    tr, va = np.arange(n - nv), np.arange(n - nv, n)
    m = MLPRegressor(hidden_layer_sizes=(64, 32), alpha=alpha, learning_rate_init=1e-3, random_state=seed,
                     batch_size=min(256, len(tr)))
    best, best_m, wait = -np.inf, None, 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(max_epochs):
            m.partial_fit(a[tr], z[tr])
            sc = r2(z[va], m.predict(a[va]))
            if sc > best + 1e-4:
                best, best_m, wait = sc, copy.deepcopy(m), 0
            else:
                wait += 1
                if wait >= patience:
                    break
    return best_m or m, best


def mlp_cv(F, y, folds, seed: int) -> dict:
    """Small MLP per outer fold: the L2 penalty is chosen from MLP_ALPHAS by the validation R² of
    the blocked early-stopping split (the last 15 % of the training rows); predictions are clipped
    to the training range (a network extrapolates wildly on activity unlike the training blocks)."""
    preds, trues, alphas = [], [], []
    for tr, te in folds:
        a, b = _standardise(F[tr], F[te])
        mu, sd = y[tr].mean(), y[tr].std() or 1.0
        z = (y[tr] - mu) / sd
        fits = [(_mlp_fit(a, z, seed, al), al) for al in MLP_ALPHAS]
        (m, _), al = max(fits, key=lambda f: f[0][1])
        preds.append(np.clip(m.predict(b) * sd + mu, y[tr].min(), y[tr].max()))
        trues.append(y[te])
        alphas.append(al)
    out = summarize_continuous(trues, preds)
    out.update(pred_folds=preds, alpha=float(np.median(alphas)))
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


def pseudo_null(F, y, folds, C, pseudo_labels: list) -> np.ndarray:
    """Logistic decoding of pseudo-session labels (same neural data, labels from the task's generative
    model): IBL's null for trial variables (Findling et al. 2025; IBL 2025)."""
    out = []
    for lab in pseudo_labels:
        if lab is None or np.unique(lab).size < 2:
            continue
        out.append(logistic_score_fixed(F, lab, folds, C))
    return np.array(out)


def dropping_curve(F, y, folds, kind: str, param: float, repeats: int, rng, cols=None) -> list[dict]:
    """Score vs number of neurons, random subsets (repeats per size) with the full model's alpha / C.
    cols(neurons) gives the feature columns of a set of neurons (lagged features); default identity."""
    n = F.shape[1] if cols is None else int(F.shape[1] // max(1, len(cols(np.array([0])))))
    rows = []
    for size in neuron_sizes(n):
        reps = 1 if size == n else repeats
        s = []
        for _ in range(reps):
            idx = rng.choice(n, size, replace=False)
            if cols is not None:
                idx = cols(idx)
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
