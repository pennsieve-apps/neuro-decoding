"""
Pseudo-session nulls for IBL-style trial variables (Findling et al. 2025, Nature; IBL 2025, Nature).

Trial labels of the IBL task are not exchangeable: the stimulus side follows blocks of a biased
prior, and the choice follows the stimulus. Shuffling trial labels breaks that structure and makes
the null too easy to beat (a slow drift in the neural data can decode the block). IBL instead
decodes **pseudo-sessions**: label sequences drawn from the task's own generative process, decoded
from the real neural data with the same pipeline. Here, for a session that has the IBL structure
(a prior column with 0.2 / 0.5 / 0.8 values and a stimulus side):

- prior (probability_left, block_type): the first 90 trials unbiased (0.5), then blocks alternating
  0.2 / 0.8 (random start) with lengths from an exponential with mean 60 truncated to 20-100 trials;
- stimulus side: left with probability probability_left; contrast: resampled from the session;
- choice: an action-kernel model fitted to the session (as Findling et al. 2025 do per mouse):
  logit P(right) = b0 + b1 * signed contrast + b2 * sign(signed contrast) + b3 * kernel, where the
  kernel is an exponentially weighted average of the previous choices (+1 right, -1 left) with a
  decay chosen by likelihood; pseudo choices are drawn trial by trial from it, so they carry the
  same dependence on the stimulus and on the mouse's own history as the real ones;
- correct / rewarded: whether the pseudo choice matches the pseudo stimulus side.

Without a choice column the reward follows the session's accuracy per contrast. Variables the model
does not cover use trial-label shuffles.
"""
from __future__ import annotations

import re

import numpy as np

PRIORS = (0.2, 0.5, 0.8)


def _find(names, pattern):
    for n in names:
        if re.search(pattern, n, re.I):
            return n
    return None


def _num(col) -> np.ndarray:
    out = np.full(len(col), np.nan)
    for i, v in enumerate(col):
        try:
            out[i] = float(v)
        except (TypeError, ValueError):
            pass
    return out


def ibl_task(trials, tvars: dict) -> dict | None:
    """The fitted task model, or None when the session does not have the IBL structure."""
    if trials is None or len(trials) < 40:
        return None
    pcol = next((c for c in trials.columns if c.lower() in ("probability_left", "probabilityleft")), None)
    if pcol is None:
        return None
    pl = trials[pcol].to_numpy(float)
    vals = set(np.round(pl[np.isfinite(pl)], 2).tolist())
    if not vals or not vals <= set(PRIORS):
        return None
    side_var = "stimulus_side" if "stimulus_side" in tvars else _find(tvars, r"stim.*side|side.*stim")
    if side_var is None:
        return None
    side = np.array([None if v is None else str(v) for v in tvars[side_var]], dtype=object)
    levels = sorted({v for v in side if v is not None})
    if len(levels) != 2:
        return None
    left = next((v for v in levels if "left" in v.lower()), levels[0])
    right = next(v for v in levels if v != left)
    # contrast per trial
    ccol = _find(trials.columns, r"^gabor_stimulus_contrast$|^contrast$|stimulus_contrast")
    if ccol is not None:
        contrast = trials[ccol].to_numpy(float)
    else:
        a = trials.get("contrast_left", trials.get("contrastLeft"))
        b = trials.get("contrast_right", trials.get("contrastRight"))
        if a is None or b is None:
            return None
        contrast = np.where(np.isfinite(a.to_numpy(float)), a.to_numpy(float), b.to_numpy(float))
    # correct trials
    rvar = _find(tvars, r"feedback|reward(ed)?$|is_mouse_rewarded|correct")
    correct = None
    if rvar is not None:
        r = tvars[rvar]
        good = {"1", "True", "true", "correct", "1.0"}
        correct = np.array([None if v is None else (str(v) in good) for v in r], dtype=object)
    cvar = _find(tvars, r"choice")
    choice_for = {}
    if cvar is not None and correct is not None:
        ch = tvars[cvar]
        for sv in (left, right):
            m = [i for i in range(len(ch)) if side[i] == sv and correct[i] is True and ch[i] is not None]
            if m:
                vals_c, counts = np.unique(np.array([ch[i] for i in m]), return_counts=True)
                choice_for[sv] = str(vals_c[np.argmax(counts)])
        if len(choice_for) != 2 or choice_for[left] == choice_for[right]:
            cvar = None
    acc = {}
    if correct is not None:
        for c in np.unique(contrast[np.isfinite(contrast)]):
            m = [i for i in range(len(contrast)) if contrast[i] == c and correct[i] is not None]
            if m:
                acc[float(c)] = float(np.mean([correct[i] for i in m]))
    prior_vars = {}
    for name, lab in tvars.items():
        if name == pcol:
            prior_vars[name] = {p: f"{p:g}" for p in PRIORS}
        elif re.search(r"block", name, re.I):
            mp = {}
            for p in PRIORS:
                labs = [lab[i] for i in range(len(lab)) if np.isfinite(pl[i]) and round(pl[i], 2) == p and lab[i] is not None]
                if labs:
                    v, c = np.unique(np.array(labs), return_counts=True)
                    mp[p] = str(v[np.argmax(c)])
            if len(mp) >= 2:
                prior_vars[name] = mp
    rlabels = {}
    if rvar is not None:
        lv = sorted({str(v) for v in tvars[rvar] if v is not None})
        good = [v for v in lv if v in {"1", "True", "true", "correct", "1.0"}]
        bad = [v for v in lv if v not in good]
        if len(good) == 1 and len(bad) == 1:
            rlabels = {True: good[0], False: bad[0]}
    cmag = np.abs(contrast)
    if np.nanmax(cmag) > 1.5:   # IBL percent (0-100) -> fraction
        cmag = cmag / 100.0
    kernel = None
    if cvar is not None:
        ch = tvars[cvar]
        r = np.array([1.0 if v == choice_for[right] else 0.0 if v == choice_for[left] else np.nan for v in ch])
        sc = np.where(side == right, 1.0, np.where(side == left, -1.0, np.nan)) * cmag
        kernel = fit_action_kernel(sc, r)
    return {"n": len(trials), "side_var": side_var, "left": left, "right": right,
            "contrasts": cmag[np.isfinite(cmag)], "acc": acc, "choice_var": cvar, "choice_for": choice_for,
            "reward_var": rvar if rlabels else None, "reward_labels": rlabels, "prior_vars": prior_vars,
            "kernel": kernel, "acc_by_magnitude": {round(float(c), 4): v for c, v in zip(
                np.unique(cmag[np.isfinite(cmag)]), [acc.get(float(c0), 0.8) for c0 in np.unique(contrast[np.isfinite(contrast)])])}}


DECAYS = (0.3, 0.5, 0.7, 0.8, 0.9, 0.95)


def _kernel_trace(r: np.ndarray, decay: float) -> np.ndarray:
    """Before each trial: (1 - decay) * sum_k decay^(k-1) x_{t-k}, x = +1 right / -1 left / 0 none."""
    x = np.where(np.isfinite(r), 2 * np.nan_to_num(r) - 1, 0.0)
    k = np.zeros(len(r))
    acc = 0.0
    for t in range(len(r)):
        k[t] = acc
        acc = decay * acc + (1 - decay) * x[t]
    return k


def _design(sc, kern):
    return np.column_stack([sc, np.sign(sc), kern])


def fit_action_kernel(sc: np.ndarray, r: np.ndarray) -> dict | None:
    """Logistic choice model with an action kernel; the decay is chosen by log-likelihood."""
    from sklearn.linear_model import LogisticRegression

    ok = np.isfinite(sc) & np.isfinite(r)
    if ok.sum() < 40 or len(np.unique(r[ok])) < 2:
        return None
    best = None
    for d in DECAYS:
        X = _design(sc, _kernel_trace(r, d))[ok]
        m = LogisticRegression(C=100.0, max_iter=1000).fit(X, r[ok])
        p = np.clip(m.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
        ll = float(np.sum(r[ok] * np.log(p) + (1 - r[ok]) * np.log(1 - p)))
        if best is None or ll > best["loglik"]:
            best = {"decay": d, "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0]), "loglik": ll}
    return best


def _blocks(n: int, rng) -> np.ndarray:
    pl = np.full(n, 0.5)
    i = min(90, n)
    cur = rng.choice([0.2, 0.8])
    while i < n:
        length = 0
        while not 20 <= length <= 100:
            length = int(round(rng.exponential(60)))
        pl[i:i + length] = cur
        i += length
        cur = 0.8 if cur == 0.2 else 0.2
    return pl


def simulate(task: dict, rng) -> dict:
    n = task["n"]
    pl = _blocks(n, rng)
    is_left = rng.random(n) < pl
    side = np.where(is_left, task["left"], task["right"]).astype(object)
    contrast = rng.choice(task["contrasts"], n) if task["contrasts"].size else np.ones(n)
    out = {task["side_var"]: side}
    kern = task.get("kernel")
    if task["choice_var"] and kern:
        # choices drawn trial by trial from the fitted action-kernel model
        sc = np.where(is_left, -1.0, 1.0) * contrast
        b = np.asarray(kern["coef"])
        d = kern["decay"]
        acc = 0.0
        right_choice = np.zeros(n, bool)
        for t in range(n):
            z = kern["intercept"] + b[0] * sc[t] + b[1] * np.sign(sc[t]) + b[2] * acc
            right_choice[t] = rng.random() < 1.0 / (1.0 + np.exp(-z))
            acc = d * acc + (1 - d) * (1.0 if right_choice[t] else -1.0)
        ok = right_choice == ~is_left
        cf = task["choice_for"]
        out[task["choice_var"]] = np.array([cf[task["right"]] if rc else cf[task["left"]] for rc in right_choice],
                                           dtype=object)
    else:
        p_ok = np.array([task["acc_by_magnitude"].get(round(float(c), 4), 0.8) for c in contrast])
        ok = rng.random(n) < p_ok
    if task["reward_var"]:
        out[task["reward_var"]] = np.array([task["reward_labels"][bool(k)] for k in ok], dtype=object)
    for name, mp in task["prior_vars"].items():
        out[name] = np.array([mp.get(round(float(v), 2), mp.get(0.5, "")) for v in pl], dtype=object)
    return out


def pseudo_labels(trials, var: str, tvars: dict, keep: np.ndarray):
    """A function rng -> pseudo labels of `var` on the kept trials, or None when the task model does
    not generate this variable."""
    task = ibl_task(trials, tvars)
    if task is None:
        return None
    covered = {task["side_var"], task["choice_var"], task["reward_var"], *task["prior_vars"]}
    if var not in covered:
        return None

    def gen(rng):
        lab = simulate(task, rng)[var][keep]
        return np.array([str(v) for v in lab])
    return gen
