"""Figures of neuro-decoding (PNG, 150 dpi, session id in the title)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

DPI = 150
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID, NULL = "#0b0b0b", "#52514e", "#d9d8d4", "#b8b7b1"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.facecolor": "white"})
    return plt


def traces(sid: str, panels: list[dict], out: Path) -> Path:
    """Decoded vs true over a window, one row per variable. panel: name, unit, t, true, preds
    [(label, values, score)], categorical: (trial index, true labels, predicted labels)."""
    plt = _plt()
    n = len(panels)
    heights = [1.4 if p.get("categorical") else 2.2 for p in panels]
    fig, axes = plt.subplots(n, 1, figsize=(10, sum(heights) + 0.9 * n), squeeze=False,
                             gridspec_kw={"height_ratios": heights})
    for ax, p in zip(axes[:, 0], panels):
        if p.get("categorical"):
            idx, yt, yp = p["categorical"]
            levels = sorted(set(yt.tolist()) | set(yp.tolist()))
            pos = {v: i for i, v in enumerate(levels)}
            ax.step(idx, [pos[v] for v in yt], where="mid", color=INK, lw=2, label="true")
            wrong = np.array([a != b for a, b in zip(yt, yp)])
            ax.scatter(idx, [pos[v] for v in yp], s=14, color=SERIES[0], zorder=3, label="decoded (held out)")
            ax.scatter(np.asarray(idx)[wrong], [pos[v] for v in np.asarray(yp)[wrong]], s=28, facecolors="none",
                       edgecolors=SERIES[7], zorder=4, label="wrong")
            ax.set_yticks(range(len(levels)), levels)
            ax.set_xlabel("trial")
            ax.set_title(f"{p['name']}: balanced accuracy {p['score']:.2f} (chance {p['chance']:.2f})", fontsize=9,
                         loc="left")
        else:
            # epoch time: consecutive analysed bins side by side; a gap in the session (bins outside the
            # epoch, e.g. immobility) is a thin vertical line and breaks the traces
            t = np.asarray(p["t"])
            dt = p.get("dt") or (np.median(np.diff(t)) if t.size > 1 else 1.0)
            x = np.arange(t.size) * dt
            gaps = np.flatnonzero(np.diff(t) > 2.5 * dt)

            def broken(v):
                v = np.asarray(v, float).copy()
                xs = x.copy()
                return np.insert(xs, gaps + 1, np.nan), np.insert(v, gaps + 1, np.nan)

            ax.plot(*broken(p["true"]), color=INK, lw=2, label="true")
            for k, (lab, v, sc) in enumerate(p["preds"]):
                ax.plot(*broken(v), color=SERIES[k], lw=1.4, alpha=0.9, label=f"{lab} (R² {sc:.2f})")
            for g in gaps:
                ax.axvline(x[g] + dt / 2, color=GRID, lw=0.8)
            ax.set_ylabel(f"{p['name']}" + (f" ({p['unit']})" if p.get("unit") else ""))
            ax.set_xlabel(f"time in the analysed epoch (s; from {t[0]:.0f} s on the session clock"
                          + (", grey lines: gaps" if gaps.size else "") + ")")
        ax.legend(frameon=False, fontsize=7, ncol=4, loc="lower right", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(f"{sid}: decoded vs true behaviour on held-out data", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def vs_null(sid: str, groups: list[dict], out: Path) -> Path:
    """One panel per variable: real score (dot) per decoder/region next to the null (5-95 % bar and
    median tick)."""
    plt = _plt()
    n = len(groups)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows), squeeze=False)
    for ax, g in zip(axes.ravel(), groups):
        xs = np.arange(len(g["items"]))
        for i, it in enumerate(g["items"]):
            if it.get("null_p05") is not None:
                ax.plot([i, i], [it["null_p05"], it["null_p95"]], color=NULL, lw=8, solid_capstyle="butt",
                        label="null 5-95%" if i == 0 else None)
                ax.plot([i - 0.15, i + 0.15], [it["null_median"]] * 2, color=MUTED, lw=1.5)
            col = SERIES[0] if (it.get("p_value") is not None and it["p_value"] < 0.05) else MUTED
            ax.scatter([i], [it["score"]], s=40, color=col, zorder=3, edgecolors="white", linewidths=1.5)
            if it.get("p_value") is not None:
                ax.annotate(f"p={it['p_value']:.3g}" if it["p_value"] >= 0.001 else "p<0.001", (i, it["score"]),
                            xytext=(6, 2), textcoords="offset points", fontsize=7, color=MUTED)
        ax.set_xticks(xs, [it["label"] for it in g["items"]], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(g["metric"])
        if g.get("chance") is not None:
            ax.axhline(g["chance"], color=MUTED, lw=0.8, ls=":")
        else:
            ax.axhline(0, color=MUTED, lw=0.8, ls=":")
        ax.set_title(g["title"], fontsize=9)
        ax.set_xlim(-0.6, len(xs) - 0.4)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    h, lab = axes.ravel()[0].get_legend_handles_labels()
    if h:
        fig.legend(h, lab, loc="upper right", frameon=False, fontsize=8)
    fig.suptitle(f"{sid}: decoding accuracy vs the null distribution", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def dropping(sid: str, curves: list[dict], out: Path) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for k, c in enumerate(curves[: len(SERIES)]):
        x = np.array([r["n_neurons"] for r in c["rows"]])
        m = np.array([r["mean"] for r in c["rows"]])
        s = np.array([r["sd"] for r in c["rows"]])
        ax.fill_between(x, m - s, m + s, color=SERIES[k], alpha=0.15, linewidth=0)
        ax.plot(x, m, color=SERIES[k], lw=2, marker="o", ms=4, label=f"{c['name']} ({c['metric']})")
    ax.set_xscale("log")
    ax.set_xlabel("number of neurons (random subsets)")
    ax.set_ylabel("decoding score")
    ax.axhline(0, color=MUTED, lw=0.8, ls=":")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"{sid}: neuron-dropping curve", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def heldout(rows: list[dict], var: str, out: Path) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(1.6 * len(rows) + 3, 3.6))
    kinds = [("within_session", "within session (own decoder)"), ("aligned", "other sessions, aligned"),
             ("unaligned", "other sessions, not aligned")]
    w = 0.25
    for k, (key, lab) in enumerate(kinds):
        ax.bar(np.arange(len(rows)) + (k - 1) * w, [max(r[key], -0.5) for r in rows], width=w - 0.03,
               color=SERIES[k], label=lab)
    ax.set_xticks(range(len(rows)), [r["session_id"] for r in rows], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(f"R² ({var}, second half)")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"Held-out session decoding of {var}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def rrr_matrix(sid: str, rows: list[dict], regions: list[str], out: Path) -> Path:
    """Source x target matrices of cross-validated R² (ridge) and optimal RRR rank."""
    plt = _plt()
    n = len(regions)
    pos = {r: i for i, r in enumerate(regions)}
    R2 = np.full((n, n), np.nan)
    RK = np.full((n, n), np.nan)
    for r in rows:
        R2[pos[r["source"]], pos[r["target"]]] = r["full_r2"]
        RK[pos[r["source"]], pos[r["target"]]] = r["optimal_rank"]
    short = [r if len(r) <= 22 else r[:20] + "…" for r in regions]
    fig, axes = plt.subplots(1, 2, figsize=(2.2 * n + 5, 1.0 * n + 2.6))
    for ax, M, title, fmt, cmap in ((axes[0], R2, "cross-validated R² (ridge)", "{:.2f}", "Blues"),
                                    (axes[1], RK, "optimal rank (dimensions)", "{:.0f}", "Oranges")):
        vmax = np.nanmax(M) if np.isfinite(M).any() else 1
        im = ax.imshow(M, cmap=cmap, vmin=0 if title.startswith("optimal") else min(0, np.nanmin(M)), vmax=vmax)
        ax.grid(False)
        ax.set_xticks(range(n), short, rotation=35, ha="right", fontsize=7)
        ax.set_yticks(range(n), short, fontsize=7)
        for i in range(n):
            for j in range(n):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=8,
                            color="white" if M[i, j] > 0.6 * vmax else INK, fontweight="bold" if i == j else None)
        ax.set_xlabel("target region (held-out units)")
        ax.set_ylabel("source region")
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"{sid}: communication subspaces between regions (diagonal: within region)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out
