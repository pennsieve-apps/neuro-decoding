"""The synthetic fixture through the processor (no Docker), both input layouts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from processor import decoders as D
from processor import main as M
from tests import make_fixture

FAST = {"N_SHUFFLES": 40, "DROP_REPEATS": 3, "THREADS": 2}
RAT1, RAT2, IBL = "sub-rat1_ses-track1", "sub-rat2_ses-track1", "sub-mouse1_ses-ibl1"


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("fixture")
    make_fixture.write(out, "both", embedding=True, short=True)
    return out


def run(monkeypatch, input_dir: Path, output_dir: Path, **params) -> int:
    for k in M.PARAMETERS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    for k, v in {**FAST, **params}.items():
        monkeypatch.setenv(k, str(v))
    return M.main()


@pytest.fixture(scope="session")
def default_run(fixture_dir, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("out")
    mp = pytest.MonkeyPatch()
    try:
        assert run(mp, fixture_dir / "linked", out) == 0
    finally:
        mp.undo()
    return out / "neuro-decoding"


def _row(df, sid, var, decoder="ridge", region="all", features=None):
    m = (df.session_id == sid) & (df.variable == var) & (df.decoder == decoder) & (df.region == region)
    if features:
        m &= df.features.str.startswith(features)
    m &= ~df.cv.str.startswith("held-out")
    assert m.sum() == 1, (sid, var, decoder, region, features)
    return df[m].iloc[0]


def test_outputs(default_run):
    for f in ("decoding.csv", "dropping.csv", "regions.csv", "rrr.csv", "heldout.csv", "heldout_position.png", "summary.json", "provenance.json"):
        assert (default_run / f).is_file(), f
    for sid in (RAT1, RAT2, IBL):
        for f in (f"{sid}_decoded_traces.png", f"{sid}_null.png", f"{sid}_dropping.png", "summary.json"):
            assert (default_run / sid / f).is_file(), (sid, f)
    df = pd.read_csv(default_run / "decoding.csv")
    assert list(df.columns) == M.CSV_COLUMNS
    summ = json.loads((default_run / "summary.json").read_text())
    for f in summ["figures"]:
        assert (default_run / f["path"]).is_file() and len(f["caption"]) > 40
    prov = json.loads((default_run / "provenance.json").read_text())
    assert prov["app"] == "neuro-decoding" and prov["sessions"][IBL]["status"] == "ok"


def test_position_decoding(default_run):
    df = pd.read_csv(default_run / "decoding.csv")
    for sid in (RAT1, RAT2):
        r = _row(df, sid, "position")
        assert r.score > 0.5 and r.p_value < 0.05 and r.score > r.null_p95, sid
        assert _row(df, sid, "position", "bayes").score > 0.6, sid
        assert _row(df, sid, "position", "knn", features="cebra_time").score > 0.5, sid
        assert _row(df, sid, "position", "knn", features="cebra_behavior").score > 0.5, sid
    ho = pd.read_csv(default_run / "heldout.csv")
    ho = ho[ho.variable == "position"]
    assert len(ho) == 2 and (ho.aligned > 0.3).all() and (ho.aligned > ho.unaligned).all()


def test_ibl_regions(default_run):
    """Planted: VISp encodes the stimulus side, MOs the choice and wheel, CA1 nothing."""
    df = pd.read_csv(default_run / "decoding.csv")
    assert _row(df, IBL, "wheel_velocity").score > 0.5
    assert _row(df, IBL, "wheel_velocity", region="MOs").score > 0.5
    assert _row(df, IBL, "wheel_velocity", region="CA1").p_value > 0.05
    st = {r: _row(df, IBL, "stimulus_side", "logistic", r).score for r in ("all", "VISp", "MOs", "CA1")}
    assert st["VISp"] > 0.8 and st["CA1"] < 0.65
    ch = _row(df, IBL, "choice", "logistic", "MOs")
    assert ch.score > 0.85 and ch.p_value < 0.05 and ch.chance == 0.5
    assert _row(df, IBL, "choice", "logistic", "CA1").p_value > 0.05
    # IBL-style task (prior blocks, contrasts): pseudo-session nulls, null-corrected scores
    assert ch.null == "pseudo-session" and ch.score_corrected > 0.3
    assert _row(df, IBL, "wheel_velocity").null == "circular shift"
    reg = pd.read_csv(default_run / "regions.csv")
    vis = reg[(reg.variable == "stimulus_side") & (reg.region == "VISp")].iloc[0]
    assert vis.fisher_p < 0.05 and vis.n_sessions == 1


def test_dropping_curve_rises(default_run):
    d = pd.read_csv(default_run / "dropping.csv")
    c = d[(d.session_id == RAT2) & (d.variable == "position")].sort_values("n_neurons")
    assert c["mean"].iloc[-1] > c["mean"].iloc[0] + 0.3


def test_flat_layout_no_embedding(fixture_dir, tmp_path, monkeypatch):
    assert run(monkeypatch, fixture_dir / "flat", tmp_path, DECODERS="ridge,logistic", N_SHUFFLES=0,
               VARIABLES="position,wheel_velocity", TRIAL_VARIABLES="choice") == 0
    df = pd.read_csv(tmp_path / "neuro-decoding" / "decoding.csv")
    assert set(df.decoder) == {"ridge", "logistic"} and set(df.variable) == {"position", "wheel_velocity", "choice"}


def test_bad_parameters_and_empty_input(fixture_dir, tmp_path, monkeypatch):
    assert run(monkeypatch, fixture_dir / "linked", tmp_path, DECODERS="svm") == 2
    assert run(monkeypatch, fixture_dir / "linked", tmp_path, CV_FOLDS=1) == 2
    assert run(monkeypatch, fixture_dir / "linked", tmp_path, TRIAL_WINDOW_S="0.5,0.1") == 2
    e = tmp_path / "e"
    e.mkdir()
    assert run(monkeypatch, e, tmp_path / "o") == 1


def test_blocked_folds_gap():
    t = np.arange(1000) * 0.1
    folds = D.blocked_folds(t, 5, 2.0)
    assert len(folds) == 5
    for tr, te in folds:
        assert np.min(np.abs(t[tr][:, None] - t[te][None, :])) > 2.0 - 1e-9
        assert np.all(np.diff(te) == 1)


def test_null_detects_nothing_in_noise():
    rng = np.random.default_rng(0)
    t = np.arange(3000) * 0.1
    F = rng.standard_normal((3000, 20))
    y = np.cumsum(rng.standard_normal(3000))
    folds = D.blocked_folds(t, 5, 2.0)
    rd = D.ridge_cv(F, y, folds)
    st = D.null_stats(rd["score"], D.circular_null(F, y, t, folds, rd["alpha"], 50, 20, rng))
    assert st["p_value"] > 0.05


def test_pseudo_sessions_follow_the_task():
    from processor import pseudo as P

    rng = np.random.default_rng(0)
    pl = P._blocks(500, rng)
    assert (pl[:90] == 0.5).all() and set(np.unique(pl[90:])) == {0.2, 0.8}
    runs = np.diff(np.flatnonzero(np.diff(pl[90:]) != 0))
    assert runs.min() >= 20 and runs.max() <= 100


def test_communication_subspace(default_run):
    """VISp and MOs share one planted latent; CA1 shares nothing."""
    r = pd.read_csv(default_run / "rrr.csv")
    r = r[r.session_id == IBL].set_index(["source", "target"])
    assert r.loc[("VISp", "MOs"), "full_r2"] > 0.05 and r.loc[("VISp", "MOs"), "optimal_rank"] <= 2
    assert r.loc[("VISp", "MOs"), "full_r2"] > r.loc[("CA1", "MOs"), "full_r2"] + 0.05
    assert r.loc[("MOs", "MOs"), "kind"] == "within"
    assert (default_run / IBL / f"{IBL}_rrr.png").is_file()
