"""
likelihood_ratio_catalog.py — TEST LR (plan 5): full-catalog likelihood ratio
via a classifier density ratio (the SBI likelihood-ratio trick, Cranmer/Brehmer/Louppe).

Two simulated universes that differ ONLY in the cold super-Earth corner:
    B = flat universe, everything kept            ("cold large rocky planets exist, flat rate")
    A = flat universe minus the TRUE corner       ("cold large rocky planets do not exist")
        corner = rocky (below silicate line) & mass > 2 M_earth & insolation < 50 I_earth
Both pass through the SAME repo detectors (Kepler transit AND best-of-HARPS/NIRPS RV, as in
script 79) and the SAME observation noise, so every difference the classifier can find is the
corner and nothing else. The classifier output s(x) gives the per-planet density ratio
r(x) = p_B(x)/p_A(x); since A = B off-corner, log r ~ 0 away from the corner — hot planets
are spectators, the statistic surgically weighs the corner region.

STEP 1  FORWARD SIM   flat pool -> joint Kepler+RV detection (cached) -> NASA-resampled
                      observation noise -> observed-box cut. A-detected = B-detected minus
                      TRUE-corner members (planets are independent, so this is exact).
STEP 2  FEATURES      x = (log10 M_obs, log10 R_obs, log10 I_obs, Teff)
STEP 3  CLASSIFIERS   HistGradientBoosting (sklearn; same GBT family as the plan's XGBoost,
                      already installed): clf_full on all 4 features and clf_loc on location
                      only (log I, Teff).  l_full = logit(s) - log(N_B/N_A);
                      l_cond = l_full - l_loc  = log ratio of p(M,R | I,Teff)  — composition
                      at fixed location; location-targeting effort cancels here the same way
                      the rocky SHARE cancelled it in TEST A.
STEP 4  STATISTIC     T = sum_i l(x_i) over the N observed planets, CONDITIONAL on N.
                      The plan's Poisson count term (Lambda_B - Lambda_A) is dropped on
                      purpose: TEST A proved absolute corner counts are follow-up-poisoned
                      (rocky 1.8x, puffy 6.3x above expectation), and the flat universe has
                      no absolute rate to calibrate it with anyway.
STEP 5  CALIBRATION   exact-frequentist: 10^4 catalogs of size N resampled from the held-out
                      A pool (null) and B pool (power). p(reject B) = P(T_B <= T_obs);
                      compatibility with A two-sided; power at alpha = 0.05.
                      HEADLINE variant: null catalogs LOCATION-MATCHED to NASA — planet i's
                      null l is drawn from held-out sim planets at NASA planet i's own
                      (log I, Teff) location (k-nearest neighbors). This conditions on the
                      observed location set, so NASA's cold-heavy targeting cannot leak into
                      the calibration either (pool-mix nulls have the flat location mix and
                      would let it leak — both are reported so the gap is visible).

Validity guardrails (printed): held-out AUC + reliability table (classifier calibration),
permutation feature importances (should scream mass-radius-insolation, not Teff),
train/calib split so null catalogs never reuse training rows.

Run:
    python scripts/likelihood_ratio_catalog.py
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
from pathlib import Path

from tools.paths import LIFESIM_OUTER_DIR, SILICON_CURVE, PSCOMPPARS_CSV
ROOT = Path(LIFESIM_OUTER_DIR)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from run.flat_universe.uniform_generator import generate_flat_catalog
from run.ppop.flat_detect import run_kepler, run_rv_best

SILICATE_CURVE = Path(SILICON_CURVE)
NASA_FILE = Path(PSCOMPPARS_CSV)
OUT_DIR = os.path.join(ROOT, "my_outputs", "likelihood_ratio_catalog")

FLAT_N_POOL = 1_000_000
RNG_SEED = 0                      # same flat universe family as script 79
RV_MAG_TARGET = 12.0
MASS_MIN = 2.0
COLD_MAX = 50.0
NASA_MASS_PREC = 0.25
NASA_RAD_PREC = 0.08
BOX = dict(r_lo=0.5, r_hi=2.2, m_lo=0.1, m_hi=12.0, f_lo=1e-2, f_hi=1e4)
TEFF_STRATA = (0.0, 7500.0)       # FGK/M window, matches scripts 79/80/81

SIGMA_LOGM_FLOOR = 0.02           # dex; conventions from script 81
SIGMA_LOGM_MISSING = 0.20
SIGMA_LOGR_FLOOR = 0.01
SIGMA_LOGI_FLOOR = 0.02

FEATURES = ["logM", "logR", "logI", "Teff"]
LOC_COLS = [2, 3]                 # (logI, Teff) columns inside the feature matrix

N_CATALOGS = 10_000
ALPHA = 0.05
S_CLIP = 1e-6
K_NEIGH = 100                     # location-matched null: kNN in (logI, Teff)
MAP_INSOL = 20.0                  # cold slice shown in panel (a)

CLF_KW = dict(max_iter=200, learning_rate=0.08, min_samples_leaf=60,
              l2_regularization=1.0, early_stopping=False, random_state=42)

CONFIGS = [("precision", True, 11), ("full", False, 12)]   # (name, precision cut, noise seed)


def load_silicate():
    d = np.loadtxt(SILICATE_CURVE, comments="#")
    m, r = d[:, 0].astype(float), d[:, 1].astype(float)
    o = np.argsort(m)
    return m[o], r[o]


def _sigma_log(err_hi, err_lo, value, floor, missing):
    e = np.maximum(np.abs(err_hi), np.abs(err_lo))
    s = (e / value) / np.log(10.0)
    s = s.where(s.notna(), missing)
    return np.maximum(s, floor)


def load_nasa(precision: bool):
    df = pd.read_csv(NASA_FILE, comment="#", low_memory=False)
    m = pd.to_numeric(df["pl_bmasse"], errors="coerce")
    r = pd.to_numeric(df["pl_rade"], errors="coerce")
    ins = pd.to_numeric(df["pl_insol"], errors="coerce")
    teff = pd.to_numeric(df["st_teff"], errors="coerce")
    prov = df.get("pl_bmassprov", pd.Series("", index=df.index)).astype(str)
    meas = prov.str.contains("Mass|Msini", case=False, na=False) & \
        ~prov.str.contains("Calc", case=False, na=False)
    me1 = pd.to_numeric(df["pl_bmasseerr1"], errors="coerce")
    me2 = pd.to_numeric(df["pl_bmasseerr2"], errors="coerce")
    re1 = pd.to_numeric(df["pl_radeerr1"], errors="coerce")
    re2 = pd.to_numeric(df["pl_radeerr2"], errors="coerce")
    ie1 = pd.to_numeric(df["pl_insolerr1"], errors="coerce")
    ie2 = pd.to_numeric(df["pl_insolerr2"], errors="coerce")

    keep = (meas & r.between(BOX["r_lo"], BOX["r_hi"]) & m.between(BOX["m_lo"], BOX["m_hi"])
            & ins.between(BOX["f_lo"], BOX["f_hi"]) & teff.between(*TEFF_STRATA))
    if precision:
        me = np.maximum(me1.abs(), me2.abs())
        re = np.maximum(re1.abs(), re2.abs())
        keep = keep & (me / m <= NASA_MASS_PREC) & (re / r <= NASA_RAD_PREC)

    sig_m = _sigma_log(me1, me2, m, SIGMA_LOGM_FLOOR, SIGMA_LOGM_MISSING)
    sr = _sigma_log(re1, re2, r, SIGMA_LOGR_FLOOR, np.nan)
    si = _sigma_log(ie1, ie2, ins, SIGMA_LOGI_FLOOR, np.nan)
    sig_r = sr.fillna(np.nanmedian(sr[keep]))
    sig_i = si.fillna(np.nanmedian(si[keep]))
    n_miss = (int(np.maximum(me1.abs(), me2.abs())[keep].isna().sum()),
              int(sr[keep].isna().sum()), int(si[keep].isna().sum()))

    tag = "precision-cut" if precision else "full measured-mass"
    print(f"    NASA ({tag}): {int(keep.sum())} planets in box + Teff window "
          f"{TEFF_STRATA} K; missing errors (m/r/I) = {n_miss} "
          f"-> fallbacks {SIGMA_LOGM_MISSING}/median/median dex")
    return dict(m=m[keep].to_numpy(), r=r[keep].to_numpy(), ins=ins[keep].to_numpy(),
                teff=teff[keep].to_numpy(),
                sig_m=sig_m[keep].to_numpy(), sig_r=sig_r[keep].to_numpy(),
                sig_i=sig_i[keep].to_numpy())


# ---------------------------------------------------------------- STEP 1: forward sim
def get_detected_pool(df):
    """TRUE parameters of joint (Kepler AND RV) detected planets, from the
    flat universe produced by run_flat_universe. Universe B is the keep-all
    population; universe A drops rocky M>2 and is not the pool used here."""
    pool = df[df["universe_type"] == "B"]
    joint = (pool["kepler_detected"].to_numpy(bool)
             & pool["rv_detected"].to_numpy(bool))
    m_sil, r_sil = load_silicate()
    det = pd.DataFrame({
        "mass": pool["mass_p"].to_numpy(float)[joint],
        "radius": pool["radius_p"].to_numpy(float)[joint],
        "flux": pool["flux_p"].to_numpy(float)[joint],
        "teff": pool["teff_s"].to_numpy(float)[joint],
    })
    rocky = det["radius"].to_numpy() < np.interp(det["mass"].to_numpy(), m_sil, r_sil)
    det["corner"] = rocky & (det["mass"].to_numpy() > MASS_MIN) & \
        (det["flux"].to_numpy() < COLD_MAX)
    print(f"    joint-detected {len(det)}/{len(pool)} ({100*len(det)/len(pool):.2f}%)")
    pi = det["corner"].mean()
    print(f"    TRUE-corner share of detected planets (pi, universe B): {100 * pi:.2f}%  "
          f"({int(det['corner'].sum())} corner planets)")
    return det


def apply_noise(det: pd.DataFrame, nasa: dict, rng) -> dict:
    """NASA-resampled per-planet noise: draw (sig_logM, sig_logR, sig_logI) triples from the
    NASA sample rows (keeps their correlations), perturb the TRUE sim values, cut back to the
    observed box (same selection NASA got)."""
    n = len(det)
    pick = rng.integers(0, len(nasa["m"]), n)
    sm, sr, si = nasa["sig_m"][pick], nasa["sig_r"][pick], nasa["sig_i"][pick]
    logm = np.log10(det["mass"].to_numpy()) + rng.normal(0.0, 1.0, n) * sm
    logr = np.log10(det["radius"].to_numpy()) + rng.normal(0.0, 1.0, n) * sr
    logi = np.log10(det["flux"].to_numpy()) + rng.normal(0.0, 1.0, n) * si
    teff = det["teff"].to_numpy()
    inbox = ((logr >= np.log10(BOX["r_lo"])) & (logr <= np.log10(BOX["r_hi"]))
             & (logm >= np.log10(BOX["m_lo"])) & (logm <= np.log10(BOX["m_hi"]))
             & (logi >= np.log10(BOX["f_lo"])) & (logi <= np.log10(BOX["f_hi"])))
    X = np.column_stack([logm, logr, logi, teff])[inbox]
    corner = det["corner"].to_numpy(bool)[inbox]
    print(f"    noise applied (NASA-resampled sigmas); observed-box keeps "
          f"{inbox.sum()}/{n} detected planets")
    return dict(X=X, corner=corner)


# ------------------------------------------------------- STEP 3: classifier machinery
def fit_ratio_classifier(XB: np.ndarray, XA: np.ndarray, cols=None):
    """A-vs-B classifier -> per-planet log density ratio l(x) = log p_B(x)/p_A(x).
    Off-corner the two classes hold the SAME rows, so s -> N_B/(N_B+N_A) there and the
    log(N_B/N_A) offset makes l(x) -> log(1 - pi) exactly, as it should."""
    if cols is not None:
        XB, XA = XB[:, cols], XA[:, cols]
    X = np.vstack([XB, XA])
    y = np.concatenate([np.ones(len(XB)), np.zeros(len(XA))])
    clf = HistGradientBoostingClassifier(**CLF_KW).fit(X, y)
    offset = np.log(len(XB) / len(XA))
    return clf, offset


def log_ratio(clf, offset, X, cols=None):
    if cols is not None:
        X = X[:, cols]
    s = np.clip(clf.predict_proba(X)[:, 1], S_CLIP, 1.0 - S_CLIP)
    return np.log(s / (1.0 - s)) - offset


def reliability_table(clf, offset, XB, XA, cols=None, n_bins=10):
    XB_ = XB[:, cols] if cols is not None else XB
    XA_ = XA[:, cols] if cols is not None else XA
    X = np.vstack([XB_, XA_])
    y = np.concatenate([np.ones(len(XB_)), np.zeros(len(XA_))])
    s = clf.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, s)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (s >= lo) & (s < hi if hi < 1 else s <= hi)
        if sel.sum() >= 30:
            rows.append((0.5 * (lo + hi), s[sel].mean(), y[sel].mean(), int(sel.sum())))
    return auc, rows


# ------------------------------------------------ STEP 4+5: statistic and calibration
def calibrate(ell_pool_A, ell_pool_B, t_obs, n_cat, rng):
    """Pool-mix null: catalogs of size N resampled from the held-out pools (flat-selection
    location mix). Rejecting B needs T_obs in B's LOW tail."""
    n = t_obs["n"]
    TA = ell_pool_A[rng.integers(0, len(ell_pool_A), (n_cat, n))].sum(axis=1)
    TB = ell_pool_B[rng.integers(0, len(ell_pool_B), (n_cat, n))].sum(axis=1)
    return _pvals(TA, TB, t_obs["T"], n_cat)


def _pvals(TA, TB, t, n_cat):
    p_reject_B = (1 + np.sum(TB <= t)) / (1 + n_cat)
    p_lo = (1 + np.sum(TA <= t)) / (1 + n_cat)
    p_hi = (1 + np.sum(TA >= t)) / (1 + n_cat)
    p_compat_A = min(1.0, 2.0 * min(p_lo, p_hi))
    crit = np.quantile(TB, ALPHA)
    power = float(np.mean(TA <= crit))
    return dict(TA=TA, TB=TB, T=t, p_reject_B=p_reject_B, p_compat_A=p_compat_A,
                power=power, crit=crit)


def calibrate_matched(neigh_A, neigh_B, t_obs, n_cat, rng):
    """Fully conditional null: planet i's null l is drawn from its OWN location-matched
    neighbor pool (N, K). Conditions on N and on the observed location set."""
    n, k = neigh_A.shape
    rows = np.arange(n)[None, :]
    TA = neigh_A[rows, rng.integers(0, k, (n_cat, n))].sum(axis=1)
    TB = neigh_B[rows, rng.integers(0, k, (n_cat, n))].sum(axis=1)
    return _pvals(TA, TB, t_obs, n_cat)


def neighbor_ell(Xn, X_pool, ell_pool, scale, k):
    nn = NearestNeighbors(n_neighbors=k).fit(X_pool[:, LOC_COLS] / scale)
    dist, idx = nn.kneighbors(Xn[:, LOC_COLS] / scale)
    return ell_pool[idx], float(np.median(dist[:, -1]))


def run_config(det, nasa, name, noise_seed):
    print(f"\n  ================ TEST LR — {name} NASA sample ================")
    rng = np.random.default_rng(noise_seed)

    print("  STEP 1 — forward-sim noise + observed-box cut")
    obs = apply_noise(det, nasa, rng)
    X, corner = obs["X"], obs["corner"]

    print("  STEP 2 — features x = (logM, logR, logI, Teff); split train/calib 50/50")
    half = rng.permutation(len(X))
    tr, ca = half[: len(X) // 2], half[len(X) // 2:]
    XB_tr, XB_ca = X[tr], X[ca]
    cor_tr, cor_ca = corner[tr], corner[ca]
    XA_tr, XA_ca = XB_tr[~cor_tr], XB_ca[~cor_ca]
    n_cor_tr = int(cor_tr.sum())
    print(f"    train B/A = {len(XB_tr)}/{len(XA_tr)} (corner in train: {n_cor_tr}); "
          f"calib B/A = {len(XB_ca)}/{len(XA_ca)}")
    if n_cor_tr < 50:
        print("    [WARN] <50 corner planets in training — raise FLAT_N_POOL")

    print("  STEP 3 — classifiers (HistGB): full(4 feats) + location-only(logI, Teff)")
    clf_full, off_full = fit_ratio_classifier(XB_tr, XA_tr)
    clf_loc, off_loc = fit_ratio_classifier(XB_tr, XA_tr, cols=LOC_COLS)
    auc_full, rel_full = reliability_table(clf_full, off_full, XB_ca, XA_ca)
    auc_loc, _ = reliability_table(clf_loc, off_loc, XB_ca, XA_ca, cols=LOC_COLS)
    print(f"    held-out AUC: full = {auc_full:.4f}, location-only = {auc_loc:.4f} "
          "(both barely >0.5 by design: A and B differ only in the small corner)")
    print("    reliability (held-out): bin-center | mean predicted s | observed B-fraction | n")
    for c, sp, yo, nn in rel_full:
        print(f"      {c:>5.2f} | {sp:>6.3f} | {yo:>6.3f} | {nn}")

    sub = np.random.default_rng(1).permutation(len(XB_ca))[:20000]
    Xi = np.vstack([XB_ca[sub], XA_ca[np.random.default_rng(2).permutation(len(XA_ca))[:20000]]])
    yi = np.concatenate([np.ones(min(20000, len(XB_ca))), np.zeros(min(20000, len(XA_ca)))])
    imp = permutation_importance(clf_full, Xi, yi, n_repeats=3, random_state=3,
                                 scoring="roc_auc")
    imp_txt = ", ".join(f"{f}={v:.4f}" for f, v in zip(FEATURES, imp.importances_mean))
    print(f"    permutation importance (AUC drop): {imp_txt}")

    ell_full_ca = log_ratio(clf_full, off_full, XB_ca)
    ell_loc_ca = log_ratio(clf_loc, off_loc, XB_ca, cols=LOC_COLS)
    ell_cond_ca = ell_full_ca - ell_loc_ca

    Xn = np.column_stack([np.log10(nasa["m"]), np.log10(nasa["r"]),
                          np.log10(nasa["ins"]), nasa["teff"]])
    ell_full_n = log_ratio(clf_full, off_full, Xn)
    ell_cond_n = ell_full_n - log_ratio(clf_loc, off_loc, Xn, cols=LOC_COLS)
    n_nasa = len(Xn)

    print("  STEP 4 — T = sum_i l(x_i) | N  (count term dropped: absolute corner counts are"
          " follow-up-poisoned per TEST A; flat universe carries no absolute rate)")
    print(f"    dropped count factor for reference: Lambda_B/Lambda_A = "
          f"{len(XB_ca) / len(XA_ca):.4f}")

    print(f"  STEP 5 — calibration: {N_CATALOGS} catalogs x N={n_nasa} from held-out pools")
    rng_cal = np.random.default_rng(100 + noise_seed)
    res = {}

    scale = XB_ca[:, LOC_COLS].std(axis=0)
    neigh_B, dB = neighbor_ell(Xn, XB_ca, ell_cond_ca, scale, K_NEIGH)
    neigh_A, dA = neighbor_ell(Xn, XB_ca[~cor_ca], ell_cond_ca[~cor_ca], scale, K_NEIGH)
    res["cond_loc"] = calibrate_matched(neigh_A, neigh_B, float(ell_cond_n.sum()),
                                        N_CATALOGS, rng_cal)
    for stat, pool, t_nasa in [
        ("shape", ell_full_ca, float(ell_full_n.sum())),
        ("cond", ell_cond_ca, float(ell_cond_n.sum())),
    ]:
        res[stat] = calibrate(pool[~cor_ca], pool, dict(T=t_nasa, n=n_nasa),
                              N_CATALOGS, rng_cal)

    labels = [
        ("shape", "T_shape    (all axes; location-targeting effort NOT removed — the"
                  " confound demo)"),
        ("cond", "T_cond     (per-planet weight conditioned; null still has the FLAT"
                 " location mix)"),
        ("cond_loc", "T_cond|loc (null catalogs location-matched to NASA — fully"
                     " conditional HEADLINE)"),
    ]
    for stat, label in labels:
        r = res[stat]
        print(f"    {label}")
        print(f"      T_NASA = {r['T']:+8.2f} | null A: {np.mean(r['TA']):+7.2f} "
              f"± {np.std(r['TA']):.2f} | alt B: {np.mean(r['TB']):+7.2f} ± {np.std(r['TB']):.2f}")
        print(f"      p(reject B, one-sided) = {r['p_reject_B']:.4f} | "
              f"p(compat A, two-sided) = {r['p_compat_A']:.4f} | "
              f"power(A vs B at 5%) = {r['power']:.2f}")
    print(f"    location matching: K={K_NEIGH} neighbors in standardized (logI, Teff); "
          f"median K-th neighbor distance A/B = {dA:.3f}/{dB:.3f} (scaled units; <~0.3 = "
          "matching is tight)")

    order = np.argsort(ell_cond_n)[::-1]
    print("    top NASA contributors to T_cond (planets that look like corner members):")
    for i in order[:6]:
        print(f"      M={nasa['m'][i]:6.2f} M⊕  R={nasa['r'][i]:5.2f} R⊕  "
              f"I={nasa['ins'][i]:8.1f} I⊕  Teff={nasa['teff'][i]:6.0f} K  "
              f"l_cond={ell_cond_n[i]:+.3f}")
    pos = ell_cond_n[ell_cond_n > 0].sum()
    neg = ell_cond_n[ell_cond_n < 0].sum()
    print(f"    T_cond split: positive planets {pos:+.2f}, negative planets {neg:+.2f}")

    per_planet = pd.DataFrame(dict(m=nasa["m"], r=nasa["r"], ins=nasa["ins"],
                                   teff=nasa["teff"], ell_full=ell_full_n,
                                   ell_cond=ell_cond_n))
    csv = os.path.join(OUT_DIR, f"nasa_per_planet_ell_{name}.csv")
    per_planet.to_csv(csv, index=False)
    print(f"    per-planet NASA log-ratios saved -> {csv}")

    return dict(name=name, res=res, auc_full=auc_full, auc_loc=auc_loc, rel=rel_full,
                imp=imp.importances_mean, clf_full=clf_full, off_full=off_full,
                clf_loc=clf_loc, off_loc=off_loc, nasa=nasa, ell_cond_n=ell_cond_n,
                n_nasa=n_nasa, X_obs=X, corner_obs=corner, XB_ca=XB_ca, cor_ca=cor_ca,
                ell_cond_ca=ell_cond_ca)


# ---------------------------------------------------------------------------- figure
def panel_map(ax, fig, cfg, m_sil, r_sil):
    teff0 = float(np.median(cfg["nasa"]["teff"]))
    lm = np.linspace(np.log10(BOX["m_lo"]), np.log10(BOX["m_hi"]), 90)
    rr = np.linspace(BOX["r_lo"], BOX["r_hi"], 90)
    LM, RR = np.meshgrid(lm, rr)
    G = np.column_stack([LM.ravel(), np.log10(RR).ravel(),
                         np.full(LM.size, np.log10(MAP_INSOL)), np.full(LM.size, teff0)])
    ell = log_ratio(cfg["clf_full"], cfg["off_full"], G) \
        - log_ratio(cfg["clf_loc"], cfg["off_loc"], G, cols=LOC_COLS)
    Z = ell.reshape(LM.shape)
    vmax = max(np.abs(Z).max(), 0.1)
    pc = ax.pcolormesh(10.0 ** LM, RR, Z, cmap="RdBu_r",
                       norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax))
    fig.colorbar(pc, ax=ax, label=r"$\ell_{\rm cond}(x)=\log\,p_B/p_A\,(M,R\,|\,I,T_{\rm eff})$")
    ms = np.logspace(np.log10(BOX["m_lo"]), np.log10(BOX["m_hi"]), 200)
    ax.plot(ms, np.interp(ms, m_sil, r_sil), color="k", lw=1.5, label="silicate line")
    ax.axvline(MASS_MIN, color="red", ls="--", lw=1.2, label=f"M = {MASS_MIN:g} M⊕")
    nasa = cfg["nasa"]
    cold = nasa["ins"] < COLD_MAX
    ax.scatter(nasa["m"][cold], nasa["r"][cold], s=28, facecolor="tab:green",
               edgecolor="k", lw=0.4, zorder=5, label=f"NASA cold (I<{COLD_MAX:g})")
    ax.set_xscale("log")
    ax.set_xlim(BOX["m_lo"], BOX["m_hi"]); ax.set_ylim(BOX["r_lo"], BOX["r_hi"])
    ax.set_xlabel(r"planet mass [$M_\oplus$]"); ax.set_ylabel(r"planet radius [$R_\oplus$]")
    ax.set_title(f"(a) what the classifier learned — per-planet log-LR at "
                 f"I={MAP_INSOL:g} I⊕, Teff={teff0:.0f} K\n"
                 "(red = looks like a corner member; blue/white = spectator)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")


def panel_calibration(ax, r, stat_label, n_nasa, extra=None):
    bins = np.linspace(min(r["TA"].min(), r["TB"].min(), r["T"]) - 1,
                       max(r["TA"].max(), r["TB"].max(), r["T"]) + 1, 60)
    ax.hist(r["TA"], bins=bins, density=True, color="tab:blue", alpha=0.55,
            label="universe A catalogs (corner empty)")
    ax.hist(r["TB"], bins=bins, density=True, color="tab:orange", alpha=0.55,
            label="universe B catalogs (flat corner)")
    ax.axvline(r["T"], color="tab:green", lw=2.5, label=f"NASA (N={n_nasa})")
    ax.axvline(r["crit"], color="0.4", ls=":", lw=1.2, label=f"5% critical value of B")
    txt = (f"p(reject B) = {r['p_reject_B']:.4f}\n"
           f"p(compat A) = {r['p_compat_A']:.4f}\npower = {r['power']:.2f}")
    if extra:
        txt += "\n" + extra
    ax.text(0.02, 0.975, txt, transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round", fc="whitesmoke", ec="0.7"))
    ax.set_xlabel(r"catalog statistic  $T=\sum_i \ell(x_i)$"); ax.set_ylabel("density")
    ax.set_title(stat_label, fontsize=10)
    ax.grid(alpha=0.2); ax.legend(fontsize=8, loc="upper right")


def panel_reliability(ax, cfg):
    c = np.array([[x, sp, yo] for x, sp, yo, _ in cfg["rel"]])
    ax.plot([0, 1], [0, 1], ls="--", color="0.6", label="perfect calibration")
    ax.plot(c[:, 1], c[:, 2], marker="o", color="tab:blue", label="clf_full (held-out)")
    ax.set_xlabel("mean predicted P(B)"); ax.set_ylabel("observed B fraction")
    imp_txt = "\n".join(f"{f}: {v:.4f}" for f, v in zip(FEATURES, cfg["imp"]))
    ax.text(0.98, 0.02, f"AUC full = {cfg['auc_full']:.4f}\nAUC loc = {cfg['auc_loc']:.4f}\n"
            f"permutation importance\n(AUC drop):\n{imp_txt}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            bbox=dict(boxstyle="round", fc="whitesmoke", ec="0.7"))
    ax.set_title("(d) classifier validity — reliability + feature importances", fontsize=10)
    ax.grid(alpha=0.2); ax.legend(fontsize=8, loc="upper left")


def make_figure(cfg_prec, cfg_full, m_sil, r_sil):
    fig, axes = plt.subplots(2, 2, figsize=(17, 11.5))
    panel_map(axes[0, 0], fig, cfg_prec, m_sil, r_sil)
    extra = (f"pool-mix-null T_cond: p(reject B) = "
             f"{cfg_prec['res']['cond']['p_reject_B']:.4f}\n"
             f"full sample (N={cfg_full['n_nasa']}): p(reject B) = "
             f"{cfg_full['res']['cond_loc']['p_reject_B']:.4f}, "
             f"p(compat A) = {cfg_full['res']['cond_loc']['p_compat_A']:.4f}")
    panel_calibration(axes[0, 1], cfg_prec["res"]["cond_loc"],
                      "(b) HEADLINE  T_cond | locations — composition at NASA's own "
                      "(I, Teff) locations; null location-matched — precision NASA",
                      cfg_prec["n_nasa"], extra=extra)
    panel_calibration(axes[1, 0], cfg_prec["res"]["shape"],
                      "(c) T_shape — all axes, effort NOT removed (shown to expose the "
                      "confound, not as the test) — precision NASA",
                      cfg_prec["n_nasa"])
    panel_reliability(axes[1, 1], cfg_prec)
    fig.suptitle("TEST LR — full-catalog likelihood ratio via classifier density ratio "
                 "(SBI likelihood-ratio trick)\n"
                 "A = flat universe MINUS cold rocky M>2 corner, B = flat universe; both through "
                 "the repo Kepler+RV joint detectors + NASA-resampled noise; "
                 r"$T=\sum_i \ell(x_i)$ conditional on N + observed locations (count term "
                 "dropped — TEST A showed absolute corner counts are follow-up-confounded)",
                 fontsize=11)
    fig.text(0.5, 0.005,
             "Caveats: detectors are toy transit+RV proxies for NASA's heterogeneous selection "
             "(T_cond shields location-targeting effort but NOT composition-dependent publication "
             "bias within a location); classifier imperfection loses power, never validity "
             "(p-values calibrated by simulation on held-out planets); corner defined in TRUE "
             "parameters via the silicate line; NASA Msini treated as mass (transiting: sin i ≈ 1).",
             ha="center", fontsize=8.5)
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    out_png = os.path.join(OUT_DIR, "full_catalog_lr.png")
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"\n--> Saved: {out_png}")


# ------------------------------------------------- explainer figures (MRI-space bridge)
# Same axes and conventions as the mass-radius / radius-insolation detection figures the
# audience already knows (scripts 44/53/75/79); one pipeline concept per figure.
EXPL_SUB = 900


def _mr_axis(ax, m_sil, r_sil, shade_corner=True):
    ms = np.logspace(np.log10(BOX["m_lo"]), np.log10(BOX["m_hi"]), 200)
    ax.plot(ms, np.interp(ms, m_sil, r_sil), "k-", lw=1.4, label="silicate line")
    ax.axvline(MASS_MIN, color="red", ls="--", lw=1.1, label=f"M = {MASS_MIN:g} M⊕")
    if shade_corner:
        mc = np.logspace(np.log10(MASS_MIN), np.log10(BOX["m_hi"]), 120)
        ax.fill_between(mc, BOX["r_lo"], np.clip(np.interp(mc, m_sil, r_sil),
                                                 BOX["r_lo"], BOX["r_hi"]),
                        color="red", alpha=0.07, zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(BOX["m_lo"], BOX["m_hi"]); ax.set_ylim(BOX["r_lo"], BOX["r_hi"])
    ax.set_xlabel(r"planet mass [$M_\oplus$]")
    ax.grid(alpha=0.2)


def explainer_universes(cfg, m_sil, r_sil):
    """E1 — the whole test as three M-R scatter panels (cold slice, observed values)."""
    X, corner, nasa = cfg["X_obs"], cfg["corner_obs"], cfg["nasa"]
    m, r, i = 10.0 ** X[:, 0], 10.0 ** X[:, 1], 10.0 ** X[:, 2]
    cold = i < COLD_MAX
    rng = np.random.default_rng(5)
    pick = rng.permutation(np.where(cold)[0])[:EXPL_SUB]
    pc, pn = pick[corner[pick]], pick[~corner[pick]]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), sharey=True)
    for ax in axes:
        _mr_axis(ax, m_sil, r_sil)
    axes[0].scatter(m[pn], r[pn], s=8, color="0.65", alpha=0.55, label="other cold planets")
    axes[0].scatter(m[pc], r[pc], s=16, color="tab:red", alpha=0.8,
                    label="corner planets (cold rocky M>2)")
    axes[0].set_title("(a) universe B — flat occurrence:\ncorner planets EXIST (red)",
                      fontsize=10)
    axes[1].scatter(m[pn], r[pn], s=8, color="0.65", alpha=0.55)
    axes[1].set_title("(b) universe A — same universe,\ncorner planets REMOVED at birth",
                      fontsize=10)
    ncold = nasa["ins"] < COLD_MAX
    axes[2].scatter(nasa["m"][ncold], nasa["r"][ncold], s=42, facecolor="tab:green",
                    edgecolor="k", lw=0.5, label=f"NASA cold planets (N={int(ncold.sum())})")
    axes[2].set_title("(c) NASA — which universe\ndoes it look like?", fontsize=10)
    axes[0].set_ylabel(r"planet radius [$R_\oplus$]")
    for ax in axes:
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("TEST LR in one look — the COLD (I<50 I⊕) mass-radius plane, all three "
                 "through the SAME detectors and the SAME NASA-sized noise\n"
                 "(shaded = the disputed corner; same axes as the M-R detection figures)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = os.path.join(OUT_DIR, "explainer_1_three_universes_mr.png")
    fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"--> Saved: {out}")


def _ell_grid(cfg, insol, teff0):
    lm = np.linspace(np.log10(BOX["m_lo"]), np.log10(BOX["m_hi"]), 90)
    rr = np.linspace(BOX["r_lo"], BOX["r_hi"], 90)
    LM, RR = np.meshgrid(lm, rr)
    G = np.column_stack([LM.ravel(), np.log10(RR).ravel(),
                         np.full(LM.size, np.log10(insol)), np.full(LM.size, teff0)])
    Z = (log_ratio(cfg["clf_full"], cfg["off_full"], G)
         - log_ratio(cfg["clf_loc"], cfg["off_loc"], G, cols=LOC_COLS)).reshape(LM.shape)
    return 10.0 ** LM, RR, Z


def explainer_scores(cfg, m_sil, r_sil):
    """E2 — the classifier score in the audience's own planes: M-R at a cold and a hot
    slice, then NASA planets colored by score in the (insolation, radius) plane."""
    nasa = cfg["nasa"]
    teff0 = float(np.median(nasa["teff"]))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), constrained_layout=True)
    Zs = [(_ell_grid(cfg, 20.0, teff0), f"(a) score map, COLD slice (I=20 I⊕)\n"
           "red = 'you look like a corner planet'"),
          (_ell_grid(cfg, 200.0, teff0), "(b) score map, HOT slice (I=200 I⊕)\n"
           "blank — hot planets are spectators, the score ignores them")]
    vmax = max(np.abs(Z).max() for (_, _, Z), _ in Zs)
    vmax = max(vmax, 0.1)
    for ax, ((M, R, Z), title) in zip(axes[:2], Zs):
        pc = ax.pcolormesh(M, R, Z, cmap="RdBu_r",
                           norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax))
        _mr_axis(ax, m_sil, r_sil, shade_corner=False)
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel(r"planet radius [$R_\oplus$]")
    fig.colorbar(pc, ax=axes[:2], label=r"per-planet score $\ell_{\rm cond}$", shrink=0.9)

    ax = axes[2]
    ell = cfg["ell_cond_n"]
    sc = ax.scatter(nasa["ins"], nasa["r"], c=ell, cmap="RdBu_r",
                    norm=TwoSlopeNorm(0.0, vmin=-max(np.abs(ell).max(), 0.1),
                                      vmax=max(np.abs(ell).max(), 0.1)),
                    s=24 + 40 * np.abs(ell), edgecolor="k", lw=0.4)
    ax.axvline(COLD_MAX, color="red", ls="--", lw=1.3)
    ax.text(COLD_MAX * 0.75, BOX["r_hi"] - 0.05, "← cold", color="red", ha="right",
            va="top", fontsize=9)
    ax.set_xscale("log"); ax.set_xlim(BOX["f_lo"], BOX["f_hi"])
    ax.set_ylim(BOX["r_lo"], BOX["r_hi"])
    ax.set_xlabel(r"insolation [$I_\oplus$]"); ax.grid(alpha=0.2)
    fig.colorbar(sc, ax=ax, label="NASA per-planet score")
    ax.set_title("(c) every NASA planet scored, in the familiar\n"
                 "(insolation, radius) plane — size = |score|", fontsize=10)
    fig.suptitle("What the classifier actually learned — scores drawn in the same planes as "
                 "the detection-fraction maps (script 79)\n"
                 "sum of the scores in (c) is the whole test statistic T", fontsize=12)
    out = os.path.join(OUT_DIR, "explainer_2_score_maps.png")
    fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"--> Saved: {out}")


def explainer_verdict(cfg, m_sil, r_sil):
    """E3 — the verdict without statistics jargon: corner-looking COUNTS per catalog, and
    the catalog scores as a number line."""
    XB_ca, cor_ca, nasa = cfg["XB_ca"], cfg["cor_ca"], cfg["nasa"]
    n = cfg["n_nasa"]
    m, r, i = 10.0 ** XB_ca[:, 0], 10.0 ** XB_ca[:, 1], 10.0 ** XB_ca[:, 2]
    obs_corner = (r < np.interp(m, m_sil, r_sil)) & (m > MASS_MIN) & (i < COLD_MAX)
    nasa_corner = int(((nasa["r"] < np.interp(nasa["m"], m_sil, r_sil))
                       & (nasa["m"] > MASS_MIN) & (nasa["ins"] < COLD_MAX)).sum())

    Xn = np.column_stack([np.log10(nasa["m"]), np.log10(nasa["r"]),
                          np.log10(nasa["ins"]), nasa["teff"]])
    scale = XB_ca[:, LOC_COLS].std(axis=0)
    rng = np.random.default_rng(6)
    counts = {}
    for lbl, pool_mask in [("universe A\n(corner empty)", ~cor_ca),
                           ("universe B\n(flat corner)", np.ones(len(cor_ca), bool))]:
        nn = NearestNeighbors(n_neighbors=K_NEIGH).fit(XB_ca[pool_mask][:, LOC_COLS] / scale)
        idx = nn.kneighbors(Xn[:, LOC_COLS] / scale, return_distance=False)
        flags = obs_corner[pool_mask][idx]                       # (N, K)
        pick = rng.integers(0, K_NEIGH, (800, n))
        counts[lbl] = flags[np.arange(n)[None, :], pick].sum(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    ax = axes[0]
    labels = list(counts) + ["NASA\n(observed)"]
    med = [np.median(c) for c in counts.values()] + [nasa_corner]
    lo = [np.percentile(c, 2.5) for c in counts.values()] + [nasa_corner]
    hi = [np.percentile(c, 97.5) for c in counts.values()] + [nasa_corner]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    ax.bar(labels, med, color=colors,
           yerr=[np.array(med) - np.array(lo), np.array(hi) - np.array(med)], capsize=6)
    for x, v in enumerate(med):
        ax.text(x, v + 0.4, f"{v:.0f}", ha="center", fontsize=10)
    ax.set_ylabel(f"corner-looking planets per {n}-planet catalog")
    ax.set_title("(a) plain counts: cold rocky M>2 planets (observed values)\n"
                 "fake catalogs built AT NASA's own planet locations", fontsize=10)
    ax.grid(alpha=0.2, axis="y")

    ax = axes[1]
    r_loc = cfg["res"]["cond_loc"]
    for y, (T, color, lbl) in enumerate([(r_loc["TA"], "tab:blue", "universe A catalogs"),
                                         (r_loc["TB"], "tab:orange", "universe B catalogs")]):
        sub = T[rng.permutation(len(T))[:400]]
        ax.scatter(sub, np.full(len(sub), y) + rng.uniform(-0.18, 0.18, len(sub)),
                   s=6, color=color, alpha=0.4, label=lbl)
    ax.axvline(r_loc["T"], color="tab:green", lw=2.5, label="NASA")
    ax.annotate("NASA lands here", xy=(r_loc["T"], 0.5), xytext=(r_loc["T"] + 25, 0.5),
                arrowprops=dict(arrowstyle="->", color="tab:green"), color="tab:green",
                fontsize=10)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["universe A\ncatalogs", "universe B\ncatalogs"])
    ax.set_xlabel("total catalog score  T  (sum of the per-planet scores)")
    ax.set_title(f"(b) the number line: each dot = one fake catalog\n"
                 f"p(NASA is a B catalog) = {r_loc['p_reject_B']:.4f}", fontsize=10)
    ax.grid(alpha=0.2, axis="x"); ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("The verdict in two everyday pictures — counts, then a number line "
                 f"(precision NASA, N={n})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = os.path.join(OUT_DIR, "explainer_3_counts_numberline.png")
    fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"--> Saved: {out}")


def main(df):
    os.makedirs(OUT_DIR, exist_ok=True)
    m_sil, r_sil = load_silicate()

    print("STEP 1 — FORWARD SIM (shared): flat pool -> joint Kepler+RV detection")
    det = get_detected_pool(df)

    print("\n--> loading NASA samples")
    nasa_prec = load_nasa(True)
    nasa_full = load_nasa(False)

    cfgs = {}
    for name, precision, seed in CONFIGS:
        nasa = nasa_prec if precision else nasa_full
        cfgs[name] = run_config(det, nasa, name, seed)

    make_figure(cfgs["precision"], cfgs["full"], m_sil, r_sil)
    explainer_universes(cfgs["precision"], m_sil, r_sil)
    explainer_scores(cfgs["precision"], m_sil, r_sil)
    explainer_verdict(cfgs["precision"], m_sil, r_sil)

    rp = cfgs["precision"]["res"]["cond_loc"]
    rs = cfgs["precision"]["res"]["shape"]
    rf = cfgs["full"]["res"]["cond_loc"]
    rsf = cfgs["full"]["res"]["shape"]
    print("\n================ CAVEMAN SUMMARY ================")
    print("  We made two fake universes. Same everything, except one has cold big rocky")
    print("  planets (B) and one does not (A). Both go through our detectors + NASA-size")
    print("  noise. A classifier learns the ONLY difference: the corner. Each NASA planet")
    print("  then gets a score: 'do you look like a corner planet?'. Sum of scores = T.")
    print("  Null catalogs are built AT NASA's own planet locations, so 'NASA hunts cold")
    print("  planets harder' cannot fake the answer in either direction.")
    print(f"  NASA (precision, N={cfgs['precision']['n_nasa']}): "
          f"p={rp['p_reject_B']:.4f} under B -> "
          f"{'B REJECTED: the corner is missing beyond selection' if rp['p_reject_B'] < ALPHA else 'B not rejected at 5%'}.")
    print(f"  Compatibility with empty-corner A: p={rp['p_compat_A']:.3f} "
          f"({'NASA looks like A' if rp['p_compat_A'] > ALPHA else 'NASA is not exactly A either'}).")
    print(f"  The un-shielded statistic flips with sample choice (T_shape: precision "
          f"p={rs['p_reject_B']:.4f}, full p={rsf['p_reject_B']:.4f}) —")
    print("  that flip IS the follow-up confound; the conditional statistic does not flip.")
    print(f"  Full sample (N={cfgs['full']['n_nasa']}): p(reject B)={rf['p_reject_B']:.4f}, "
          f"p(compat A)={rf['p_compat_A']:.4f}.")
    print("\nCAVEATS")
    print("  1. p-value is exact-frequentist w.r.t. OUR simulator; toy detectors stand in for")
    print("     NASA's messy selection. T_cond cancels location-targeting effort (the TEST A")
    print("     share lesson); composition-dependent publication bias within a location remains.")
    print("  2. Corner membership uses the silicate line + TRUE parameters; M-R model enters")
    print("     only via the silicate curve, not a mass-from-radius draw (flat M ⊥ R).")
    print("  3. Homogeneous-subsample rerun, M-R swap, coverage test = plan steps 6-8, not built.")
    print("  4. Classifier imperfection costs power only — the null calibration keeps validity.")


if __name__ == "__main__":
    from run.flat_universe.run_flat_universe import main as run_flat
    main(run_flat(seed=RNG_SEED, n_planets=FLAT_N_POOL))
DOWNLOAD_NASA_DATA = False  # Set to True to download fresh data, False to use local CSV
