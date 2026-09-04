"""
bayesian_cold_rocky_desert.py — three-universe Bayesian model comparison in the
cold rocky desert, resolved over three insolation panels (paper Figure 1 format).

Object (per-cell Bayes, single detected planet at location theta=(M,R) in insolation bin b):
    P(theta | detected, U_k) = l_b(theta) * pi_k(theta) / Z_{k,b}
      likelihood  l_b(theta)  = joint transit+RV detection fraction AMONG TRANSITING on the
                                MR plane, measured on the uniform-parameter universe.
      prior       pi_k(theta) = the universe's TRUE planet density (one of three).
      evidence    Z_{k,b}     = sum_theta pi_k l_b  (from the model, NOT from NASA).
    NASA is the DATA, not the marginal. All figures consider TRANSITING planets only
    (the detector's geometric transit flag); NASA's sample is transiting-confirmed.

Three universes (priors), all built on the uniform-parameter universe:
    rocky_formation  Otegi rocky R=1.03 M^0.29 (0.15 dex), ALL planets kept
    escape_only      same pool MINUS rocky (below silicate) & true mass>2
    uniform          mass INDEPENDENT of radius (log-uniform M x uniform R)

Model comparison (normalization-free; "mixture, not count"):
    per insolation bin, among DETECTED planets the volatile fraction conditions out absolute
    occurrence and survey effort. NASA gives k_b volatile of n_b detected; universe k predicts
    p_{k,b}. L_k = prol_b Binomial(k_b; n_b, p_{k,b}); posterior P(U_k|NASA), equal 1/2 prior;
    pairwise Bayes factors. Headline = the I<50 cold rocky desert (mass>2).

Run:
    python scripts/statistical_analysis/bayesian_cold_rocky_desert.py
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
from pathlib import Path

from tools.paths import LIFESIM_OUTER_DIR, PSCOMPPARS_CSV
ROOT = Path(LIFESIM_OUTER_DIR)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from tools.paths import SILICON_CURVE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.stats import binom
from scipy.ndimage import gaussian_filter

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from run.ppop.uniform_generator import generate_flat_catalog
from run.ppop.flat_detect import run_kepler, run_tess, run_rv_best

# Transit leg of the joint detector. This module IS the Kepler analysis; the parallel TESS
# analysis (42_bayesian_cold_rocky_desert_tess.py) imports this module and calls main("tess").
# Everything downstream reads MISSION at call time, so the wrapper only sets this one global.
MISSION = "kepler"
_TRANSIT = {"kepler": run_kepler, "tess": run_tess}
_MISSION_LABEL = {"kepler": "Kepler", "tess": "TESS"}
_OUT_NAME = {"kepler": "bayesian_cold_rocky_desert",
             "tess": "42_bayesian_cold_rocky_desert_tess"}


def _mlabel():
    return _MISSION_LABEL[MISSION]


def _out_dir():
    return os.path.join(ROOT, "output/plots", _OUT_NAME[MISSION])


SILICATE_CURVE = Path(SILICON_CURVE)
NASA_FILE = Path(PSCOMPPARS_CSV)

BOX = dict(r_lo=0.5, r_hi=2.2, m_lo=0.1, m_hi=12.0, f_lo=1e-2, f_hi=1e4)
FLAT_N_POOL = 10_000_000       # 10x: the cold among-transiting denominator is thin (~2% transit);
                               # this fills every MR cell in the I<10 panel above MIN_CELL.
CHUNK = 2_000_000              # generate+detect in chunks to bound peak memory (~1.5 GB/chunk)
RNG_SEED = 0
RV_MAG_TARGET = 12.0
MASS_MIN = 2.0                 # super-Earth threshold (cold rocky desert cut)
COLD_MAX = 50.0                # cold rocky desert boundary [I_earth]
OTEGI_C, OTEGI_BETA, OTEGI_SCATTER = 1.03, 0.29, 0.15
MASS_FRAC_ERR = 0.20           # log-normal measurement noise (script 72 convention)
RAD_FRAC_ERR = 0.046
NASA_MASS_PREC = 0.25
NASA_RAD_PREC = 0.08
N_FRAC_REP = 4000               # noise realizations for the predicted volatile fraction
                                # (matches the 4,000-draw convention of script 52's density figure
                                # and the main paper's own Section-4.2 MC procedure)

# Nested insolation panels, matching paper Figure 1 (rocky_mr_insolation_3panel): the cold
# rocky desert is the I<50 panel, I<10 its extreme, I>50 the hot control.
INSOL_BINS = [("I < 10", BOX["f_lo"], 10.0),
              ("I < 50", BOX["f_lo"], 50.0),
              ("I > 50", 50.0, BOX["f_hi"])]

UNIVERSES = [
    ("rocky_formation", "Primordial-rocky", "tab:blue"),
    ("escape_only", "Escape-only", "tab:orange"),
]

# LINEAR mass grid (paper Fig.1 axis). With the 10M pool every cell in every panel clears
# MIN_CELL, so the detectability field is fully colored (no blanks).
M_EDGES = np.linspace(BOX["m_lo"], BOX["m_hi"], 17)
R_EDGES = np.linspace(BOX["r_lo"], BOX["r_hi"], 15)
M_CENT = 0.5 * (M_EDGES[:-1] + M_EDGES[1:])
R_CENT = 0.5 * (R_EDGES[:-1] + R_EDGES[1:])
MIN_CELL = 5                   # display gate; at 10M even the sparsest cold cell has >=~10 samples


def load_silicate():
    d = np.loadtxt(SILICATE_CURVE, comments="#")
    m, r = d[:, 0].astype(float), d[:, 1].astype(float)
    o = np.argsort(m)
    return m[o], r[o]


def is_volatile(mass, radius, m_sil, r_sil):
    return radius > np.interp(mass, m_sil, r_sil)


def nasa_frac_sigma(nasa, lo, hi, mass_min, m_sil, r_sil, rng, n_rep=N_FRAC_REP):
    """Standard deviation of the observed volatile fraction v/n from NASA's own measurement
    error alone: redraw each planet's mass/radius within its own published asymmetric error
    bars (no resampling of which planets are in the set), reapply the mass cut on the redrawn
    mass so boundary planets can flip in/out, and take the spread over n_rep redraws (same
    per-planet-error method as the main paper's NASA Monte Carlo bells)."""
    sel = (nasa["ins"] >= lo) & (nasa["ins"] < hi)
    m, r = nasa["m"][sel], nasa["r"][sel]
    me1, me2, re1, re2 = nasa["me1"][sel], nasa["me2"][sel], nasa["re1"][sel], nasa["re2"][sel]
    n = m.size
    if n == 0:
        return np.nan, np.nan
    fr = []
    for _ in range(n_rep):
        zm, zr = rng.normal(size=n), rng.normal(size=n)
        mb = np.clip(m + np.where(zm >= 0, zm * me1, zm * me2), 1e-3, None)
        rb = np.clip(r + np.where(zr >= 0, zr * re1, zr * re2), 1e-3, None)
        if mass_min:
            k = mb > mass_min
            mb, rb = mb[k], rb[k]
        if mb.size < 1:
            continue
        fr.append(float(is_volatile(mb, rb, m_sil, r_sil).mean()))
    if not fr:
        return np.nan, np.nan
    return float(np.mean(fr)), float(np.std(fr))


def load_nasa(precision: bool):
    df = pd.read_csv(NASA_FILE, comment="#", low_memory=False)
    m = pd.to_numeric(df["pl_bmasse"], errors="coerce")
    r = pd.to_numeric(df["pl_rade"], errors="coerce")
    ins = pd.to_numeric(df["pl_insol"], errors="coerce")
    me1 = pd.to_numeric(df["pl_bmasseerr1"], errors="coerce").abs()
    me2 = pd.to_numeric(df["pl_bmasseerr2"], errors="coerce").abs()
    re1 = pd.to_numeric(df["pl_radeerr1"], errors="coerce").abs()
    re2 = pd.to_numeric(df["pl_radeerr2"], errors="coerce").abs()
    me = np.maximum(me1, me2)
    re = np.maximum(re1, re2)
    prov = df.get("pl_bmassprov", pd.Series("", index=df.index)).astype(str)
    meas = prov.str.contains("Mass|Msini", case=False, na=False) & \
        ~prov.str.contains("Calc", case=False, na=False)
    keep = (meas & r.between(BOX["r_lo"], BOX["r_hi"]) & m.between(BOX["m_lo"], BOX["m_hi"])
            & ins.between(BOX["f_lo"], BOX["f_hi"]))
    if precision:
        keep = keep & (me / m <= NASA_MASS_PREC) & (re / r <= NASA_RAD_PREC)
    k = keep.to_numpy(bool)

    def fill(err, base, frac):
        return np.where(np.isfinite(err) & (err > 0), err, frac * base)[k]

    return dict(
        m=m.to_numpy(float)[k], r=r.to_numpy(float)[k], ins=ins.to_numpy(float)[k],
        me=fill(me, m, MASS_FRAC_ERR), re=fill(re, r, RAD_FRAC_ERR),
        me1=fill(me1, m, MASS_FRAC_ERR), me2=fill(me2, m, MASS_FRAC_ERR),
        re1=fill(re1, r, RAD_FRAC_ERR), re2=fill(re2, r, RAD_FRAC_ERR),
        n=int(k.sum()))


def _detect_chunk(mass_model, n, seed, **kw):
    """Generate n planets -> box cut -> joint Kepler+RV detection. Per-row detectors, so this is
    equivalent to processing one big pool but with bounded memory."""
    pool = generate_flat_catalog(n_planets=n, seed=seed, mass_model=mass_model, **kw)
    r = pd.to_numeric(pool["radius_p"], errors="coerce")
    m = pd.to_numeric(pool["mass_p"], errors="coerce")
    f = pd.to_numeric(pool["flux_p"], errors="coerce")
    keep = (r.between(BOX["r_lo"], BOX["r_hi"]) & m.between(BOX["m_lo"], BOX["m_hi"])
            & f.between(BOX["f_lo"], BOX["f_hi"]))
    pool = pool[keep].copy()
    kep = _TRANSIT[MISSION](pool)
    rd = run_rv_best(pool, mag_target=RV_MAG_TARGET)["detected"].to_numpy(bool)
    return dict(
        mass=pd.to_numeric(pool["mass_p"], errors="coerce").to_numpy(float),
        radius=pd.to_numeric(pool["radius_p"], errors="coerce").to_numpy(float),
        flux=pd.to_numeric(pool["flux_p"], errors="coerce").to_numpy(float),
        teff=pd.to_numeric(pool["teff_s"], errors="coerce").to_numpy(float),
        det=kep["detected"].to_numpy(bool) & rd,
        transit=kep["transiting_geometric"].to_numpy(bool),
    )


def make_pool(mass_model, **kw):
    """Uniform-parameter pool -> box cut -> joint Kepler+RV detection (cached to npz), built in
    CHUNK-sized pieces (distinct per-chunk seeds) to bound peak memory. Returns TRUE arrays, the
    joint-detected mask, and the geometric transit mask (detected ⊂ transit)."""
    tag = "_".join([mass_model] + [f"{k}{v}" for k, v in sorted(kw.items())]
                   + [f"N{FLAT_N_POOL}", f"s{RNG_SEED}"])
    cache = os.path.join(_out_dir(), f"pool_{tag}.npz")
    if os.path.exists(cache):
        print(f"    loaded cached pool: {os.path.basename(cache)}")
        z = np.load(cache)
        return {fld: z[fld] for fld in z.files}
    parts, done, ci = [], 0, 0
    while done < FLAT_N_POOL:
        n = min(CHUNK, FLAT_N_POOL - done)
        parts.append(_detect_chunk(mass_model, n, RNG_SEED + ci, **kw))
        done += n
        ci += 1
        print(f"      chunk {ci}: {done:>9d}/{FLAT_N_POOL} generated+detected")
    result = {fld: np.concatenate([p[fld] for p in parts]) for fld in parts[0]}
    np.savez(cache, **result)
    return result


def build_universes(m_sil, r_sil):
    print(f"--> building Otegi (rocky-formation) pool N={FLAT_N_POOL} ...")
    otegi = make_pool("powerlaw", mr_C=OTEGI_C, mr_beta=OTEGI_BETA, mass_scatter_dex=OTEGI_SCATTER)
    print(f"--> building independent (uniform) pool N={FLAT_N_POOL} ...")
    indep = make_pool("independent")

    rocky_true = ~is_volatile(otegi["mass"], otegi["radius"], m_sil, r_sil)
    keep_escape = ~(rocky_true & (otegi["mass"] > MASS_MIN))     # remove born-rocky super-Earths

    univ = {"rocky_formation": otegi,
            "escape_only": {k: (v[keep_escape] if isinstance(v, np.ndarray) else v)
                            for k, v in otegi.items()},
            "uniform": indep}
    for key, _, _ in UNIVERSES:
        u = univ[key]
        print(f"    {key:<16} pool={u['mass'].size:>8}  transiting={int(u['transit'].sum()):>7}"
              f"  detected={int(u['det'].sum()):>6}")
    return univ


def detection_map(pool, lo, hi):
    """l_b(M,R) = joint detected fraction AMONG TRANSITING on the linear-mass MR grid, in
    insolation bin [lo,hi). Denominator = transiting count (paper fig:flat convention)."""
    tsel = (pool["flux"] >= lo) & (pool["flux"] < hi) & pool["transit"]
    m, r, det = pool["mass"][tsel], pool["radius"][tsel], pool["det"][tsel]
    num, _, _ = np.histogram2d(m, r, bins=[M_EDGES, R_EDGES], weights=det.astype(float))
    tot, _, _ = np.histogram2d(m, r, bins=[M_EDGES, R_EDGES])
    D = np.where(tot >= MIN_CELL, num / np.maximum(tot, 1), np.nan)
    return D, tot


def predicted_frac(u, lo, hi, mass_min, m_sil, r_sil, rng, n_rep=N_FRAC_REP):
    """Detected volatile fraction p_{k,b} for one universe/bin/mass-cut, averaged over
    measurement-noise realizations (detected planets all transit)."""
    sel = u["det"] & (u["flux"] >= lo) & (u["flux"] < hi)
    m0, r0 = u["mass"][sel], u["radius"][sel]
    if m0.size == 0:
        return np.nan, np.nan, 0
    fr, cnt = [], []
    for _ in range(n_rep):
        mo = m0 * np.exp(rng.normal(0.0, MASS_FRAC_ERR, m0.size))
        ro = r0 * np.exp(rng.normal(0.0, RAD_FRAC_ERR, r0.size))
        if mass_min:
            k = mo > mass_min
            mo, ro = mo[k], ro[k]
        if mo.size < 5:
            continue
        fr.append(float(is_volatile(mo, ro, m_sil, r_sil).mean()))
        cnt.append(mo.size)
    if not fr:
        return np.nan, np.nan, 0
    return float(np.mean(fr)), float(np.std(fr)), int(np.mean(cnt))


def nasa_bin(nasa, lo, hi, mass_min, m_sil, r_sil):
    sel = (nasa["ins"] >= lo) & (nasa["ins"] < hi)
    if mass_min:
        sel = sel & (nasa["m"] > mass_min)
    m, r = nasa["m"][sel], nasa["r"][sel]
    vol = is_volatile(m, r, m_sil, r_sil)
    return int(vol.sum()), int(sel.sum())


def model_posterior(p_by_univ, k, n):
    keys = list(p_by_univ)
    logl = np.array([binom.logpmf(k, n, float(np.clip(p_by_univ[key], 1e-4, 1 - 1e-4)))
                     for key in keys])
    post = np.exp(logl - logl.max())
    post = post / post.sum()
    return dict(zip(keys, logl)), dict(zip(keys, post))


# ----------------------------------------------------------------------- reporting
def desert_table(univ, nasa, m_sil, r_sil, rng, tag):
    print(f"\n  ================ Bayesian model comparison — {tag} ================")
    print("  volatile fraction f_k, NASA v/n, binomial likelihood L_k, and likelihood odds O=L/sum L\n")
    segments = [(lbl, lo, hi, MASS_MIN) for lbl, lo, hi in INSOL_BINS]
    segments += [("all insolation", BOX["f_lo"], BOX["f_hi"], None)]
    rows = []
    for lbl, lo, hi, mcut in segments:
        p_raw = {key: predicted_frac(univ[key], lo, hi, mcut, m_sil, r_sil, rng)
                 for key, _, _ in UNIVERSES}
        p_by = {key: v[0] for key, v in p_raw.items()}
        p_sigma = {key: v[1] for key, v in p_raw.items()}
        k, n = nasa_bin(nasa, lo, hi, mcut, m_sil, r_sil)
        if n == 0:
            print(f"  {lbl:<24} NASA n=0 — skipped")
            continue
        _, nasa_sigma = nasa_frac_sigma(nasa, lo, hi, mcut, m_sil, r_sil, rng)
        logl, post = model_posterior(p_by, k, n)
        L = {key: float(np.exp(logl[key])) for key in logl}
        ftxt = "  ".join(f"{lbl2[:5]}={p_by[key]:.3f}+-{p_sigma[key]:.3f}" for key, lbl2, _ in UNIVERSES)
        Ltxt = "  ".join(f"{lbl2[:5]}={L[key]:.2e}" for key, lbl2, _ in UNIVERSES)
        postxt = "  ".join(f"{lbl2[:5]}={post[key]:.2f}" for key, lbl2, _ in UNIVERSES)
        cut = " (M>2)" if mcut else ""
        print(f"  {lbl + cut:<24} NASA vol {k}/{n} (v/n={k/n:.3f}+-{nasa_sigma:.3f})")
        print(f"      f_k : {ftxt}")
        print(f"      L_k : {Ltxt}")
        print(f"      odds O   : {postxt}")
        rows.append(dict(label=lbl + cut, k=k, n=n, p=p_by, p_sigma=p_sigma,
                          nasa_sigma=nasa_sigma, post=post, logl=logl, L=L))
    return rows


def save_stats_table(rows, tag):
    """Per-panel f_k, L_k, O_esc, and Bayes factor to CSV, so the paper table traces to code.
    rocky_formation is the paper's Primordial-rocky (f_prim); escape_only is Escape-only (f_esc)."""
    recs = []
    for r in rows:
        Le, Lp = r["L"]["escape_only"], r["L"]["rocky_formation"]
        recs.append(dict(
            panel=r["label"], v=r["k"], n=r["n"], v_over_n=round(r["k"] / r["n"], 3),
            v_over_n_sigma=round(r["nasa_sigma"], 3),
            f_esc=round(r["p"]["escape_only"], 3), f_esc_sigma=round(r["p_sigma"]["escape_only"], 3),
            f_prim=round(r["p"]["rocky_formation"], 3), f_prim_sigma=round(r["p_sigma"]["rocky_formation"], 3),
            L_esc=Le, L_prim=Lp,
            O_esc=round(r["post"]["escape_only"], 3),
            BF_esc_over_prim=Le / max(Lp, 1e-300)))
    df = pd.DataFrame(recs)
    out = os.path.join(_out_dir(), "model_stats.csv")
    df.to_csv(out, index=False)
    print(f"--> Saved: {out}")
    return df


def print_bayes_factors(rows):
    print("\n  ---- pairwise Bayes factors (headline: I<50 cold rocky desert, M>2) ----")
    desert = next((r for r in rows if r["label"].startswith("I < 50")), None)
    if desert is None:
        return
    logl = desert["logl"]
    for a, la, _ in UNIVERSES:
        for b, lb, _ in UNIVERSES:
            if a < b:
                d = (logl[a] - logl[b]) / np.log(10)
                print(f"    {la} vs {lb}: BF = {np.exp(logl[a] - logl[b]):.2g}  (log10 = {d:+.2f})")


# -------------------------------------------------------------------------- figures
def _nasa_overlay(ax, nasa, lo, hi, norm, mass_min=None):
    """NASA planets in insolation bin: grey error bars + points colored by log insolation.
    mass_min, if given, applies the same super-Earth cut used by the panel it is drawn on."""
    nsel = (nasa["ins"] >= lo) & (nasa["ins"] < hi)
    if mass_min:
        nsel = nsel & (nasa["m"] > mass_min)
    if nsel.sum() == 0:
        return None
    ax.errorbar(nasa["m"][nsel], nasa["r"][nsel],
                xerr=[nasa["me2"][nsel], nasa["me1"][nsel]],
                yerr=[nasa["re2"][nsel], nasa["re1"][nsel]],
                fmt="none", ecolor="0.8", elinewidth=0.7, capsize=0, zorder=4)
    return ax.scatter(nasa["m"][nsel], nasa["r"][nsel], c=np.log10(nasa["ins"][nsel]),
                      cmap="plasma", norm=norm, s=34, edgecolor="k", lw=0.5, zorder=5)


def _field_image(ax, field, vmax, levels):
    """Figure-3-style filled background: smooth viridis image + white contour lines. `field` is
    a rate/intensity on the (M,R) grid; nan cells (none at 10M) render transparent."""
    ax.imshow(field.T, origin="lower",
              extent=[BOX["m_lo"], BOX["m_hi"], BOX["r_lo"], BOX["r_hi"]],
              aspect="auto", cmap="viridis", vmin=0, vmax=vmax, interpolation="bilinear", zorder=0)
    sm = gaussian_filter(np.nan_to_num(field, nan=0.0), 0.8)
    X, Y = np.meshgrid(M_CENT, R_CENT)
    ax.contour(X, Y, sm.T, levels=levels, colors="white", linewidths=0.7, alpha=0.75, zorder=2)


def fig_likelihood_maps(univ, nasa):
    """1x3 by insolation: the detectability field l_b(M,R) (detected fraction among transiting,
    uniform-parameter universe, ALL masses) in Figure-3 format — filled viridis + white contours,
    NASA planets (mass > MASS_MIN only) colored by log insolation with error bars."""
    Ds = [detection_map(univ["uniform"], lo, hi)[0] for _, lo, hi in INSOL_BINS]
    vmax = max((np.nanmax(D) for D in Ds if np.isfinite(D).any()), default=1.0)
    levels = np.round(np.linspace(0.2, vmax, 4), 2)
    norm = Normalize(vmin=np.log10(BOX["f_lo"]), vmax=np.log10(BOX["f_hi"]))
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), sharey=True, constrained_layout=True)
    sc = None
    for ax, (lbl, lo, hi), D in zip(axes, INSOL_BINS, Ds):
        _field_image(ax, D, vmax, levels)
        s = _nasa_overlay(ax, nasa, lo, hi, norm, mass_min=MASS_MIN)
        if s is not None:
            sc = s
        ax.set_xlim(BOX["m_lo"], BOX["m_hi"]); ax.set_ylim(BOX["r_lo"], BOX["r_hi"])
        ax.set_xlabel(r"planet mass [$M_\oplus$]")
        ax.set_title(f"{lbl} $I_\\oplus$", fontsize=12)
    axes[0].set_ylabel(r"planet radius [$R_\oplus$]")
    cb = fig.colorbar(axes[0].images[0], ax=axes, location="right", shrink=0.9)
    cb.set_label(f"detected fraction among transiting  $\\ell_b(M,R)$  ({_mlabel()} transit + RV)")
    if sc is not None:
        cb2 = fig.colorbar(sc, ax=axes, location="bottom", shrink=0.45, pad=0.02, aspect=45)
        cb2.set_label(r"NASA planets: log(Insolation Flux [$I_\oplus$])")
    fig.suptitle(f"Detectability (likelihood) on the MR plane — {_mlabel()} transit + RV, "
                 "transiting only\n"
                 "viridis = detected fraction among transiting (uniform-parameter universe); "
                 "white contours",
                 fontsize=13)
    out = os.path.join(_out_dir(), "likelihood_detection_maps.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"--> Saved: {out}")


def fig_posterior_predictive(univ, nasa, m_sil, r_sil):
    """3x3 (universe x insolation) in Figure-3 format: filled viridis predicted-detected
    (transiting, mass > MASS_MIN) density per panel (self-normalized, so every cell carries a
    color), white contours, NASA planets colored by log insolation with error bars, silicate
    line. Linear mass axis; no red highlight (per request)."""
    ms = np.linspace(BOX["m_lo"], BOX["m_hi"], 250)
    r_sil_line = np.interp(ms, m_sil, r_sil)
    norm = Normalize(vmin=np.log10(BOX["f_lo"]), vmax=np.log10(BOX["f_hi"]))
    X, Y = np.meshgrid(M_CENT, R_CENT)
    fig, axes = plt.subplots(len(UNIVERSES), 3, figsize=(17, 9.5), sharex=True, sharey=True,
                             constrained_layout=True)
    sc = None
    for i, (key, plabel, _) in enumerate(UNIVERSES):
        u = univ[key]
        for j, (blabel, lo, hi) in enumerate(INSOL_BINS):
            ax = axes[i, j]
            sel = (u["det"] & (u["flux"] >= lo) & (u["flux"] < hi)     # detected ⊂ transiting
                   & (u["mass"] > MASS_MIN))
            H, _, _ = np.histogram2d(u["mass"][sel], u["radius"][sel], bins=[M_EDGES, R_EDGES])
            # log stretch: the uniform-in-R track dams a density spike at the M=12 box wall that a
            # linear+clip scale saturates into a flat slab; log1p renders it as a smooth gradient.
            Hs = gaussian_filter(H, 1.0)
            L = np.log1p(Hs)
            field = L / L.max() if L.max() > 0 else L
            ax.imshow(field.T, origin="lower",
                      extent=[BOX["m_lo"], BOX["m_hi"], BOX["r_lo"], BOX["r_hi"]],
                      aspect="auto", cmap="viridis", vmin=0, vmax=1,
                      interpolation="bilinear", zorder=0)
            ax.contour(X, Y, field.T, levels=[0.25, 0.5, 0.75], colors="white",
                       linewidths=0.7, alpha=0.7, zorder=2)
            ax.plot(ms, r_sil_line, "k--", lw=1.3, zorder=3)
            s = _nasa_overlay(ax, nasa, lo, hi, norm)
            if s is not None:
                sc = s
            ax.set_xlim(BOX["m_lo"], BOX["m_hi"]); ax.set_ylim(BOX["r_lo"], BOX["r_hi"])
            if i == 0:
                ax.set_title(f"{blabel} $I_\\oplus$", fontsize=12)
            if j == 0:
                ax.set_ylabel(f"{plabel}\n" + r"radius [$R_\oplus$]", fontsize=11)
            if i == len(UNIVERSES) - 1:
                ax.set_xlabel(r"planet mass [$M_\oplus$]")
    cb = fig.colorbar(axes[0, 0].images[0], ax=axes, location="right", shrink=0.85)
    cb.set_label("predicted-detected density (per-panel normalized)")
    if sc is not None:
        cb2 = fig.colorbar(sc, ax=axes, location="bottom", shrink=0.4, pad=0.02, aspect=50)
        cb2.set_label(r"NASA planets: log(Insolation Flux [$I_\oplus$])")
    fig.suptitle(f"Posterior-predictive detected density ({_mlabel()} transit + RV, viridis, "
                 f"transiting only, $M>{MASS_MIN:.0f}\\,M_\\oplus$) vs confirmed NASA planets "
                 "(points colored by log insolation, with error bars)\n"
                 "rows = universes (priors); columns = insolation panels; "
                 "black dashed = silicate line", fontsize=13)
    out = os.path.join(_out_dir(), "posterior_predictive_maps.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"--> Saved: {out}")


def fig_model_odds(rows):
    labels = [r["label"] for r in rows]
    x = np.arange(len(labels))
    w = 0.26
    fig, ax = plt.subplots(figsize=(13, 5.6))
    for i, (key, plabel, color) in enumerate(UNIVERSES):
        ax.bar(x + (i - (len(UNIVERSES) - 1) / 2) * w, [r["post"][key] for r in rows], w,
               color=color, label=plabel)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(" ", "\n", 1) for l in labels], fontsize=8.5)
    ax.set_ylabel(r"likelihood odds  $O = L / \sum_j L_j$")
    ax.set_ylim(0, 1.05)
    ax.axhline(1 / len(UNIVERSES), color="0.6", ls=":", lw=1,
               label=f"even split (1/{len(UNIVERSES)})")
    ax.set_title(f"Which universe does NASA prefer? ({_mlabel()} transit + RV; normalized "
                 "binomial composition likelihood)\n"
                 "headline = I<50 cold rocky desert (M>2); transiting planets only", fontsize=11)
    ax.grid(alpha=0.2, axis="y"); ax.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(_out_dir(), "model_odds.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"--> Saved: {out}")


def fig_model_stats(rows):
    """Companion to the odds bars, and the direct answer to 'why does O saturate when the maps
    look alike': (top) the two universes' predicted volatile fractions f_k with the observed
    NASA v/n marked, showing how close the fractions sit; (bottom) the binomial likelihoods L_k
    on a log axis, showing the same small f gap turn into orders of magnitude in L. Every plotted
    quantity is labelled on its bar/marker; the derived odds (O_esc, BF) live in the note's table."""
    esc, prim = "escape_only", "rocky_formation"
    labels = [r["label"] for r in rows]
    x = np.arange(len(labels))
    w = 0.26
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 8.0), sharex=True, gridspec_kw=dict(height_ratios=[1.0, 1.0]))
    for i, (key, plabel, color) in enumerate(UNIVERSES):
        xb = x + (i - (len(UNIVERSES) - 1) / 2) * w
        vals = [r["p"][key] for r in rows]
        ax1.bar(xb, vals, w, color=color, label=f"$f$  {plabel}")
        for xi, v in zip(xb, vals):
            ax1.annotate(f"{v:.2f}", (xi, v), textcoords="offset points", xytext=(0, 2),
                         ha="center", fontsize=8, color=color)
    ax1.plot(x, [r["k"] / r["n"] for r in rows], "kD", ms=8, zorder=5, label=r"observed $v/n$")
    for xi, r in zip(x, rows):
        vn = r["k"] / r["n"]
        ytop = max([vn] + [r["p"][key] for key, _, _ in UNIVERSES]) + 0.09
        ax1.annotate(f"{r['k']}/{r['n']} = {vn:.2f}", (xi, ytop),
                     ha="center", fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax1.set_ylabel("volatile fraction")
    ax1.set_ylim(0, 1.12)
    ax1.set_xticks(x)
    ax1.grid(alpha=0.2, axis="y"); ax1.legend(fontsize=9, ncol=3)
    ax1.set_title(r"Predicted volatile fractions $f_k$ vs observed $v/n$ "
                  "(the two universes sit close)", fontsize=11)
    for i, (key, plabel, color) in enumerate(UNIVERSES):
        xb = x + (i - (len(UNIVERSES) - 1) / 2) * w
        vals = [max(r["L"][key], 1e-300) for r in rows]
        ax2.bar(xb, vals, w, color=color, label=f"$L$  {plabel}")
        for xi, v in zip(xb, vals):
            ax2.annotate(f"{v:.1e}", (xi, v), textcoords="offset points", xytext=(0, 2),
                         ha="center", fontsize=8, color=color)
    ax2.set_yscale("log")
    ax2.set_ylabel(r"binomial likelihood $L_k$")
    ax2.set_xticks(x)
    ax2.set_xticklabels([l.replace(" (M>2)", "\n(M>2)") for l in labels])
    ax2.grid(alpha=0.2, axis="y"); ax2.legend(fontsize=9)
    ax2.set_title(r"Binomial likelihoods $L_k$ (log axis): the small $f$ gap becomes orders "
                  "of magnitude", fontsize=11)
    fig.suptitle(f"Why the odds saturate — {_mlabel()} transit + RV, precision-cut sample",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(_out_dir(), "model_stats.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"--> Saved: {out}")


def main(mission="kepler"):
    global MISSION
    MISSION = mission
    os.makedirs(_out_dir(), exist_ok=True)
    m_sil, r_sil = load_silicate()
    rng = np.random.default_rng(RNG_SEED)

    print(f"=== Cold rocky desert Bayesian comparison — transit mission: {_mlabel()} ===")
    univ = build_universes(m_sil, r_sil)

    # Table-6 reconciliation targets are Kepler-derived; only meaningful for the Kepler run.
    if MISSION == "kepler":
        print("\n--> reconciliation vs paper Table 6 (Flat All / Flat Vol detected f_vol):")
        for lbl, lo, hi, mcut, ref in [("all bins, no cut", BOX["f_lo"], BOX["f_hi"], None,
                                        "All 0.43 / Vol 0.70"),
                                       ("cold I<50, M>2", BOX["f_lo"], COLD_MAX, MASS_MIN,
                                        "All 0.44 / Vol 0.74")]:
            pa = predicted_frac(univ["rocky_formation"], lo, hi, mcut, m_sil, r_sil, rng)[0]
            pv = predicted_frac(univ["escape_only"], lo, hi, mcut, m_sil, r_sil, rng)[0]
            print(f"    {lbl:<20} rocky-formation f_vol={pa:.2f}  escape-only f_vol={pv:.2f}   "
                  f"[paper Table 6: {ref}]")

    rows_prec, nasa_prec = None, None
    for precision, tag in [(True, f"{_mlabel()} — precision-cut NASA (primary)"),
                           (False, f"{_mlabel()} — full measured-mass NASA (sensitivity)")]:
        nasa = load_nasa(precision)
        print(f"\n--> {tag}: N={nasa['n']} in box")
        rows = desert_table(univ, nasa, m_sil, r_sil, rng, tag)
        print_bayes_factors(rows)
        if precision:
            rows_prec, nasa_prec = rows, nasa

    fig_likelihood_maps(univ, nasa_prec)
    fig_posterior_predictive(univ, nasa_prec, m_sil, r_sil)
    fig_model_odds(rows_prec)
    fig_model_stats(rows_prec)
    save_stats_table(rows_prec, "precision-cut (primary)")

    print("\n  CAVEATS:")
    print("  - Figures consider TRANSITING planets only (detector geometric transit flag); the")
    print("    likelihood map is the detected fraction among transiting (paper fig:flat).")
    print("  - Blanks were the MIN_CELL quality mask, not empty cells: cold planets transit ~2%,")
    print(f"    so the I<10 transiting subsample was thin. N={FLAT_N_POOL:,} fills every MR cell.")
    print("    Posterior-predictive dark regions for the two track universes are PHYSICAL (the MR")
    print("    relation is a curve); only the uniform universe's fill needed more simulations.")
    print("  - Normalization-free: the binomial conditions on n_b, so absolute occurrence and")
    print("    survey effort cancel. No absolute-rate prior is used.")
    print("  - Point-estimate binomial => overconfident; beta-binomial + NASA error propagation")
    print("    is the honest next step (I<10 especially is data-poor).")
    print("  - escape_only removes TRUE rocky M>2; measurement noise keeps p<1 so one confirmed")
    print("    cold rocky planet (LHS 1140 b) does not send its likelihood to 0.")


if __name__ == "__main__":
    main()
