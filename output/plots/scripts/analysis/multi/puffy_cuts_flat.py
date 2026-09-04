"""
puffy_cuts_flat.py — puffy-fraction "which universe does NASA imply?" under four cuts,
with the FLAT and P-Pop universes separated into two rows so each panel holds only 2 models + NASA.

Same machinery as script 70 (transit+RV detected, NASA-like measurement error propagated), but:
  * puffy fraction only (density dropped, per request),
  * 2 rows x 4 columns:  row 1 = FLAT (flat A, flat B),  row 2 = P-Pop (P-Pop A, P-Pop B),
  * columns = cuts:  all | mass>2 M_earth | insolation<50 I_earth | mass>2 & insol<50.

Run:
    python scripts/puffy_cuts_flat.py
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

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from run.ppop.uniform_generator import generate_flat_catalog
from run.ppop.flat_detect import run_kepler, run_rv_best

SILICATE_CURVE = Path(SILICON_CURVE)
PPOP_CATALOG = ROOT / "run" / "kepler" / "data" / "Gaia" / "kepler_catalog_0.csv"
NASA_FILE = Path(PSCOMPPARS_CSV)
OUT_DIR = os.path.join(ROOT, "output/plots", "puffy_cuts_flat")

N_SAMPLE = 20000
N_REPEATS = 10000
MASS_THRESHOLD = 2.0
MASS_FRAC_ERR = 0.20
RAD_FRAC_ERR = 0.046
NASA_MASS_PREC = 0.25           # keep NASA planets with fractional mass error <= this
NASA_RAD_PREC = 0.08            # keep NASA planets with fractional radius error <= this
FLAT_N_POOL = 300000
RNG_SEED = 0
RV_MAG_TARGET = 12.0
BOX = dict(r_lo=0.5, r_hi=2.2, m_lo=0.1, m_hi=12.0, f_lo=1e-2, f_hi=1e4)

ROWS = [("flat", "FLAT universe", [(True, "flat A", "tab:orange"), (False, "flat B", "tab:blue")]),
        ("ppop", "P-Pop universe", [(True, "P-Pop A", "tab:red"), (False, "P-Pop B", "tab:purple")])]
CUTS = [("all (no cut)", {}),
        ("mass > 2 M⊕", dict(mass_min=2.0)),
        ("insolation < 50 I⊕", dict(insol_max=50.0)),
        ("mass>2 & insol<50", dict(mass_min=2.0, insol_max=50.0))]


def load_silicate():
    d = np.loadtxt(SILICATE_CURVE, comments="#")
    m, r = d[:, 0].astype(float), d[:, 1].astype(float)
    o = np.argsort(m)
    return m[o], r[o]


def puffy_frac(m_obs, r_obs, m_sil, r_sil):
    return float((r_obs > np.interp(m_obs, m_sil, r_sil)).mean())


def build_pool(population, m_sil, r_sil):
    if population == "flat":
        pool = generate_flat_catalog(n_planets=FLAT_N_POOL, seed=RNG_SEED)
    else:
        cols = ["radius_p", "mass_p", "p_orb", "inc_p", "ecc_p", "semimajor_p", "radius_s",
                "mass_s", "temp_s", "teff_s", "distance_s", "l_sun", "flux_p", "detected"]
        pool = pd.read_csv(PPOP_CATALOG, usecols=lambda c: c in cols, low_memory=False)
    r = pd.to_numeric(pool["radius_p"], errors="coerce")
    m = pd.to_numeric(pool["mass_p"], errors="coerce")
    keep = r.between(BOX["r_lo"], BOX["r_hi"]) & m.between(BOX["m_lo"], BOX["m_hi"])
    f = pd.to_numeric(pool["flux_p"], errors="coerce")
    keep = keep & (f.isna() | f.between(BOX["f_lo"], BOX["f_hi"]))
    pool = pool[keep].copy()
    mass = pd.to_numeric(pool["mass_p"], errors="coerce").to_numpy(float)
    radius = pd.to_numeric(pool["radius_p"], errors="coerce").to_numpy(float)
    flux = pd.to_numeric(pool["flux_p"], errors="coerce").to_numpy(float)
    puffy = radius > np.interp(mass, m_sil, r_sil)
    if population == "flat":
        td = run_kepler(pool)["detected"].to_numpy(bool)
    else:
        td = pd.to_numeric(pool["detected"], errors="coerce").fillna(0).astype(bool).to_numpy()
    rd = run_rv_best(pool, mag_target=RV_MAG_TARGET)["detected"].to_numpy(bool)
    return mass, radius, flux, puffy, td & rd


def mc_universe(arrays, drop, cut, m_sil, r_sil, rng):
    mass, radius, flux, puffy, det = arrays
    keep = ~((~puffy) & (mass > MASS_THRESHOLD)) if drop else np.ones(len(mass), bool)
    if cut.get("insol_max"):
        keep = keep & (flux < cut["insol_max"])
    idx = np.flatnonzero(keep)
    mass_min = cut.get("mass_min")
    out = np.full(N_REPEATS, np.nan); cnt = np.zeros(N_REPEATS)
    if idx.size:
        m_u, r_u, det_u = mass[idx], radius[idx], det[idx]
        L = idx.size
        for i in range(N_REPEATS):
            s = rng.integers(0, L, N_SAMPLE)
            sel = s[det_u[s]]
            if sel.size == 0:
                continue
            m_obs = m_u[sel] * np.exp(rng.normal(0, MASS_FRAC_ERR, sel.size))
            r_obs = r_u[sel] * np.exp(rng.normal(0, RAD_FRAC_ERR, sel.size))
            if mass_min:
                k = m_obs > mass_min
                m_obs, r_obs = m_obs[k], r_obs[k]
            if m_obs.size < 5:
                continue
            out[i] = puffy_frac(m_obs, r_obs, m_sil, r_sil); cnt[i] = m_obs.size
    return out[np.isfinite(out)], float(np.nanmean(cnt[cnt > 0])) if (cnt > 0).any() else 0.0


def load_nasa():
    df = pd.read_csv(NASA_FILE, comment="#", low_memory=False)
    m = pd.to_numeric(df["pl_bmasse"], errors="coerce")
    r = pd.to_numeric(df["pl_rade"], errors="coerce")
    ins = pd.to_numeric(df["pl_insol"], errors="coerce")
    me1 = pd.to_numeric(df["pl_bmasseerr1"], errors="coerce").abs()
    me2 = pd.to_numeric(df["pl_bmasseerr2"], errors="coerce").abs()
    re1 = pd.to_numeric(df["pl_radeerr1"], errors="coerce").abs()
    re2 = pd.to_numeric(df["pl_radeerr2"], errors="coerce").abs()
    prov = df.get("pl_bmassprov", pd.Series("", index=df.index)).astype(str)
    meas = prov.str.contains("Mass|Msini", case=False, na=False) & ~prov.str.contains("Calc", case=False, na=False)
    # measurement-precision test: larger (conservative) error side must be within tolerance
    prec = (np.maximum(me1, me2) / m <= NASA_MASS_PREC) & (np.maximum(re1, re2) / r <= NASA_RAD_PREC)
    keep = meas & prec & r.between(BOX["r_lo"], BOX["r_hi"]) & m.between(BOX["m_lo"], BOX["m_hi"])
    d = dict(m=m[keep].to_numpy(), r=r[keep].to_numpy(), ins=ins[keep].to_numpy(),
             me1=me1[keep].to_numpy(), me2=me2[keep].to_numpy(), re1=re1[keep].to_numpy(), re2=re2[keep].to_numpy())
    d["me1"] = np.where(np.isfinite(d["me1"]), d["me1"], MASS_FRAC_ERR * d["m"])
    d["me2"] = np.where(np.isfinite(d["me2"]), d["me2"], MASS_FRAC_ERR * d["m"])
    d["re1"] = np.where(np.isfinite(d["re1"]), d["re1"], RAD_FRAC_ERR * d["r"])
    d["re2"] = np.where(np.isfinite(d["re2"]), d["re2"], RAD_FRAC_ERR * d["r"])
    return d


def mc_nasa(nasa, cut, m_sil, r_sil, rng):
    sub = (nasa["ins"] < cut["insol_max"]) if cut.get("insol_max") else np.ones(len(nasa["m"]), bool)
    m, r = nasa["m"][sub], nasa["r"][sub]
    me1, me2, re1, re2 = nasa["me1"][sub], nasa["me2"][sub], nasa["re1"][sub], nasa["re2"][sub]
    n = len(m); mass_min = cut.get("mass_min")
    out = np.full(N_REPEATS, np.nan); cnt = np.zeros(N_REPEATS)
    # NO bootstrap resample: the fixed set of n precision-passing planets is kept every universe;
    # only each planet's OWN asymmetric measurement error is re-perturbed -> narrower bell (the "real" universe).
    for i in range(N_REPEATS):
        zm, zr = rng.normal(size=n), rng.normal(size=n)
        mb = np.clip(m + np.where(zm >= 0, zm * me1, zm * me2), 1e-3, None)
        rb = np.clip(r + np.where(zr >= 0, zr * re1, zr * re2), 1e-3, None)
        if mass_min:
            k = mb > mass_min; mb, rb = mb[k], rb[k]
        if mb.size < 5:
            continue
        out[i] = puffy_frac(mb, rb, m_sil, r_sil); cnt[i] = mb.size
    good = np.isfinite(out)
    # report the mean per-draw count AFTER the observed-mass cut (matches mc_universe's N_eff),
    # not the insolation-only window count — the bell width is set by this effective N
    n_eff = int(round(cnt[good].mean())) if good.any() else 0
    return out[good], n_eff


def gauss(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)) if sd > 0 else np.zeros_like(x)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    m_sil, r_sil = load_silicate()
    rng = np.random.default_rng(RNG_SEED)
    print("--> building pools + detectors...")
    pools = {pop: build_pool(pop, m_sil, r_sil) for pop in ["flat", "ppop"]}
    nasa = load_nasa()

    # precompute NASA bells per cut (shared by both rows)
    nasa_cut = {}
    for cut_label, cut in CUTS:
        nasa_cut[cut_label] = mc_nasa(nasa, cut, m_sil, r_sil, rng)

    # ---- pass 1: compute every panel, collect all values for shared axes ----
    panels = {}
    all_vals = []
    print(f"\n  {'row':<7}{'cut':<20}{'universe':<9}{'μ':>7}{'σ':>7}{'tension':>9}{'N_eff':>7}")
    for ri, (pop, row_label, models) in enumerate(ROWS):
        for ci, (cut_label, cut) in enumerate(CUTS):
            nv, n_nasa = nasa_cut[cut_label]
            n_mu, n_sd = nv.mean(), nv.std()
            series = []
            for drop, label, colour in models:
                s, neff = mc_universe(pools[pop], drop, cut, m_sil, r_sil, rng)
                tens = (abs(s.mean() - n_mu) / np.sqrt(s.std() ** 2 + n_sd ** 2)
                        if s.size and (s.std() + n_sd) > 0 else np.nan)
                series.append((label, colour, s, neff, tens))
                if s.size:
                    all_vals.append(s)
                    print(f"  {row_label.split()[0]:<7}{cut_label:<20}{label:<9}"
                          f"{s.mean():>7.3f}{s.std():>7.3f}{tens:>8.1f}σ{neff:>7.0f}")
            all_vals.append(nv)
            panels[(ri, ci)] = dict(series=series, nv=nv, n_nasa=n_nasa, n_mu=n_mu, n_sd=n_sd)
        print()

    # shared x-range + bins (so every panel is on the same scale and the bells are directly comparable)
    cat = np.concatenate(all_vals)
    lo, hi = cat.min(), cat.max(); pad = 0.04 * (hi - lo)
    gx_lo, gx_hi = lo - pad, hi + pad
    edges = np.linspace(gx_lo, gx_hi, 60)
    xs = np.linspace(gx_lo, gx_hi, 400)

    # shared y-max: tallest histogram bar / gaussian peak across all panels (so nothing clips)
    y_max = 0.0
    for P in panels.values():
        for _, _, s, _, _ in P["series"]:
            if s.size:
                y_max = max(y_max, np.histogram(s, bins=edges, density=True)[0].max(),
                            gauss(xs, s.mean(), s.std()).max())
        y_max = max(y_max, np.histogram(P["nv"], bins=edges, density=True)[0].max(),
                    gauss(xs, P["n_mu"], P["n_sd"]).max())
    gy_hi = y_max * 1.06

    # ---- pass 2: plot every panel on the shared x/y axes ----
    fig, axes = plt.subplots(2, 4, figsize=(22, 10), sharex=True, sharey=True)
    for ri, (pop, row_label, models) in enumerate(ROWS):
        for ci, (cut_label, cut) in enumerate(CUTS):
            ax = axes[ri, ci]
            P = panels[(ri, ci)]
            for label, colour, s, neff, tens in P["series"]:
                if s.size == 0:
                    continue
                ax.hist(s, bins=edges, density=True, color=colour, alpha=0.30)
                ax.plot(xs, gauss(xs, s.mean(), s.std()), color=colour, lw=2.0,
                        label=f"{label}: μ={s.mean():.2f} σ={s.std():.3f} ({tens:.1f}σ)")
            ax.hist(P["nv"], bins=edges, density=True, color="tab:green", alpha=0.34)
            ax.plot(xs, gauss(xs, P["n_mu"], P["n_sd"]), color="tab:green", lw=2.4,
                    label=f"NASA: μ={P['n_mu']:.2f} σ={P['n_sd']:.3f} (N={P['n_nasa']})")
            ax.set_xlim(gx_lo, gx_hi); ax.set_ylim(0, gy_hi)
            ax.set_title((f"{cut_label}\n" if ri == 0 else "") + f"{row_label} — puffy fraction", fontsize=10)
            ax.grid(alpha=0.2); ax.legend(fontsize=7.6, loc="upper left")
            if ci == 0:
                ax.set_ylabel("density over repeated draws")
            if ri == 1:
                ax.set_xlabel("puffy fraction")

    fig.suptitle("Puffy fraction: which universe does NASA imply, under four cuts — FLAT (top) vs P-Pop (bottom)\n"
                 "transit+RV detected, measurement-error propagated; closest bell to green NASA wins",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_png = os.path.join(OUT_DIR, "puffy_cuts_flat_ppop.png")
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"--> Saved: {out_png}")


if __name__ == "__main__":
    main()
