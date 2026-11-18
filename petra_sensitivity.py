#!/usr/bin/env python3
"""
Petra terracotta pipelines: closed-form reanalysis of the partial-flow interpretation in Ortloff (2020)
==================================================================================
Water 2020, 12, 3498.  doi:10.3390/w12123498

What this does
--------------
Treats each pipeline as a PARTIALLY-FULL CIRCULAR PIPE and uses closed-form
hydraulics (no CFD):
  * circular-segment geometry (A, T, P, R)
  * critical flow:   Q^2 * T / (g * A^3) = 1
  * normal flow:     S = f V^2 / (8 g R)  with Colebrook-White friction factor
  * critical slope Sc: the slope at which uniform flow occurs at critical depth
and then asks: how robust is the claim "slope S ~ critical slope" when the
unknown inputs (diameter, roughness, discharge, slope) vary over a plausible range?

Metric used everywhere:  ratio = S / Sc
    ratio ~ 1  -> near-critical     ratio > 1 -> steep (supercritical)
    ratio < 1  -> mild (subcritical) -> may become full / pressurised

Run:    python petra_sensitivity.py            (about 10-30 s)
Output: ./petra_outputs/  (figures + CSV + printed tables)

NOTES
-----
* Each input carries a status: PUBLISHED (page in Ortloff 2020), INFERRED
  (derived from published values) or ASSUMED (plausible range chosen by the author).
* Page references should be checked against the original paper.
* Colebrook-White is an extrapolation for relative roughness eps/(4R) > ~0.05.
  Ortloff's eps/D = 0.07-0.1 is a lumped roughness (joints + wear), so results
  at those values are indicative.
* Output figures: fig1_critical_slope_map.png (manuscript Fig. 1),
  fig2_tornado.png (Fig. 2), fig3_montecarlo.png (Fig. 3).
"""
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------
G = 9.82       # m/s^2 -- value Ortloff states (p. 22). Use 9.81 if you prefer.
NU = 1.0e-6    # m^2/s, water ~20 C   (ASSUMED)
OUT = Path("petra_outputs")
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(42)

# --------------------------------------------------------------------------
# INPUT TABLE  (edit here)  -- value / unit / source / status
# --------------------------------------------------------------------------
INPUTS = [
    # name, value, unit, source, status
    ("Siq: pipe diameter D",           0.14, "m",   "Ortloff 2020, p.12-13",          "PUBLISHED"),
    ("Pipe diameter observed range",   "0.14-0.23", "m", "Ortloff 2020, p.16",       "PUBLISHED"),
    ("Siq: slope (between basins)",    3.0,  "deg", "Ortloff 2020, p.13-14",          "PUBLISHED"),
    ("Siq: Q partial critical flow",   96/3600, "m3/s", "Ortloff 2020, p.14-15",      "PUBLISHED"),
    ("Siq: Q full flow @2deg / @1deg", "0.023 / 0.017", "m3/s", "Ortloff 2020, p.13", "PUBLISHED"),
    ("Ain Mousa output range",         "0.01-0.04", "m3/s", "Ortloff 2020, p.12 (Oleson)", "PUBLISHED"),
    ("eps/D (clean wall)",             0.01, "-",   "Ortloff 2020, p.12",             "PUBLISHED"),
    ("eps/D (with socket joints)",     0.07, "-",   "Ortloff 2020, p.13",             "PUBLISHED"),
    ("Mataha: slope",                  2.8,  "deg", "Ortloff 2020, p.20-23",          "PUBLISHED"),
    ("Mataha: Q (critical case C)",    0.062, "m3/s", "Ortloff 2020, p.22-23",        "PUBLISHED"),
    ("Mataha: eps/D",                  0.1,  "-",   "Ortloff 2020, p.21",             "PUBLISHED"),
    ("Mataha: pipe diameter D",        0.23, "m",   "NOT STATED; from V=1.5 m/s & Q=0.062, and sqrt(gD)=1.5 (p.23)", "INFERRED"),
    ("Kinematic viscosity",            NU,   "m2/s", "water at ~20 C",                "ASSUMED"),
]


# --------------------------------------------------------------------------
# GEOMETRY  (circular segment, depth ratio y/D)
# --------------------------------------------------------------------------
def geom(y_over_D, D):
    """Flow area A, free-surface width T, wetted perimeter P, hydraulic radius R."""
    yd = np.clip(np.asarray(y_over_D, float), 1e-6, 1.0 - 1e-9)
    th = 2.0 * np.arccos(1.0 - 2.0 * yd)
    A = D ** 2 / 8.0 * (th - np.sin(th))
    T = D * np.sin(th / 2.0)
    P = D * th / 2.0
    return A, T, P, A / P


# universal critical-flow curve:  Qc / (sqrt(g) D^2.5) = sqrt(A*^3 / T*)
_YG = np.linspace(0.001, 0.995, 4000)
_A, _T, _P, _R = geom(_YG, 1.0)
_QSTAR = np.sqrt(_A ** 3 / _T)


def critical_depth_ratio(Q, D):
    """yc/D for discharge Q in a pipe of diameter D (clipped at 0.995)."""
    q = np.asarray(Q, float) / (np.sqrt(G) * np.asarray(D, float) ** 2.5)
    return np.interp(q, _QSTAR, _YG)


def critical_discharge(y_over_D, D):
    A, T, _, _ = geom(y_over_D, D)
    return np.sqrt(G * A ** 3 / T)


# --------------------------------------------------------------------------
# FRICTION  (Colebrook-White, fixed-point iteration, vectorised)
# --------------------------------------------------------------------------
def friction_factor(rel_rough, Re, n_iter=60):
    rel_rough, Re = np.broadcast_arrays(np.asarray(rel_rough, float), np.asarray(Re, float))
    f = np.full(Re.shape, 0.05)
    for _ in range(n_iter):
        f_new = (-2.0 * np.log10(rel_rough / 3.7 + 2.51 / (Re * np.sqrt(f)))) ** -2
        f = 0.5 * f + 0.5 * f_new
    return f


def critical_slope(Q, D, eps_over_D):
    """Slope at which UNIFORM flow of discharge Q occurs at critical depth."""
    Q, D, eD = np.broadcast_arrays(*(np.asarray(x, float) for x in (Q, D, eps_over_D)))
    yc = critical_depth_ratio(Q, D)
    A, T, P, R = geom(yc, D)
    V = Q / A
    dh = 4.0 * R
    f = friction_factor(eD * D / dh, V * dh / NU)
    return f * V ** 2 / (8.0 * G * R), yc


def normal_flow_curve(D, eps_over_D, S, y_grid):
    """Uniform-flow discharge Qn(y) for each sample (rows) on a y/D grid (cols)."""
    A, T, P, R = geom(y_grid[None, :], D)
    dh = 4.0 * R
    rel = eps_over_D * D / dh
    f = np.full(A.shape, 0.05)
    V = np.sqrt(8.0 * G * R * S / f)
    for _ in range(60):
        Re = V * dh / NU
        f_new = (-2.0 * np.log10(rel / 3.7 + 2.51 / (Re * np.sqrt(f)))) ** -2
        f = 0.5 * f + 0.5 * f_new
        V = np.sqrt(8.0 * G * R * S / f)
    return A * V


def analyse(Q, D, eps_over_D, S, M=160):
    """Full classification for arrays/scalars of (Q [m3/s], D [m], eps/D, S [-])."""
    Q, D, eD, S = np.broadcast_arrays(*(np.atleast_1d(np.asarray(x, float)) for x in (Q, D, eps_over_D, S)))
    yg = np.linspace(0.02, 0.99, M)
    Qn = normal_flow_curve(D[:, None], eD[:, None], S[:, None], yg)
    Qopen_max = Qn.max(axis=1)                   # max open-channel capacity at slope S
    mask = Qn >= Q[:, None]
    pressurised = ~mask.any(axis=1)              # cannot pass as free-surface flow
    first = mask.argmax(axis=1)
    i = np.clip(first, 1, M - 1)
    rows = np.arange(len(Q))
    q0, q1 = Qn[rows, i - 1], Qn[rows, i]
    yn = yg[i - 1] + (Q - q0) / (q1 - q0) * (yg[i] - yg[i - 1])
    yn = np.where(first == 0, yg[0], yn)
    yn = np.where(pressurised, np.nan, yn)
    A, T, _, _ = geom(np.nan_to_num(yn, nan=0.5), D)
    Fr = np.where(pressurised, np.nan, (Q / A) / np.sqrt(G * A / T))
    Sc, yc = critical_slope(Q, D, eD)
    ratio = S / Sc
    # classification (near-critical band is an ASSUMED tolerance: 0.8-1.25)
    status = np.where(pressurised, "pressurised/full",
              np.where(ratio > 1.25, "supercritical",
              np.where(ratio < 0.8, "subcritical", "near-critical")))
    return dict(yn=yn, Fr=Fr, Sc=Sc, yc=yc, ratio=ratio, status=status,
                pressurised=pressurised, Qopen_max=Qopen_max)


def deg2S(deg):
    """Friction slope along the pipe for a pipe inclined at `deg` degrees."""
    return np.sin(np.radians(deg))


# --------------------------------------------------------------------------
# PART 1 -- SANITY CHECKS against numbers printed in the paper
# --------------------------------------------------------------------------
def part1_checks():
    print("=" * 78)
    print("PART 1  Sanity checks on Ortloff (2020) numbers")
    print("=" * 78)

    D = 0.14
    full_area = np.pi * D ** 2 / 4
    print("\n[1a] Mean velocity = Q / (full pipe area), D = 0.14 m  (paper p.13-14)")
    for Q, paper in [(0.017, 1.1), (0.023, 1.5), (96 / 3600, 1.73)]:
        print(f"     Q={Q:.4f} m3/s -> V={Q / full_area:5.2f} m/s   (paper: ~{paper})")

    print("\n[1b] Critical discharge of a PARTLY-FULL 0.14 m pipe vs the paper's 96 m3/h")
    print("     y/D     Qc [m3/h]")
    for yd in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99):
        print(f"     {yd:4.2f}   {critical_discharge(yd, D) * 3600:8.1f}")
    yc = critical_depth_ratio(96 / 3600, D)
    print(f"     -> Q = 96 m3/h is critical at y_c/D = {float(yc):.3f}   (paper: 0.8-0.9)")

    print("\n[1c] Mataha pipe diameter implied by the paper's own numbers (p.22-23)")
    Q, V = 0.062, 1.5
    print(f"     from Q/V:       A = {Q / V:.4f} m2 -> D = {np.sqrt(4 * Q / V / np.pi):.3f} m")
    print(f"     from sqrt(gD)=1.5, g={G}: D = {1.5 ** 2 / G:.3f} m")
    print("     -> consistent with D ~ 0.23 m (top of the 0.14-0.23 m range on p.16),")
    print("        not 0.14 m. (INFERRED: say so in the paper.)")
    for Dm in (0.14, 0.23):
        yc = float(critical_depth_ratio(0.062, Dm))
        A, T, _, _ = geom(yc, Dm)
        print(f"     D={Dm:.2f}: critical depth for 0.062 m3/s -> y_c/D = {yc:.3f}"
              f"{'  (paper: ~0.8)' if Dm == 0.23 else ''}")

    print("\n[1d] Slope vs elevation drop, Mataha (p.20): 910 m -> ~785 m over 1.5 km")
    drop = 910 - 785
    print(f"     drop {drop} m over 1.5 km -> angle = {np.degrees(np.arctan(drop / 1500)):.1f} deg "
          f"(paper states 2.8 deg -> drop = {1500 * np.tan(np.radians(2.8)):.0f} m)")
    print(f"     2.8 deg at {drop} m drop needs a pipe ~{drop / np.sin(np.radians(2.8)) / 1000:.2f} km long")


# --------------------------------------------------------------------------
# PART 2 -- Reproduce / test the paper's regime claims with this model
# --------------------------------------------------------------------------
def part2_regimes():
    print("\n" + "=" * 78)
    print("PART 2  Regime of the paper's scenarios under this (analytic) model")
    print("=" * 78)
    rows = []

    def show(label, Q, D, eD, deg):
        r = analyse(Q, D, eD, deg2S(deg))
        yn = r["yn"][0]
        print(f"{label:<38} D={D:.2f} eps/D={eD:<5} slope={deg:>3.1f}deg Q={Q:.4f} | "
              f"Sc={r['Sc'][0] * 100:5.2f}% S/Sc={r['ratio'][0]:5.2f} "
              f"yn/D={'  full' if np.isnan(yn) else f'{yn:5.2f}'} "
              f"Fr={'  n/a' if np.isnan(r['Fr'][0]) else f'{r['Fr'][0]:5.2f}'} -> {r['status'][0]}")
        rows.append([label, D, eD, deg, Q, r["Sc"][0], r["ratio"][0], yn, r["Fr"][0], r["status"][0]])

    print("\n-- Siq pipeline (paper: partial, near-critical, Fr~1, ~96 m3/h, ~3 deg) --")
    for eD in (0.01, 0.07):
        show("Siq, 96 m3/h", 96 / 3600, 0.14, eD, 3.0)
    print("\n-- Wadi Mataha, three slope options (paper: A full-flow, B supercritical+jump, C critical) --")
    for Dm in (0.14, 0.23):
        print(f"   [D = {Dm} m{' (INFERRED)' if Dm == 0.23 else ''}]")
        show("Mataha A (1 deg)", 0.05, Dm, 0.1, 1.0)
        show("Mataha B (4 deg)", 0.08, Dm, 0.1, 4.0)
        show("Mataha C (2.8 deg)", 0.062, Dm, 0.1, 2.8)

    with open(OUT / "part2_regimes.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "D_m", "eps_over_D", "slope_deg", "Q_m3s", "Sc", "S_over_Sc", "yn_over_D", "Fr", "status"])
        w.writerows(rows)


# --------------------------------------------------------------------------
# PART 3 -- One-at-a-time sensitivity (tornado)
# --------------------------------------------------------------------------
SYSTEMS = {
    # baseline = values from the paper;  ranges = published / assumed plausible
    "Siq (Q=96 m3/h)": dict(
        base=dict(D=0.14, eD=0.07, deg=3.0, Q=96 / 3600),
        rng=dict(D=(0.14, 0.23), eD=(0.01, 0.10), deg=(2.0, 4.0), Q=(0.010, 0.040)),
        note="D: p.16 range; eps/D: p.12-13/21; slope: ~3 deg +-1; Q: Ain Mousa range p.12"),
    "Mataha (Q=0.062)": dict(
        base=dict(D=0.23, eD=0.10, deg=2.8, Q=0.062),
        rng=dict(D=(0.14, 0.23), eD=(0.01, 0.10), deg=(2.0, 4.8), Q=(0.050, 0.080)),
        note="D: p.16 range (baseline INFERRED); slope up to 4.8 deg from p.20 elevations; Q: p.21-23 cases"),
}
LABEL = dict(D="diameter D", eD="roughness eps/D", deg="slope", Q="discharge Q")


def ratio_for(p):
    Sc, _ = critical_slope(p["Q"], p["D"], p["eD"])
    return float(deg2S(p["deg"]) / Sc)


def part3_tornado():
    print("\n" + "=" * 78)
    print("PART 3  One-at-a-time sensitivity of S/Sc  (1 = exactly critical)")
    print("=" * 78)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, (name, cfg) in zip(axes, SYSTEMS.items()):
        b = cfg["base"]
        r0 = ratio_for(b)
        print(f"\n{name}   baseline S/Sc = {r0:.2f}\n  ({cfg['note']})")
        rows = []
        for k, (lo, hi) in cfg["rng"].items():
            rl = ratio_for({**b, k: lo})
            rh = ratio_for({**b, k: hi})
            rows.append((k, rl, rh))
            print(f"  {LABEL[k]:<16} {lo:>7} -> {hi:<7}  S/Sc: {rl:5.2f} -> {rh:5.2f}")
        rows.sort(key=lambda t: abs(np.log10(t[2]) - np.log10(t[1])))
        for i, (k, rl, rh) in enumerate(rows):
            ax.barh(i, np.log10(rh) - np.log10(r0), left=np.log10(r0), color="#c0392b", alpha=.8)
            ax.barh(i, np.log10(rl) - np.log10(r0), left=np.log10(r0), color="#2c7fb8", alpha=.8)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([LABEL[k] for k, _, _ in rows])
        ax.axvline(0, color="k", lw=.8)
        ax.axvspan(np.log10(.8), np.log10(1.25), color="g", alpha=.12, label="near-critical band (0.8-1.25)")
        ax.set_xlabel("log10(S / Sc)")
        ax.set_title(f"{name}\nbaseline S/Sc = {r0:.2f}", fontsize=10)
        ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_tornado.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# PART 4 -- Monte Carlo
# --------------------------------------------------------------------------
def loguniform(lo, hi, n):
    return np.exp(RNG.uniform(np.log(lo), np.log(hi), n))


def part4_montecarlo(N=4000):
    print("\n" + "=" * 78)
    print(f"PART 4  Monte Carlo (N = {N}); inputs sampled independently over the ranges below")
    print("=" * 78)
    spaces = {
        "Siq":    dict(D=(0.14, 0.23), eD=(0.005, 0.10), deg=(2.0, 4.0), Q=(0.010, 0.040)),
        "Mataha": dict(D=(0.14, 0.23), eD=(0.010, 0.10), deg=(2.0, 4.8), Q=(0.040, 0.080)),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    summary = []
    for ax, (name, sp) in zip(axes, spaces.items()):
        D = RNG.uniform(*sp["D"], N)
        eD = loguniform(*sp["eD"], N)
        deg = RNG.uniform(*sp["deg"], N)
        Q = RNG.uniform(*sp["Q"], N)
        r = analyse(Q, D, eD, deg2S(deg), M=120)
        st = r["status"]
        print(f"\n{name}: D {sp['D']} m, eps/D {sp['eD']} (log-uniform), slope {sp['deg']} deg, Q {sp['Q']} m3/s")
        for s in ("subcritical", "near-critical", "supercritical", "pressurised/full"):
            frac = np.mean(st == s) * 100
            print(f"   {s:<18}{frac:6.1f} %")
            summary.append([name, s, frac])
        lr = np.log10(r["ratio"])
        print(f"   median S/Sc = {10 ** np.median(lr):.2f}   5-95 %: {10 ** np.percentile(lr, 5):.2f} - {10 ** np.percentile(lr, 95):.2f}")
        ax.hist(lr, bins=50, color="#34495e", alpha=.85)
        ax.axvspan(np.log10(.8), np.log10(1.25), color="g", alpha=.2)
        ax.axvline(0, color="k", lw=.8)
        ax.set_title(f"{name}: spread of S/Sc under input uncertainty", fontsize=10)
        ax.set_xlabel("log10(S / Sc)   (green = near-critical band)")
    axes[0].set_ylabel("samples")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_montecarlo.png", dpi=200)
    plt.close(fig)
    with open(OUT / "part4_montecarlo_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "regime", "percent_of_samples"])
        w.writerows(summary)


# --------------------------------------------------------------------------
# PART 5 -- Critical-slope map
# --------------------------------------------------------------------------
def part5_map():
    Dg = np.linspace(0.14, 0.23, 60)
    eg = np.logspace(np.log10(0.005), np.log10(0.1), 60)
    DD, EE = np.meshgrid(Dg, eg)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, Q, ttl in ((axes[0], 96 / 3600, "Q = 96 m3/h (Siq case)"), (axes[1], 0.062, "Q = 0.062 m3/s (Mataha case)")):
        Sc, _ = critical_slope(Q, DD, EE)
        ang = np.degrees(np.arcsin(Sc))
        cs = ax.contourf(DD * 100, EE, ang, levels=20, cmap="viridis")
        ax.set_yscale("log")
        ax.set_xlabel("pipe diameter D [cm]")
        ax.set_ylabel("eps/D")
        ax.set_title(f"Critical slope [deg], {ttl}", fontsize=10)
        fig.colorbar(cs, ax=ax)
        obs = 3.0 if Q < 0.05 else 2.8
        ax.contour(DD * 100, EE, ang, levels=[obs], colors="r", linewidths=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_critical_slope_map.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    print("INPUT TABLE (verify every PUBLISHED entry against the PDF)")
    for n, v, u, s, st in INPUTS:
        print(f"  [{st:<9}] {n:<34} {v!s:<16} {u:<5} {s}")
    with open(OUT / "inputs_table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "value", "unit", "source", "status"])
        w.writerows(INPUTS)
    part1_checks()
    part2_regimes()
    part3_tornado()
    part4_montecarlo(N)
    part5_map()
    print("\nDone. Files written to", OUT.resolve())


if __name__ == "__main__":
    main()
