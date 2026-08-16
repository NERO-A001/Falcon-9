

import numpy as np
import matplotlib.pyplot as plt

try:
    import scienceplots  # noqa: F401
    plt.style.use(['science', 'notebook', 'grid'])
except Exception:
    plt.style.use('seaborn-v0_8-darkgrid')

# Rocket
dry_mass1 = 25600.0
dry_mass2 = 4000.0
propellent_mass1 = 395700.0
propellent_mass2 = 92670.0

payload_mass = 15000.0          # satellite stack
fairing_mass = 1900.0           # jettisoned during stage 2

m0 = (dry_mass1 + propellent_mass1 +
      dry_mass2 + propellent_mass2 +
      payload_mass + fairing_mass)

A_e = 10.8                      # reference area, m^2
SL_thrust = 7607000.0
V_thrust = 8227000.0
SL_Isp = 282.0
V_Isp = 311.0
P2_Isp = 348.0
Merlin_thrust = 934000.0

MAX_G = 4.0                     # throttle cap (structural / payload limit)
FAIRING_JETT_ALT = 110000.0


# EARTH
RE = 6371000.0
g0 = 9.80665
mu = g0 * RE**2
t0 = 288.15
L = 0.0065
L3 = 0.001
P0 = 101325.0
M_air = 0.0289644
R = 8.31447
R_specific = R / M_air
Cd = 0.3

# MISSION / GUIDANCE
TARGET_ALT = 284600.0                       # altitude the guidance holds
TARGET_SMA = RE + 284600.0                  # cutoff when SMA reaches this
SOE_TARGET = -mu / (2.0 * TARGET_SMA)

TAU = 600.0                                 # altitude-error time constant, s
K_P = 0.20                                  # gain on flight-path-angle error
MAX_MISMATCH = np.deg2rad(35.0)

t_vertical = 10.0
kick_duration = 6.0
kick_angle = np.deg2rad(6.26)               # tuned for near-circular insertion

dt = 0.25
T_MAX = 900.0
COAST_STEPS = 5600                          # ~one revolution after cutoff
COAST_DT = 1.0


def get_gravity(height):
    return g0 * RE**2 / (RE + height)**2


def get_air_density(height):
    h = max(height, 0.0)
    if h < 11000.0:
        Temp = t0 - L * h
        pressure = P0 * (Temp / t0) ** ((g0 * M_air) / (R * L))
    elif h < 20000.0:
        Temp = 216.65
        pressure = 22632.0 * np.exp(-g0 * M_air * (h - 11000.0) / (R * Temp))
    elif h < 32000.0:
        T20 = 216.65
        Temp = T20 + L3 * (h - 20000.0)
        pressure = 5475.0 * (T20 / Temp) ** (g0 * M_air / (R * L3))
    else:
        Temp = 228.65
        pressure = 868.0 * np.exp(-(h - 32000.0) / 7000.0)
    return pressure / (R_specific * Temp), pressure


def get_thrust(height, mass, stage, cutoff):
    """Stage selected by `stage`, never by absolute mass."""
    if cutoff or stage not in (1, 2):
        return 0.0, 0.0

    if stage == 1:
        _, pressure = get_air_density(height)
        thrust = V_thrust - (V_thrust - SL_thrust) * (pressure / P0)
        Isp = V_Isp - (V_Isp - SL_Isp) * (pressure / P0)
    else:
        thrust = Merlin_thrust
        Isp = P2_Isp

    a_cap = MAX_G * g0                       # throttle to hold the g cap
    if thrust / mass > a_cap:
        thrust = a_cap * mass

    return thrust, thrust / (Isp * g0)


def commanded_angle(t, height, v, gamma, stage):
    if stage == 1:
        if t < t_vertical:
            return np.pi / 2
        if t < t_vertical + kick_duration:
            return np.pi / 2 - kick_angle * (t - t_vertical) / kick_duration
        return gamma                                     # pure gravity turn

    # stage 2: cascade altitude -> vertical speed -> flight-path angle
    if v < 1.0:
        return gamma
    vv_target = np.clip((TARGET_ALT - height) / TAU, -400.0, 400.0)
    gamma_target = np.arcsin(np.clip(vv_target / v, -1.0, 1.0))
    cmd = gamma + K_P * (gamma_target - gamma)
    return np.clip(cmd, gamma - MAX_MISMATCH, gamma + MAX_MISMATCH)


def derivative(S, t, stage, cutoff):
    height, phi, v, mass, gamma = S
    r = RE + height

    density, _ = get_air_density(height)
    drag = 0.5 * density * v**2 * A_e * Cd
    thrust, mdot = get_thrust(height, mass, stage, cutoff)
    g = get_gravity(height)

    mismatch = commanded_angle(t, height, v, gamma, stage) - gamma

    dhdt = v * np.sin(gamma)
    dphidt = v * np.cos(gamma) / r
    dvdt = (thrust * np.cos(mismatch) - drag) / mass - g * np.sin(gamma)
    dmdt = -mdot

    if v < 1.0:
        dgdt = 0.0
    else:
        dgdt = ((thrust * np.sin(mismatch)) / (v * mass)
                - (g * np.cos(gamma)) / v
                + (v * np.cos(gamma)) / r)          # <-- curvature term

    return np.array([dhdt, dphidt, dvdt, dmdt, dgdt])


def rk4(S, t, h, stage, cutoff):
    k1 = derivative(S, t, stage, cutoff)
    k2 = derivative(S + k1 * h / 2, t + h / 2, stage, cutoff)
    k3 = derivative(S + k2 * h / 2, t + h / 2, stage, cutoff)
    k4 = derivative(S + k3 * h, t + h, stage, cutoff)
    return S + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def elements(height, v, gamma, phi):
    r = RE + height
    SOE = v**2 / 2.0 - mu / r
    SAM = r * v * np.cos(gamma)
    p = SAM**2 / mu
    e = np.sqrt(max(0.0, 1.0 + 2.0 * SOE * SAM**2 / mu**2))

    out = dict(r=r, SOE=SOE, SAM=SAM, e=e, p=p, SMA=np.nan,
               perigee=np.nan, apogee=np.nan, period=np.nan,
               nu=np.nan, omega=np.nan)

    if SOE < 0.0:
        SMA = -mu / (2.0 * SOE)
        out["SMA"] = SMA
        out["perigee"] = SMA * (1.0 - e) - RE
        out["apogee"] = SMA * (1.0 + e) - RE
        out["period"] = 2.0 * np.pi * np.sqrt(SMA**3 / mu)

    if e > 1e-9:
        nu = np.arccos(np.clip((p / r - 1.0) / e, -1.0, 1.0))
        if gamma < 0.0:                      # inbound leg resolves the sign
            nu = -nu
    else:
        nu = 0.0
    out["nu"] = nu
    out["omega"] = phi - nu                  # perigee direction, inertial
    return out


def delta_v_budget():
    m1_0 = m0
    m1_f = m0 - propellent_mass1
    m2_0 = m1_f - dry_mass1
    m2_f = m2_0 - propellent_mass2
    Isp1 = 0.5 * (SL_Isp + V_Isp)
    dv1 = Isp1 * g0 * np.log(m1_0 / m1_f)
    dv2 = P2_Isp * g0 * np.log(m2_0 / m2_f)
    return dv1, dv2, dv1 + dv2


# ======================================================================
# INTEGRATION
# ======================================================================
def run():
    S = np.array([0.0, 0.0, 0.0, m0, np.pi / 2])
    t = 0.0
    stage, cutoff, fairing_gone = 1, False, False
    p1, p2 = propellent_mass1, propellent_mass2
    sep_time = seco_time = np.nan
    insertion = None

    keys = ("t", "h", "phi", "v", "m", "g", "SOE", "SAM",
            "per", "apo", "e", "q", "acc", "burning")
    hist = {k: [] for k in keys}

    def log(t, S, stage, cutoff):
        h, phi, v, m, gam = S
        density, _ = get_air_density(h)
        thrust, _ = get_thrust(h, m, stage, cutoff)
        hist["t"].append(t); hist["h"].append(h); hist["phi"].append(phi)
        hist["v"].append(v); hist["m"].append(m); hist["g"].append(gam)
        hist["q"].append(0.5 * density * v**2)
        hist["acc"].append(thrust / m / g0)
        hist["burning"].append(thrust > 0.0)
        if stage == 2 or cutoff:
            el = elements(h, v, gam, phi)
            hist["SOE"].append(el["SOE"]); hist["SAM"].append(el["SAM"])
            hist["per"].append(el["perigee"]); hist["apo"].append(el["apogee"])
            hist["e"].append(el["e"])
        else:
            for k in ("SOE", "SAM", "per", "apo", "e"):
                hist[k].append(np.nan)

    log(t, S, stage, cutoff)

    # ---------------- powered flight ---------------------------------
    while t < T_MAX and not cutoff:
        Sn = rk4(S, t, dt, stage, cutoff)
        burned = S[3] - Sn[3]
        if stage == 1 and burned > p1:
            Sn[3] = S[3] - p1
            burned = p1
        if stage == 2 and burned > p2:
            Sn[3] = S[3] - p2
            burned = p2

        # ---- cutoff: locate the target-energy crossing exactly ------
        if stage == 2:
            if elements(Sn[0], Sn[2], Sn[4], Sn[1])["SOE"] >= SOE_TARGET:
                lo, hi = 0.0, dt
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    Sm = rk4(S, t, mid, stage, cutoff)
                    if elements(Sm[0], Sm[2], Sm[4], Sm[1])["SOE"] >= SOE_TARGET:
                        hi = mid
                    else:
                        lo = mid
                Sn = rk4(S, t, hi, stage, cutoff)
                p2 -= (S[3] - Sn[3])
                S, t = Sn, t + hi
                cutoff, seco_time = True, t
                insertion = dict(elements(S[0], S[2], S[4], S[1]),
                                 t=t, height=S[0], v=S[2], gamma=S[4],
                                 mass=S[3], prop_left=p2, reason="target energy reached")
                log(t, S, stage, cutoff)
                break

        if stage == 1:
            p1 -= burned
        else:
            p2 -= burned
        S, t = Sn, t + dt

        if stage == 1 and p1 <= 1e-6:                    # MECO / separation
            sep_time = t
            S[3] -= dry_mass1
            stage = 2
        if stage == 2 and not fairing_gone and S[0] > FAIRING_JETT_ALT:
            S[3] -= fairing_mass
            fairing_gone = True
        if stage == 2 and p2 <= 1e-6:                    # ran dry
            cutoff, seco_time = True, t
            insertion = dict(elements(S[0], S[2], S[4], S[1]),
                             t=t, height=S[0], v=S[2], gamma=S[4],
                             mass=S[3], prop_left=0.0, reason="propellant depleted")

        log(t, S, stage, cutoff)
        if S[0] < 0.0:
            break

    # ---------------- ballistic coast --------------------------------
    if insertion is not None and insertion["SOE"] < 0:
        for _ in range(COAST_STEPS):
            S = rk4(S, t, COAST_DT, 0, True)
            t += COAST_DT
            log(t, S, 0, True)
            if S[0] < 0.0:
                break

    for k in hist:
        hist[k] = np.array(hist[k])
    return hist, insertion, sep_time, seco_time


def report(hist, ins, sep_time, seco_time):
    W = 64
    print("=" * W)
    print("  FALCON 9 ASCENT — ORBITAL INSERTION REPORT")
    print("=" * W)

    dv1, dv2, dvt = delta_v_budget()
    print(f"  Lift-off mass            {m0:14,.0f} kg")
    print(f"  Payload + fairing        {payload_mass + fairing_mass:14,.0f} kg")
    print(f"  Ideal dv, stage 1        {dv1:14,.0f} m/s")
    print(f"  Ideal dv, stage 2        {dv2:14,.0f} m/s")
    print(f"  Ideal dv, total          {dvt:14,.0f} m/s")

    imaxq = int(np.nanargmax(hist["q"]))
    print(f"\n  Max dynamic pressure     {hist['q'][imaxq]/1000:14.2f} kPa "
          f"at t = {hist['t'][imaxq]:.1f} s")
    print(f"  Max axial acceleration   {np.nanmax(hist['acc']):14.2f} g")

    if np.isfinite(sep_time):
        i = int(np.argmin(np.abs(hist["t"] - sep_time)))
        print(f"\n  MECO / SEPARATION        t = {sep_time:10.2f} s")
        print(f"      altitude             {hist['h'][i]/1000:14.2f} km")
        print(f"      velocity             {hist['v'][i]:14.1f} m/s")
        print(f"      flight-path angle    {np.rad2deg(hist['g'][i]):14.2f} deg")
    else:
        print("\n  MECO                     never reached")

    if ins is None:
        print("\n  INSERTION                FAILED — no cutoff occurred")
        print("=" * W)
        return

    v_circ = np.sqrt(mu / ins["r"])
    print(f"\n  SECO                     t = {ins['t']:10.2f} s   ({ins['reason']})")
    print(f"      altitude             {ins['height']/1000:14.2f} km")
    print(f"      velocity             {ins['v']:14.1f} m/s")
    print(f"      circular velocity    {v_circ:14.1f} m/s")
    print(f"      excess over v_circ   {ins['v'] - v_circ:14.1f} m/s")
    print(f"      flight-path angle    {np.rad2deg(ins['gamma']):14.4f} deg")
    print(f"      propellant remaining {ins['prop_left']:14,.0f} kg")

    print("\n  ORBIT")
    if ins["SOE"] >= 0.0:
        print("      classification       "
              f"{'HYPERBOLIC / ESCAPE' if ins['e'] > 1 else 'PARABOLIC':>14}")
        print(f"      specific energy      {ins['SOE']:14.4e} J/kg  (positive)")
        print("      *** NOT A CLOSED ORBIT ***")
        print("=" * W)
        return

    print(f"      semi-major axis      {ins['SMA']/1000:14.2f} km")
    print(f"      eccentricity         {ins['e']:14.6f}")
    print(f"      perigee altitude     {ins['perigee']/1000:14.2f} km")
    print(f"      apogee altitude      {ins['apogee']/1000:14.2f} km")
    print(f"      period               {ins['period']/60:14.2f} min")
    print(f"      specific energy      {ins['SOE']:14.4e} J/kg")
    print(f"      angular momentum     {ins['SAM']:14.4e} m^2/s")
    print(f"      true anomaly         {np.rad2deg(ins['nu']):14.2f} deg")

    if ins["perigee"] < 100000:
        cls = "SUBORBITAL (perigee in atmosphere)"
    elif ins["perigee"] < 2000000:
        cls = "LOW EARTH ORBIT"
    elif ins["perigee"] < 35786000:
        cls = "MEDIUM EARTH ORBIT"
    else:
        cls = "HIGH / GEO+"
    print(f"      classification       {cls:>14}")

    m = hist["t"] > seco_time
    if m.sum() > 5:
        E, H = hist["SOE"][m], hist["SAM"][m]
        print("\n  INTEGRATOR CHECK (ballistic coast, conserved quantities)")
        print(f"      SOE relative drift   {abs(E[-1]-E[0])/abs(E[0]):14.3e}")
        print(f"      SAM relative drift   {abs(H[-1]-H[0])/abs(H[0]):14.3e}")
    print("=" * W)

def plot_timeseries(hist, sep_time, seco_time):
    fig, axs = plt.subplots(4, 2, figsize=(13, 14))
    t = hist["t"]
    burn = hist["burning"]

    def marks(ax):
        if np.isfinite(sep_time):
            ax.axvline(sep_time, color="k", ls="--", lw=0.9, alpha=0.6)
        if np.isfinite(seco_time):
            ax.axvline(seco_time, color="r", ls="--", lw=0.9, alpha=0.6)

    panels = [
        (axs[0, 0], hist["m"] / 1000, "Mass [t]"),
        (axs[0, 1], hist["h"] / 1000, "Altitude [km]"),
        (axs[1, 0], hist["v"], "Velocity [m/s]"),
        (axs[1, 1], np.rad2deg(hist["g"]), "Flight-path angle [deg]"),
        (axs[2, 0], hist["q"] / 1000, "Dynamic pressure [kPa]"),
        (axs[2, 1], hist["acc"], "Axial acceleration [g]"),
        (axs[3, 0], hist["SOE"], "Specific orbital energy [J/kg]"),
        (axs[3, 1], hist["per"] / 1000, "Perigee / apogee altitude [km]"),
    ]
    for ax, y, title in panels:
        ax.plot(t, y, lw=1.4)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("t [s]")
        marks(ax)

    axs[3, 1].plot(t, hist["apo"] / 1000, lw=1.4, alpha=0.8, label="apogee")
    axs[3, 1].plot(t, hist["per"] / 1000, lw=1.4, label="perigee")
    axs[3, 1].legend(fontsize=8)
    axs[1, 1].axhline(0, color="r", lw=0.8, alpha=0.5)

    # ascent panels: zoom to powered flight. element panels: full timeline.
    t_end = (seco_time * 1.10) if np.isfinite(seco_time) else t[-1]
    for ax in (axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1], axs[2, 1]):
        ax.set_xlim(0, t_end)
    axs[2, 0].set_xlim(0, min(300, t[-1]))
    if np.isfinite(seco_time):
        axs[3, 0].set_xlim(seco_time * 0.3, t[-1])
        axs[3, 1].set_xlim(seco_time * 0.3, t[-1])
        axs[3, 1].set_ylim(-400, 800)

    fig.suptitle("Falcon 9 two-stage ascent  "
                 "(black dashed = MECO, red dashed = SECO)", fontsize=13)
    fig.tight_layout()
    return fig


def plot_orbit(hist, ins):
    fig, axs = plt.subplots(1, 2, figsize=(15, 7.5))
    r = RE + hist["h"]
    burn = hist["burning"]

    # ---- inertial view ---------------------------------------------
    ax = axs[0]
    th = np.linspace(0, 2 * np.pi, 500)
    ax.fill(RE * np.cos(th) / 1e6, RE * np.sin(th) / 1e6,
            color="#3a6ea5", alpha=0.6, zorder=1, label="Earth")

    ax.plot(r[~burn] * np.cos(hist["phi"][~burn]) / 1e6,
            r[~burn] * np.sin(hist["phi"][~burn]) / 1e6,
            color="darkorange", lw=3.0, zorder=2, label="integrated coast")
    ax.plot(r[burn] * np.cos(hist["phi"][burn]) / 1e6,
            r[burn] * np.sin(hist["phi"][burn]) / 1e6,
            color="crimson", lw=3.0, zorder=3, label="powered ascent")

    if ins is not None and ins["SOE"] < 0 and np.isfinite(ins["omega"]):
        nu = np.linspace(0, 2 * np.pi, 800)
        rc = ins["p"] / (1.0 + ins["e"] * np.cos(nu))
        ang = nu + ins["omega"]
        ax.plot(rc * np.cos(ang) / 1e6, rc * np.sin(ang) / 1e6,
                color="k", ls=(0, (6, 6)), lw=1.2, zorder=4,
                label="analytic conic from SECO elements")

    if ins is not None:
        i = int(np.argmin(np.abs(hist["t"] - ins["t"])))
        ax.plot(r[i] * np.cos(hist["phi"][i]) / 1e6,
                r[i] * np.sin(hist["phi"][i]) / 1e6,
                "o", color="yellow", mec="k", ms=9, zorder=5, label="SECO")

    ax.set_aspect("equal")
    ax.set_xlabel("x [1000 km]")
    ax.set_ylabel("y [1000 km]")
    ax.set_title("Inertial frame — integrated arc vs analytic conic")
    ax.legend(fontsize=8, loc="lower left", framealpha=0.9)

    
    ax = axs[1]
    downrange = RE * hist["phi"] / 1000
    ax.plot(downrange[burn], hist["h"][burn] / 1000,
            color="crimson", lw=2.2, label="powered")
    ax.plot(downrange[~burn], hist["h"][~burn] / 1000,
            color="darkorange", lw=1.6, label="coast")
    ax.axhline(0, color="#3a6ea5", lw=3)
    if ins is not None and np.isfinite(ins["perigee"]):
        ax.axhline(ins["perigee"] / 1000, color="g", ls=":", lw=1.2, label="perigee")
        ax.axhline(ins["apogee"] / 1000, color="purple", ls=":", lw=1.2, label="apogee")
    ax.set_xlabel("downrange [km]")
    ax.set_ylabel("altitude [km]")
    ax.set_title("Ascent profile (altitude exaggerated)")
    ax.set_xlim(0, downrange[hist["burning"]].max() * 1.05)
    ax.set_ylim(0, None)
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    hist, ins, sep_time, seco_time = run()
    report(hist, ins, sep_time, seco_time)
    plot_timeseries(hist, sep_time, seco_time)
    plot_orbit(hist, ins)
    plt.show()