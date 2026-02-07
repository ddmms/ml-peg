import numpy as np

M_WATER = 18.01528   # g/mol
M_ETOH  = 46.06844   # g/mol

# Pick densities consistent with your reference conditions (T,P!)
# If your ref curve is at ~20°C, these are around:
RHO_WATER_PURE = 0.9982   # g/cm^3
RHO_ETH_PURE   = 0.7893   # g/cm^3

def x_to_phi_ethanol(x, rho_mix,
                     *, M_eth=M_ETOH, M_water=M_WATER,
                     rho_eth=RHO_ETH_PURE, rho_water=RHO_WATER_PURE):
    """
    Convert ethanol mole fraction x to ethanol volume fraction phi using
    mixture density rho_mix and pure-component densities as volume proxies.
    """
    x = np.asarray(x, dtype=float)
    rho_mix = np.asarray(rho_mix, dtype=float)

    m_eth = x * M_eth
    m_wat = (1.0 - x) * M_water

    V_mix = (m_eth + m_wat) / rho_mix         # cm^3 per "1 mol mixture basis"
    V_eth = m_eth / rho_eth                   # cm^3 (proxy)
    phi = V_eth / V_mix
    return phi

def weight_to_mole_fraction(w):
    """
    Convert ethanol weight fraction -> mole fraction.

    w = mass_ethanol / total_mass
    """
    n_e = w / M_ETOH
    n_w = (1 - w) / M_WATER
    return n_e / (n_e + n_w)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(np.mean(d * d)))


def _interp_1d(x_src: np.ndarray, y_src: np.ndarray, x_tgt: np.ndarray) -> np.ndarray:
    """
    Linear interpolation. Requires x_tgt within [min(x_src), max(x_src)].
    """
    if np.any(x_tgt < x_src.min() - 1e-12) or np.any(x_tgt > x_src.max() + 1e-12):
        raise ValueError("Target x values fall outside reference interpolation range.")
    return np.interp(x_tgt, x_src, y_src)


def _endpoints_at_0_1(x: np.ndarray, y: np.ndarray, tol: float = 1e-8) -> tuple[float, float]:
    """
    Return y(x=0) and y(x=1). Requires that x includes (approximately) 0 and 1.
    """
    i0 = np.where(np.isclose(x, 0.0, atol=tol))[0]
    i1 = np.where(np.isclose(x, 1.0, atol=tol))[0]
    if len(i0) != 1 or len(i1) != 1:
        raise ValueError("Curve must include x=0 and x=1 to define linear baseline.")
    return float(y[i0[0]]), float(y[i1[0]])


def _linear_baseline(x: np.ndarray, y0: float, y1: float) -> np.ndarray:
    return y0 + x * (y1 - y0)


def _excess_curve(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Excess relative to the dataset's own pure endpoints (x=0 and x=1).
    """
    y0, y1 = _endpoints_at_0_1(x, y)
    return y - _linear_baseline(x, y0, y1)


def _peak_x_quadratic(x: np.ndarray, y: np.ndarray) -> float:
    """
    Estimate x position of minimum y.
    - If min is interior and we have neighbors, fit quadratic through 3 points.
    - Otherwise fall back to argmin x.
    """
    if len(x) < 3:
        return float(x[int(np.argmin(y))])

    i = int(np.argmin(y))
    if i == 0 or i == len(x) - 1:
        return float(x[i])

    # Fit a parabola to (i-1, i, i+1)
    xs = x[i - 1 : i + 2]
    ys = y[i - 1 : i + 2]

    # y = ax^2 + bx + c
    a, b, _c = np.polyfit(xs, ys, deg=2)
    if abs(a) < 1e-16:
        return float(x[i])

    xv = -b / (2.0 * a)

    # Clamp to local bracket so we don't get silly extrapolation
    xv = float(np.clip(xv, xs.min(), xs.max()))
    return xv
