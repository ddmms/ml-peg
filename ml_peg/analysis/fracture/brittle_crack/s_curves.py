"""
S-curve metric of the brittle_crack benchmark (A' variant).

An S-curve is K/K_G against a reaction coordinate x (a bond length on the reference
topology, in units of the model's own bulk bond) along a traced crack path. Model and
DFT reference are compared on the reference's window: the K error E_K, the bond-breaking
energy error E_E (the elastic term c k^2 with one c~ = c / (2 gamma L b_bulk) from the
reference for every curve), the coverage of the window, and the energy-norm displacement
error E_D near the tip. Coverage enters ML-PEG's raw value by padding:
x_eff = c max(x, g) + (1 - c) b. Ported from the crackcell metric.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ase.units import J, m
import numpy as np

BACKSTEP_TOL = (
    0.03  # A: coordinate back-steps below this do not end an increasing stretch
)
N_GRID = 200
R_D = 8.0  # A: E_D pairs within this distance of the tip (in the crack plane)
WAKE = 1.5  # A: E_D pairs at most this far behind the tip


def drop_backsteps(b, tol: float = BACKSTEP_TOL) -> np.ndarray:
    """
    Return the indices to keep, skipping points less than ``tol`` below the maximum.

    Parameters
    ----------
    b
        Coordinate along the path.
    tol
        Back-step tolerance (A).

    Returns
    -------
    numpy.ndarray
        Indices of the kept points; a larger drop starts a new stretch and is kept.
    """
    b = np.asarray(b, float)
    if not len(b):
        return np.arange(0)
    keep, top = [0], b[0]
    for i in range(1, len(b)):
        if b[i] > top:
            keep.append(i)
            top = b[i]
        elif b[i] < top - tol:
            keep.append(i)
            top = b[i]
    return np.array(keep, int)


@dataclass
class SCurve:
    """
    An S-curve: load, coordinate and energy along a traced path.

    Attributes
    ----------
    k
        K/K_G with the curve's own K_G.
    K_abs
        Absolute K (eV/A^2.5).
    b
        Coordinate (A).
    E
        Energy (eV).
    K_G
        Griffith K (eV/A^2.5).
    a0
        Lattice constant (A).
    gamma
        Surface energy (J/m^2).
    front
        Cluster length along the crack front (A).
    """

    k: np.ndarray
    K_abs: np.ndarray
    b: np.ndarray
    E: np.ndarray
    K_G: float
    a0: float
    gamma: float
    front: float

    def __post_init__(self) -> None:
        """Convert the per-point fields to float arrays."""
        for name in ("k", "K_abs", "b", "E"):
            setattr(self, name, np.asarray(getattr(self, name), float))

    @property
    def b_bulk(self) -> float:
        """
        Bulk bond length of the curve's lattice.

        Returns
        -------
        float
            Bulk bond, sqrt(3) a0 / 4 (A).
        """
        return float(np.sqrt(3.0) * self.a0 / 4.0)

    def prefix(self) -> int:
        """
        Return the number of leading points along which b increases strictly.

        Returns
        -------
        int
            Length of the increasing prefix.
        """
        bad = np.nonzero(np.diff(self.b) <= 1e-6)[0]
        return int(bad[0] + 1) if len(bad) else len(self.b)

    def runs(self) -> list[tuple[int, int]]:
        """
        Return the maximal stretches along which b increases strictly.

        Returns
        -------
        list[tuple[int, int]]
            Half-open index ranges [i0, i1).
        """
        out, i0 = [], 0
        for i, up in enumerate(np.diff(self.b) > 1e-6):
            if not up:
                out.append((i0, i + 1))
                i0 = i + 1
        out.append((i0, len(self.b)))
        return out

    def take(self, idx) -> SCurve:
        """
        Return the curve at the given point indices.

        Parameters
        ----------
        idx
            Point indices.

        Returns
        -------
        SCurve
            The selected points.
        """
        idx = np.asarray(idx, int)
        return SCurve(
            self.k[idx],
            self.K_abs[idx],
            self.b[idx],
            self.E[idx],
            self.K_G,
            self.a0,
            self.gamma,
            self.front,
        )

    def without_backsteps(self, tol: float = BACKSTEP_TOL) -> SCurve:
        """
        Return the curve without its back-steps below ``tol``.

        Parameters
        ----------
        tol
            Back-step tolerance (A).

        Returns
        -------
        SCurve
            The curve with small dips removed.
        """
        return self.take(drop_backsteps(self.b, tol))

    def slice(self, i0: int, i1: int) -> SCurve:
        """
        Return the points i0 to i1 (exclusive).

        Parameters
        ----------
        i0
            First index.
        i1
            End index.

        Returns
        -------
        SCurve
            The slice.
        """
        return self.take(np.arange(i0, i1))

    def folds(self, rel_prominence: float = 1e-3) -> list[int]:
        """
        Return the extrema of k along the prefix, without noise pairs.

        Parameters
        ----------
        rel_prominence
            Extremum pairs closer in k than this fraction of the k range are removed.

        Returns
        -------
        list[int]
            Indices of the folds.
        """
        n = self.prefix()
        k = self.k[:n]
        dk = np.diff(k)
        ext = [i for i in range(1, n - 1) if dk[i - 1] * dk[i] < 0]
        tol = rel_prominence * (k.max() - k.min()) if n else 0.0
        while len(ext) >= 2:
            gaps = [abs(k[ext[j + 1]] - k[ext[j]]) for j in range(len(ext) - 1)]
            j = int(np.argmin(gaps))
            if gaps[j] >= tol:
                break
            del ext[j : j + 2]
        return ext


def pair_distance(atoms, pair: tuple[int, int]) -> float:
    """
    Return the distance of an atom pair, minimum image along the front (z).

    Parameters
    ----------
    atoms
        Structure.
    pair
        Atom indices.

    Returns
    -------
    float
        Distance (A).
    """
    d = atoms.positions[pair[1]] - atoms.positions[pair[0]]
    cz = atoms.cell[2, 2]
    d[2] -= cz * np.round(d[2] / cz)
    return float(np.linalg.norm(d))


def curve_from_frames(frames, pair, a0: float, gamma: float, k_g: float) -> SCurve:
    """
    Build the S-curve of a traced path on the coordinate ``pair``.

    Parameters
    ----------
    frames
        Path frames with info ``K_over_KG`` and ``REF_energy``.
    pair
        Atom indices of the coordinate bond.
    a0
        Lattice constant of the curve's material (A).
    gamma
        Surface energy (J/m^2).
    k_g
        Griffith K of the curve's material (eV/A^2.5).

    Returns
    -------
    SCurve
        The curve.
    """
    k = np.array([f.info["K_over_KG"] for f in frames], float)
    b = [pair_distance(f, tuple(pair)) for f in frames]
    e = [f.info["REF_energy"] for f in frames]
    return SCurve(k, k_g * k, b, e, k_g, a0, gamma, float(frames[0].cell[2, 2]))


def _x(c: SCurve) -> np.ndarray:
    """
    Return the coordinate in units of the curve's bulk bond.

    Parameters
    ----------
    c
        Curve.

    Returns
    -------
    numpy.ndarray
        Coordinate b / b_bulk.
    """
    return c.b / c.b_bulk


def _first_folds(c: SCurve) -> tuple[int, int]:
    """
    Return the indices of the first K+ fold and the K- fold after it.

    Parameters
    ----------
    c
        Curve.

    Returns
    -------
    tuple[int, int]
        (K+, K-) indices.
    """
    f = c.folds()
    up = next((i for i in f if c.k[i] > c.k[i - 1]), None)
    if up is None:
        raise ValueError("reference curve has no K+ fold")
    down = next((i for i in f if i > up), None)
    if down is None:
        raise ValueError("reference curve has no K- fold after K+")
    return up, down


def window(ref: SCurve) -> tuple[float, float]:
    """
    Return the scored window: the reference's whole increasing range.

    Parameters
    ----------
    ref
        Reference curve.

    Returns
    -------
    tuple[float, float]
        (x_lo, x_hi) in units of the bulk bond.
    """
    ref = ref.without_backsteps()
    x, n = _x(ref), ref.prefix()
    if n < 2:
        raise ValueError("reference curve has fewer than two increasing points")
    return float(x[0]), float(x[n - 1])


def _elastic_c(c: SCurve, x_closed: float | None = None) -> float:
    """
    Fit c in E = E0 + c k^2 on the curve's own closed branch.

    Parameters
    ----------
    c
        Curve.
    x_closed
        End of the closed branch for a curve without a K+ fold.

    Returns
    -------
    float
        Elastic coefficient c (eV).
    """
    f = c.folds()
    up = next((i for i in f if c.k[i] > c.k[i - 1]), None)
    if up is None:
        n = c.prefix()
        up = (
            int(np.searchsorted(_x(c)[:n], x_closed, side="right")) - 1
            if x_closed is not None
            else n - 1
        )
        up = max(up, 2)
    kk, ee = c.k[: up + 1], c.E[: up + 1]
    a = np.vstack([np.ones_like(kk), kk**2]).T
    return float(np.linalg.lstsq(a, ee, rcond=None)[0][1])


def _elastic_c_on(
    c: SCurve, x: np.ndarray, n: int, x_lo: float, x_closed: float
) -> float:
    """
    Fit the elastic coefficient on the reference's closed-branch range.

    Parameters
    ----------
    c
        Curve.
    x
        Curve coordinate (bulk-bond units) on its prefix.
    n
        Prefix length.
    x_lo
        Start of the range.
    x_closed
        End of the range.

    Returns
    -------
    float
        Elastic coefficient c (eV); the own closed-branch fit if fewer than 3
        points fall in the range.
    """
    sel = (x >= x_lo) & (x <= x_closed)
    if sel.sum() < 3:
        return _elastic_c(c, x_closed)
    kk, ee = c.k[:n][sel], c.E[:n][sel]
    a = np.vstack([np.ones_like(kk), kk**2]).T
    return float(np.linalg.lstsq(a, ee, rcond=None)[0][1])


def scored(c: SCurve, ref: SCurve) -> SCurve:
    """
    Return the increasing stretch of c with the largest overlap with the window.

    Parameters
    ----------
    c
        Curve.
    ref
        Reference curve.

    Returns
    -------
    SCurve
        The scored stretch (the prefix if no stretch overlaps).
    """
    c = c.without_backsteps()
    x_lo, x_hi = window(ref)
    b_lo, b_hi = x_lo * c.b_bulk, x_hi * c.b_bulk
    runs = c.runs()
    overlap = [max(0.0, min(c.b[i1 - 1], b_hi) - max(c.b[i0], b_lo)) for i0, i1 in runs]
    best = int(np.argmax(overlap))
    return c.slice(*runs[best]) if overlap[best] > 0 else c.slice(0, c.prefix())


def _unit(c: SCurve) -> float:
    """
    Return the A' energy unit of a curve.

    Parameters
    ----------
    c
        Curve.

    Returns
    -------
    float
        2 gamma L_front b_bulk (eV), the curve's own constants.
    """
    return 2.0 * c.gamma * (J / m**2) * c.front * c.b_bulk


def _closed_range(ref: SCurve) -> tuple[float, float]:
    """
    Return the reference's closed branch: the window start to its K+ fold.

    Parameters
    ----------
    ref
        Reference curve.

    Returns
    -------
    tuple[float, float]
        (x_lo, x_closed); the whole window for a reference without folds.
    """
    x_lo, x_hi = window(ref)
    try:
        x_closed = float(_x(ref)[_first_folds(ref)[0]])
    except ValueError:
        x_closed = x_hi
    return x_lo, x_closed


def reference_c_tilde(ref: SCurve) -> float:
    """
    Fit the reference's dimensionless elastic coefficient c~ = c / (2 gamma L b_bulk).

    Parameters
    ----------
    ref
        Reference curve.

    Returns
    -------
    float
        Coefficient c~, fitted on the reference's closed branch.
    """
    r = scored(ref, ref)
    n = r.prefix()
    x_lo, x_closed = _closed_range(ref)
    return _elastic_c_on(r, _x(r)[:n], n, x_lo, x_closed) / _unit(r)


def coords(c: SCurve, ref: SCurve, c_tilde: float) -> tuple[np.ndarray, ...]:
    """
    Return coordinate, load and bond-breaking energy of a curve on the window.

    Parameters
    ----------
    c
        Curve.
    ref
        Reference curve.
    c_tilde
        Elastic coefficient c~ (from the reference).

    Returns
    -------
    tuple[numpy.ndarray, ...]
        Arrays (x, k, e): e = (E - c~ unit k^2) from the window start, in units of
        2 gamma L b_bulk.
    """
    c = scored(c, ref)
    n = c.prefix()
    x = _x(c)[:n]
    x_lo, _ = window(ref)
    unit = _unit(c)
    e = c.E[:n] - c_tilde * unit * c.k[:n] ** 2
    e = (e - np.interp(x_lo, x, e)) / unit
    return x, c.k[:n], e


def errors(model: SCurve, ref: SCurve) -> dict:
    """
    Compute K and energy errors of a model curve against the reference.

    Parameters
    ----------
    model
        Model curve.
    ref
        Reference curve.

    Returns
    -------
    dict
        ``E_K``, ``E_E`` (RMS over a uniform grid on the covered window) and
        ``coverage``.
    """
    x_lo, x_hi = window(ref)
    grid = np.linspace(x_lo, x_hi, N_GRID)
    ct = reference_c_tilde(ref)
    xr, yr, er = coords(ref, ref, ct)
    xm, ym, em = coords(model, ref, ct)
    cov = (grid >= xm[0]) & (grid <= xm[-1])
    if not cov.any():
        return {"E_K": float("nan"), "E_E": float("nan"), "coverage": 0.0}
    g = grid[cov]
    dk = np.interp(g, xm, ym) - np.interp(g, xr, yr)
    de = np.interp(g, xm, em) - np.interp(g, xr, er)
    return {
        "E_K": float(np.sqrt(np.mean(dk**2))),
        "E_E": float(np.sqrt(np.mean(de**2))),
        "coverage": float(cov.mean()),
    }


def si_bonds(
    reference, numbers, a0: float, front: float, shell: int = 1, tol: float = 0.15
):
    """
    Return heavy-atom pairs of the unrelaxed lattice at a neighbour distance.

    Parameters
    ----------
    reference
        Unrelaxed positions (N, 3).
    numbers
        Atomic numbers.
    a0
        Lattice constant (A).
    front
        Period along the front (A).
    shell
        1 for nearest neighbours, 2 for second neighbours (same fcc sublattice).
    tol
        Relative distance tolerance.

    Returns
    -------
    numpy.ndarray
        Index pairs (n, 2), i < j; with one front period each periodic image is a
        separate pair.
    """
    ref = np.asarray(reference)
    si = np.flatnonzero(np.asarray(numbers)[: len(ref)] != 1)
    b0 = np.sqrt(3.0) * a0 / 4.0 if shell == 1 else a0 / np.sqrt(2.0)
    p = ref[si]
    d = p[None, :, :] - p[:, None, :]
    d[..., 2] -= front * np.round(d[..., 2] / front)
    pairs = []
    for shift in (-1.0, 0.0, 1.0):
        e = d.copy()
        e[..., 2] += shift * front
        r = np.linalg.norm(e, axis=2)
        a, b = np.nonzero(np.triu(np.abs(r - b0) < tol * b0, k=1))
        pairs.append(np.stack([si[a], si[b]], axis=1))
    return np.concatenate(pairs)


def ed_pairs(reference, numbers, tip, a0: float, front: float) -> np.ndarray:
    """
    Return the E_D stencil: second-neighbour pairs near the tip, not in the wake.

    Parameters
    ----------
    reference
        Unrelaxed positions (N, 3).
    numbers
        Atomic numbers.
    tip
        Field origin (x, y).
    a0
        Lattice constant (A).
    front
        Period along the front (A).

    Returns
    -------
    numpy.ndarray
        Index pairs (n, 2) with midpoints within R_D of the tip and x >= x_tip - WAKE.
    """
    p = si_bonds(reference, numbers, a0, front, shell=2)
    ref = np.asarray(reference)
    mid = 0.5 * (ref[p[:, 0]] + ref[p[:, 1]])
    tip = np.asarray(tip)
    keep = np.linalg.norm(mid[:, :2] - tip, axis=1) < R_D
    keep &= mid[:, 0] >= tip[0] - WAKE
    return p[keep]


def bond_gradient(ubar, bonds, a0: float, front: float) -> np.ndarray:
    """
    Return the bond strain of a corrector field on a set of pairs.

    Parameters
    ----------
    ubar
        Corrector displacements (N, 3).
    bonds
        Index pairs.
    a0
        Lattice constant (A).
    front
        Period along the front (A).

    Returns
    -------
    numpy.ndarray
        Strain (u_j - u_i) / a0, minimum image along the front.
    """
    d = ubar[bonds[:, 1]] - ubar[bonds[:, 0]]
    d[:, 2] -= front * np.round(d[:, 2] / front)
    return d / a0


class Track:
    """
    Traced path for the energy norm: positions and loads on a cluster topology.

    Parameters
    ----------
    reference
        Unrelaxed positions of the cluster (N, 3).
    tip
        Field origin (x, y).
    field
        Callable K -> (N, 3) K-field displacements at absolute load K.
    a0
        Lattice constant (A).
    front
        Period along the front (A).
    positions
        Frame positions (n_frames, N, 3).
    k_abs
        Absolute loads per frame (eV/A^2.5).
    pair
        Coordinate bond.
    """

    def __init__(
        self,
        reference,
        tip,
        field: Callable,
        a0: float,
        front: float,
        positions,
        k_abs,
        pair,
    ) -> None:
        """
        Store the path and compute its coordinate.

        Parameters
        ----------
        reference
            Unrelaxed positions of the cluster (N, 3).
        tip
            Field origin (x, y).
        field
            Callable K -> (N, 3) K-field displacements at absolute load K.
        a0
            Lattice constant (A).
        front
            Period along the front (A).
        positions
            Frame positions (n_frames, N, 3).
        k_abs
            Absolute loads per frame (eV/A^2.5).
        pair
            Coordinate bond.
        """
        self.reference, self.tip, self.field = (
            np.asarray(reference),
            np.asarray(tip),
            field,
        )
        self.a0, self.front = float(a0), float(front)
        self.positions = np.asarray(positions, float)
        self.K = np.asarray(k_abs, float)
        self.pair = tuple(int(x) for x in pair)
        d = self.positions[:, self.pair[1]] - self.positions[:, self.pair[0]]
        d[:, 2] -= self.front * np.round(d[:, 2] / self.front)
        self.b = np.linalg.norm(d, axis=1)

    def _with(self, keep) -> Track:
        """
        Return the track at the given frame indices.

        Parameters
        ----------
        keep
            Frame indices.

        Returns
        -------
        Track
            The selected frames.
        """
        return Track(
            self.reference,
            self.tip,
            self.field,
            self.a0,
            self.front,
            self.positions[keep],
            self.K[keep],
            self.pair,
        )

    def without_backsteps(self) -> Track:
        """
        Return the track without back-steps below the tolerance.

        Returns
        -------
        Track
            The track with small dips removed.
        """
        return self._with(drop_backsteps(self.b))

    def runs(self) -> list[tuple[int, int]]:
        """
        Return the maximal stretches along which b increases strictly.

        Returns
        -------
        list[tuple[int, int]]
            Half-open index ranges.
        """
        out, i0 = [], 0
        for i, up in enumerate(np.diff(self.b) > 1e-6):
            if not up:
                out.append((i0, i + 1))
                i0 = i + 1
        out.append((i0, len(self.b)))
        return out

    def best_run(self, lo: float, hi: float):
        """
        Return the increasing run with the largest overlap with [lo, hi].

        Parameters
        ----------
        lo
            Lower end (A).
        hi
            Upper end (A).

        Returns
        -------
        tuple[int, int] | None
            The run, or None if none overlaps.
        """
        runs = self.runs()
        ov = [
            max(0.0, min(self.b[i1 - 1], hi) - max(self.b[i0], lo)) for i0, i1 in runs
        ]
        j = int(np.argmax(ov))
        return runs[j] if ov[j] > 0 else None

    def at_b(self, b: float, run) -> tuple[np.ndarray, float]:
        """
        Interpolate positions and load linearly in b on a run.

        Parameters
        ----------
        b
            Coordinate (A).
        run
            Run (i0, i1).

        Returns
        -------
        tuple[numpy.ndarray, float]
            Positions and K; ValueError if b is outside the run.
        """
        i0, i1 = run
        bs = self.b[i0:i1]
        if len(bs) < 2 or not bs[0] <= b <= bs[-1]:
            raise ValueError(f"b = {b} outside the run")
        i = i0 + min(int(np.searchsorted(bs, b, side="right")) - 1, len(bs) - 2)
        w = (b - self.b[i]) / (self.b[i + 1] - self.b[i])
        pos = (1 - w) * self.positions[i] + w * self.positions[i + 1]
        return pos, float((1 - w) * self.K[i] + w * self.K[i + 1])

    def strain(self, b: float, bonds, run) -> np.ndarray:
        """
        Return the corrector bond strain at coordinate b.

        Parameters
        ----------
        b
            Coordinate (A).
        bonds
            Index pairs.
        run
            Run (i0, i1).

        Returns
        -------
        numpy.ndarray
            Bond strain of u_bar = r - r_ref - u_CLE(K).
        """
        pos, k = self.at_b(b, run)
        return bond_gradient(
            pos - self.reference - self.field(k), bonds, self.a0, self.front
        )


def ed_on_grid(
    model: Track, ref: Track, numbers, window_x, n_grid: int = N_GRID
) -> dict:
    """
    Compute the energy-norm displacement error, matched at equal b / b_bulk.

    Parameters
    ----------
    model
        Model track.
    ref
        Reference track.
    numbers
        Atomic numbers of the cluster.
    window_x
        Window (x_lo, x_hi) in bulk-bond units.
    n_grid
        Grid points.

    Returns
    -------
    dict
        ``E_D`` (RMS over the evaluated grid points, None if none) and ``coverage``.
    """
    bonds = ed_pairs(ref.reference, numbers, ref.tip, ref.a0, ref.front)
    sm, sr = np.sqrt(3.0) * model.a0 / 4.0, np.sqrt(3.0) * ref.a0 / 4.0
    lo, hi = window_x
    grid = np.linspace(lo, hi, n_grid)
    model, ref = model.without_backsteps(), ref.without_backsteps()
    rm, rr = model.best_run(lo * sm, hi * sm), ref.best_run(lo * sr, hi * sr)
    vals = []
    for x in grid:
        if rm is None or rr is None:
            break
        try:
            dm, dr = model.strain(x * sm, bonds, rm), ref.strain(x * sr, bonds, rr)
        except ValueError:
            continue
        vals.append(float(np.sqrt(np.sum(np.linalg.norm(dm - dr, axis=1) ** 2))))
    if not vals:
        return {"E_D": None, "coverage": 0.0}
    return {
        "E_D": float(np.sqrt(np.mean(np.square(vals)))),
        "coverage": len(vals) / n_grid,
    }


def normalised(x: float, g: float, b: float) -> float:
    """
    Return the ML-PEG score of a raw value (lower is better).

    Parameters
    ----------
    x
        Raw value.
    g
        Good threshold.
    b
        Bad threshold.

    Returns
    -------
    float
        Score: 1 at or below g, 0 at or above b, linear between; 0 for NaN.
    """
    if x is None or not np.isfinite(x):
        return 0.0
    return float(np.clip((b - x) / (b - g), 0.0, 1.0))


def padded(x: float, c: float, g: float, b: float) -> float:
    """
    Return the raw value with the uncovered fraction counted at b.

    Parameters
    ----------
    x
        Error on the covered part.
    c
        Coverage (0-1).
    g
        Good threshold.
    b
        Bad threshold.

    Returns
    -------
    float
        Value c max(x, g) + (1 - c) b, so its score is c times the score of x.
    """
    if c <= 0 or x is None or not np.isfinite(x):
        return float(b)
    return float(c * max(x, g) + (1.0 - c) * b)
