"""Analysis module for copper water interface calculations."""

from __future__ import annotations

import pickle

from ase.io import read
import matplotlib.pyplot as plt
import mdtraj as mdt
import numpy as np
from scipy import signal


def _acfs_to_spectra(
    acfs: np.ndarray,
    nw: int,
    npad: int = 0,
    d: float = 1.0,
    f_w: callable = np.hanning,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate spectra from autocorrelation functions using FFT.

    Optionally processes multiple spectra at the same time. In that case,
    `acfs` is a 2D array of shape number of ACFs by length of ACFs. This is a
    low-level function, use `get_spectra` for a more convenient interface.

    Parameters
    ----------
    acfs
        Array, 1D or 2D, symmetric full ACFs.
    nw
        One-sided number of points of ACF window.
    npad
        One-sided number of additional padding zeros. Default is 0.
    d
        Real-space step. Default is 1.0.
    f_w
        Function to evaluate the double-sided symmetric window. Default is np.hanning.

    Returns
    -------
    tuple
        Frequency and intensity arrays.
    """
    # Make sure we're processing a 2D array.
    ndim = len(acfs.shape)
    if ndim == 1:
        acfs = acfs[np.newaxis, :]
    elif ndim == 2:
        pass
    else:
        raise ValueError("1D or 2D array required.")

    # number of spectra, total length of full ACF
    n, length = acfs.shape

    # window width
    ww = 2 * nw + 1

    assert ww <= length, "Window cannot be wider than data."

    # slice ACF data
    data = acfs[:, length // 2 - nw : length // 2 + nw + 1].copy()
    length_trim = data.shape[1]

    # multiply by the window
    data *= f_w(ww)

    # pad with optional zeros along time axis
    # one extra zero for symmetry - keep the ACF an even function
    data = np.pad(data, ((0, 0), (npad + 1, npad)), "constant", constant_values=0.0)
    assert data.shape == (n, length_trim + 2 * npad + 1)

    # window width including zero padding
    wwp = data.shape[1]

    # frequencies, with the provided real-space step
    frequency = np.fft.rfftfreq(wwp, d=d)

    # FFT ACF to spectrum
    # N.B.: For an ACF that is an even function, imaginary part is strictly zero.
    #       This is general, though.
    data_fft = np.fft.rfft(data)
    intensity = np.abs(data_fft)

    # Make result consistent with input in 1D case.
    if ndim == 1:
        intensity = intensity[0, :]

    # Normalize intensities to 1:
    intensity = intensity / intensity.sum()

    return frequency, intensity


def get_acfs(source: list[np.ndarray]) -> np.ndarray:
    """
    Calculate averaged autocorrelation function for a number of timeseries.

    The `source` yields individual timeseries and each of those is a
    timeseries of vector quantity. This means that if you
    want only a single autocorrelation function, you need to wrap the input
    array in something iterable, like a list.

    Parameters
    ----------
    source
        Iterator over timeseries of dimension N by 3.

    Returns
    -------
    ndarray
        ACFs, as many as timeseries yielded by `source`.
    """
    acfs = []
    for data in source:
        n = len(data[:, 0])
        norm = n - np.abs(np.arange(1 - n, n), dtype=float)
        cfs = [
            signal.correlate(data[:, d], data[:, d], mode="full", method="auto") / norm
            for d in range(3)
        ]
        acfs.append(np.array(cfs).sum(axis=0))
    return np.array(acfs).mean(axis=0)


def get_spectra(
    acfs: np.ndarray,
    dt: float,
    dt_window: float,
    dt_pad: float = 0.0,
    f_w: callable = np.hanning,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate spectra from autocorrelation functions using FFT.

    Processes multiple spectra at the same time, depending on the data in `acfs`.

    Parameters
    ----------
    acfs
        CFs numpy array.
    dt
        Time step, in femtoseconds.
    dt_window
        Total (double-sided) width of window, in femtoseconds.
    dt_pad
        Additional (double-sided) width of padding, in femtoseconds. Default is 0.0.
    f_w
        Function used to generate a (symmetric) window. Default is np.hanning.

    Returns
    -------
    tuple
        Frequency (nu) and intensity arrays.
    """
    c = 299792458.0  # m / s

    # check the input
    if dt_window <= 0.0:
        raise ValueError("`Dt` must be positive.")
    if dt_pad < 0.0:
        raise ValueError("`Dt_pad` must not be negative.")
    if dt * len(acfs) < dt_window:
        msg = (
            "The window ({:.0f} fs) must be narrower than the data "
            "({:.0f} fs). Alas, it is not."
        )
        raise ValueError(msg.format(dt_window, dt * len(acfs)))

    nw = int(dt_window / dt / 2.0)
    npad = int(dt_pad / dt / 2.0)

    frequency, intensity = _acfs_to_spectra(acfs, nw, npad=npad, d=1e-15 * dt, f_w=f_w)

    # convert frequency from Hz to cm^-1
    nu = frequency / (100.0 * c)

    return nu, intensity


def get_unique_atom_types(topology: mdt.Topology) -> list[str]:
    """
    Determine the unique atom types in a trajectory.

    Parameters
    ----------
    topology
        Topology object from mdtraj trajectory.

    Returns
    -------
    list
        List of unique atom type names.
    """
    return list({atom.name for atom in topology.atoms})


def get_unique_elements(topology: mdt.Topology) -> list[str]:
    """
    Determine the unique elements in a trajectory.

    Parameters
    ----------
    topology
        Topology object from mdtraj trajectory.

    Returns
    -------
    list
        List of unique element symbols.
    """
    return list({atom.element.symbol for atom in topology.atoms})


def compute_all_errors(
    ref_fnc: dict[str, tuple[np.ndarray, np.ndarray]],
    test_fnc: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, list]:
    """
    Compute MAEs relative to reference data.

    Given two sets of functions returns the mean absolute error
    of the test functions relative to the reference (first argument).

    Parameters
    ----------
    ref_fnc
        Reference functions dictionary.
    test_fnc
        Test functions dictionary.

    Returns
    -------
    dict
        Dictionary of errors with MAE values for each pair.
    """
    error = {}
    for name, data in ref_fnc.items():
        # Rescale array in case of different resolution
        test_data = np.interp(data[0], test_fnc[name][0], test_fnc[name][1])
        diff = data[1] - test_data
        mae = np.sum(np.absolute(diff)) / (np.sum(data[1]) + np.sum(test_data))
        error[name] = [data[0], diff, mae]
    return error


def print_errors(error: dict[str, list]) -> None:
    """
    Print the errors in a human readable way.

    Parameters
    ----------
    error
        Dictionary of errors with MAE values.
    """
    n = 22
    print(n * "=")
    print(f"{'Score Summary':{n}s}")
    print(n * "=")
    print("Label   | Accuracy [%]")
    print(n * "_")

    all_err = []
    for name, data in error.items():
        err = (1 - data[2]) * 100
        print(f"{name:7s} | {err:3.4}")
        all_err.append(err)

    # Mean of errors
    name = "Mean"
    err = np.mean(all_err)
    print(f"{name:7s} | {err:3.4}")

    print(n * "=")


def plot_all_f_and_errors(
    ref: dict[str, tuple[np.ndarray, np.ndarray]],
    test: dict[str, tuple[np.ndarray, np.ndarray]],
    error: dict[str, list],
    observable: str,
) -> None:
    """
    Loop over all functions and plot them, along with the error.

    Parameters
    ----------
    ref
        Reference functions dictionary.
    test
        Test functions dictionary.
    error
        Error dictionary.
    observable
        Observable name for plotting (e.g., "RDF", "VDOS").
    """
    for name, ref_data in ref.items():
        plot_f_and_errors(ref_data, test[name], error[name], name, observable)


def plot_f_and_errors(
    ref: tuple[np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray],
    error: list,
    title: str,
    observable: str,
) -> None:
    """
    Handle the plotting of the functions and their errors.

    Parameters
    ----------
    ref
        Reference function data.
    test
        Test function data.
    error
        Error data.
    title
        Title for the plot.
    observable
        Observable name for plotting (e.g., "RDF", "VDOS").
    """
    # Plot settings
    cm2in = 1 / 2.54
    fig = plt.figure(figsize=(8 * cm2in, 12 * cm2in), constrained_layout=True)
    gs = fig.add_gridspec(ncols=1, nrows=2, height_ratios=[2.0, 1.0])

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])

    # Plot reference and test property
    ax0.plot(ref[0], ref[1], color="black", label="Reference " + str(title), lw=2)
    ax0.plot(
        test[0],
        test[1],
        color="red",
        dashes=(0.5, 1.5),
        dash_capstyle="round",
        label="Test " + str(title),
        lw=2,
    )

    # Plot error
    ax1.plot(error[0], error[1], color="black", lw=2)

    # Formatting
    ax0.set_ylabel(observable)
    ax0.set_xticklabels([])
    ax1.set_ylabel("Absolute Error")
    if observable == "VDOS":
        ax0.set_ylim(ymin=0)
        ax0.set_xlim([0, 4500])
        ax1.set_xlim([0, 4500])
        ax1.set_xlabel(r"Frequency (cm$^{-1}$)")
    if observable == "RDF":
        ax1.set_xlabel(r"Distance ($\mathrm{\AA{}}$)")

    ax0.set_title("Species: " + str(title))

    plt.savefig(observable + "-" + str(title) + ".pdf")


def plot_mae_errors(error: dict[str, list], fn_out: str = "accuracy-all.pdf") -> None:
    """
    Plot the summed absolute errors for each element.

    Parameters
    ----------
    error
        Dictionary of errors with MAE values.
    fn_out
        Output filename for the plot. Default is "accuracy-all.pdf".
    """

    def autolabel(rects: list, form: str = r"{:1.1f}") -> None:
        """
        Attach a text label in each bar in rects, displaying its value.

        Parameters
        ----------
        rects
            Bar rectangles from matplotlib.
        form
            Format string for label. Default is r"{:1.1f}".
        """
        for rect in rects:
            width = rect.get_width()
            plt.annotate(
                form.format(width),
                xy=(0.0, rect.get_y() + rect.get_height() / 2),
                xytext=(3, 0),  # 3 points vertical offset
                textcoords="offset points",
                ha="left",
                va="center",
                color="w",
                fontsize=8,
                fontweight="bold",
            )

    # Plot settings
    cm2in = 1 / 2.54
    cm = plt.cm.viridis(np.linspace(0, 1.0, 8))[::-1]
    fig, ax = plt.subplots(
        ncols=1, nrows=1, constrained_layout=True, figsize=(8 * cm2in, 6 * cm2in)
    )

    # height of the bars
    height = 0.6
    # Label and their locations
    y = np.arange(len(error) + 1)
    labels = list(error.keys())

    # Convert errors to percent
    errors = np.array([(1 - error[key][2]) * 100 for key in labels][::-1])

    # Plot individual errors and mean
    rects = ax.barh(y[1:], errors, height, color=cm[4])
    autolabel(rects)
    rects = ax.barh(y[0], errors.mean(), height, color=cm[6])
    autolabel(rects)

    # Formatting
    ax.set_yticks(y)
    ax.set_yticklabels(np.append(labels, "All")[::-1])
    ax.set_xlim([0, 100])
    ax.set_xlabel("Accuracy (%)")
    ax.set_frame_on(False)
    ax.grid(axis="x")

    if fn_out is not None:
        plt.savefig(fn_out)


def run_vdos_test(
    ref_trj: mdt.Trajectory,
    ref_dt: float,
    test_trj: mdt.Trajectory,
    test_dt: float,
    fn_out: str = "vdos-res.pkl",
) -> None:
    """
    Perform the VDOS scoring and save results.

    Parameters
    ----------
    ref_trj
        Reference trajectory.
    ref_dt
        Reference time step.
    test_trj
        Test trajectory.
    test_dt
        Test time step.
    fn_out
        Output filename for results. Default is "vdos-res.pkl".
    """
    # Compute all RDFs
    ref_vdos = compute_all_vdos(ref_trj, ref_dt)
    test_vdos = compute_all_vdos(test_trj, test_dt)

    # Compute the errors
    vdos_errors = compute_all_errors(ref_vdos, test_vdos)

    # Plot the errors
    plot_all_f_and_errors(ref_vdos, test_vdos, vdos_errors, observable="VDOS")
    plot_mae_errors(vdos_errors, fn_out="vdos-all.pdf")

    # Print the errors
    print_errors(vdos_errors)

    # Save results
    results = {"ref_vdos": ref_vdos, "test_vdos": test_vdos, "vdos_errors": vdos_errors}
    with open(fn_out, "wb") as f_out:
        pickle.dump(results, f_out)


def run_vdos_single(
    ref_trj: mdt.Trajectory,
    ref_dt: float,
    fn_out: str = "vdos-res.pkl",
) -> None:
    """
    Perform the VDOS scoring and save results.

    Parameters
    ----------
    ref_trj
        Reference trajectory.
    ref_dt
        Reference time step.
    fn_out
        Output filename for results. Default is "vdos-res.pkl".
    """
    # Compute all RDFs
    ref_vdos = compute_all_vdos(ref_trj, ref_dt)

    for name, ref_data in ref_vdos.items():
        # Plot settings
        cm2in = 1 / 2.54
        fig = plt.figure(figsize=(8 * cm2in, 6 * cm2in), constrained_layout=True)
        gs = fig.add_gridspec(ncols=1, nrows=1)

        ax0 = fig.add_subplot(gs[0, 0])

        # Plot reference and test property
        ax0.plot(ref_data[0], ref_data[1], color="black", lw=2)

        # Formatting
        ax0.set_ylabel("vdos")
        ax0.set_xticklabels([])
        ax0.set_ylim(ymin=0)
        ax0.set_xlim([0, 4500])
        ax0.set_title("Species: " + str(name))

        plt.savefig("VDOS-single-" + str(name) + ".pdf")


def compute_all_vdos(
    trj: mdt.Trajectory,
    dt: float = 1,
    dt_window: float = 2000.0,
    dt_pad: float = 2000.0,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Compute VDOS separately for all atom types in a trajectory.

    Parameters
    ----------
    trj
        MDTraj trajectory object.
    dt
        Time step. Default is 1.
    dt_window
        Total window width. Default is 2000.0.
    dt_pad
        Padding width. Default is 2000.0.

    Returns
    -------
    dict
        Dictionary of VDOS data for each atom type.
    """
    top = trj.topology

    vdos_all = {}

    # Determine the unique atom types
    atom_types = get_unique_atom_types(top)

    for t1 in atom_types:
        # select indices of the atom type
        idx_t1 = top.select("name " + t1)

        # calculate velocity autocorrelation functions,
        # averaged over atoms of this species
        cfs = get_acfs(trj.xyz.transpose(1, 0, 2)[idx_t1])

        # calculate the spectrum for this species and store it
        nu, intensity = get_spectra(
            cfs,
            dt=dt,
            dt_window=dt_window,
            dt_pad=dt_pad,
        )
        vdos_all[t1] = nu, intensity

    return vdos_all


def parse_velocities_all(filename: str) -> list:
    """
    Read velocities from an extended XYZ with momenta using ASE.

    Reads an extended XYZ with momenta using ASE and returns
    per-frame velocities calculated as momenta / mass.

    Parameters
    ----------
    filename
        Path to the XYZ file.

    Returns
    -------
    list
        List of structures with velocities.
    """
    traj = read(filename, index=":")  # read all frames
    structures = []

    for atoms in traj:
        momenta = atoms.get_momenta()  # (N, 3)
        masses = atoms.get_masses()[:, None]  # (N, 1) for broadcasting
        velocities = momenta / masses  # v = p / m

        symbols = atoms.get_chemical_symbols()

        frame = [
            (sym, v[0], v[1], v[2]) for sym, v in zip(symbols, velocities, strict=False)
        ]
        structures.append(frame)

    return structures


def write_velocities_traj_all(structures: list, output_file: str) -> None:
    """
    Write velocities trajectory to file.

    Parameters
    ----------
    structures
        List of structures with velocities.
    output_file
        Path to output file.
    """
    with open(output_file, "w") as file:
        for structure in structures:
            file.write(f"{len(structure)}\n")
            file.write('Properties=species:S:1:vel:R:3 pbc="T T T"\n')
            for atom_name, vx, vy, vz in structure:
                file.write(f"{atom_name:2s} {vx:12.7f} {vy:12.7f} {vz:12.7f}\n")


def compute_all_rdfs(
    trj: mdt.Trajectory,
    n_bins: int = 150,
    **kwargs,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Compute RDFs between all pairs of atom types in a trajectory.

    Parameters
    ----------
    trj
        MDTraj trajectory object.
    n_bins
        Number of bins for RDF calculation. Default is 150.
    **kwargs
        Additional keyword arguments passed to mdt.compute_rdf.

    Returns
    -------
    dict
        Dictionary of RDF data for each atom type pair.
    """
    top = trj.topology

    rdfs_all = {}

    # Determine the unique atom types
    atom_types = get_unique_atom_types(trj.topology)

    for i1, t1 in enumerate(atom_types):
        # select indices of the first atom type
        idx_t1 = top.select("name " + t1)

        # unique atom type pairs only
        for i2 in range(i1, len(atom_types)):
            t2 = atom_types[i2]

            # select indices of the second atom type
            idx_t2 = top.select("name " + t2)

            # prepare all pairs of indices
            pairs = trj.topology.select_pairs(idx_t1, idx_t2)

            # single atom with itself -> no RDF
            if len(pairs) == 0:
                continue

            # OM: not sure this should be done here
            min_dimension = trj[0].unitcell_lengths.min() / 2

            r, g_r = mdt.compute_rdf(
                trj, pairs, (0, min_dimension), n_bins=n_bins, **kwargs
            )

            rdfs_all[t1 + "-" + t2] = r, g_r

    return rdfs_all


def run_rdf_test(
    ref_trj: mdt.Trajectory,
    test_trj: mdt.Trajectory,
    fn_out: str = "rdf-res.pkl",
) -> None:
    """
    Perform the RDF scoring and save results.

    Parameters
    ----------
    ref_trj
        Reference trajectory.
    test_trj
        Test trajectory.
    fn_out
        Output filename for results. Default is "rdf-res.pkl".
    """
    # Compute all RDFs
    ref_rdf = compute_all_rdfs(ref_trj)
    test_rdf = compute_all_rdfs(test_trj)

    # Compute the errors
    rdf_errors = compute_all_errors(ref_rdf, test_rdf)

    # Plot the errors
    plot_all_f_and_errors(ref_rdf, test_rdf, rdf_errors, observable="RDF")
    plot_mae_errors(rdf_errors, fn_out="rdf-all.pdf")

    # Print the errors
    print_errors(rdf_errors)

    # Save results
    results = {"ref_rdf": ref_rdf, "test_rdf": test_rdf, "rdf_errors": rdf_errors}
    with open(fn_out, "wb") as f_out:
        pickle.dump(results, f_out)


def load_with_cell(
    filename_or_filenames: str | list[str],
    start: int | None = None,
    stop: int | None = None,
    step: int | None = None,
    **kwargs,
) -> mdt.Trajectory:
    """
    Load trajectory and inject cell dimensions from topology PDB file if not present.

    All arguments and keyword arguments are passed on to `mdtraj.load`. The `top`
    keyword argument is used to load a PDB file and get cell information from it.

    Parameters
    ----------
    filename_or_filenames
        Filename or list of filenames to load.
    start
        Starting frame index. Default is None.
    stop
        Stopping frame index. Default is None.
    step
        Frame step. Default is None.
    **kwargs
        Additional keyword arguments passed to mdtraj.load.

    Returns
    -------
    mdtraj.Trajectory
        Loaded trajectory with cell information.
    """
    # load the "topology frame" to get cell dimensions
    top = kwargs.get("top")
    if top is not None and isinstance(top, str):
        # load first frame from file as topology
        frame_top = mdt.load_frame(top, 0)
        unitcell_lengths = frame_top.unitcell_lengths
        unitcell_angles = frame_top.unitcell_angles
        if (unitcell_lengths is None) or (unitcell_angles is None):
            raise ValueError("Frame providing topology is missing cell information.")
    else:
        raise ValueError("Provide a PDB with cell dimensions.")

    # load the trajectory itself
    trj = mdt.load(filename_or_filenames, **kwargs)
    trj = trj[start:stop:step]

    # inject the cell information
    len_trj = len(trj)
    trj.unitcell_lengths = unitcell_lengths.repeat(len_trj, axis=0)
    trj.unitcell_angles = unitcell_angles.repeat(len_trj, axis=0)

    return trj


def error_score_percentage(mae: float) -> float:
    """
    Calculate error score as a percentage from mean absolute error.

    Parameters
    ----------
    mae
        Mean absolute error value.

    Returns
    -------
    float
        Error score as a percentage (0-100).
    """
    return (1 - mae) * 100


############################
### NEW FUNCTIONS FOR VACFS
############################


def compute_all_errors_vacf(ref_fnc, test_fnc):
    """
    Compute MAEs relative to reference data.

    Given two sets of functions returns the mean absolute error
    of the test functions relative to the reference (first argument).

    Parameters
    ----------
    ref_fnc : dict
        Reference function data.
    test_fnc : dict
        Test function data.

    Returns
    -------
    dict
        Dictionary containing error data for each function.
    """
    error = {}
    for name, data in ref_fnc.items():
        # Rescale array in case of different resolution
        test_data = np.interp(data[0], test_fnc[name][0], test_fnc[name][1])
        diff = data[1] - test_data
        mae = np.sum(np.abs(diff)) / (
            np.sum(np.abs(data[1])) + np.sum(np.abs(test_data))
        )
        error[name] = [data[0], diff, mae]
    return error


def run_vacfs_test(ref_trj, ref_dt, test_trj, test_dt, fn_out="vcaf-res.pkl"):
    """
    Perform the VCAF scoring and save results.

    Parameters
    ----------
    ref_trj : mdtraj.Trajectory
        Reference trajectory.
    ref_dt : float
        Reference trajectory time step.
    test_trj : mdtraj.Trajectory
        Test trajectory.
    test_dt : float
        Test trajectory time step.
    fn_out : str, default "vcaf-res.pkl"
        Output filename for results.
    """
    # Compute all RDFs
    ref_vacf = compute_all_vacfs(ref_trj, ref_dt)
    test_vacf = compute_all_vacfs(test_trj, test_dt)

    # Compute the errors
    vacf_errors = compute_all_errors_vacf(ref_vacf, test_vacf)

    # Plot the errors
    plot_all_f_and_errors(ref_vacf, test_vacf, vacf_errors, observable="VACF")
    plot_mae_errors(vacf_errors, fn_out="vacfs-all.pdf")

    # Print the errors
    print_errors(vacf_errors)

    # Save results
    results = {"ref_vacf": ref_vacf, "test_vacf": test_vacf, "vacf_errors": vacf_errors}
    with open(fn_out, "wb") as f_out:
        pickle.dump(results, f_out)


def compute_all_vacfs(trj, dt=1):
    """
    Compute VDOS separately for all atom types in a trajectory.

    Parameters
    ----------
    trj : mdtraj.Trajectory
        Trajectory data.
    dt : float, default 1
        Time step.

    Returns
    -------
    dict
        Dictionary containing VACF data for all atom types.
    """
    top = trj.topology

    vacfs_all = {}

    # Determine the unique atom types
    atom_types = get_unique_atom_types(top)

    for t1 in atom_types:
        # select indices of the atom type
        idx_t1 = top.select("name " + t1)

        # velocities assumed in Å/fs
        # shape: (n_frames, n_atoms, 3)
        vel = trj.xyz.transpose(1, 0, 2)[idx_t1]

        # compute velocity autocorrelation
        # expected shape: (n_lags,)
        vacf = get_acfs(vel)

        # time axis
        t = np.arange(len(vacf)) * dt  # fs

        # center index
        i0 = len(vacf) // 2

        # keep only t >= 0
        vacf = vacf[i0:]
        t = t[: len(vacf)]

        # normalize to sort out any units missmatch
        vacf /= vacf[0]

        vacfs_all[t1] = (t, vacf)

    return vacfs_all
