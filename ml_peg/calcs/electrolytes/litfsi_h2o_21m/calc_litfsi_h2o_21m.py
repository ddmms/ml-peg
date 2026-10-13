"""
Run calculations for the WiSE 21 m LiTFSI/H2O electrolyte benchmark.

The benchmark has two halves, which can be run together or independently.

**Reference MD** (``test_reference_md``, marked ``multi_day``) runs the
protocol that generated the reference data, and writes its products into
``outputs/<model>/``:

* ``p64_w170`` (1534 atoms, 27.4938 A cubic): NVT 50 ps equilibration ->
  NVT 50 ps production, held at the experimental volume throughout. No NPT:
  S(q) and the coordination numbers are compared at the reference density
  rather than at each model's own. Produces ``nvt_trajectory.extxyz``.
* ``p16_w42`` (382 atoms): NVT 50 ps equilibration -> NPT 200 ps production.
  NPT is used only on this smaller cell, where the density converges far more
  cheaply. Produces ``density.json``, averaged over the last 150 ps, and
  ``npt_final.xyz``, the last frame of the run, which the app shows as the
  model's structure.

Both cells were generated and equilibrated once, at the experimental density,
with the procedure described in Brugnoli et al., arXiv:2603.22099 (SI,
Section 1); they are downloaded from the ml-peg data bucket, so every model
starts from the same configuration.

Models not trained on dispersion-corrected data are run with a D3(BJ)
correction added at runtime (``Model.add_d3_calculator``), as in the other
ml-peg MD benchmarks; the analysis labels them ``-D3``.

MD is driven by janus-core at dt = 0.5 fs, T = 298.15 K and P = 1 atm: NVT
with the Nose-Hoover chain (``NVT_NH``, thermostat time 50 fs) and NPT with
the Martyna-Tobias-Klein chain (``NPT_MTK``, barostat time 500 fs; plain
``NPT`` in janus-core is Melchionna and is not used).

**Extraction** (the remaining tests, fast) reads those files back and computes
three observables for each registered MLIP model:

* Density from the NPT run (p16_w42).
* Li-O coordination numbers from the NVT trajectory via radial distribution
  functions g(r) for Li-O_water (O bonded to H, d_OH < 1.25 A) and
  Li-O_TFSI (O bonded to S, d_OS < 1.75 A); CN integrated to first minimum
  R_CUT = 2.83 A from the r2SCAN AIMD reference.
* X-ray structure factor S(q) computed via dynasor in TRAVIS Faber-Ziman
  convention with Cromer-Mann 4-Gaussian form factors (including H) and
  Savitzky-Golay smoothing (window=27, order=3, dq=0.02 A^-1; physical
  width 0.54 A^-1, i.e. 2.4 x the 2*pi/L spacing of the q-grid).

Reference values:

* Density: 1.7126 g/cm3 (experimental; Gilbert et al., J. Chem. Eng. Data
  62, 2056, 2017).
* Li-O CN: 2.4 (water), 1.9 (TFSI), 4.3 (total). Theoretical references at
  21 m: r2SCAN AIMD (Li128, 20 ps) gives 2.23/2.02/4.31; MACE potentials
  trained on the same r2SCAN dataset converge to 2.35-2.39/1.89-1.91/
  4.27-4.29 over >2 ns. The total is consistent with the experimental 4.21
  at ~18.5 m (Watanabe et al., JPCB 125, 7477, 2021, neutron diffraction
  with isotopic substitution); the experimental water/TFSI split at that
  concentration is not used, as it disagrees systematically with all MD
  partitionings and is measured at a different composition.
* S(q): SAXS (Zhang et al., J. Phys. Chem. B 125, 4501, 2021).

The MD stage handling and the observables are implemented in
``litfsi_utils.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import warnings

from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write
import numpy as np
import pytest

from ml_peg.calcs.electrolytes.litfsi_h2o_21m.litfsi_utils import (
    compute_rdf,
    compute_sq_travis_style,
    identify_o_types,
    run_stage,
    write_density_json,
)
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

warnings.filterwarnings("ignore")

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"

# --- Density references ------------------------------------------------------

RHO_EXP = 1.7126  # g/cm3 (Gilbert et al., JCED 2017)

# --- Reference MD protocol ---------------------------------------------------

TEMPERATURE_K = 298.15
PRESSURE_BAR = 1.01325  # 1 atm
PRESSURE_GPA = PRESSURE_BAR * 1e-4  # janus-core takes pressure in GPa
TIMESTEP_FS = 0.5

# Nose-Hoover damping times: 100*dt and 1000*dt
THERMOSTAT_TIME_FS = 50.0
BAROSTAT_TIME_FS = 500.0
NH_CHAIN = 3  # length of both the thermostat and the barostat sub-chains

# Step counts at 0.5 fs/step
NVT_EQUIL_STEPS = 100_000  # 50 ps, both cells
NVT_PROD_STEPS = 100_000  # 50 ps, p64_w170 only (S(q), RDF)
NPT_PROD_STEPS = 400_000  # 200 ps, p16_w42 only (density)

# IO cadence
STATS_EVERY = 100  # 0.05 ps
TRAJ_EVERY = 200  # 0.1 ps -> 501 frames over the 50 ps production run

SEED = 42

# Density is averaged over the last 150 ps of the 200 ps NPT run.
DENSITY_WINDOW_PS = (50.0, 200.0)

CELL_P64 = "p64_w170"
CELL_P16 = "p16_w42"


# =============================================================================
# Reference MD (slow: generates the data read by the tests below)
# =============================================================================


def load_initial_structure(cell: str) -> Atoms:
    """
    Load the equilibrated starting configuration of a cell.

    Both cells were generated and equilibrated once, at the experimental
    density and identically for every model (Brugnoli et al.,
    arXiv:2603.22099, SI Section 1). They are downloaded from the ml-peg data
    bucket and cached.

    Parameters
    ----------
    cell : str
        Cell name, ``"p64_w170"`` (1534 atoms) or ``"p16_w42"`` (382 atoms).

    Returns
    -------
    Atoms
        ASE Atoms at rho = 1.7126 g/cm3.

    Raises
    ------
    FileNotFoundError
        If the structure is missing from the downloaded data.
    """
    input_dir = (
        download_s3_data(
            filename="litfsi_h2o_21m.zip",
            key="inputs/electrolytes/litfsi_h2o_21m/litfsi_h2o_21m.zip",
        )
        / "litfsi_h2o_21m"
    )
    path = input_dir / f"{cell}_equilibrated.xyz"
    if not path.exists():
        raise FileNotFoundError(
            f"Starting structure for {cell} not found at {path}. "
            "Both cells (p64_w170, p16_w42) are needed to run the reference MD."
        )
    struct = ase_read(path)

    # Both cells are neutral, closed-shell (equal numbers of Li+ and TFSI-).
    # Models trained on molecular data, such as orb-v3-consv-omol, refuse to
    # evaluate a structure that does not carry these keys.
    struct.info.setdefault("charge", 0)
    struct.info.setdefault("spin", 1)
    return struct


def run_reference_md(model_name: str, model: Any) -> None:
    """
    Run the reference protocol for one model and write its data products.

    Produces ``nvt_trajectory.extxyz`` (p64_w170), ``density.json`` and
    ``npt_final.xyz`` (p16_w42) under ``OUT_PATH/<model_name>/``, which the
    extraction tests and the analysis then read back.

    Parameters
    ----------
    model_name : str
        Registry name of the MLIP model.
    model : Any
        Model object from :func:`load_models`.
    """
    # Imported lazily: janus-core pulls in torch, which is not needed by the
    # extraction tests.
    from janus_core.calculations.md import NPT_MTK, NVT_NH

    calc = model.get_calculator(precision="low")
    # Runtime D3(BJ) for the models not trained on dispersion-corrected data,
    # as in the other ml-peg MD benchmarks; the others are returned unchanged.
    calc = model.add_d3_calculator(calc)

    data_dir = OUT_PATH / model_name
    work_dir = data_dir / "md"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Settings shared by every stage.
    md_kwargs = {
        "temp": TEMPERATURE_K,
        "timestep": TIMESTEP_FS,
        "thermostat_time": THERMOSTAT_TIME_FS,
        "stats_every": STATS_EVERY,
        "traj_every": TRAJ_EVERY,
        "seed": SEED,
    }

    for cell in (CELL_P64, CELL_P16):
        struct = load_initial_structure(cell)
        # Assigned, not copied: copying a MatterSim calculator sends it through
        # __setstate__, which rebuilds the potential from a checkpoint
        # (defaulting to the 1M weights) instead of reusing the loaded one.
        struct.calc = calc

        # The equilibration trajectory is written but not analysed.
        # (traj_every=0 is not a way to disable it: ASE reads a non-positive
        # interval as "call once at step 0", which would divide by zero in
        # janus-core's trajectory writer.)
        struct = run_stage(
            NVT_NH,
            struct,
            steps=NVT_EQUIL_STEPS,
            file_prefix=work_dir / f"{cell}-nvt_equil",
            **md_kwargs,
        )

        if cell == CELL_P64:
            # Production at the experimental volume: this is what S(q) and the
            # RDF are computed from.
            prefix = work_dir / f"{cell}-nvt_prod"
            run_stage(
                NVT_NH,
                struct,
                steps=NVT_PROD_STEPS,
                file_prefix=prefix,
                **md_kwargs,
            )
            traj = ase_read(f"{prefix}-traj.extxyz", index=":")
            ase_write(data_dir / "nvt_trajectory.extxyz", traj, format="extxyz")
        else:
            # NPT only on the small cell, for the density.
            prefix = work_dir / f"{cell}-npt_prod"
            final = run_stage(
                NPT_MTK,
                struct,
                steps=NPT_PROD_STEPS,
                file_prefix=prefix,
                pressure=PRESSURE_GPA,
                barostat_time=BAROSTAT_TIME_FS,
                thermostat_chain=NH_CHAIN,
                barostat_chain=NH_CHAIN,
                **md_kwargs,
            )
            # Last frame at the model's own density, for the structure viewer.
            ase_write(data_dir / "npt_final.xyz", final, format="extxyz")
            write_density_json(
                model_name,
                Path(f"{prefix}-stats.dat"),
                data_dir / "density.json",
                cell=CELL_P16,
                rho_exp=RHO_EXP,
                window_ps=DENSITY_WINDOW_PS,
            )


@pytest.mark.multi_day
@pytest.mark.parametrize("mlip", MODELS.items())
def test_reference_md(mlip: tuple[str, Any]) -> None:
    """
    Regenerate the benchmark data for one model with the reference protocol.

    350 ps of MD in total on 1534 and 382 atoms. Measured on one A100 (80 GB)
    in float32: 5-190 GPU-hours per model without dispersion (median 31 h;
    mace-mp-0a 24 h, pet-oam-xl 187 h). The runtime D3 term costs about as
    much as a medium-sized MLIP, so it roughly triples the time of the models
    that need it (mace-mp-0a: 75 h).

    Parameters
    ----------
    mlip : tuple[str, Any]
        ``(model_name, model)`` pair from the ml-peg registry.
    """
    model_name, model = mlip
    run_reference_md(model_name, model)

    assert (OUT_PATH / model_name / "nvt_trajectory.extxyz").exists()
    assert (OUT_PATH / model_name / "density.json").exists()
    assert (OUT_PATH / model_name / "npt_final.xyz").exists()


# =============================================================================
# Observables (fast: read the reference MD products back)
# =============================================================================


@pytest.mark.fast
@pytest.mark.parametrize("model_name", MODELS)
def test_density(model_name: str) -> None:
    """
    Check the NPT density written by the reference MD for one model.

    Parameters
    ----------
    model_name : str
        Name of the MLIP model in the registry.
    """
    density_path = OUT_PATH / model_name / "density.json"
    if not density_path.exists():
        pytest.skip(
            f"No density data for {model_name} at {density_path}. Generate it "
            "with test_reference_md (-m very_slow)."
        )

    with open(density_path) as f:
        result = json.load(f)

    # No bound on the error: a model whose liquid decomposes in NPT yields a
    # valid (badly scored) result, not a failed calculation.
    assert result["n_samples"] > 0, f"No density samples for {model_name}"
    assert np.isfinite(result["rho_error_pct"]), f"Invalid density for {model_name}"
    assert result["rho_mean"] > 0, f"Negative density for {model_name}"


@pytest.mark.fast
@pytest.mark.parametrize("model_name", MODELS)
def test_compute_rdf(model_name: str) -> None:
    """
    Compute and save Li-O RDFs and coordination numbers for one model.

    Parameters
    ----------
    model_name : str
        Name of the MLIP model in the registry.
    """
    traj_path = OUT_PATH / model_name / "nvt_trajectory.extxyz"
    if not traj_path.exists():
        pytest.skip(
            f"No NVT trajectory for {model_name} at {traj_path}. Generate it "
            "with test_reference_md (-m very_slow)."
        )

    first_frame = ase_read(str(traj_path), index=0, format="extxyz")
    o_water_idx, o_tfsi_idx = identify_o_types(first_frame)

    assert len(o_water_idx) > 0, f"No O_water found for {model_name}"
    assert len(o_tfsi_idx) > 0, f"No O_TFSI found for {model_name}"

    result = compute_rdf(traj_path, o_water_idx, o_tfsi_idx)

    out_dir = OUT_PATH / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "rdf.json", "w") as f:
        json.dump(
            {k: v for k, v in result.items() if not isinstance(v, list)}, f, indent=2
        )
    np.savez(
        out_dir / "rdf.npz",
        r=np.array(result["r"]),
        gr_LiO_total=np.array(result["gr_LiO_total"]),
        gr_LiO_water=np.array(result["gr_LiO_water"]),
        gr_LiO_TFSI=np.array(result["gr_LiO_TFSI"]),
    )

    assert 2.0 < result["coord_LiO_total"] < 8.0, (
        f"Unexpected Li-O_total CN={result['coord_LiO_total']:.2f} for {model_name}"
    )


@pytest.mark.fast
@pytest.mark.parametrize("model_name", MODELS)
def test_compute_xray_sq(model_name: str) -> None:
    """
    Compute and save X-ray S(q) in Faber-Ziman convention for one model.

    Parameters
    ----------
    model_name : str
        Name of the MLIP model in the registry.
    """
    traj_path = OUT_PATH / model_name / "nvt_trajectory.extxyz"
    if not traj_path.exists():
        pytest.skip(
            f"No NVT trajectory for {model_name} at {traj_path}. Generate it "
            "with test_reference_md (-m very_slow)."
        )

    result = compute_sq_travis_style(traj_path)
    result["model"] = model_name
    result["traj_path"] = str(traj_path)

    out_dir = OUT_PATH / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "xray_sq.json", "w") as f:
        json.dump(result, f, indent=2)
    np.savez(
        out_dir / "xray_sq.npz",
        q=np.array(result["q"]),
        Sq=np.array(result["Sq"]),
    )

    assert len(result["q"]) > 10, f"Too few q-points for {model_name}"
