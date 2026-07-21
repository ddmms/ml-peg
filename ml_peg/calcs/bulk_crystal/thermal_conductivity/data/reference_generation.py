"""Reference data generation for thermal conductivity calculations using phono3py."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib
from io import BytesIO
import os
from pathlib import Path
import re
import sys
import time

from ase import Atoms
import h5py
from phono3py.cui.load import load
import requests
from tqdm import tqdm

DATA_PATH = Path(__file__).parent

sys.path.append(str(DATA_PATH.parent))
tc = importlib.import_module("thermal_conductivity")

PH3_PARAMS_PATH = DATA_PATH / "PBE"

MAT_ID_FILE = DATA_PATH / "mat_id-phonondb.txt"
STRUCTURE_FILE = DATA_PATH / "phononDB-PBE-structures.extxyz"
SCRAPED_YAMLS_FILE = PH3_PARAMS_PATH / "scraped_yamls.txt"

SCRAPE = False

FAST_ONLY = True

SKIP_EXISTING = True

TEMPERATURES = [300]


def get_original_qmesh(mat_id_txt: str, name: str, spg_no: int) -> list[int]:
    """
    Determine the original q-point mesh for a given material.

    Uses its name and space group number.

    Parameters
    ----------
    mat_id_txt : str
        The content of the mat_id-phonondb.txt file as a string.
    name : str
        The name of the material (e.g., "Si", "Ge", etc.).
    spg_no : int
        The space group number of the material.

    Returns
    -------
    list[int]
        The original q-point mesh corresponding to the given space group number.
        For example:
        -For space group numbers 216 and 225, the original qpoint mesh is [19, 19, 19].
        -For space group number 186, the original qpoint mesh is [19, 19, 15].
        -If the space group number is not recognized, a ValueError is raised.
    """
    if spg_no == 216 or spg_no == 225:
        return [19, 19, 19]
    if spg_no == 186:
        return [19, 19, 15]

    raise ValueError(f"Unsupported space group number: {spg_no} for {name}")


def get_fast_qmesh(mat_id_txt: str, name: str, spg_no: int) -> list[int]:
    """
    Determine the "fast" q-point mesh for a given material.

    Uses its name and space group number. The "fast" q-point mesh is a reduced mesh
    used for quicker calculations, and it is determined based on the space group number
    of the material. This function checks the space group number and returns the
    corresponding fast q-point mesh.

    Parameters
    ----------
    mat_id_txt : str
        The content of the mat_id-phonondb.txt file as a string.
    name : str
        The name of the material (e.g., "Si", "Ge", etc.).
    spg_no : int
        The space group number of the material.

    Returns
    -------
    list[int]
        The fast q-point mesh corresponding to the given space group number.
        For example:
        - For space group numbers 216 and 225, the fast q-point mesh is [9, 9, 9].
        - For space group number 186, the fast q-point mesh is [9, 9, 7].
        - If the space group number is not recognized, a ValueError is raised.
    """
    if spg_no == 216 or spg_no == 225:
        return [9, 9, 9]
    if spg_no == 186:
        return [9, 9, 7]

    raise ValueError(f"Unsupported space group number: {spg_no} for {name}")


def get_mat_id(mat_id_txt: str, name: str, spg_no: int) -> str:
    """
    Extract the material ID from the mat_id_txt.

    Uses the material name and space group number.

    Parameters
    ----------
    mat_id_txt : str
        The content of the mat_id-phonondb.txt file as a string.
    name : str
        The name of the material (e.g., "Si", "Ge", etc.).
    spg_no : int
        The space group number of the material.

    Returns
    -------
    str
        The material ID corresponding to the given name and space group number, or "N/A"
        if not found.
    """
    for line in mat_id_txt:
        if " ".join([name, str(spg_no)]) in line:
            return line.split()[-1]  # Assuming the MP ID is the first element
    return "N/A"


def create_yaml(mat_id: str, yaml_str: str) -> Path:
    """
    Save the phono3py parameters as a YAML file for a given material ID.

    Parameters
    ----------
    mat_id : str
        Material ID to use for naming the YAML file and directory.
    yaml_str : str
        The YAML content as a string to be saved to the file.

    Returns
    -------
    Path
        The path to the saved YAML file.
    """
    os.makedirs(PH3_PARAMS_PATH / mat_id, exist_ok=True)
    with open(PH3_PARAMS_PATH / mat_id / "phono3py_params.yaml", "w") as f:
        f.write(yaml_str)
    return PH3_PARAMS_PATH / mat_id / "phono3py_params.yaml"


def scrape_phono3py_data(
    phono3py_yaml_path: str = "data/DFT",
) -> tuple[list[str], list[str]]:
    """
    Scrape phono3py parameters from the phonondb README and save them as YAML files.

    The README contains links to compressed YAML files with phonon properties
    and thermal conductivity parameters for a set of materials. This function
    extracts those links, downloads the YAML files, decompresses them, and saves
    them locally. It also extracts material names and space group numbers for
    later use in matching with material IDs.

    Parameters
    ----------
    phono3py_yaml_path : str, optional
        Local directory to save the scraped YAML files, by default "data/DFT".

    Returns
    -------
    tuple[list[str], list[str]]
        A tuple containing a list of YAML strings and a corresponding list of material
        names.
    """
    raw_url = "https://raw.githubusercontent.com/atztogo/phonondb/refs/heads/main/mdr/phono3py_103compounds_fd_PBE/README.md"
    params_link_regex = r"\[\s*phono3py_params\.yaml\.xz\s*\]\((https?://[^)\s]+)\)"
    name_regex = r"(?m)^\|\s*[A-Za-z0-9]+-([A-Z][a-z]?(?:[A-Z][a-z]?|\d+)*)\s*\|"

    phonondb_readme_path = PH3_PARAMS_PATH / "phonon3db_readme.md"
    if os.path.exists(phonondb_readme_path):
        print(
            "Found existing phonon3db README at "
            f"{phonondb_readme_path}. Loading from file instead of downloading."
        )
        with open(phonondb_readme_path) as f:
            phonondb_readme = f.read()
    else:
        r = requests.get(raw_url, timeout=30)
        r.raise_for_status()
        phonondb_readme = r.text
        with open(phonondb_readme_path, "w") as f:
            f.write(phonondb_readme)

    params_links = re.findall(params_link_regex, phonondb_readme)
    name_list = re.findall(name_regex, phonondb_readme)

    if os.path.exists(SCRAPED_YAMLS_FILE):
        print(
            "Found existing scraped YAMLs at "
            f"{SCRAPED_YAMLS_FILE}. Loading from file instead of downloading."
        )
        with open(SCRAPED_YAMLS_FILE) as f:
            yaml_list = [y for y in f.read().split("END\n") if y.strip()]

        return yaml_list, name_list

    try:
        import lzma
    except ImportError:
        raise ImportError("lzma module is required to decompress .xz files.") from None

    def _download_one(link: str) -> str:
        """
        Download and decompress one phono3py params YAML, retrying on failure.

        Parameters
        ----------
        link : str
            URL of the ``.yaml.xz`` file.

        Returns
        -------
        str
            Decompressed YAML content.
        """
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                xz_bytes = requests.get(link, timeout=30).content
                return lzma.decompress(xz_bytes).decode("utf-8")
            except Exception as exc:
                last_exc = exc
                time.sleep(2**attempt)
        raise RuntimeError(f"Failed to download {link} after 3 attempts") from last_exc

    # Downloads are I/O-bound, so thread them. map preserves order so yaml_list
    # stays aligned with params_links.
    with ThreadPoolExecutor(max_workers=8) as executor:
        yaml_list = list(
            tqdm(
                executor.map(_download_one, params_links),
                total=len(params_links),
                desc="Download",
            )
        )

    return yaml_list, name_list


yaml_list, name_list = scrape_phono3py_data()


with open(SCRAPED_YAMLS_FILE, "w") as f:
    for yaml_str in yaml_list:
        f.write(yaml_str + "\nEND\n")


with open(MAT_ID_FILE) as f:
    mat_id_txt = f.readlines()


with open(STRUCTURE_FILE, "w") as f:
    f.write("")  # Clear the file before appending new structures

N = len(name_list)

if N != len(yaml_list):
    raise ValueError(
        "Number of names "
        f"({len(name_list)}) does not match number of YAMLs ({len(yaml_list)})!"
    )
pbar = tqdm(range(N), desc="Processing phono3py")
for i in pbar:
    yaml_str = yaml_list[i]
    name = name_list[i]
    pbar.set_postfix_str(name)

    ph3 = load(BytesIO(yaml_str.encode("utf-8")), produce_fc=True, symmetrize_fc=True)

    symm_no = ph3.symmetry.dataset.number
    mat_id = get_mat_id(mat_id_txt, name, symm_no)

    original_qmesh = get_original_qmesh(mat_id_txt, name, symm_no)
    fast_qmesh = get_fast_qmesh(mat_id_txt, name, symm_no)

    info_dict = {
        "name": name,
        "primitive_matrix": ph3.primitive_matrix,
        "fc2_supercell": ph3.supercell_matrix,
        "fc3_supercell": ph3.supercell_matrix,
        "symm.no": ph3.symmetry.dataset.number,
        tc.TCKeys.mat_id: mat_id,
        "q_point_mesh": original_qmesh,
        "fast_q_point_mesh": fast_qmesh,
    }

    yaml_file = create_yaml(mat_id, yaml_str)
    if SKIP_EXISTING:
        if FAST_ONLY and os.path.exists(yaml_file.parent / "fast_kappa.hdf5"):
            print(
                "Skipping {mat_id}: fast_kappa.hdf5 already exists at "
                f"{yaml_file.parent / 'fast_kappa.hdf5'}"
            )
            continue
        elif os.path.exists(yaml_file.parent / "kappa.hdf5") and os.path.exists(
            yaml_file.parent / "fast_kappa.hdf5"
        ):
            print(
                "Skipping {mat_id}: kappa.hdf5 and fast_kappa.hdf5 already exist at "
                f"{yaml_file.parent / 'kappa.hdf5'}"
            )
            continue
    pbar.set_postfix_str(f"{name}-{mat_id}")

    atoms = Atoms(
        positions=ph3.unitcell.positions,
        cell=ph3.unitcell.cell,
        symbols=ph3.unitcell.symbols,
        pbc=True,
    )
    atoms.info.update(info_dict)

    atoms.write(STRUCTURE_FILE, format="extxyz", append=True)

    ph3.mesh_numbers = fast_qmesh

    with tc.tqdm_gridpoints(desc="Conducitivity calc"):
        ph3, fast_kappa_dict, _fast_cond = tc.calculate_conductivity(
            ph3, temperatures=TEMPERATURES, log_level=2
        )

    fast_kappa_dict = {
        tc.TCKeys.kappa_tot_avg: tc.calculate_kappa_avg(
            fast_kappa_dict[tc.TCKeys.kappa_tot_rta]
        ),
        tc.TCKeys.mode_kappa_tot_avg: tc.calculate_kappa_avg(
            fast_kappa_dict[tc.TCKeys.mode_kappa_tot_rta]
        ),
        tc.TCKeys.q_points: fast_kappa_dict[tc.TCKeys.q_points],
        tc.TCKeys.temperatures: TEMPERATURES,
        tc.TCKeys.ph_freqs: _fast_cond.frequencies,
        tc.TCKeys.heat_capacity: _fast_cond.mode_heat_capacities,
        tc.TCKeys.spg_num: ph3.symmetry.dataset.number,
        tc.TCKeys.name: name,
        "q_mesh": fast_qmesh,
        "q_point_mesh": fast_qmesh,
        "weights": fast_kappa_dict[tc.TCKeys.mode_weights],
        tc.TCKeys.mode_weights: fast_kappa_dict[tc.TCKeys.mode_weights],
    }

    with h5py.File(PH3_PARAMS_PATH / mat_id / "fast_kappa.hdf5", "w") as f:
        tc.dict_to_hdf5(fast_kappa_dict, f)

    if not FAST_ONLY:
        ph3.mesh_numbers = original_qmesh

        with tc.tqdm_gridpoints(desc="Conducitivity calc"):
            ph3, original_kappa_dict, _cond = tc.calculate_conductivity(
                ph3, temperatures=TEMPERATURES, log_level=2
            )

        original_kappa_dict = {
            tc.TCKeys.kappa_tot_avg: tc.calculate_kappa_avg(
                original_kappa_dict[tc.TCKeys.kappa_tot_rta]
            ),
            tc.TCKeys.mode_kappa_tot_avg: tc.calculate_kappa_avg(
                original_kappa_dict[tc.TCKeys.mode_kappa_tot_rta]
            ),
            tc.TCKeys.q_points: original_kappa_dict[tc.TCKeys.q_points],
            tc.TCKeys.temperatures: TEMPERATURES,
            tc.TCKeys.ph_freqs: _cond.frequencies,
            tc.TCKeys.heat_capacity: _cond.mode_heat_capacities,
            tc.TCKeys.spg_num: ph3.symmetry.dataset.number,
            tc.TCKeys.name: name,
            "q_mesh": original_qmesh,
            "q_point_mesh": original_qmesh,
            "weights": original_kappa_dict[tc.TCKeys.mode_weights],
            tc.TCKeys.mode_weights: original_kappa_dict[tc.TCKeys.mode_weights],
        }

        with h5py.File(PH3_PARAMS_PATH / mat_id / "kappa.hdf5", "w") as f:
            tc.dict_to_hdf5(original_kappa_dict, f)
