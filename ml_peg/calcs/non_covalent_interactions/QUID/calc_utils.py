import hashlib
import json
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from ase import Atoms


CACHE_ENABLED = os.getenv('QUID_DISABLE_CACHE', '0') != '1'


def set_cache_enabled(flag: bool):
    global CACHE_ENABLED
    CACHE_ENABLED = bool(flag)


def _calc_identity_hash(calc) -> str:
    """
    Build a stable hash from calculator class and parameters to avoid cache
    collisions across different parameterizations (e.g., different models).
    """
    ident = {
        'class': f"{calc.__class__.__module__}.{calc.__class__.__name__}",
    }
    # ASE calculators commonly expose 'todict' or 'parameters'
    try:
        if hasattr(calc, 'todict'):
            d = calc.todict()
        elif hasattr(calc, 'parameters'):
            d = getattr(calc, 'parameters')
        else:
            d = {}
    except Exception:
        d = {}

    # Extend with common attributes if present
    for k in ('model', 'model_path', 'foundation_model', 'method', 'dtype', 'device'):
        if hasattr(calc, k):
            try:
                d[k] = getattr(calc, k)
            except Exception:
                pass
    # Make JSON serializable
    def _ser(obj):
        try:
            json.dumps(obj)
            return obj
        except Exception:
            return repr(obj)
    ident['params'] = _ser(d)
    js = json.dumps(ident, sort_keys=True)
    return hashlib.sha1(js.encode('utf-8')).hexdigest()[:16]


def _cache_dir(calc) -> str:
    root = os.path.join(os.path.dirname(__file__), 'cache')
    cls = calc.__class__.__name__
    h = _calc_identity_hash(calc)
    d = os.path.join(root, f"{cls}__{h}")
    os.makedirs(d, exist_ok=True)
    return d


def _atoms_fingerprint(atoms: Atoms) -> str:
    Z = atoms.get_atomic_numbers().astype(np.int64)
    R = atoms.get_positions().astype(np.float64)
    key = Z.tobytes() + R.tobytes()
    return hashlib.sha1(key).hexdigest()


def energy_cached(atoms: Atoms, calc, name: Optional[str] = None) -> float:
    """
    Compute energy in eV with a calculator, caching by atomic numbers and positions.
    """
    if name is None:
        name = 'structure'
    fp = _atoms_fingerprint(atoms)
    cdir = _cache_dir(calc)
    cpath = os.path.join(cdir, f'{name}_{fp}.json')
    if CACHE_ENABLED and os.path.exists(cpath):
        try:
            with open(cpath, 'r') as f:
                data = json.load(f)
            e = float(data['energy_eV'])
            return e
        except Exception:
            pass

    atoms = atoms.copy()
    atoms.calc = calc
    e = float(atoms.get_potential_energy())  # ASE default unit: eV
    if CACHE_ENABLED:
        try:
            with open(cpath, 'w') as f:
                json.dump({'energy_eV': e}, f)
        except Exception:
            pass
    return e


def compute_interaction_energy_ev(dimer: Atoms, big: Atoms, small: Atoms, calc) -> Tuple[float, float, float, float]:
    """
    Returns (E_dimer, E_big, E_small, E_int) in eV.
    """
    Ed = energy_cached(dimer, calc, name='dimer')
    Eb = energy_cached(big, calc, name='big')
    Es = energy_cached(small, calc, name='small')
    return Ed, Eb, Es, (Ed - Eb - Es)


# Calculator factory
def create_calculator(name: str, **kwargs):
    name = name.strip().lower()
    if name == 'emt':
        from ase.calculators.emt import EMT
        return EMT(**kwargs)
    if name == 'xtb' or name == 'gfn2-xtb' or name == 'gfn2':
        # Requires xtb-python installed
        from ase.calculators.xtb import XTB
        params = dict(method='GFN2-xTB')
        params.update(kwargs)
        return XTB(**params)
    if name == 'mace':
        # Expect user to provide model path
        try:
            from mace.calculators import mace_mp
        except Exception as e:
            raise RuntimeError('MACE calculator requested but not available. Install mace.') from e
        model = kwargs.get('model', None)
        device = kwargs.get('device', 'cpu')
        if model is None:
            raise ValueError('create_calculator(mace): please provide model=path_to_model.pt')
        return mace_mp(model_path=model, device=device)

    # Dynamic import hook for custom calculators, e.g. --calc-module mypkg.mod:factory
    if name.startswith('py:'):
        # Format: py:module.path:factory?arg=value,...
        spec = name[3:]
        if ':' not in spec:
            raise ValueError('Invalid py: spec. Use py:module.path:factory')
        mod_name, factory = spec.split(':', 1)
        mod = __import__(mod_name, fromlist=['*'])
        fn = getattr(mod, factory)
        return fn(**kwargs)

    raise ValueError(f'Unknown calculator: {name}')
