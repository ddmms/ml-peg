#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Generator, List, Optional, Tuple

import h5py
import numpy as np
from ase import Atoms


@dataclass
class QuidRecord:
    label: str
    scope: str  # 'equilibrium' or 'dissociation'
    q: Optional[str]  # non-equilibrium distance label, e.g., 'q_1.25'
    formula: Optional[str]
    dimer: Atoms
    big: Atoms
    small: Atoms
    eint_ev: Optional[Dict[str, float]]  # interaction energies by method (in eV)
    sapt: Optional[Dict[str, float]]
    properties: Optional[Dict[str, float]]


def _atoms_from_group(g, which: Optional[str] = None) -> Atoms:
    """
    Build ASE Atoms from an HDF5 group. Supports multiple layouts:
    1) g['atoms'] (dataset of atomic numbers) and g['positions'] (dataset Nx3)
    2) g['atoms'] (group) and g['positions'] (group) with datasets under keys like 'dimer', 'big_monomer', 'small_monomer'
    3) g has a subgroup (e.g. 'dimer') that itself contains datasets 'atoms' and 'positions'
    """
    # Layout 2: atoms/positions as groups keyed by which
    if 'atoms' in g and isinstance(g['atoms'], h5py.Group) and 'positions' in g and isinstance(g['positions'], h5py.Group):
        if which is None:
            # Try a sensible default
            candidates = list(g['atoms'].keys())
            if 'dimer' in candidates:
                which = 'dimer'
            elif candidates:
                which = candidates[0]
        if which not in g['atoms'] or which not in g['positions']:
            raise KeyError(f"Group missing atoms/positions for key '{which}'. Available: atoms={list(g['atoms'].keys())}, positions={list(g['positions'].keys())}")
        Z = np.array(g['atoms'][which], dtype=int)
        R = np.array(g['positions'][which], dtype=float)
        return Atoms(numbers=Z, positions=R)

    # Layout 1: direct datasets
    if 'atoms' in g and 'positions' in g and isinstance(g['atoms'], h5py.Dataset) and isinstance(g['positions'], h5py.Dataset):
        Z = np.array(g['atoms'], dtype=int)
        R = np.array(g['positions'], dtype=float)
        return Atoms(numbers=Z, positions=R)

    # Layout 3: subgroup like 'dimer' contains datasets
    if which and which in g and isinstance(g[which], h5py.Group):
        sg = g[which]
        if 'atoms' in sg and 'positions' in sg:
            Z = np.array(sg['atoms'], dtype=int)
            R = np.array(sg['positions'], dtype=float)
            return Atoms(numbers=Z, positions=R)

    # Try alternative dataset names
    keys = set(g.keys())
    cand_Z = [k for k in keys if 'atom' in k.lower() and 'number' in k.lower()]
    cand_R = [k for k in keys if 'pos' in k.lower()]
    if cand_Z and cand_R:
        Z = np.array(g[cand_Z[0]], dtype=int)
        R = np.array(g[cand_R[0]], dtype=float)
        return Atoms(numbers=Z, positions=R)

    raise KeyError('Could not find atoms/positions; keys: ' + str(sorted(g.keys())))


def _read_optional_dict(g, key: str) -> Optional[Dict[str, float]]:
    if key not in g:
        return None
    sub = g[key]
    if isinstance(sub, h5py.Dataset):
        # A single value or array; represent generically
        try:
            val = sub[()]
            if np.isscalar(val):
                return {key: float(val)}
        except Exception:
            pass
        return None
    # If group, collect numeric leaves
    out: Dict[str, float] = {}
    for k in sub.keys():
        try:
            v = sub[k][()]
            if np.isscalar(v) or (isinstance(v, np.ndarray) and v.shape == ()):  # scalar dataset
                out[str(k)] = float(v)
        except Exception:
            continue
    return out or None


def _get_formula(g) -> Optional[str]:
    if 'formula' in g:
        try:
            v = g['formula'][()]
            if isinstance(v, bytes):
                return v.decode('utf-8')
            if isinstance(v, str):
                return v
        except Exception:
            return None
    return None


def iterate_equilibrium(h5: h5py.File) -> Generator[QuidRecord, None, None]:
    for label, grp in h5.items():
        if label == 'dissociation':
            continue
        if not isinstance(grp, h5py.Group):
            continue
        # Two possible layouts:
        # A) grp has children groups: 'dimer', 'big_monomer', 'small_monomer'
        # B) grp itself is the dimer; and has subgroups 'big_monomer', 'small_monomer'
        # Prefer layout with shared 'atoms'/'positions' groups containing named entries
        try:
            dimer = _atoms_from_group(grp, which='dimer')
            big = _atoms_from_group(grp, which='big_monomer')
            small = _atoms_from_group(grp, which='small_monomer')
        except Exception:
            # Fallback: nested groups per species
            if 'dimer' in grp:
                dimer = _atoms_from_group(grp['dimer'])
            else:
                dimer = _atoms_from_group(grp)
            if 'big_monomer' in grp:
                big = _atoms_from_group(grp['big_monomer'])
            else:
                raise KeyError(f'{label}: missing big_monomer group')
            if 'small_monomer' in grp:
                small = _atoms_from_group(grp['small_monomer'])
            else:
                raise KeyError(f'{label}: missing small_monomer group')

        rec = QuidRecord(
            label=label,
            scope='equilibrium',
            q=None,
            formula=_get_formula(grp),
            dimer=dimer,
            big=big,
            small=small,
            eint_ev=_read_optional_dict(grp, 'Eint'),
            sapt=_read_optional_dict(grp, 'SAPT'),
            properties=_read_optional_dict(grp, 'properties'),
        )
        yield rec


def iterate_dissociation(h5: h5py.File) -> Generator[QuidRecord, None, None]:
    if 'dissociation' not in h5:
        return
    diss = h5['dissociation']
    for label, grp in diss.items():
        if not isinstance(grp, h5py.Group):
            continue
        # group of q points
        for qlabel, qgrp in grp.items():
            if not isinstance(qgrp, h5py.Group):
                continue
            # Prefer shared atoms/positions groups with named entries
            try:
                dimer = _atoms_from_group(qgrp, which='dimer')
                big = _atoms_from_group(qgrp, which='big_monomer')
                small = _atoms_from_group(qgrp, which='small_monomer')
            except Exception:
                # Fallback to subgroup-per-species
                if 'dimer' in qgrp:
                    dimer = _atoms_from_group(qgrp['dimer'])
                else:
                    dimer = _atoms_from_group(qgrp)
                if 'big_monomer' in qgrp:
                    big = _atoms_from_group(qgrp['big_monomer'])
                else:
                    raise KeyError(f'{label}/{qlabel}: missing big_monomer')
                if 'small_monomer' in qgrp:
                    small = _atoms_from_group(qgrp['small_monomer'])
                else:
                    raise KeyError(f'{label}/{qlabel}: missing small_monomer')

            rec = QuidRecord(
                label=label,
                scope='dissociation',
                q=str(qlabel),
                formula=_get_formula(qgrp),
                dimer=dimer,
                big=big,
                small=small,
                eint_ev=_read_optional_dict(qgrp, 'Eint'),
                sapt=_read_optional_dict(qgrp, 'SAPT'),
                properties=_read_optional_dict(qgrp, 'properties'),
            )
            yield rec


def inspect_layout(h5: h5py.File, depth: int = 2) -> Dict:
    def walk(g, d):
        if d == 0:
            return {}
        out = {}
        for k, v in g.items():
            if isinstance(v, h5py.Group):
                out[k] = {'type': 'group', 'children': walk(v, d - 1)}
            else:
                try:
                    shape = list(v.shape)
                except Exception:
                    shape = None
                out[k] = {'type': 'dataset', 'shape': shape}
        return out
    return walk(h5, depth)


def main():
    ap = argparse.ArgumentParser(description='Parse QUID.h5 and list or export entries')
    ap.add_argument('--h5', default=os.path.join(os.path.dirname(__file__), 'QUID.h5'))
    ap.add_argument('--inspect', action='store_true', help='Print top-level layout')
    ap.add_argument('--list', action='store_true', help='List equilibrium labels')
    ap.add_argument('--list-diss', action='store_true', help='List dissociation labels')
    ap.add_argument('--list-eint-keys', action='store_true', help='List unique Eint method keys and counts from equilibrium set')
    ap.add_argument('--export-json', help='Export minimal metadata to JSON')
    args = ap.parse_args()

    if not os.path.exists(args.h5):
        raise SystemExit(f'File not found: {args.h5}. Place QUID.h5 here or pass --h5 path.')

    with h5py.File(args.h5, 'r') as h5:
        if args.inspect:
            layout = inspect_layout(h5, depth=3)
            print(json.dumps(layout, indent=2))
            return

        if args.list:
            labels = [rec.label for rec in iterate_equilibrium(h5)]
            labels = sorted(set(labels))
            print('\n'.join(labels))
            return

        if args.list_diss:
            labels = [rec.label for rec in iterate_dissociation(h5)]
            labels = sorted(set(labels))
            print('\n'.join(labels))
            return

        if args.list_eint_keys:
            from collections import defaultdict
            counts = defaultdict(int)
            examples = {}
            total = 0
            for rec in iterate_equilibrium(h5):
                if rec.eint_ev:
                    for k in rec.eint_ev.keys():
                        counts[k] += 1
                        if k not in examples:
                            examples[k] = rec.label
                        total += 1
            print('Eint keys found (equilibrium):')
            for k in sorted(counts.keys()):
                print(f'- {k} : {counts[k]} entries (example: {examples.get(k)})')
            if not counts:
                print('No Eint keys found.')
            return

        if args.export_json:
            data = []
            for rec in iterate_equilibrium(h5):
                d = {
                    'label': rec.label,
                    'scope': rec.scope,
                    'q': rec.q,
                    'formula': rec.formula,
                    'eint_methods': list(rec.eint_ev.keys()) if rec.eint_ev else [],
                }
                data.append(d)
            with open(args.export_json, 'w') as f:
                json.dump(data, f, indent=2)
            print(f'Wrote {len(data)} entries to {args.export_json}')
            return

        # Default: show a brief summary
        eq = list(iterate_equilibrium(h5))
        nd = list(iterate_dissociation(h5))
        print(f'Equilibrium dimers: {len(eq)}')
        if eq[:3]:
            print('  Example:', ', '.join(r.label for r in eq[:3]))
        print(f'Dissociation entries: {len(nd)}')
        if nd[:3]:
            tag = ', '.join(f'{r.label}/{r.q}' for r in nd[:3])
            print('  Example:', tag)


if __name__ == '__main__':
    main()
