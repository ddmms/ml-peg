from mace.calculators import mace_mp 
from ase import units
from ase.io import read

macemp = mace_mp(model='/scratch/fd385/work/MACE-MATERIALS-PROJECT/MP0B3/mace-mp-0b3-medium.model', dispersion=True, default_dtype="float64")

atoms=read('POSCAR')
atoms.calc = macemp

energy=atoms.get_potential_energy()

with open("energy-mace", "a") as f:
    print(energy, file=f)
f.close()
