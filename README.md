# VAP

* Author: Xin Chen
* Email: Bismarrck@me.com

The Virtual-Atom Approach for constructing symmetry function based atomistic 
neural network potentials.

## 1. Introduction

Available datasets:

* Ni: 461 structures.
* Mo: 284 structures.
* SNAP: 3973 structures.

The original data can be found on this 
[repo](https://github.com/materialsvirtuallab/snap).

Available pre-trained models:

* Ni: energy, force
* Mo: energy, force, stress
* Ni-Mo: energy, force, stress

The $\eta$ and $\omega$ used in these models are:

* $\eta$: `[0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0, 40.0]`
* $\omega$: `[0.0, 3.0]`

## 2. Requirements

* Python >= 3.7.0
* TensorFlow == 1.13
* Scikit
* Numpy
* ASE >= 3.17

## 3. Usage

A benchmark function `prediction_phase_benchmark` is provided. You can measure
the performance of your own machine with this function.

To calculate properties of an arbitrary $\mathrm{Ni}_{x}\mathrm{Mo}_{y}$ 
structure, simply do:

```python
from vap import TensorAlloyCalculator
from ase.build import bulk
from ase.units import GPa

calc = TensorAlloyCalculator("NiMo.pb")
atoms = bulk("Ni", cubic=True)
atoms.calc = calc
print(atoms.get_total_energy())
print(atoms.get_forces())
print(atoms.get_stress() / GPa)
```

To use the provided datasets:

```python
from ase.db import connect

db = connect("datasets/snap.db")
size = len(db)
for atoms_id in range(1, 1 + size):
    atoms = db.get_atoms(id=atoms_id, add_additional_information=True)
```

The random seed for splitting datasets is a constant, **611**

```python
from sklearn.model_selection import train_test_split
from ase.db import connect

db = connect("datasets/snap.db")
size = len(db)
_, test_id_list = train_test_split(
    range(1, 1 + size), 
    random_state=611, 
    test_size=300)

for atoms_id in test_id_list:
    atoms = db.get_atoms(id=atoms_id, add_additional_information=True)
```
