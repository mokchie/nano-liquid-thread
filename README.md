#LAMMPS simulations of nano liquid threads in supersaturated environment

This repository contains the LAMMPS input scripts and pre- and post-processing codes used in the study 
> *Instability of nano liquid threads in a saturated or supersaturated vapor environment*
> 
> *Chaojie Mo, Xiaocong Yang, Qing-fei Fu, Longfei Chen*
> 
> Journal of Fluid Mechanics (submitted in June 2025, revised in Dec 2025)

The scripts allow reproduction of the numerical procedures and post-processing workflow described in the paper.

## 1. Contents

```text
.
|-- README.md
|-- pythonmod
|   |-- create_atom.py        (pre-processing module)
|   |-- variables3.py         (read/set variables in LAMMPS input files)
|   `-- readvel3.py           (read LAMMPS output data files)
|
|-- long_thread               (long liquid thread simulations)
|   |-- S2.0l                 (supersaturation ratio S = 2.0)
|   |   |-- atoms             (initial atoms.data and atoms.xyz)
|   |   |-- data              (LAMMPS output directory)
|   |   |-- domain.py         (pre-processing: generate liquid thread)
|   |   |-- in.run            (LAMMPS input script)
|   |   |-- plotone.py        (thread profile evolution)
|   |   |-- plotspectrum.py   (spectrum computation)
|   |   |-- rho1398.data      (equilibrium liquid argon)
|   |   `-- rho6.44.data      (equilibrium vapor argon)
|   |
|   |-- S3.0l                 (supersaturation ratio S = 3.0)
|   |   |-- atoms
|   |   |-- data
|   |   |-- domain.py
|   |   |-- in.run
|   |   |-- plotone.py
|   |   |-- plotspectrum.py
|   |   |-- rho1398.data
|   |   `-- rho9.66.data
|   |
|   `-- S4.0l                 (supersaturation ratio S = 4.0)
|       |-- atoms
|       |-- data
|       |-- domain.py
|       |-- in.run
|       |-- plotone.py
|       |-- plotspectrum.py
|       |-- rho12.88.data
|       `-- rho1398.data
|
`-- short_thread              (short liquid thread simulations)
    |-- S2.0s                 (supersaturation ratio S = 2.0)
    |   |-- atoms
    |   |-- data
    |   |-- domain.py
    |   |-- in.run
    |   |-- plothmax.py       (maximum perturbation amplitude)
    |   |-- plotone.py
    |   |-- rho1398.data
    |   `-- rho6.44.data
    |
    |-- S3.0s                 (supersaturation ratio S = 3.0)
    |   |-- atoms
    |   |-- data
    |   |-- domain.py
    |   |-- in.run
    |   |-- plothmax.py
    |   |-- plotone.py
    |   |-- rho1398.data
    |   `-- rho9.66.data
    |
    `-- S4.0s                 (supersaturation ratio S = 4.0)
        |-- atoms
        |-- data
        |-- domain.py
        |-- in.run
        |-- plot2H1.py        (first-order Fourier coefficient)
        |-- plothmax.py
        |-- plotone.py
        |-- rho12.88.data
        `-- rho1398.data
```
---

## 2. Requirements

### LAMMPS
- LAMMPS version: **≥ 2. Aug. 2023**
- Required packages:
  - MC
  - MOLECULE


### Post-processing
- Python ≥ 3.8
- Required packages:
  - numpy
  - scipy
  - matplotlib
- The path of "pythonmod" directory should be added to the environment variable "PYTHONPATH".

---

## 3. Building the liquid thread model

    python domain.py

---

## 4. Running the simulations

    mpirun -np 64 lmp -in in.run

---

## 5. Post-processing

Examples:

    python plotone.py

This read the data output by lammps and plot the profile evolution of the liquid thread.

    python plotspectrum.py

This read the data output by lammps and compute the spectrum and plot the spectrum curves.

    python plothmax.py

This read the data output by lammps and plot the evolution of the maximum perturbation magnitude.

    python plot2H1.py

This read the data output by lammps and plot the evolution of the first-order Fourier coefficient.

---

## 6. Relation to figures in the paper

| Figure | Script | Description |
| ------ | ------ | ----------- |
| Fig. 6 and Fig. 8 |  plotone.py | Profile evolution |
| Fig. 7 | plotspectrum.py | Time evolution of the perturbation spectrum |
| Fig. 9 | plothmax.py | Time evolution of the maximum perturbation magnitude |
| Fig. 9 inset | plot2H1.py | Time evolution of the first-order Fourier coefficient |

## 7. Contact
For questions regarding the simulations or scripts, please contact:
Name: Chaojie Mo
Email: mochaojie@gmail.com or mochaojie@buaa.edu.cn


