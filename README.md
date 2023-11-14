# Second-Harmonic Nonlinear Electrochemical Impedance Spectroscopy (2nd-NLEIS)
----------------------------------------------------------------

This repository can be cited with: [![DOI](https://zenodo.org/badge/709980608.svg)](https://zenodo.org/doi/10.5281/zenodo.10050482)


For further information or if this code is used, please go to or cite the following paper:
* Paper citation will be available upon acceptance

----------------------------------------------------------------

### Part I: Analytical theory and equivalent circuit representations for planar and porous electrodes
-------------
### Abstract
-------------

Analytical theory for second harmonic nonlinear electrochemical impedance spectroscopy (2nd- NLEIS) of planar and porous electrodes is developed for interfaces governed by Butler-Volmer kinetics, a Helmholtz (mainly) or Gouy-Chapman (introduced) double layer, and transport by ion migration and diffusion. A continuum of analytical EIS and 2nd-NLEIS models is presented, from nonlinear Randles circuits with or without diffusion impedances to nonlinear macrohomogeneous porous electrode theory that is shown to be analogous to a nonlinear transmission-line model. EIS and 2nd-NLEIS for planar electrodes share classic charge transfer RC and diffusion time-scales, whereas porous electrode EIS and 2nd-NLEIS share three characteristic time constants. In both cases, the magnitude of 2nd-NLEIS is proportional to nonlinear charge transfer asymmetry and thermodynamic curvature parameters. The phase behavior of 2nd-NLEIS is more complex and model-sensitive than in EIS, with half-cell NLEIS spectra potentially traversing all four quadrants of a Nyquist plot. We explore the power of simultaneously analyzing the linear EIS and 2nd-NLEIS spectra for two-electrode configurations, where the full-cell linear EIS signal arises from the sum of the half-cell spectra, while the 2nd-NLEIS signal arises from their difference.

----------------------------------------------------------------
### Part II: Model-based Analysis of Lithium-Ion Battery Experiments
-------------
### Abstract
-------------
Quantitative analysis of electrochemical impedance spectroscopy (EIS) and 2nd-harmonic nonlinear EIS (2nd-NLEIS) data from commercial Li-ion batteries is performed using the porous electrode half-cell models developed in Part I. Because EIS and 2nd-NLEIS signals have opposite parity, the full-cell EIS model relies on the sum of cathode and anode half-cells whereas the full- cell 2nd-NLEIS model requires subtraction of the anode half-cell from the cathode. The full-cell EIS model produces a low error fit to EIS measurements, but importing EIS best-fit parameters into the 2nd-NLEIS model fails to ensure robust model-data convergence. In contrast, simultaneously fitting opposite parity EIS and 2nd-NLEIS models to the corresponding magnitude- normalized experimental data provides a lower total error fit, more internally-self-consistent parameters, and better assignment of parameters to individual electrodes than EIS analysis alone. Our results quantify the extent that mildly aging of cells (<1% capacity loss) results in substantial increases in cathode charge transfer resistance, and for the first time, a breakdown in cathode charge transfer symmetry at 30% and lower state-of-charge (SoC). New avenues for model-based analysis are discussed for full-cell diagnostic and we identify several open questions.

### Software Dependencies
----------------------------------------------------------------
This repository was developed using the following versions of the subsequent softwares:

* Python 3.9.13
* Conda 23.1.0
* Git Bash for MacOS

The conda environment used for this work can be recreated with the following commands:

```conda env create -f environment.yml```

```conda activate impedance```

----------------------------------------------------------------
### Folders
----------------------------------------------------------------
**NLEIS_toolbox**: This folder contains essential data and an illustrative Jupyter notebook, serving as a user guide for utilizing the NLEIS toolbox. Users familiar with impedance.py will find the material readily adaptable, ensuring a user-friendly experience.

**Part I**: This folder contains one Supplementary Jupyter Notebook for recreating figures in Part I of the manuscript. 

**Part II**: This folder contains two Supplementary Jupyter Notebooks for part II of the manuscript. The main Supplementary Notebook provide the necessary code to generate all the figures in Part II. The Sobol Notebook provides code to reproduce the sobol sampling method disscussed in the manuscript

**impedance**: This code represents the adapted version of impedance.py, now integrated with the NLEIS toolbox.


----------------------------------------------------------------
### Credits
----------------------------------------------------------------

This work adopted and builted a 2nd-NLEIS toolbox based on [impedance.py](https://github.com/ECSHackWeek/impedance.py) (Murbach, M., Gerwe, B., Dawson-Elli, N., & Tsui, L. (2020). impedance.py: A Python package for electrochemical impedance analysis. Journal of Open Source Software, 5. https://doi.org/10.21105/joss.02349)

The 2nd-NLEIS toolbox will be be fully integrated into impedance.py in the future.

----------------------------------------------------------------
