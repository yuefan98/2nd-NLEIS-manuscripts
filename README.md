# Second-Harmonic Nonlinear Electrochemical Impedance Spectroscopy (2nd-NLEIS)
----------------------------------------------------------------

This repository can be cited with: 

For further information or if this code is used, please go to or cite the following paper:

----------------------------------------------------------------

### Part I: Analytical theory and equivalent circuit representations for planar and porous electrodes
-------------
### Abstract
-------------

----------------------------------------------------------------
### Part II: Experimental Approach and Analysis of lithium-ion battery Experiments
-------------
### Abstract
-------------


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
NLEIS_toolbox: This folder contains essential data and an illustrative Jupyter notebook, serving as a user guide for utilizing the NLEIS toolbox. Users familiar with impedance.py will find the material readily adaptable, ensuring a user-friendly experience.

Part I: This folder contains one Supplementary Jupyter Notebook for recreating figures in Part I of the manuscript. 

Part II: This folder contains two Supplementary Jupyter Notebooks for part II of the manuscript. The main Supplementary Notebook provide the necessary code to generate all the figures in Part II. The Sobol Notebook provides code to reproduce the sobol sampling method disscussed in the manuscript

impedance: This code represents the adapted version of impedance.py, now integrated with the NLEIS toolbox.


----------------------------------------------------------------
### Credits
----------------------------------------------------------------

This work adopted and builted a 2nd-NLEIS toolbox based on [impedance.py](https://github.com/ECSHackWeek/impedance.py) (Murbach, M., Gerwe, B., Dawson-Elli, N., & Tsui, L. (2020). impedance.py: A Python package for electrochemical impedance analysis. Journal of Open Source Software, 5. https://doi.org/10.21105/joss.02349)

The 2nd-NLEIS toolbox will be be fully integrated into impedance.py in the future.

----------------------------------------------------------------
