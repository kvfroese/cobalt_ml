# Linear MLIP Implementation for Cobalt Structures

This is an implementation of a linear machine learning interatomic potential, using the Behler–Parrinello method, with the use of invariant geometric descriptors. The symmetry and cutoff functions used in the project came from the BP paper *Generalized Neural-Network Representation of High-Dimensional Potential-Energy Surfaces*. The data we have used for learning is a good majority of the COMPASS dataset for cobalt (II) coordination structures.

## How to Run

It is recommended to run the program via terminal with a config file. If no config file is specified, config/default.ini will be chosen.

1. Atomic parser for Orca (split_orca.py)
    - From an output file folder (default orca_outputs) this will parse the file for coordinate and energy information. It is likely that not all files will be able to be read correctly. This will dump all parsed information into a pickle file (default folder atom_geometry)
2. MLIP (mlip.py)
    - This file generates numerical descriptors using symmetry functions and their derivatives, saving info to another pickle file (default folder desc_out).
3. Testing MLIP (test_mlip.py)
    - This performs six unit tests on the MLIP functions themselves.
4. Ridge regression (regression.py)
    - This file processes the descriptor and energy info to perform linear regression, outputting to another pickle file (into desc_out as well).
5. Figure generation (generate_figures.py)
    - This file was used to prepare all figures for the written report.
6. Various utility modules, called by other files.

## Further info

If you run across this and are not our professor and have some questions, we are happy to discuss. As stated in the description this was made for a final project, for a class about machine learning in chemistry.