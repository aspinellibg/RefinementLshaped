**Description**

The code provided in this repository accompanies the paper "_A novel L-shaped refinement chain cuts method for two-stage stochastic programs_" (see the reference below).

Specifically, the repository contains the following files and folders:

- Folder "scens": contains the scenario files.
- Folder "network": contains the network instances.
- File "model_stoch_network_design.py": Python implementation of the proposed methods, including the disjoint partition approach, the fixed-scenario partition approach, and the iterative refinement algorithm.
- File "MRSNDP_run.py": main script used to run the computational experiments.

For example, consider the network instance r04.1 with 256 scenarios.

To solve the Benders master problem at a fixed refinement level using disjoint groups of S scenarios each, run:
python3 MRSNDP_run.py network/r04.1.dow scens/scen_dem_cap_.1/scens_256_r04.1.dow bendersGrouping_disjoint S

To solve the Benders master problem at a fixed refinement level using fixed-scenario groups of S scenarios each (where the first scenario is included in every group), run:
python3 MRSNDP_run.py network/r04.1.dow scens/scen_dem_cap_.1/scens_256_r04.1.dow bendersGrouping_fixedscenarios S

To run the iterative refinement algorithm between two consecutive levels of the refinement chain using disjoint groups of sizes S_lower (lower level) and S_upper (upper level), run:
python3 MRSNDP_run.py network/r04.1.dow scens/scen_dem_cap_.1/scens_256_r04.1.dow bendersGrouping_disjoint_ALGO S_lower S_upper

**Reference**

M. Hewitt, F. Maggioni, and A. Spinelli (2026). _A novel L-shaped refinement chain cuts method for two-stage stochastic programs_. Under review. Preprint available at https://arxiv.org/abs/2606.02469
