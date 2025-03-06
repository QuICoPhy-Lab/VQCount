"""
Example of solving the optimization version of the NAE3SAT problem using QAOA.
"""

import cotengra as ctg
from vqcount.counting.problem import PositiveNaeThreeSatGraph
from vqcount.qaoa.launcher import QAOALauncher

# PARAMETERS

# problem parameters
problem = "nae3sat"
qubit = 6
alpha = 1

# QAOA parameters
ini_method = "tqa"
qaoa_version = "qaoa"
depth = 3

# optimization parameters
use_mps_contract = True
use_mps_sampling = True
optimizer = "SLSQP"
backend = "numpy"

# sampling parameters
num_samples = 1000

# COTENGRA PARAMETERS

# contraction parameters
contract_kwargs = {
    "minimize": "size",
    "methods": ["greedy", "kahypar"],
    "reconf_opts": {},
    "optlib": "random",
    "max_repeats": 64,
    "parallel": True,
    "max_time": "rate:1e6",
}

# sampling parameters
sampling_kwargs = {
    "minimize": "flops",
    "methods": ["greedy"],
    "reconf_opts": {},
    "optlib": "random",
    "max_repeats": 32,
    "parallel": True,
    "max_time": "rate:1e6",
}

contract_opt = ctg.ReusableHyperOptimizer(**contract_kwargs)
sampling_opt = ctg.ReusableHyperOptimizer(**sampling_kwargs)

# GENERATION

graph = PositiveNaeThreeSatGraph(qubit, alpha * qubit, int(alpha * 3), 3)
print("3-SAT formula:\n", graph.cnf_ini)
print()

# COMPUTATION

QAOA = QAOALauncher(
    graph,
    depth,
    qaoa_version=qaoa_version,
    optimizer=optimizer,
    backend=backend,
)
theta_ini = QAOA.initialize_qaoa(
    ini_method=ini_method, opt=contract_opt, use_mps=use_mps_contract
)
print("Initialization is done!")
energy, theta = QAOA.optimize_qaoa(opt=contract_opt, use_mps=use_mps_contract)
print("Optimization is done!\n")

print("Energy:", energy)
