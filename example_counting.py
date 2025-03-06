"""
Example of counting the number of solutions of a NAE3SAT problem using the VQCount algorithm.
"""

import cotengra as ctg
from vqcount.counting.gen import generate_satisfiable_sat_problem
from vqcount.vqcount import VQCount

# PARAMETERS

# problem parameters
problem = "nae3sat"
qubit = 6
alpha = 2

# QAOA parameters
ini_method = "tqa"
qaoa_version = "vqcount-qaoa"
depth = 3

# optimization parameters
use_mps_contract = True
use_mps_sampling = True
optimizer = "SLSQP"
backend = "numpy"

# sampling parameters
reoptimize = False
num_sol_sampled_per_step = 50

# stats parameters
compute_stats = False

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

graph = generate_satisfiable_sat_problem(problem, qubit, alpha)
print("3-SAT formula:\n", graph.cnf_ini)
print()

# COMPUTATION

QAOA = VQCount(
    graph,
    depth,
    qaoa_version=qaoa_version,
    max_bond=None,
    optimizer=optimizer,
    backend=backend,
)
print("Instantiation is done.")

theta_ini = QAOA.initialize_qaoa(ini_method, opt=contract_opt, use_mps=use_mps_contract)
print("Initialization is done.")

energy_opt, theta_opt = QAOA.optimize_qaoa(opt=contract_opt, use_mps=use_mps_contract)
print("Optimization is done.")

approx_num_sol, stats = QAOA.counter(
    num_sol_sampled_per_step=num_sol_sampled_per_step,
    reoptimize=reoptimize,
    compute_stats=compute_stats,
    contract_opt=contract_opt,
    use_mps_contract=use_mps_contract,
    sampling_opt=sampling_opt,
    use_mps_sampling=use_mps_sampling,
)
print("Counting is done.\n")

print("Approximate number of solutions:", approx_num_sol)
print("Exact number of solutions:", graph.exact_num_sol)
print()

if compute_stats:
    for key, value in stats.items():
        print(key)
        print(value)
        print()
