"""
Implementation of the VQCount algorithm for the approximate counting of solutions to a combinatorial problem.
"""

import json
import time
from collections import Counter

import numpy as np

from .counting.jvv import find_next_jvv_assignment, update_exact_sol_pdf
from .counting.sat import find_tvd, postselect_approx_pdf, verify_sat_state
from .qaoa.instantiation import instantiate_ansatz
from .qaoa.launcher import QAOALauncher


class VQCount(QAOALauncher):
    """
    Class to count the number of solutions to a problem using the VQCount algorithm.

    Parameters:
    -----------
    graph: ProblemGraph
        Graph representing the instance of the problem from the following:
        - "nae3sat": positive Not-All-Equal 3-SAT problem.
        - "1in3sat": positive 1-in-3-SAT problem.
    depth: int
        Number of layers of gates to apply (depth 'p').
    qaoa_version: str
        Type of QAOA circuit to create from the following:
        - "qaoa": X-mixer QAOA circuit.
        - "gm-qaoa": Grover-mixer QAOA circuit.
    optimizer: str, optional
        SciPy optimizer to use for the minimization of the energy. Default is "SLSQP".
    backend: str, optional
        Backend to use for the simulation of the QAOA circuit from Quimb's possible backends. Default is "numpy".
    max_bond: int, optional
        Maximum bond in the contraction of the QAOA circuit. If None, the bond is not limited.
    tau: float, optional
        Thresold for the optimization of the energy. If the energy is below this value, the optimization stops.

    Attributes:
    -----------
    theta_ini: np.ndarray
        Initial parameters of the QAOA circuit.
    theta_opt: np.ndarray
        Optimal parameters of the QAOA circuit.
    compute_time: dict
        Dictionary of the computation time of the different steps of the QAOA circuit.
    """

    def __init__(
        self,
        graph,
        depth,
        qaoa_version="regular",
        optimizer="SLSQP",
        backend="numpy",
        max_bond=None,
        tau=None,
    ):
        super().__init__(
            graph,
            depth,
            qaoa_version=qaoa_version,
            optimizer=optimizer,
            backend=backend,
            max_bond=max_bond,
            tau=tau,
        )

    def pruned_sampler(
        self, num_sol_sampled, ansatz, wavefunction=None, opt=None, use_mps=False
    ):
        """
        Sample without replacement from the QAOA circuit until a given number of solutions (postselected samples) is found.

        Parameters:
        -----------
        num_sol_sampled: int
            Number of solutions to sample.
        ansatz: ansatz: qtn.Circuit, optional
            QAOA ansatz to sample. Used to save the marginals found. If None, the QAOA ansatz is instantiated from the optimal parameters.
        wavefunction: qtn.TensorNetwork, optional
            Dense representation of the QAOA circuit. If None, the wavefunction is generated from the circuit.
        opt: str, optional
            Contraction path optimizer. Default is Quimb's default optimizer.
        use_mps: bool, optional
            If True, the QAOA circuit is instantiated in the MPS format. Default is False.

        Returns:
        --------
        counts: counter
            Counter of the states sampled.
        postselected_counts: counter
            Counter of the postselected states sampled.
        distinct_postselected_counts: counter
            Counter of the distinct postselected states sampled (each count is one since only distinct solutions are kept).
        """

        counts = Counter()
        postselected_counts = Counter()

        while sum(postselected_counts.values()) < np.floor(num_sol_sampled):

            if wavefunction is None:
                samples = list(
                    self.sample_qaoa(
                        int(num_sol_sampled),
                        ansatz=ansatz,
                        opt=opt,
                        use_mps=use_mps,
                    )
                )
            else:
                samples = list(
                    self.simulate_counts_qaoa(
                        int(num_sol_sampled),
                        ansatz=ansatz,
                        wavefunction=wavefunction,
                    )
                )

            for sample in samples:
                counts[sample] += 1

                # verify if the sample was already found
                if sample in list(postselected_counts.keys()):
                    postselected_counts[sample] += 1

                # test if the sample is a solution
                elif verify_sat_state(sample, self.graph.cnf):
                    postselected_counts[sample] += 1

                # avoid looping overgm-qaoaremove non-distinct solutions
        distinct_postselected_counts = Counter()
        for sample in postselected_counts:
            distinct_postselected_counts[sample] = 1

        return counts, postselected_counts, distinct_postselected_counts

    def counter(
        self,
        num_sol_sampled_per_step,
        reoptimize=False,
        compute_stats=False,
        dense=True,
        contract_opt=None,
        use_mps_contract=None,
        sampling_opt=None,
        use_mps_sampling=False,
    ):
        """
        Given a number of solutions to sample per step of the self-reduction procedure, find the approximate number of solutions to the given problem.

        Parameters:
        -----------
        num_sol_sampled_per_step: int
            Number of solutions to sample per step of the self-reduction procedure.
        reoptimize: bool, optional
            If True, reoptimize the QAOA circuit after each subproblem. Default is False.
        compute_stats: bool, optional
            If True, return the statistics of the sampling. Default is False.
        dense: bool, optional
            If True, construct the dense representation of the QAOA circuit. Default is True.
        contract_opt: str, optional
            Contraction path optimizer. Default is Quimb's default optimizer.
        use_mps_contract: bool, optional
            If True, initialize the QAOA circuit as a Matrix Product State (MPS).
        sampling_opt: str, optional
            Contraction path optimizer for the sampling. Default is Quimb's default optimizer.
        use_mps_sampling: bool, optional
            If True, the QAOA circuit is instantiated in the MPS format. Default is False.

        Returns:
        --------
        approx_num_sol: float
            Approximate number of solutions to the problem.
        stats: dict
            Dictionnary with the statistics of the counting process.
        """

        if self.theta_opt is not None:
            theta = self.theta_opt
        elif self.theta_ini is not None:
            theta = self.theta_ini
        else:
            raise ValueError(
                "Please initialize or initialize and run QAOA before sampling."
            )

        n = self.graph.num_nodes

        assignements = ""
        assumptions = []
        probabilities = []

        stats_obj = StatsCounter(self.energy_opt, theta.tolist())

        # iterator on subproblems
        for _ in range(n):

            # instantiate the vqcount ansatz
            start_instant = time.time()
            ansatz = instantiate_ansatz(
                self.graph,
                self.depth,
                theta[: self.depth],
                theta[self.depth :],
                qaoa_version=self.qaoa_version,
                mps=use_mps_sampling,
                assumptions=assumptions,
            )
            end_instant = time.time()

            # construct the dense representation of the QAOA circuit
            start_dense = time.time()
            if dense or compute_stats:
                wavefunction = ansatz.to_dense(
                    optimize=contract_opt, backend=self.backend
                )
            else:
                wavefunction = None
            end_dense = time.time()

            # sample from the circuit
            start_sample = time.time()
            counts, postselected_counts, distinct_postselected_counts = (
                self.pruned_sampler(
                    num_sol_sampled_per_step,
                    ansatz,
                    wavefunction=wavefunction,
                    opt=sampling_opt,
                    use_mps=use_mps_sampling,
                )
            )
            end_sample = time.time()

            # find the probability for each subproblem
            prob, next_assigment = find_next_jvv_assignment(
                distinct_postselected_counts, assignements
            )

            # reoptimize the QAOA circuit
            start_opt = time.time()
            if reoptimize:
                energy_opt, theta_opt = self.optimize_qaoa(
                    opt=contract_opt, use_mps=use_mps_contract, assumptions=assumptions
                )
                theta = theta_opt
            else:
                energy_opt = np.nan
                theta_opt = np.empty_like(theta)
                theta_opt[:] = np.nan
            end_opt = time.time()

            # compute statistics
            if compute_stats:
                start_stats = time.time()
                stats_obj.update_stats(
                    self.graph,
                    counts,
                    postselected_counts,
                    distinct_postselected_counts,
                    wavefunction,
                    energy_opt,
                    theta_opt.tolist(),
                    assumptions,
                )
                end_stats = time.time()

                stats_obj.counter_time["instantiation"] += end_instant - start_instant
                stats_obj.counter_time["sampling"] += end_sample - start_sample
                stats_obj.counter_time["optimization"] += end_opt - start_opt
                stats_obj.counter_time["dense"] += end_dense - start_dense
                stats_obj.counter_time["stats"] = end_stats - start_stats

            # update the assignements, assumptions and probabilities
            assignements += next_assigment
            assumptions.append(assignements[-1])
            probabilities.append(prob)

        # compute the approximate count
        approx_num_sol = 1 / np.prod(probabilities)

        # return statistics
        stats = stats_obj.return_stats()

        return approx_num_sol, stats


class StatsCounter:
    """
    Save the statistics of the counting process for the method counter of the VQCount class.

    Parameters:
    -----------
    energy: float
        Initial energy of the given QAOA circuit.
    theta_opt: list
        Initial parameters of the QAOA circuit.

    Attributes:
    -----------
    num_samples_per_step: list
        Number of samples generated per step of the self-reduction procedure.
    num_postselected_samples_per_step: list
        Number of postselected samples generated per step of the self-reduction procedure.
    num_distinct_postselected_samples_per_step: list
        Number of distinct pruned samples generated.
    success_rate_per_step: list
        Success rate of the sampling per step of the self-reduction procedure.
    unpostselected_nonuniformity_per_step: list
        Unpostselected nonuniformity of the sampling per step of the self-reduction procedure.
    nonuniformity_per_step: list
        Nonuniformity of the sampling per step of the self-reduction procedure
    assumptions: list
        Assumptions made during the self-reduction procedure, i.e. the qubits to fix in the QAOA circuit.
    counts: list
        Counters of the states sampled.
    postselected_counts: list
        Counters of the postselected states sampled.
    distinct_postselected_counts:
        Counters of the distinct postselected states sampled (each count is one since only distinct solutions are kept).
    counter_time: dict
        Computation time of the different steps of the counting process.
    energy_per_step: list
        Energy of the QAOA circuit for each subproblem.
    theta_opt_per_step: list
        Optimized parameters of the QAOA circuit for each subproblem.
    """

    def __init__(self, energy, theta_opt):
        self.num_samples_per_step = []
        self.num_postselected_samples_per_step = []
        self.num_distinct_postselected_samples_per_step = []
        self.success_rate_per_step = []
        self.unpostselected_nonuniformity_per_step = []
        self.nonuniformity_per_step = []
        self.assumptions = []
        self.counts = []
        self.postselected_counts = []
        self.distinct_postselected_counts = []

        self.energy_per_step = [energy]
        self.theta_opt_per_step = [theta_opt]

        self.counter_time = {
            "instantiation": 0,
            "sampling": 0,
            "optimization": 0,
            "dense": 0,
            "stats": 0,
        }

    def update_stats(
        self,
        graph,
        counts,
        postselected_counts,
        distinct_postselected_counts,
        wavefunction,
        energy,
        theta_opt,
        assumptions,
    ):
        """
        Save the statistics of the counting process for one step of the self-reduction procedure.

        Parameters:
        -----------
        graph: ProblemGraph
            Graph representing the instance of the problem from the following:
            - "nae3sat": positive Not-All-Equal 3-SAT problem.
            - "1in3sat": positive 1-in-3-SAT problem.
        counts: counter
            Counter of the states sampled.
        postselected_counts: counter
            Counter of the postselected states sampled.
        distinct_postselected_counts: counter
            Counter of the distinct postselected states sampled.
        wavefunction: np.ndarray
            Wavefunction of the QAOA circuit.
        energy: float
            Energy of the QAOA circuit.
        theta_opt: list
            Optimized parameters of the QAOA circuit.
        assumptions
            Assumptions made during the self-reduction procedure, i.e. the qubits to fix in the QAOA circuit.
        """

        # do not compute the nonuniformity to avoid memory issues for large graphs
        if graph.num_nodes < 31:
            # update the exact solution probability density function
            exact_sol_pdf = update_exact_sol_pdf(graph.exact_sol_pdf, assumptions)

            # compute the approximate solution probability density function
            approx_sol_pdf = np.squeeze(np.abs(wavefunction) ** 2)

            # postselect the approximate solution probabilitity density function
            postselected_approx_sol_pdf = postselect_approx_pdf(
                approx_sol_pdf, exact_sol_pdf
            )

            # compute the nonuniformity
            unpostselected_nonuniformity = find_tvd(approx_sol_pdf, exact_sol_pdf)
            nonuniformity = find_tvd(postselected_approx_sol_pdf, exact_sol_pdf)

            self.unpostselected_nonuniformity_per_step.append(
                unpostselected_nonuniformity
            )
            self.nonuniformity_per_step.append(nonuniformity)

        self.energy_per_step.append(energy)
        self.theta_opt_per_step.append(theta_opt)

        num_samples = sum(counts.values())
        num_postselected_samples = sum(postselected_counts.values())
        num_distinct_postselected_samples = sum(distinct_postselected_counts.values())

        success_rate_per_step = num_postselected_samples / num_samples

        self.num_samples_per_step.append(num_samples)
        self.num_postselected_samples_per_step.append(num_postselected_samples)
        self.num_distinct_postselected_samples_per_step.append(
            num_distinct_postselected_samples
        )

        self.success_rate_per_step.append(success_rate_per_step)
        self.assumptions = assumptions

        self.counts.append(counts)
        self.postselected_counts.append(postselected_counts)
        self.distinct_postselected_counts.append(distinct_postselected_counts)

    def return_stats(self):
        """
        Return the statistics of the counting process.

        Returns:
        --------
        stats: dict
            Dictionnary with the statistics of the counting process.
        """

        stats = {
            "num_samples_per_step": self.num_samples_per_step,
            "num_postselected_samples_per_step": self.num_postselected_samples_per_step,
            "num_distinct_postselected_samples_per_step": self.num_distinct_postselected_samples_per_step,
            "success_rate_per_step": self.success_rate_per_step,
            "unpostselected_nonuniformity_per_step": self.unpostselected_nonuniformity_per_step,
            "nonuniformity_per_step": self.nonuniformity_per_step,
            "assumptions": self.assumptions,
            "counter_time": self.counter_time,
            "energy": self.energy_per_step,
            "theta": self.theta_opt_per_step,
            "postselected_counts": self.postselected_counts,
            "distinct_postselected_counts": self.distinct_postselected_counts,
        }

        return stats

    def save_stats(self, file_path):
        """
        Save the statistics of the counting process in a file.

        Parameters:
        -----------
        file_path: str
            Path of the file to save the statistics.
        """

        with open(file_path, "w") as f:
            json.dump(self.return_stats(), f)
