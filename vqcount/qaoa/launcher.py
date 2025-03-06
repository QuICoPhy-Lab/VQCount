"""
Launcher for QAOA. Main class for the simulation of QAOA circuits.
"""

import time
from collections import Counter

import quimb.tensor as qtn
from quimb import simulate_counts

from .circuit import create_qaoa_circ
from .contraction import compute_energy, minimize_energy
from .initialization import initialize_qaoa_parameters
from .mps import create_qaoa_mps


class QAOALauncher:
    """
    This class regroups the main methods of QAOA applied to a particular combinatorial optimization problem. The launcher can initialize, run, and sample the QAOA circuit.

    Parameters:
    -----------
    graph: ProblemGraph
        Graph representing the instance of the problem from the following:
        - "nae3sat": Not-All-Equal 3-SAT problem.
        - "1in3sat": 1-in-3-SAT problem.
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
        qaoa_version,
        optimizer="SLSQP",
        backend="numpy",
        max_bond=None,
        tau=None,
    ):
        self.graph = graph
        self.depth = depth
        self.qaoa_version = qaoa_version
        self.optimizer = optimizer
        self.backend = backend
        self.max_bond = max_bond
        self.tau = tau

        # QAOA angles of the form (gamma_1, ..., gamma_p, beta_1, ..., beta_p)
        self.theta_ini = None
        self.theta_opt = None

        self.compute_time = {
            "initialization": 0,
            "energy": 0,
            "minimisation": 0,
            "sampling": 0,
        }

    def instantiate_qaoa(self, theta, use_mps=False, **ansatz_opts):
        """
        Instantiate the QAOA ansatz (circuit or MPS).

        Parameters:
        -----------
        theta: np.ndarray
            Parameters of the QAOA circuit to instantiate.
        use_mps: bool, optional
            If True, initialize the QAOA circuit as a Matrix Product State (MPS) instead of as a general tensor networks.
        ansatz_opts: dict
            Additional options to pass to the QAOA ansatz.

        Returns:
        --------
        circ: qtn.Circuit
            Instantiated QAOA circuit.
        """

        # create the QAOA circuit or MPS
        if use_mps:
            psi0 = create_qaoa_mps(
                self.graph,
                self.depth,
                theta[: self.depth],
                theta[self.depth :],
                qaoa_version=self.qaoa_version,
                **ansatz_opts,
            )
            circ = qtn.Circuit(psi0=psi0)
        else:
            circ = create_qaoa_circ(
                self.graph,
                self.depth,
                theta[: self.depth],
                theta[self.depth :],
                qaoa_version=self.qaoa_version,
                **ansatz_opts,
            )

        return circ

    def initialize_qaoa(self, ini_method, opt=None, use_mps=False, **ansatz_opts):
        """
        Initialize QAOA.

        Parameters:
        -----------
        ini_method: str
            Method to use for the initialization of the QAOA circuit from the following:
            - "random": Random initialization of the parameters.
            - "tqa": Trotterized Quantum Annealing initialization of the parameters.
        opt: str, optional
            Contraction path optimizer. Default is Quimb's default optimizer.
        use_mps: bool, optional
            If True, initialize the QAOA circuit as a Matrix Product State (MPS) instead of as a tensor network.
        ansatz_opts: dict
            Additional options to pass to the QAOA ansatz.

        Returns:
        --------
        theta_ini: np.ndarray
            Initial parameters of the QAOA circuit.
        """

        # initialize the QAOA circuit
        start_ini = time.time()
        theta_ini = initialize_qaoa_parameters(
            self.graph,
            self.depth,
            ini_method,
            qaoa_version=self.qaoa_version,
            opt=opt,
            backend=self.backend,
            use_mps=use_mps,
            max_bond=self.max_bond,
            **ansatz_opts,
        )
        end_ini = time.time()

        self.compute_time["initialization"] = end_ini - start_ini

        # save the initial parameters
        self.theta_ini = theta_ini

        return theta_ini

    def optimize_qaoa(self, opt=None, use_mps=False, energy=True, **ansatz_opts):
        """
        Optimize the qaoa.

        Parameters:
        -----------
        opt: str, optional
            Contraction path optimizer. Default is Quimb's default optimizer.
        use_mps: bool, optional
            If True, initialize the QAOA circuit as a Matrix Product State (MPS) instead of as a general tensor networks.
        energy: bool, optional
            If True, compute the final energy of the QAOA circuit.
        ansatz_opts: dict
            Additional options to pass to the QAOA ansatz.

        Returns:
        --------
        energy: float
            Energy of the QAOA circuit.
        theta_opt: np.ndarray
            Optimal parameters of the QAOA circuit.
        """

        if self.theta_ini is None:
            raise ValueError("Please initialize QAOA before running.")

        # minimize the energy
        start_minim = time.time()
        theta_opt, energy_opt = minimize_energy(
            self.theta_ini,
            self.graph,
            qaoa_version=self.qaoa_version,
            optimizer=self.optimizer,
            opt=opt,
            backend=self.backend,
            max_bond=self.max_bond,
            use_mps=use_mps,
            tau=self.tau,
            **ansatz_opts,
        )
        end_minim = time.time()

        if energy:
            # compute the final energy (useful for contraction time)
            start_energy = time.time()
            energy_opt = compute_energy(
                theta_opt,
                self.graph,
                qaoa_version=self.qaoa_version,
                opt=opt,
                backend=self.backend,
                use_mps=use_mps,
                max_bond=self.max_bond,
                **ansatz_opts,
            )
            end_energy = time.time()
        else:
            start_energy = None
            end_energy = None
            energy_opt = None

        self.compute_time["minimisation"] = end_minim - start_minim
        self.compute_time["energy"] = end_energy - start_energy

        # save the optimal parameters
        self.energy_opt = energy_opt
        self.theta_opt = theta_opt

        return energy_opt, theta_opt

    def sample_qaoa(self, shots, ansatz=None, opt=None, use_mps=True, **ansatz_opts):
        """
        Sample the qaoa shot by shot.

        Parameters:
        -----------
        shots: int
            Number of samples to take.
        ansatz: qtn.Circuit, optional
            QAOA ansatz to sample. Used to save the marginals found. If None, the QAOA ansatz is instantiated from the optimal parameters.
        opt: str, optional
            Contraction path optimizer. Default is Quimb's default optimizer.
        use_mps: bool, optional
            If True, initialize the QAOA circuit as a Matrix Product State (MPS) instead of as a tensor network.
        ansatz_opts: dict
            Additional options to pass to the QAOA ansatz.

        Returns:
        --------
        counts: Counter
            Counter of the samples.
        """

        if ansatz is None:
            if self.theta_opt is not None:
                ansatz = self.instantiate_qaoa(self.theta_opt, use_mps, **ansatz_opts)
            elif self.theta_ini is not None:
                ansatz = self.instantiate_qaoa(self.theta_ini, use_mps, **ansatz_opts)
            else:
                raise ValueError(
                    "Please initialize or initialize and run QAOA before sampling."
                )

        # sample the QAOA circuit
        start_sampling = time.time()
        counts = ansatz.sample(
            shots, optimize=opt, backend=self.backend, max_marginal_storage=2**28
        )
        end_sampling = time.time()

        self.compute_time["sampling"] = end_sampling - start_sampling
        self.counts = counts

        return counts

    def simulate_counts_qaoa(
        self,
        shots,
        ansatz=None,
        wavefunction=None,
        opt=None,
        use_mps=True,
        **ansatz_opts
    ):
        """
        Simulate the counts for the qaoa using the dense representation of the wavefunction.

        Parameters:
        -----------
        shots: int
            Number of samples to take.
        ansatz: qtn.Circuit, optional
            QAOA ansatz to sample. Used to save the marginals found. If None, the QAOA ansatz is instantiated from the optimal parameters.
        wavefunction: qtn.TensorNetwork, optional
            Dense representation of the QAOA circuit. If None, the wavefunction is generated from the circuit.
        opt: str, optional
            Contraction path optimizer. Default is Quimb's default optimizer.
        use_mps: bool, optional
            If True, initialize the QAOA circuit as a Matrix Product State (MPS) instead of as a general tensor networks.

        Returns:
        --------
        counts: Counter
            Counter of the samples
        """

        if ansatz is None:
            if self.theta_opt is not None:
                ansatz = self.instantiate_qaoa(self.theta_opt, use_mps, **ansatz_opts)
            elif self.theta_ini is not None:
                ansatz = self.instantiate_qaoa(self.theta_ini, use_mps, **ansatz_opts)
            else:
                raise ValueError(
                    "Please initialize or initialize and run QAOA before sampling."
                )

        if wavefunction is None:
            wavefunction = ansatz.to_dense(optimize=opt, backend=self.backend)

        # sample the QAOA circuit
        start_sampling = time.time()
        counts = Counter(simulate_counts(wavefunction, shots))
        end_sampling = time.time()

        self.compute_time["sampling"] = end_sampling - start_sampling
        self.counts = counts

        return counts
