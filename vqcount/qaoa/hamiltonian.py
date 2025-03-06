"""
Implementation of the problem Hamiltonian of QAOA for different problems.
"""

import quimb as qu


def hamiltonian(graph):
    """
    Create the problem Hamiltonian for a given problem.

    Parameters:
    -----------
    graph: ProblemGraph
        Graph representing the instance of the problem

    Returns:
    --------
    hamiltonian: Hamiltonian
        Hamiltonian of the problem instance.

    """

    if graph.problem == "nae3sat":
        return IsingHamiltonian(graph)

    if graph.problem == "mono1in3sat":
        return IsingWithFieldHamiltonian(graph)

    raise ValueError("The problem given is not implemented yet.")


class ProblemHamiltonian:
    """
    Base class for problem Hamiltonians. Each specific problem Hamiltonian should inherit from this class and implement the methods defined there.
    """

    def __init__(self, graph):
        """
        Initializes the problem Hamiltonian with a problem graph.

        Parameters:
        -----------
        graph: ProblemGraph
            Graph representing the instance of the problem.
        """
        self.graph = graph

        self.problem_hamiltonian()

    @property
    def num_qubit(self):
        """
        Returns the number of qubits needed to represent the problem.
        """
        raise NotImplementedError("This method should be implemented in a subclass")

    def problem_hamiltonian(self):
        """
        Computes the problem Hamiltonian. Add the gates as an attribute to the ProblemHamiltonian.
        """
        raise NotImplementedError("This method should be implemented in a subclass")

    def operators(self):
        """
        Returns a list of the operators of the problem Hamiltonian. Necessary for the contraction of the QAOA ansatz (see contraction.py).

        Returns:
        --------
        ops: list[str]
            List of operators of problem Hamiltonian.
        qubits: list[Tuple[int]]
            List of qubits on which the gates act.
        """
        raise NotImplementedError("This method should be implemented in a subclass")

    def gates(self):
        """
        Returns the gates of the problem Hamiltonian. Necessary for the QAOA circuit and QAOA MPS creation.

        Returns:
        --------
        coefs: list[float]
            List of coefficients of the gates.
        ops: list[str]
            List of gates.
        qubits: list[Tuple[int]]
            List of qubits on which the gates act.
        """
        raise NotImplementedError("This method should be implemented in a subclass")


class IsingHamiltonian(ProblemHamiltonian):
    r"""
    Implementation of the Ising Hamiltonian without local field

        H = \sum_{i, j} J_{ij} \sigma_i^z \sigma_j^z,

    where interaction terms are given by the problem instance. The method operator() returns the necessary list of operators for the contraction of the QAOA ansatz (see contraction.py). The method gates() returns the necessary list of gates for the QAOA circuit and QAOA MPS creation (see circuit.py and mps.py).

    Parameters:
    -----------
    G: ProblemGraph
        Graph representing the instance of the problem.
    """

    @property
    def num_qubit(self):
        """Number of qubits to represent the problem."""
        return self.graph.num_nodes

    def problem_hamiltonian(self):
        """
        Computes a dictionnary of RZZ gates with its parameter representing the problem Hamiltonian. Add the RZZ gates as an attribute to the ProblemHamiltonian.
        """

        rzz_gates = {}

        for edge, weight in list(self.graph.terms.items()):
            rzz_gates[edge] = weight

        self.rzz_gates = rzz_gates

    def operators(self):
        """
        Returns a list of the operators the problem Hamiltonian. Necessary for the contraction of the QAOA ansatz (see contraction.py).

        Returns:
        --------
        ops: list[str]
            List of operators of problem Hamiltonian.
        qubits: list[Tuple[int]]
            List of qubits on which the gates act.
        """

        qubits = []
        ops = []

        for qubit, value in self.rzz_gates.items():
            qubits.append(qubit)
            ops.append(value * qu.pauli("Z") & qu.pauli("Z"))

        return ops, qubits

    def gates(self):
        """
        Returns the gates of the problem Hamiltonian. Necessary for the QAOA circuit and QAOA MPS creation (see circuit.py and mps.py).

        Returns:
        --------
        coefs: list[float]
            List of coefficients of the gates.
        ops: list[str]
            List of gates.
        qubits: list[Tuple[int]]
            List of qubits on which the gates act.
        """

        qubits = []
        ops = []
        coefs = []

        for qubit, value in self.rzz_gates.items():
            qubits.append(qubit)
            ops.append("rzz")
            coefs.append(1 / 2 * value)

        return coefs, ops, qubits


class IsingWithFieldHamiltonian(ProblemHamiltonian):
    r"""
    Implementation of the Ising Hamiltonian with local field

        H = \sum_{i, j} J_{ij} \sigma_i^z \sigma_j^z + \sum_i h_i \sigma_i^z,

    where interactions and field terms are given by the problem instance. The method operator() returns the necessary list of operators for the contraction of the QAOA ansatz (see contraction.py). The method gates() returns the necessary list of gates for the QAOA circuit and QAOA MPS creation (see circuit.py and mps.py).

    Parameters:
    -----------
    G: ProblemGraph
        Graph representing the instance of the problem.
    """

    @property
    def num_qubit(self):
        """Number of qubits to represent the problem."""
        return self.graph.num_nodes

    def problem_hamiltonian(self):
        """
        Computes a dictionnary of RZ and RZZ gates with its parameter representing the problem Hamiltonian.

        Returns:
        --------
        rz_gates: dict[int, float]
            RZ gates representing the interaction terms of problem Hamiltonian, where the keys are the qubits and the values are the parameters of the RZ gates.
        rzz_gates: dict[tuple[int, int], float]
            RZZ gates representing the field terms of the problem Hamiltonian, where the keys are the edges of the graph and the values are the parameters of the RZZ gates.
        """

        rz_gates = {}
        rzz_gates = {}

        for edge, weight in list(self.graph.terms.items()):
            if len(edge) == 2:
                rzz_gates[edge] = weight
            elif len(edge) == 1:
                rz_gates[edge] = weight

        self.rz_gates = rz_gates
        self.rzz_gates = rzz_gates

    def operators(self):
        """
        Returns a list of the operators the problem Hamiltonian. Necessary for the contraction of the QAOA ansatz (see contraction.py).

        Returns:
        --------
        ops: list[str]
            List of operators of problem Hamiltonian.
        qubits: list[Tuple[int]]
            List of qubits on which the gates act.
        """

        qubits = []
        ops = []
        localham_rz = {}
        localham_rzz = {}

        for qubit, value in self.rzz_gates.items():
            qubits.append(qubit)
            ops.append(value * qu.pauli("Z") & qu.pauli("Z"))
            localham_rzz[qubit] = value * qu.pauli("Z") & qu.pauli("Z")

        for qubit, value in self.rz_gates.items():
            qubits.append(qubit)
            ops.append(value * qu.pauli("Z"))
            localham_rz[qubit[0]] = value * qu.pauli("Z")

        localham = qu.tensor.LocalHamGen(localham_rzz, H1=localham_rz)

        qubits = []
        ops = []

        for qubit, op in localham.items():
            qubits.append(qubit)
            ops.append(op)

        return ops, qubits

    def gates(self):
        """
        Returns the gates of the problem Hamiltonian. Necessary for the QAOA circuit and QAOA MPS creation (see circuit.py and mps.py).

        Returns:
        --------
        coefs: list[float]
            List of coefficients of the gates.
        ops: list[str]
            List of gates.
        qubits: list[Tuple[int]]
            List of qubits on which the gates act.
        """

        qubits = []
        ops = []
        coefs = []

        for qubit, value in self.rzz_gates.items():
            qubits.append(qubit)
            ops.append("rzz")
            coefs.append(2 * value)

        for qubit, value in self.rz_gates.items():
            qubits.append(qubit)
            ops.append("rz")
            coefs.append(2 * value)

        return coefs, ops, qubits
