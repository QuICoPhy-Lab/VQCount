"""
Implementation of different types of circuits for the quantum alternating operator ansatz (QAOA) with tensor networks.
"""

import numpy as np
import quimb as qu
import quimb.tensor as qtn

from .hamiltonian import hamiltonian


def create_qaoa_circ(
    graph, depth, gammas, betas, qaoa_version, assumptions=[], **circuit_opts
):
    """
    Creates the appropriate QAOA circuit based on user input.

    Parameters
    ----------
    graph: ProblemGraph
        Graph representing the instance of the problem.
    depth: int
        Number of layers of gates to apply (depth 'p').
    gammas: List of floats
        Interaction angles (problem Hamiltonian) for each layer.
    betas: List of floats
        Rotation angles (mixer Hamiltonian) for each layer.
    qaoa_version: str
        Type of QAOA circuit to create. Choose from 'qaoa', 'gm-qaoa', 'vqcount-qaoa', 'vqcount-gm-qaoa'.
    assumptions: List of str
        Qubits to fix in the QAOA circuit for the VQCount algorithm.
    circuit_opts: dict
        Supplied to :class:`~quimb.tensor.circuit.Circuit`.

    Returns
    -------
    circ: Circuit
        QAOA circuit.
    """

    if qaoa_version == "qaoa":
        circ = create_reg_qaoa_circ(graph, depth, gammas, betas, **circuit_opts)
    elif qaoa_version == "gm-qaoa":
        circ = create_gm_qaoa_circ(graph, depth, gammas, betas, **circuit_opts)
    elif qaoa_version == "vqcount-qaoa":
        circ = create_reduced_reg_qaoa_circ(
            graph, depth, gammas, betas, assumptions=assumptions, **circuit_opts
        )
    elif qaoa_version == "vqcount-gm-qaoa":
        circ = create_reduced_gm_qaoa_circ(
            graph, depth, gammas, betas, assumptions=assumptions, **circuit_opts
        )
    else:
        raise ValueError("The QAOA version given is not valid.")

    return circ


def create_reg_qaoa_circ(
    graph,
    depth,
    gammas,
    betas,
    **circuit_opts,
):
    """
    Creates the original QAOA circuit, i.e. with the X-mixer.

    Parameters
    ----------
    graph: ProblemGraph
        Graph representing the instance of the problem.
    depth: int
        Number of layers of gates to apply (depth 'p').
    gammas: iterable of floats
        Interaction angles (problem Hamiltonian) for each layer.
    betas: iterable of floats
        Rotation angles (mixer Hamiltonian) for each layer.
    circuit_opts: dict
        Supplied to :class:`~quimb.tensor.circuit.Circuit`.

    Returns
    -------
    circ: Circuit
        QAOA circuit.
    """

    circuit_opts.setdefault("gate_opts", {})
    circuit_opts["gate_opts"].setdefault("contract", False)

    hamil = hamiltonian(graph)
    n = hamil.num_qubit

    circ = qtn.Circuit(n, **circuit_opts)

    gates = []

    # layer of hadamards to get into plus state
    for i in range(n):
        gates.append((0, "h", i))

    circ.apply_gates(gates)

    for p in range(depth):

        gates = []

        # problem Hamiltonian
        coefs, ops, qubits = hamil.gates()

        for coef, op, qubit in zip(coefs, ops, qubits):
            gates.append((p, op, coef * gammas[p], *qubit))

        # mixer Hamiltonian
        for i in range(n):
            gates.append((p, "rx", 2 * betas[p], i))

        circ.apply_gates(gates)

    return circ


def create_gm_qaoa_circ(
    graph,
    depth,
    gammas,
    betas,
    **circuit_opts,
):
    """
    ONLY WORKS UP TO AROUND 14 QUBITS DUE TO THE APPLICATION OF AN N-CONTROL GATE. USE THE MPS VERSION FOR SIMULATION TO HIGHER NUMBER OF QUBITS. Creates the Grover-Mixer QAOA (GM-QAOA) circuit.

    Parameters
    ----------
    graph: ProblemGraph
        Graph representing the instance of the problem.
    depth: int
        Number of layers of gates to apply (depth 'p').
    gammas: iterable of floats
        Interaction angles (problem Hamiltonian) for each layer.
    betas: iterable of floats
        Rotation angles (mixer Hamiltonian) for each layer.
    circuit_opts: dict
        Supplied to :class:`~quimb.tensor.circuit.Circuit`.

    Returns
    -------
    circ: Circuit
        QAOA circuit.
    """

    circuit_opts.setdefault("gate_opts", {})
    circuit_opts["gate_opts"].setdefault("contract", False)

    hamil = hamiltonian(graph)
    n = hamil.num_qubit

    circ = qtn.Circuit(n, **circuit_opts)

    gates = []

    # layer of hadamards to get into plus state
    for i in range(n):
        gates.append((0, "h", i))

    circ.apply_gates(gates)

    for p in range(depth):

        gates = []

        # problem Hamiltonian
        coefs, ops, qubits = hamil.gates()
        for coef, op, qubit in zip(coefs, ops, qubits):
            gates.append((p, op, coef * gammas[p], *qubit))

        # mixer Hamiltonian
        for i in range(n):
            gates.append((p, "h", i))
            gates.append((p, "x", i))

        circ.apply_gates(gates)

        # multi-control phase-shift gate
        ncrz_gate = np.eye(2**n, dtype=complex)
        ncrz_gate[-1, -1] = np.exp(-2j * betas[p])

        circ.apply_gate_raw(ncrz_gate, range(0, n), gate_round=p, tags="NCRZ")

        gates = []

        for i in range(n):
            gates.append((p, "x", i))
            gates.append((p, "h", i))

        circ.apply_gates(gates)

    return circ


def create_reduced_reg_qaoa_circ(
    graph,
    depth,
    gammas,
    betas,
    assumptions,
    **circuit_opts,
):
    """
    Creates the reduced orignal QAOA circuit, i.e. with the X-mixer, following the self-reduction procedure of the VQCount algorithm.

    Parameters
    ----------
    graph: ProblemGraph
        Graph representing the instance of the problem.
    depth: int
        Number of layers of gates to apply (depth 'p').
    gammas: iterable of floats
        Interaction angles (problem Hamiltonian) for each layer.
    betas: iterable of floats
        Rotation angles (mixer Hamiltonian) for each layer.
    assumptions: iterable of str
        Qubits to fix in the QAOA circuit for the VQCount algorithm.
    circuit_opts: dict
        Supplied to :class:`~quimb.tensor.circuit.Circuit`.

    Returns
    -------
    circ: Circuit
        QAOA circuit.
    """

    circuit_opts.setdefault("gate_opts", {})
    circuit_opts["gate_opts"].setdefault("contract", False)

    hamil = hamiltonian(graph)
    n = hamil.num_qubit

    circ = qtn.Circuit(n, **circuit_opts)

    gates = []

    # layer of hadamards to get into plus state
    for i in range(len(assumptions), n):
        gates.append((0, "h", i))

    for i, var in enumerate(assumptions):
        if var == str(1):
            gates.append((0, "x", i))

    circ.apply_gates(gates)

    for p in range(depth):

        gates = []

        # problem Hamiltonian
        coefs, ops, qubits = hamil.gates()

        for coef, op, qubit in zip(coefs, ops, qubits):
            gates.append((p, op, coef * gammas[p], *qubit))

        # mixer Hamiltonian
        for i in range(len(assumptions), n):
            gates.append((p, "rx", 2 * betas[p], i))

        circ.apply_gates(gates)

    return circ


def create_reduced_gm_qaoa_circ(
    graph,
    depth,
    gammas,
    betas,
    assumptions,
    **circuit_opts,
):
    """
    ONLY WORKS UP TO AROUND 14 QUBITS DUE TO THE APPLICATION OF AN N-CONTROL GATE. USE THE MPS VERSION FOR SIMULATION TO HIGHER NUMBER OF QUBITS. Creates the reduced Grover-Mixer QAOA (GM-QAOA) circuit following the self-reduction procedure of the VQCount algorithm.

    Parameters
    ----------
    graph: ProblemGraph
        Graph representing the instance of the problem.
    depth: int
        Number of layers of gates to apply (depth 'p').
    gammas: iterable of floats
        Interaction angles (problem Hamiltonian) for each layer.
    betas: iterable of floats
        Rotation angles (mixer Hamiltonian) for each layer.
    assumptions: iterable of str
        Qubits to fix in the GM-QAOA circuit for the VQCount algorithm.
    circuit_opts: dict
        Supplied to :class:`~quimb.tensor.circuit.Circuit`.

    Returns
    -------
    circ: Circuit
        QAOA circuit.
    """

    circuit_opts.setdefault("gate_opts", {})
    circuit_opts["gate_opts"].setdefault("contract", False)

    hamil = hamiltonian(graph)
    n = hamil.num_qubit

    circ = qtn.Circuit(n, **circuit_opts)

    gates = []

    # layer of hadamards to get into plus state
    for i in range(len(assumptions), n):
        gates.append((0, "h", i))

    for i, var in enumerate(assumptions):
        if var == str(1):
            gates.append((0, "x", i))

    circ.apply_gates(gates)

    for p in range(depth):

        gates = []

        # problem Hamiltonian
        coefs, ops, qubits = hamil.gates()
        for coef, op, qubit in zip(coefs, ops, qubits):
            gates.append((p, op, coef * gammas[p], *qubit))

        # mixer Hamiltonian
        for i in range(len(assumptions), n):
            gates.append((p, "h", i))
            gates.append((p, "x", i))

        circ.apply_gates(gates)

        # multi-control phase-shift gate
        ncrz_gate = qu.ncontrolled_gate(
            n - len(assumptions) - 1, np.diag([1, np.exp(-2j * betas[p])])
        )

        circ.apply_gate_raw(
            ncrz_gate, range(len(assumptions), n), gate_round=p, tags="NCRZ"
        )

        gates = []

        for i in range(len(assumptions), n):
            gates.append((p, "x", i))
            gates.append((p, "h", i))

        circ.apply_gates(gates)

    return circ
