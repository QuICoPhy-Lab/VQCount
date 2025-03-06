"""
Implementation of different types of Matrix Product States (MPS) representation for the quantum alternating operator ansatz (QAOA) with the MPS/MPO method.
"""

import numpy as np
import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.circuit import rx_gate_param_gen, rz_gate_param_gen, rzz_param_gen
from quimb.tensor.tensor_builder import MPS_computational_state

from .hamiltonian import hamiltonian


def create_qaoa_mps(graph, depth, gammas, betas, qaoa_version, assumptions=[]):
    """
    Creates the appropriate QAOA MPS based on user input.

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
    qaoa_version: str
        Type of QAOA MPS to create. Choose from 'qaoa', 'gm-qaoa', 'vqcount-qaoa', 'vqcount-gm-qaoa'.
    assumptions: iterable of str
        Qubits to fix in the QAOA circuit for the VQCount algorithm.

    Returns
    -------
    psi: MatrixProductState
        QAOA MPS.
    """

    if qaoa_version == "qaoa":
        psi = create_reg_qaoa_mps(graph, depth, gammas, betas)
    elif qaoa_version == "gm-qaoa":
        psi = create_gm_qaoa_mps(graph, depth, gammas, betas)
    elif qaoa_version == "vqcount-qaoa":
        psi = create_reduced_reg_qaoa_mps(
            graph, depth, gammas, betas, assumptions=assumptions
        )
    elif qaoa_version == "vqcount-gm-qaoa":
        psi = create_reduced_gm_qaoa_mps(
            graph, depth, gammas, betas, assumptions=assumptions
        )
    else:
        raise ValueError("The QAOA version given is not valid.")

    return psi


def create_reg_qaoa_mps(
    graph,
    depth,
    gammas,
    betas,
):
    """
    Creates the original QAOA MPS, i.e. with the X-mixer.

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

    Returns
    -------
    psi: MatrixProductState
        QAOA MPS.
    """

    hamil = hamiltonian(graph)
    n = hamil.num_qubit

    # initial MPS
    psi0 = MPS_computational_state("0" * n, tags="PSI0")

    coefs, ops, qubits = hamil.gates()

    # layer of hadamards to get into plus state
    for i in range(n):
        psi0.gate_(qu.hadamard(), i, contract="swap+split", tags="H")

    for p in range(depth):

        # problem Hamiltonian
        for coef, op, qubit in zip(coefs, ops, qubits):
            if op == "rzz":
                psi0.gate_with_auto_swap_(rzz_param_gen([coef * gammas[p]]), qubit)

            elif op == "rz":
                psi0.gate_(
                    rz_gate_param_gen([coef * gammas[p]]),
                    qubit,
                    contract="swap+split",
                    tags="RZ",
                )

        # mixer Hamiltonian
        for i in range(n):
            psi0.gate_(
                rx_gate_param_gen([2 * betas[p]]), i, contract="swap+split", tags="RX"
            )

        psi0.normalize()

    return psi0


def create_gm_qaoa_mps(
    graph,
    depth,
    gammas,
    betas,
):
    """
    Creates the Grover-Mixer QAOA (GM-QAOA) MPS.

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

    Returns
    -------
    psi: MatrixProductState
        QAOA MPS.
    """

    hamil = hamiltonian(graph)
    n = hamil.num_qubit

    # initial MPS
    psi0 = MPS_computational_state("0" * n, tags="PSI0")

    # layer of hadamards to get into plus state
    for i in range(n):
        psi0.gate_(qu.hadamard(), i, contract="swap+split", tags="H")

    for p in range(depth):

        # problem Hamiltonian
        coefs, ops, qubits = hamil.gates()

        for coef, op, qubit in zip(coefs, ops, qubits):
            if op == "rzz":
                psi0.gate_with_auto_swap_(rzz_param_gen([coef * gammas[p]]), qubit)

            elif op == "rz":
                psi0.gate_(
                    rz_gate_param_gen([coef * gammas[p]]),
                    qubit,
                    contract="swap+split",
                    tags="RZ",
                )

        # mixer Hamiltonian
        for i in range(n):
            psi0.gate_(qu.hadamard(), i, contract="swap+split", tags="H")
            psi0.gate_(qu.pauli("X"), i, contract="swap+split", tags="X")

        # multi-control phase-shift gate
        ncrz_gate = [CP_TOP()]
        for i in range(n - 2):
            ncrz_gate.append(ADD())
        ncrz_gate.append(RZ(2 * betas[p]))

        ncrz_gate = qtn.tensor_1d.MatrixProductOperator(ncrz_gate, "udrl", tags="NCRZ")

        psi = ncrz_gate.apply(psi0)
        del psi0

        for i in range(n):
            psi.gate_(qu.pauli("X"), i, contract="swap+split", tags="X")
            psi.gate_(qu.hadamard(), i, contract="swap+split", tags="H")

        psi.normalize()

        psi0 = psi
        del psi

    return psi0


def create_reduced_reg_qaoa_mps(
    graph,
    depth,
    gammas,
    betas,
    assumptions,
):
    """
    Creates the reduced orignal QAOA MPS, i.e. with the X-mixer, following the self-reduction procedure of the VQCount algorithm.

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

    Returns
    -------
    psi: MatrixProductState
        QAOA MPS.
    """

    hamil = hamiltonian(graph)
    n = hamil.num_qubit

    # initial MPS
    psi0 = MPS_computational_state("0" * n, tags="PSI0")

    coefs, ops, qubits = hamil.gates()

    for i, var in enumerate(assumptions):
        if var == str(1):
            psi0.gate_(qu.pauli("X"), i, contract="swap+split", tags="X")

    # layer of hadamards to get into plus state
    for i in range(len(assumptions), n):
        psi0.gate_(qu.hadamard(), i, contract="swap+split", tags="H")

    for p in range(depth):

        # problem Hamiltonian
        for coef, op, qubit in zip(coefs, ops, qubits):
            if op == "rzz":
                psi0.gate_with_auto_swap_(rzz_param_gen([coef * gammas[p]]), qubit)

            elif op == "rz":
                psi0.gate_(
                    rz_gate_param_gen([coef * gammas[p]]),
                    qubit,
                    contract="swap+split",
                    tags="RZ",
                )

        # mixer Hamiltonian
        for i in range(len(assumptions), n):
            psi0.gate_(
                rx_gate_param_gen([2 * betas[p]]), i, contract="swap+split", tags="RX"
            )

        psi0.normalize()

    return psi0


def create_reduced_gm_qaoa_mps(
    graph,
    depth,
    gammas,
    betas,
    assumptions,
):
    """
    Creates the reduced Grover-Mixer QAOA (GM-QAOA) circuit following the self-reduction procedure of the VQCount algorithm.

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

    Returns
    -------
    psi: MatrixProductState
        QAOA MPS.
    """

    hamil = hamiltonian(graph)
    n = hamil.num_qubit

    # initial MPS
    psi0 = MPS_computational_state("0" * n, tags="PSI0")

    for i, var in enumerate(assumptions):
        if var == str(1):
            psi0.gate_(qu.pauli("X"), i, contract="swap+split", tags="X")

    # layer of hadamards to get into plus state
    for i in range(len(assumptions), n):
        psi0.gate_(qu.hadamard(), i, contract="swap+split", tags="H")

    for p in range(depth):

        # problem Hamiltonian
        coefs, ops, qubits = hamil.gates()

        for coef, op, qubit in zip(coefs, ops, qubits):
            if op == "rzz":
                psi0.gate_with_auto_swap_(rzz_param_gen([coef * gammas[p]]), qubit)

            elif op == "rz":
                psi0.gate_(
                    rz_gate_param_gen([coef * gammas[p]]),
                    qubit,
                    contract="swap+split",
                    tags="RZ",
                )

        # mixer Hamiltonian
        for i in range(len(assumptions), n):
            psi0.gate_(qu.hadamard(), i, contract="swap+split", tags="H")
            psi0.gate_(qu.pauli("X"), i, contract="swap+split", tags="X")

        ncrz_gate = []

        # multi-control phase-shift gate
        if len(assumptions) != 0:
            ncrz_gate.append(ID_TOP())

            for i in range(len(assumptions) - 1):
                ncrz_gate.append(ID_MID())

            if n - len(assumptions) > 1:
                ncrz_gate.append(CP_MID())

        else:
            ncrz_gate.append(CP_TOP())

        for i in range(n - len(assumptions) - 2):
            ncrz_gate.append(ADD())

        if n - len(assumptions) == 1:
            ncrz_gate.append(RZ_BOT(2 * betas[p]))
        else:
            ncrz_gate.append(RZ(2 * betas[p]))

        ncrz_gate = qtn.tensor_1d.MatrixProductOperator(ncrz_gate, "udrl", tags="NCRZ")

        psi = ncrz_gate.apply(psi0)
        del psi0

        for i in range(len(assumptions), n):
            psi.gate_(qu.pauli("X"), i, contract="swap+split", tags="X")
            psi.gate_(qu.hadamard(), i, contract="swap+split", tags="H")

        psi.normalize()

        psi0 = psi
        del psi

    return psi0


def RZ(beta):
    "Z-Rotation gate"
    rz = np.zeros((2, 2, 1, 2), dtype="complex")
    rz[0, 0, 0, 0] = 1
    rz[1, 1, 0, 0] = 1
    rz[0, 0, 0, 1] = 1
    rz[1, 1, 0, 1] = np.exp(-1j * beta)
    return rz


def RZ_BOT(beta):
    "Z-Rotation gate"
    rz = np.zeros((2, 2, 1, 2), dtype="complex")
    rz[0, 0, 0, 0] = 1
    rz[1, 1, 0, 0] = np.exp(-1j * beta)
    rz[0, 0, 0, 1] = 1
    rz[1, 1, 0, 1] = np.exp(-1j * beta)
    return rz


def ADD():
    """ADD gate"""
    add = np.zeros((2, 2, 2, 2), dtype="complex")
    add[0, 0, 0, 0] = 1
    add[0, 0, 1, 0] = 0
    add[0, 0, 1, 1] = 0
    add[0, 0, 0, 1] = 1
    add[1, 1, 0, 0] = 1
    add[1, 1, 1, 0] = 0
    add[1, 1, 0, 1] = 0
    add[1, 1, 1, 1] = 1
    return add


def CP_TOP():
    """COPY gate"""
    cp = np.zeros((2, 2, 2, 1), dtype="complex")
    cp[0, 0, 0, 0] = 1
    cp[1, 1, 1, 0] = 1
    return cp


def CP_MID():
    """COPY gate"""
    cp = np.zeros((2, 2, 2, 2), dtype="complex")
    cp[0, 0, 0, 0] = 1
    cp[1, 1, 1, 0] = 1
    return cp


def ID_TOP():
    """Identity gate"""
    id = np.zeros((2, 2, 2, 1), dtype="complex")
    id[0, 0, 0, 0] = 1
    id[1, 1, 0, 0] = 1
    return id


def ID_MID():
    """Identity gate"""
    id = np.zeros((2, 2, 2, 2), dtype="complex")
    id[0, 0, 0, 0] = 1
    id[1, 1, 0, 0] = 1
    return id
