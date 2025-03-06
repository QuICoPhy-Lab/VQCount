"""
Instantiate the QAOA ansatz (circuit or MPS).
"""

import quimb.tensor as qtn

from .circuit import create_qaoa_circ
from .mps import create_qaoa_mps


def instantiate_ansatz(
    graph, depth, gammas, betas, qaoa_version, mps=False, **ansatz_opts
):
    """
    Instantiate the QAOA ansatz (circuit or MPS).

    Parameters:
    -----------
    theta: np.ndarray
        Parameters of the QAOA circuit to instantiate.
    mps: bool, optional
        If True, initialize the QAOA circuit as a Matrix Product State (MPS) instead of as a general tensor networks.

    Returns:
    --------
    circ: qtn.Circuit
        QAOA circuit instantiated.
    """

    # create the QAOA circuit or MPS
    if mps:
        psi0 = create_qaoa_mps(
            graph,
            depth,
            gammas,
            betas,
            qaoa_version,
            **ansatz_opts,
        )
        circ = qtn.Circuit(psi0=psi0)
    else:
        circ = create_qaoa_circ(
            graph,
            depth,
            gammas,
            betas,
            qaoa_version=qaoa_version,
            **ansatz_opts,
        )

    return circ
