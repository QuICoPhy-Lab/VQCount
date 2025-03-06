"""
Routines for the approximate counting of a SAT problem with the JVV algorithm (see https://www.sciencedirect.com/science/article/pii/030439758690174X).
"""

import numpy as np


def find_next_jvv_assignment(counts, assignment):
    """
    In a tree of partial assignements, compute the probability of the solutions of the following subtree and fix the bit with the highest probability. This function is used for approximate counting of a SAT problem with the JVV algorithm. The count should only include solutions with the same prefix as the current assignment.

    Parameters
    ----------
    counts : dict
        Dictionary of the bitstrings and their counts, where the bitstrings are solutions to the SAT problem.
    assignment : str
        Bitstring of the current assignement.

    Returns
    -------
    prob : float
        Probability of the solutions of the following subtree.
    next_assignment : str
        Bit of the next assignment with the highest probability.
    """

    bitstrings = list(counts.keys())
    assignment_position = len(assignment)

    # test if all the state are non-solutions
    if len(counts) == 0:
        raise ValueError("All the states are non-solutions.")

    count_one = 0
    count_zero = 0

    # count the number of solutions in the following subtree
    for j in bitstrings:
        if j[0 : assignment_position + 1] == (assignment + str(1)):
            count_one += counts[j]

        elif j[0 : assignment_position + 1] == (assignment + str(0)):
            count_zero += counts[j]

    # only count solutions with the same prefix as the current assignment
    count_total = count_one + count_zero

    # find the bit with the highest probability
    if count_one > count_zero:
        prob = count_one / count_total
        next_assignment = "1"
    else:
        prob = count_zero / count_total
        next_assignment = "0"

    return prob, next_assignment


def update_exact_sol_pdf(exact_sol_pdf, assignments):
    """
    Update the probability distribution of the solutions by setting the probability of the bitstrings that do not satisfy the assignments to zero. The assignments are qubits that are fixed. This function assumes that the qubits are fixed to 0 or 1 in order from the first qubit to the last qubit.

    Parameters
    ----------
    exact_sol_pdf : list
        Probability distribution of the solutions.
    assignments : list
        Bitstring of the current assignement.

    Returns
    -------
    exact_sol_pdf : numpy.ndarray
        Updated probability distribution of the solutions.
    """

    num_state = len(exact_sol_pdf)
    num_qubit = int(np.log2(num_state))

    exact_sol_pdf = np.copy(exact_sol_pdf)

    # generate all possible bitstrings of size n
    all_bitstrings = np.arange(num_state, dtype=np.uint64)

    for i, assignment in enumerate(assignments):
        # find bitstrings where the bit at the specified position is k
        mask = (all_bitstrings >> (num_qubit - i - 1)) & 1 == 1 - int(assignment)

        # filter the probability distribution based on the mask
        exact_sol_pdf[mask] = 0

    # normalize the probability distribution
    exact_sol_pdf = exact_sol_pdf / np.sum(exact_sol_pdf)

    return exact_sol_pdf
