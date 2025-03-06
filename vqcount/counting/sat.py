"""
Routines for analysing SAT problems.
"""

import tempfile
import warnings

import numpy as np
from pysat.formula import CNF
from pysat.solvers import Glucose4

from ..utils.ganak_wrapper import ganak_counter


def verify_sat_state(bitstring, cnf):
    """
    Verify if a bitstring is a solution of the given SAT formula.

    Parameters
    ----------
    bitstring : str
        Bitstring of the state.
    cnf : list or np.ndarray
        List of clauses of the SAT formula, with one-based indexing. Negative integers represent negated literals.

    Returns
    -------
    bool
        True if the state is a solution, False otherwise.
    """

    if len(cnf) == 0:
        raise ValueError("The CNF formula is empty.")

    # convert the bitstring into a NumPy array of boolean values
    bit_values = np.array(list(bitstring), dtype=int) > 0

    # iterate through each clause in the formula
    for clause in cnf:
        clause_satisfied = False

        # check if at least one literal in the clause is satisfied
        for literal in clause:
            index = abs(literal) - 1  # Convert to zero-based index
            is_positive = literal > 0

            # Use NumPy's efficient indexing and boolean operations
            literal_value = bit_values[index]
            if (literal_value and is_positive) or (
                not literal_value and not is_positive
            ):
                clause_satisfied = True
                break

        # If any clause is not satisfied, the whole formula is not satisfied
        if not clause_satisfied:
            return False

    # If all clauses are satisfied, the formula is satisfied
    return True


def count_sat(cnf):
    """
    Count the number of solutions of a SAT problem with the Ganak solver (see https://github.com/meelgroup/ganak).

    Parameters
    ----------
    cnf : list or np.ndarray
        CNF formula.

    Returns
    -------
    num_sol : int
        Number of solutions.
    """

    if len(cnf) == 0:
        raise ValueError("The CNF formula is empty.")

    if isinstance(cnf, np.ndarray):
        cnf = cnf.tolist()

    cnf = CNF(from_clauses=cnf)

    # create a temporary file to store the CNF formula
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        cnf.to_file(temp_file.name)

        # count the number of solutions with Ganak
        num_sol = ganak_counter(temp_file.name)

    return num_sol


def enumerate_sat(cnf, compute_pdf=True):
    """
    Enumerate all the solutions of a SAT problem with the PySat solver (see https://pysathq.github.io/). The probability distribution of the solutions is also computed.

    Parameters
    ----------
    cnf : list
        CNF formula.
    compute_pdf : bool, optional
        Compute the probability distribution of the solutions. The default is True.

    Returns
    -------
    sol_bitstring : list
        List of the bitstring of the solutions.
    sol_pdf : numpy.ndarray
        Probability distribution of the solutions.
    """

    if len(cnf) == 0:
        raise ValueError("The CNF formula is empty.")

    if isinstance(cnf, np.ndarray):
        cnf = cnf.tolist()

    # create CNF object
    cnf = CNF(from_clauses=cnf)

    # initialize solver
    solver = Glucose4()
    solver.append_formula(cnf.clauses)

    # enumerate all the solutions
    sol_bitstring = []
    sol_int = []
    for sol in solver.enum_models():
        bitstring = "".join(["1" if x > 0 else "0" for x in sol])

        sol_bitstring.append(bitstring)
        sol_int.append(int(bitstring, 2))

    # delete solver to free memory
    solver.delete()

    numvar = cnf.nv

    # compute the probability distribution of the solutions
    if compute_pdf:
        # only consider satisfiable problems
        if len(sol_int) != 0:
            sol_pdf = np.zeros(2**numvar)

            sol_pdf[sol_int] = 1 / len(sol_int)

        else:
            sol_pdf = None
            warnings.warn("The CNF formula is unsatisfiable.")

    else:
        sol_pdf = None

    return sol_bitstring, sol_pdf


def find_prob_sat(cnf, counts):
    """
    Find the probability of the solutions found in a given set of bitstrings for a SAT problem.

    Parameters
    ----------
    cnf : list
        CNF formula.
    counts : dict
        Dictionary of the bitstrings and their counts.

    Returns
    -------
    prob_sol : float
        Probability of the solutions.
    """

    num_sol = 0
    count_sol = 0

    for bitstring, value in counts.items():
        if verify_sat_state(bitstring, cnf):
            num_sol += 1
            count_sol += value

    prob_sol = count_sol / sum(counts.values())

    return prob_sol


def find_tvd(pdf_a, pdf_b):
    """
    Calculate the total variation distance between to probability distributions.

    Parameters
    ----------
    pdf_a : numpy.ndarray
        Probability distribution.
    pdf_b : numpy.ndarray
        Probability distribution.

    Returns
    -------
    tvd : float
        Total variation distance.
    """

    if pdf_b is not None and pdf_a is not None:
        tvd = sum(abs(pdf_a - pdf_b)) / 2
    else:
        tvd = None
        warnings.warn("One or both of the probability distributions is None.")

    return tvd


def postselect_approx_pdf(approx_pdf, exact_pdf):
    """
    Postselect the approximate probability distribution by setting the probability of the bitstrings that are not solutions to zero.

    Parameters
    ----------
    approx_pdf : numpy.ndarray
        Approximate probability distribution.
    exact_pdf : numpy.ndarray
        Exact probability distribution.

    Returns
    -------
    postselected_approx_pdf : numpy.ndarray
        Pruned approximate probability distribution.
    """

    if exact_pdf is not None or approx_pdf is not None:
        # remove the bitstrings that are not solutions
        postselected_approx_pdf = np.copy(approx_pdf)
        postselected_approx_pdf[exact_pdf == 0] = 0

        # normalize the probability distributions
        postselected_approx_pdf = postselected_approx_pdf / np.sum(
            postselected_approx_pdf
        )

    else:
        postselected_approx_pdf = None

    return postselected_approx_pdf
