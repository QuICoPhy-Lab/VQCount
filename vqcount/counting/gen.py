"""
Methods for the generation of combinatorial problems.
"""

import json
import os

import numpy as np

from .problem import PositiveNaeThreeSatGraph, PositiveOneInThreeSatGraph, ProblemGraph
from .sat import count_sat, enumerate_sat


def generate_satisfiable_sat_problem(problem, num_nodes, alpha):
    """
    Generate a SAT problem and find its exact solutions. Make sure it has at least one solution.

    Parameters
    ----------
    problem : str
        Type of SAT problem to generate. Choose between "nae3sat" and "1in3sat". Only supports "nae3sat" at alpha=1 or alpha=2 and "1in3sat" at alpha=2/3.
    num_nodes : int
        Number of nodes of the problem.
    alpha : float
        Ratio of clauses to nodes.

    Returns
    -------
    graph : ProblemGraph
        SAT problem.
    """

    numcau = int(alpha * num_nodes)

    # make sure that the problem has at least one solution
    for _ in range(100):
        if problem == "nae3sat":
            graph = PositiveNaeThreeSatGraph(num_nodes, numcau, int(3 * alpha), 3)
        elif problem == "1in3sat":
            graph = PositiveOneInThreeSatGraph(num_nodes)
        else:
            raise ValueError("Please choose an appropriate type of problem.")

        # finding the exact solutions of the problem
        exact_num_sol = count_sat(graph.cnf)

        # avoid memory kill for large problems
        if num_nodes < 31:
            sol_bitstrings, exact_sol_pdf = enumerate_sat(graph.cnf, compute_pdf=True)
        else:
            sol_bitstrings, exact_sol_pdf = enumerate_sat(graph.cnf, compute_pdf=False)

        # check if the number of solutions is consistent with the list of solutions
        if len(sol_bitstrings) != exact_num_sol:
            raise ValueError(
                "The number of exact solutions is not consistent with the list of solutions."
            )

        graph.exact_num_sol = exact_num_sol
        graph.exact_sol_bit = sol_bitstrings
        graph.exact_sol_pdf = exact_sol_pdf

        if exact_num_sol != 0:
            return graph

        print("The initial problem is unsatisfiable: change of seed.")

    raise ValueError("All generated problems are unsatisfiables.")


def generate_sat_problems(file_path, problem, num_node, alpha, num_graph):
    """
    Generate a set of satisfiable SAT problems and save them in a file.

    Parameters
    ----------
    file_path : str
        The path to the file where the problems will be saved.
    problem : str
        The type of SAT problem to generate. Choose between "nae3sat" and "1in3sat". Only supports "nae3sat" at alpha=1 or alpha=2 and "1in3sat" at alpha=2/3.
    num_node : int
        Number of nodes of the problem.
    alpha : list
        Ratio of clauses to nodes.
    num_graph : int
        The number of problems to generate for each ratio.
    """

    cnf = []
    for g in range(1, num_graph + 1):

        i = 0
        while i < 50:
            data = {}

            graph = generate_satisfiable_sat_problem(problem, num_node, alpha)

            # make sure that the problem was not already generated
            if sorted(graph.cnf_ini.tolist()) not in cnf:
                data["problem"] = graph.problem
                data["num_nodes"] = graph.num_nodes
                data["alpha"] = alpha
                data["graph"] = g
                data["cnf_ini"] = graph.cnf_ini.tolist()
                data["cnf"] = graph.cnf.tolist()
                data["edges"] = graph.edges.tolist()
                data["terms"] = map_keys(graph.terms)
                data["exact_num_sol"] = int(graph.exact_num_sol)
                data["exact_sol_bit"] = graph.exact_sol_bit

                cnf.append(sorted(graph.cnf_ini.tolist()))

                prob_file_path = os.path.join(
                    file_path,
                    "q"
                    + str(num_node)
                    + "-a"
                    + str(round(alpha, 3))
                    + "-g"
                    + str(g)
                    + ".json",
                )

                with open(prob_file_path, "w") as f:
                    f.write(json.dumps(data))

                break

            print("The problem was already generated.")

            i += 1
            if i == 50:
                raise ValueError("The problems were not all generated.")


def load_sat_problem(file_path):
    """
    Load a SAT problem from a file.

    Parameters
    ----------
    file_path : str
        The path to the file where the problems are saved.

    Returns
    -------
    graph : ProblemGraph
        The SAT problem.
    """

    with open(file_path, "r") as f:
        data = json.load(f)
        graph = ProblemGraph()
        graph.problem = data["problem"]
        graph.num_nodes = data["num_nodes"]
        graph.cnf_ini = np.array(data["cnf_ini"])
        graph.cnf = np.array(data["cnf"])
        graph.edges = np.array(data["edges"])
        graph.terms = remap_keys(data["terms"])
        graph.exact_num_sol = data["exact_num_sol"]
        graph.exact_sol_bit = data["exact_sol_bit"]

        if graph.exact_sol_bit is not None:
            sol_int = [int(x, 2) for x in graph.exact_sol_bit]
            exact_sol_pdf = np.zeros(2**graph.num_nodes)
            exact_sol_pdf[sol_int] = 1 / len(sol_int)

            graph.exact_sol_pdf = exact_sol_pdf

        else:
            graph.exact_sol_pdf = None

    return graph


def map_keys(mapping):
    """
    Serialize the keys in tuple form of the dictionary.
    """

    key_map = []
    for k, v in mapping.items():
        if len(k) == 2:
            key_map.append({"key": (int(k[0]), int(k[1])), "value": int(v)})

        elif len(k) == 1:
            key_map.append({"key": (int(k[0]),), "value": int(v)})
        else:
            raise ValueError("The key is not valid.")

    return key_map


def remap_keys(mapping):
    """
    Deserialize the keys in tuple form of the dictionary.
    """

    return {tuple(k["key"]): k["value"] for k in mapping}
