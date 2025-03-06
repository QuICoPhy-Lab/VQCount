"""
Implementation of differents combinatorial problems.
"""

import igraph as ig
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import qecstruct as qs
from matplotlib.lines import Line2D


def problem_graph(problem, numvar, numcau, vardeg, caudeg):
    """
    Instantiate a ProblemGraph.

    Parameters
    ----------
    problem : str
        Type of problem to generate. Choose between "nae3sat" and "1in3sat". Only supports "nae3sat" at alpha=1 or alpha=2 and "1in3sat" at alpha=2/3.
    numvar : int
        Number of variables.
    numcau : int
        Number of clauses.
    vardeg : int
        Variables degree.
    caudeg : int
        Clauses degree.

    Returns
    -------
    problem_graph : ProblemGraph
        Problem graph.
    """

    if problem == "nae3sat":
        return PositiveNaeThreeSatGraph(numvar, numcau, vardeg, caudeg)
    if problem == "1in3sat":
        return PositiveOneInThreeSatGraph(numvar)

    raise ValueError("The problem is not implemented.")


class ProblemGraph:
    """
    Base class for problem graphs.

    Attributes
    ----------
    problem : str
        Type of problem to generate.
    num_nodes : int
        Number of variables.
    cnf_ini : numpy.ndarray
        Initial CNF formula.
    cnf : numpy.ndarray
        CNF formula.
    edges : numpy.ndarray
        Edges of the Ising graph.
    terms : dict[Tuple[int, int], int]
        Couplings of the Ising graph, where the keys are the edges of the graph and the values are the weights.
    """

    def __init__(self):
        self.problem = None
        self.num_nodes = None
        self.cnf_ini = None
        self.cnf = None
        self.edges = None
        self.terms = None

    def cnf_view(self):
        """CNF formula view. Display the CNF formula graph."""

        # create adjacency matrix
        num_clauses = len(self.cnf_ini)
        num_variables = max(
            abs(literal) for clause in self.cnf_ini for literal in clause
        )
        adj_matrix = np.zeros((num_clauses, num_variables), dtype=int)

        for i, clause in enumerate(self.cnf_ini):
            adj_matrix[i, np.abs(clause) - 1] = 1

        # prepare graph and add nodes
        graph = nx.Graph()
        graph.add_nodes_from(range(num_variables), bipartite=0, color="blue")
        graph.add_nodes_from(
            range(num_variables, num_variables + num_clauses), bipartite=1, color="red"
        )

        # add edges based on the adjacency matrix
        edges = [
            (j, num_variables + i)
            for i in range(num_clauses)
            for j in range(num_variables)
            if adj_matrix[i, j]
        ]
        graph.add_edges_from(edges)

        # graph layout and drawing
        pos = nx.spring_layout(graph)
        nx.draw(
            graph,
            pos,
            node_color=[data["color"] for _, data in graph.nodes(data=True)],
            with_labels=True,
            node_size=500,
        )

        # legend setup
        plt.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label="Clauses",
                    markerfacecolor="red",
                    markersize=10,
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label="Variables",
                    markerfacecolor="blue",
                    markersize=10,
                ),
            ],
            loc="upper right",
        )

    def ising_view(self):
        """Ising formulation view. Display the Ising model graph."""

        # create the graph from edges of the object
        graph = nx.Graph(list(self.edges))

        nx.draw(graph, with_labels=True, node_color="blue", node_size=500)


class PositiveNaeThreeSatGraph(ProblemGraph):
    """
    This class instantiates a random bipartite regular graph representing a positive NAE3SAT problem using qecstruct. It then maps the bipartite regular graph to an Ising model using the Ising formulation of the NAE3SAT problem (see https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2014.00005/full).

    Parameters
    ----------
    numvar : int
        Number of variables.
    numcau : int
        Number of clauses.
    vardeg : int
        Variables degree.
    caudeg : int
        Clauses degree.
    seed : int
        Seed for random number generation.

    Attributes
    ----------
    problem : str
        Type of problem to generate.
    num_nodes : int
        Number of variables.
    cnf_ini : numpy.ndarray
        3SAT formula.
    cnf : numpy.ndarray
        NAE3SAT formula.
    edges : numpy.ndarray
        Edges of the Ising graph.
    terms : dict[Tuple[int, int], int]
        Couplings of the Ising graph, where the keys are the edges of the graph and the values are the weights.
    """

    def __init__(self, numvar, numcau, vardeg, caudeg, seed=None):

        super().__init__()

        # samples a random bicubic graph
        code = qs.random_regular_code(numvar, numcau, vardeg, caudeg, qs.Rng(seed))

        # write the 3SAT formula and find the edges of the ising graph
        cnf_ini = []
        edges = []
        for row in code.par_mat().rows():
            temp_cnf = []
            for value in row:
                temp_cnf.append(value)
            cnf_ini.append(sorted(temp_cnf))
            edges.append([temp_cnf[0], temp_cnf[1]])
            edges.append([temp_cnf[1], temp_cnf[2]])
            edges.append([temp_cnf[2], temp_cnf[0]])

        # sort for consistency
        cnf_ini = sorted(cnf_ini)
        edges = sorted(edges)

        # type of problem to generate
        self.problem = "nae3sat"
        # number of variables
        self.num_nodes = numvar
        # 3SAT formula
        self.cnf_ini = np.array(cnf_ini) + 1
        # NAE3SAT formula
        self.cnf = np.vstack((self.cnf_ini, np.invert(self.cnf_ini) + 1))
        # edges of the ising graph
        self.edges = np.array(edges)
        # dictionary of edges of the ising graph
        terms = {}
        for i, j in self.edges:
            terms[(i, j)] = terms.get((i, j), 0) + 1
        self.terms = terms


class PositiveOneInThreeSatGraph(ProblemGraph):
    """
    This class instantiates a random bicubic graph representating a positive 1-in-3SAT problem. It then maps the bicubic graph to an Ising model using the Ising formulation of the monotone 1-in-3SAT problem (see https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2014.00005/full). ONLY SUPPORTS ALPHA = 2/3.

    Parameters
    ----------
    numvar : int
        Number of variables.

    Attributes
    ----------
    problem : str
        Type of problem to generate.
    num_nodes : int
        Number of variables.
    cnf_ini : numpy.ndarray
        3SAT formula.
    cnf : numpy.ndarray
        1in3SAT formula.
    edges : numpy.ndarray
        Edges of the Ising graph.
    terms : dict[Tuple[int, int], int]
        Couplings of the Ising graph, where the keys are the edges of the graph and the values are the weights.
    """

    def __init__(self, numvar):

        super().__init__()

        if numvar % 3 != 0:
            raise ValueError("The number of variable should be a multiple of 3.")

        numvar = int(2 * numvar / 3)

        cg = ig.Graph.Degree_Sequence([3] * numvar, method="vl")
        temp_edgelist = cg.get_edgelist()

        edgelist = []
        new_var = numvar
        for i, j in temp_edgelist:
            edgelist.append((new_var, i))
            edgelist.append((new_var, j))
            new_var += 1

        temp_cnf = []
        for var in range(numvar):
            temp = []
            for i, j in edgelist:
                if i == var:
                    temp.append(j)
                if j == var:
                    temp.append(i)
            temp_cnf.append(sorted(temp))

        cnf = np.array(sorted(temp_cnf)) - numvar

        # write the 3SAT formula and find the edges of the ising graph
        edges = []
        terms = {}
        for tpcnf in cnf:
            edges.append([tpcnf[0], tpcnf[1]])
            edges.append([tpcnf[1], tpcnf[2]])
            edges.append([tpcnf[2], tpcnf[0]])
            terms[(tpcnf[0], tpcnf[1])] = terms.get((tpcnf[0], tpcnf[1]), 0) + 1
            terms[(tpcnf[1], tpcnf[2])] = terms.get((tpcnf[1], tpcnf[2]), 0) + 1
            terms[(tpcnf[2], tpcnf[0])] = terms.get((tpcnf[2], tpcnf[0]), 0) + 1
            terms[(tpcnf[0],)] = terms.get((tpcnf[0],), 0) - 1
            terms[(tpcnf[1],)] = terms.get((tpcnf[1],), 0) - 1
            terms[(tpcnf[2],)] = terms.get((tpcnf[2],), 0) - 1

        # sort for consistency
        edges = sorted(edges)

        # Type of problem to generate
        self.problem = "mono1in3sat"
        # 3SAT formula
        self.cnf_ini = cnf + 1
        # 1in3SAT formula
        cnf = []
        for i, j, k in self.cnf_ini.tolist():
            cnf.append((i, j, k))
            cnf.append((i, -j, -k))
            cnf.append((-i, -j, k))
            cnf.append((-i, j, -k))
            cnf.append((-i, -j, -k))
        self.cnf = np.array(cnf)
        # edges of the ising graph
        self.edges = np.array(edges)
        # number of variables
        self.num_nodes = int(3 * numvar / 2)
        # dictionary of edges of the ising graph
        self.terms = terms
