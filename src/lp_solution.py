"""
Solver for the Bounded MST problem using Linear Programming
"""
#!/usr/bin/env pypy3

import cvxpy as cp
from itertools import * #python -m pip install more-itertools
from typing import List
import numpy as np

class Problem:
    """
    Class for creating an LP Problem

    Attributes:
        num_nodes(int): total number of nodes
        num_edges(int): total number of edges
        node_bounds(lst of ints): degree bounds of the nodes
        edges(lst of lst of ints): all edges
        x(cp.Variable): determines the indices of the edges to keep
        objective(cp.Minimize): objective to minimize the cost of the MST
        constraints(lst of cp.Constraint): constraints on the LP
        prob(cp.Problem): the LP problem

    """
    def __init__(self, N, M, bounds, edges):
        """
        Initializing a Problem class

        Inputs:
            N(int): total number of nodes
            M(int): total number of edges
            bounds(lst of ints): the degree bounds of the nodes
            edges(lst of lst of ints): all the input edges
        """
        self.num_nodes = N
        self.num_edges = M
        self.node_bounds = bounds
        self.edges = edges
        self.x = cp.Variable([M + 1])
        self.objective = (cp.Minimize(cp.sum([(self.edges[i][2] * self.x[i])
                                            for i,_ in enumerate(self.edges)])))
        self.constraints: List[cp.constraints.constraint.Constraint] = []
        self.prob = cp.Problem(self.objective, self.constraints)

def __num_edges_containing(list, n):
    """
    Given a list of edges and a node number, returns the number of edges that
    this node is in

    Inputs:
        list(lst of lst of ints): lst of edges
        n(node): node number

    Returns(int): the number of edges a given node is in
    """
    count = 0
    for (u, v, _) in list:
        if (u == n) or (v == n):
            count += 1
    return count

def __powerset(vertices):
    """
    Returns the powerset given a number of vertices

    Input:
        vertices(int): the number of vertices

    Returns(iterable): powerset of all input vertices
    """
    return (chain.from_iterable(combinations(input, r)
                                for r in range(len(vertices)+1)))

def __get_edges_connected_to(edges, k):
    """
    Returns the indices of all the edges connected to a node k

    Inputs:
        edges(lst of lst of ints): all edges
        k(int): node

    Returns(lst of ints): indices of all edges connected to node k
    """
    edge_dict = __make_edges_dict(edges)
    res = []
    for (u, v, dist) in edges:
        if (u == k) or (v == k):
            res.append(edge_dict[str((u, v, dist))])
    return res

def __solve_lp(problem, edges, E, F, W, V):
    """
    Solves the LP for the bounded MST problem

    Inputs:
        problem(Problem): Problem object
        edges(lst of lst of ints): all edges
        E(lst of lst of ints): edges decidedly not using
        F(lst of lst of ints): current edges in the solution
        W(lst of ints): vertices to be careful about the bound

    Returns: (does not return a value)
    """

    # Making subsets
    subsets = list(__powerset(V))
    edge_dict = __make_edges_dict(edges)
    problem.constraints = []

    # CONSTRAINT 1
    # Iterating through all the subsets
    for subset in subsets:
        x_indices = __get_x_indices(edges, subset)
        if x_indices:
            (problem.constraints.append(
                cp.sum([problem.x[i] for i in x_indices]) <= len(subset) - 1))

    # CONSTRAINT 2
    # Doing the same for set V
    (problem.constraints.append(cp.sum(cp.vstack(problem.x))
                                == (problem.num_nodes - 1)))

    # CONSTRAINT 3
    # The sum of all edges containing v is <= the bound of the vertex
    for vertex in W:
        #Getting x_indices of all the edges connecting to a specific vertex in W
        x_indices = __get_edges_connected_to(edges, vertex)
        if 0 in x_indices:
            x_indices.remove(0)
        if x_indices:
            #Getting bound of the vertex
            bound = problem.node_bounds[vertex]
            (problem.constraints.append(
                cp.sum([problem.x[i] for i in x_indices]) <= bound[0]))

    #Removing unwanted edges
    dont_want = edges

    for (u, v, dist) in E:
        x_ind = edge_dict[str((u, v, dist))]
        problem.constraints.append(problem.x[x_ind] >= 0)
        problem.constraints.append(problem.x[x_ind] <= 1)

        if [u, v, dist] in dont_want:
            dont_want.remove([u, v, dist])

    for (u, v, dist) in F:
        x_ind = edge_dict[str((u, v, dist))]
        problem.constraints.append(problem.x[x_ind] == 1)
        if [u, v, dist] in dont_want:
            dont_want.remove([u, v, dist])

    for (u, v, dist) in dont_want:
        x_ind = edge_dict[str((u, v, dist))]
        problem.constraints.append(problem.x[x_ind] == 0)

    # CONSTRAINT 4
    problem.constraints.append(problem.x[0] == 0)

    # Initializing and solving the problem
    problem.prob = cp.Problem(problem.objective, problem.constraints)
    problem.prob.solve()

def __get_x_indices(all_edges, vertices):
    """
    Helper function to get the indices of all the edges that all vertices
    belong in (to get x_e)

    Inputs:
        all_edges(lst of lst of ints): all edges
        vertices(lst of ints): all vertices to check for

    Returns(lst of ints): indices of edges
    """
    edge_dict = __make_edges_dict(all_edges)

    x_indices = []
    for k in vertices:
        for (u, v, dist) in all_edges:
            if (u == k) and (v in vertices):
                x_indices.append(edge_dict[str((u, v, dist))])

    return x_indices

def __make_edges_dict(edges):
    """
    Takes in a list of edges and makes a dictionary mapping the str
    representation of the edge, '(u, v, dist)', to their index

    Input:
        edges(lst of lst of ints): all edges

    Returns(dict): dict with edge strings as keys and indices as values
    """
    dict = {}
    num = 0
    for (u, v, dist) in edges:
        dict[str((u, v, dist))] = num
        num += 1
    return dict

def __read_input():
    """
    Reads the input from a .txt file

    Returns:
        N(int): total number of nodes
        M(int): total number of edges
        bounds(lst of ints): the degree bounds of the nodes
        edges(lst of lst of ints): all the input edges
    """
    N, M = [int(i) for i in input().split()]
    bounds = [[int(i) for i in input().split()] for _ in range(N)]
    edges = [[int(i) for i in input().split()] for _ in range(M)]

    return N, M, bounds, edges

def solve(problem, edges):
    """
    Solves the LP problem

    Inputs:
        problem(Problem): LP problem to solve for
        edges(lst of lst of ints): all edges

    Returns (lst of ints): indices of all edges to include in the bounded MST
    """
    N = problem.num_nodes
    bounds = problem.node_bounds
    edge_dict = __make_edges_dict(edges)
    E = edges.copy()
    E.remove([0, -1, 0])
    F = []
    W = [i for i in range(0, N+1)]
    V = W.copy()

    #Solving the LP
    while len(E) != 0:
        __solve_lp(problem, edges.copy(), E, F, W, V)
        x = np.asarray(problem.x.value)
        changed = False

        for (u, v, dist) in E:
            ind = edge_dict[str((u, v, dist))]
            if (-0.05 <= x[ind]) and (x[ind] <= 0.05):
                E.remove([u, v, dist])
                changed = True
            elif (0.95 <= x[ind]) and (x[ind] <= 1.05):
                E.remove([u, v, dist])
                changed = True
                F.append([u, v, dist])

        for v in W:
            E_edges = __num_edges_containing(E, v)
            F_edges = __num_edges_containing(F, v)
            if E_edges + F_edges <= bounds[v][0]:
                W.remove(v)
                changed = True

        if not changed:
            min_dist = 2
            min_edge = []
            zero_or_1 = 0
            for (u, v, dist) in E:
                ind = edge_dict[str((u, v, dist))]
                minus_one = x[ind] - 1
                if (abs(x[ind]) < min_dist):
                    min_dist = abs(x[ind])
                    min_edge = [u, v, dist]
                    zero_or_1 = 0
                if (abs(minus_one) < min_dist):
                    if (not (__num_edges_containing(F, u)
                             == problem.node_bounds[u][0])
                             or (__num_edges_containing(F, v)
                                 == problem.node_bounds[v][0])):
                        min_dist = abs(minus_one)
                        min_edge = [u, v, dist]
                        zero_or_1 = 1
            if zero_or_1 == 0:
                E.remove(min_edge)
            else:
                E.remove(min_edge)
                F.append(min_edge)
    res = []
    for (u, v, dist) in F:
        res.append(edge_dict[str((u, v, dist))])
    return res

def main():
    N, M, bounds, edges = __read_input()
    # adding dummy variable at bound, so that bound[1] gives bound of node 1
    bounds.insert(0, [0])
    # adding dummy edge at index 0
    edges.insert(0, [0,-1,0])
    problem = Problem(N, M, bounds, edges)
    t = solve(problem, edges)
    for i in t:
        print(i)

if __name__ == '__main__':
    main()