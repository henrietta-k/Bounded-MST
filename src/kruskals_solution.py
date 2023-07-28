"""
Solver for a Bounded MST using a modified Kruskal's algorithm
"""


class DisJointSets:
    """
    Class for creating disjoint sets to be used in union-find

    Attributes:
        parents(int): the parent nodes of all the nodes
        ranks(int): ranks of all the nodes
    """

    def __init__(self, N):
        """
        Constructor for the DisjointSets class

        Input:
            N(int): the total number of nodes
        """
        self.parents = [node for node in range(N)]
        self.ranks = [1 for _ in range(N)]

    def __find(self, u):
        """
        Finds the set that a given node belongs to

        Input:
            u(int): input node to find the set for

        Returns(int): the parent node/ set that the input node belongs to
        """
        while u != self.parents[u]:
            self.parents[u] = self.parents[self.parents[u]]
            u = self.parents[u]
        return u

    def __connected(self, u, v):
        """
        Checks if two nodes belong to the same set

        Inputs:
            u(int): the first node
            v(int): the second node

        Returns(bool): whether the two nodes belong to the same set or not
        """
        return self.__find(u) == self.__find(v)

    def __union(self, u, v):
        """
        Unionizes the set(s) that two input nodes belong to

        Inputs:
            u(int): the first node
            v(int): the second node

        Returns: (does not return a value)
        """
        root_u, root_v = self.__find(u), self.__find(v)
        if root_u == root_v:
            return
        if self.ranks[root_u] > self.ranks[root_v]:
            self.parents[root_v] = root_u
        elif self.ranks[root_v] > self.ranks[root_u]:
            self.parents[root_u] = root_v
        else:
            self.parents[root_u] = root_v
            self.ranks[root_v] += 1


def __merge_edges(e1, e2):
    """
    Merges the edges of a graph in the merge sort algorithm

    Inputs:
        e1(lst of ints): the first edge
        e2(lst of ints): the second edge

    Returns(lst of lst of ints): the list of sorted edges
    """
    e1_len = len(e1)
    e2_len = len(e2)
    e1_ind = 0
    e2_ind = 0
    res = []
    while (e1_ind != e1_len) and (e2_ind != e2_len):
        e1_curr = e1[e1_ind]
        e2_curr = e2[e2_ind]
        if e1_curr[2] < e2_curr[2]:
            res.append(e1_curr)
            e1_ind += 1
        else:
            res.append(e2_curr)
            e2_ind += 1
    if e1_ind != e1_len:
        while e1_ind != e1_len:
            res.append(e1[e1_ind])
            e1_ind += 1
    else:
        while e2_ind != e2_len:
            res.append(e2[e2_ind])
            e2_ind += 1
    return res


def __sort_edges(M, edges):
    """
    Sorting the input edges by their weights using merge sort

    Inputs:
        M(int): total number of edges
        edges(lst of lst of ints): all the input edges

    Returns(lst of lst of ints): the sorted edges
    """
    if M == 1:
        return edges
    else:
        half_ind = M // 2
        first_half = __sort_edges(half_ind, edges[0:half_ind])
        second_half = __sort_edges(M - half_ind, edges[half_ind:])
        return __merge_edges(first_half, second_half)


def __find_mst(N, edges, bounds, edges_dict):
    """
    Finds the Bounded MST using Kruskal's algorithm while maintaining the bounds

    Inputs:
        N(int): total number of nodes
        edges(lst of lst of ints): all the input edges
        bounds(lst of ints): the degree bounds of the nodes
        edges_dict(dict): dict mapping str representations of edges to its index

    Returns(lst of ints): the indices of all the edges in the MST
    """
    indices = []
    set_list = DisJointSets(N)
    for _, (u, v, dist) in enumerate(edges, 1):
        if (
            not set_list.__connected(u - 1, v - 1)
            and bounds[u][0] > 0
            and bounds[v][0] > 0
        ):
            indices.append(edges_dict[str([u, v, dist])])
            bounds[u][0] -= 1
            bounds[v][0] -= 1
            set_list.__union(u - 1, v - 1)
    return indices


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
    bounds.insert(0, [0])
    edges = [[int(i) for i in input().split()] for _ in range(M)]
    edges.insert(0, [0, 0, 0])

    return N, M, bounds, edges


def solve(N, M, bounds, edges):
    """
    Public solver function for the problem

    Inputs:
        N(int): total number of nodes
        M(int): total number of edges
        bounds(lst of ints): the degree bounds of the nodes
        edges(lst of lst of ints): all the input edges

    Returns(lst of ints): indices of all the edges to include in the MST
    """
    edges_dict = {}
    for i in range(1, len(edges)):
        edges_dict[str(edges[i])] = i

    result = __find_mst(N, __sort_edges(M, edges), bounds, edges_dict)
    return result


def main():
    N, M, bounds, edges = __read_input()
    edges = solve(N, M, bounds, edges)
    for edge in edges:
        print(edge)


if __name__ == "__main__":
    main()
