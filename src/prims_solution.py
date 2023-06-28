"""
Solver for a Bounded MST using a modified Prim's algorithm
"""

from collections import defaultdict
import heapq

def __prims(N, bounds_dict, first_node, neighbors_dict):
    """
    Using a modified Prim's algorithm to determine the min bounded MST

    Inputs:
        N(int): total number of nodes
        bounds_dict(dict): dict of all node bounds
        first_node(int): starting node
        neighbors_dict(dict): dict mapping nodes to all their neighboring nodes

    Returns(lst of ints): returns indices of all edges in the bounded MST
    """
    visited_nodes = set()
    cost_to_node = {}
    used_edges = set()
    used_up_nodes = set()

    # set the cost to infinity for each edge since all costs are under infinity
    for i in range(1, N+1):
        cost_to_node[i] = float('inf')
    # start at the first node and set its distance to 0
    cost_to_node[first_node] = 0
    q = [(0, first_node, first_node, -1)]
    # all the nodes haven't been visited yet
    not_visited = []
    for i in range(1, N+1):
        not_visited.append(i)
    while not_visited:
        # get the least cost node
        if not q:
            q = [(0, not_visited[0], not_visited[0], -1)]
        _, u, v, ind = heapq.heappop(q)
        # if visited or the edges' bound is exceeded
        # or (bounds_dict[u][0] < 1) or (bounds_dict[v][0] < 1):
        if u in visited_nodes:
            continue

        visited_nodes.add(u)
        not_visited.remove(u)

        if ind != -1:
            used_edges.add(ind)
            bounds_dict[u][0] -= 1
            bounds_dict[v][0] -= 1

        # look at the neighbors of the node and update their costs
        if (bounds_dict[u][0] > 0):
            for w, new_c, new_ind in neighbors_dict[u]:
                if w not in visited_nodes and new_c < cost_to_node[w]:
                    if (w not in used_up_nodes) and (u not in used_up_nodes):
                        cost_to_node[w] = new_c
                        heapq.heappush(q, (new_c, w, u, new_ind))
        else:
            used_up_nodes.add(u)

    return used_edges

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

def solve(N, bounds, edges):
    """
    Public solver function for the problem

    Inputs:
        N(int): total number of nodes
        bounds(lst of ints): the degree bounds of the nodes
        edges(lst of lst of ints): all the input edges

    Returns(lst of ints): indices of all the edges to include in the MST
    """
    edges_dict = defaultdict(list)
    neighbors_dict = defaultdict(list)

    for i, edge in enumerate(edges):
        edges_dict[tuple(edge)] = i + 1
        u, v, cost = edge
        neighbors_dict[u].append((v, cost, i + 1))
        neighbors_dict[v].append((u, cost, i + 1))

    bounds_dict = {}
    for i, bound in enumerate(bounds, 1):
        bounds_dict[i] = bound

    first_node = 1
    graph = __prims(N, bounds_dict, first_node, neighbors_dict)

    return graph

def main():
    N, _, bounds, edges = __read_input()
    t = solve(N, bounds, edges)
    for i in t:
        print(i)

if __name__ == '__main__':
    main()