#!/usr/bin/env python3
"""Christofides TSP implementation for complete undirected graphs."""

import sys
import math
import time
from typing import List, Tuple, Dict
import networkx as nx


def read_complete_graph(filename: str) -> List[List[float]]:
    """Read a complete undirected graph from file into an n x n distance matrix."""
    with open(filename, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if not lines:
        raise ValueError("Input file is empty")

    try:
        n = int(lines[0].split()[0])
    except Exception as e:
        raise ValueError("First line must contain integer n") from e

    dist = [[0.0 for _ in range(n)] for _ in range(n)]

    # Ignore second line if present
    idx = 1
    if idx < len(lines):
        idx += 1

    # Read edges
    for ln in lines[idx:]:
        parts = ln.split()
        if len(parts) < 3:
            continue
        u = int(parts[0]) - 1
        v = int(parts[1]) - 1
        w = float(parts[2])

        if u < 0 or u >= n or v < 0 or v >= n or u == v:
            continue

        # Keep smallest weight if multiple edges are given
        if dist[u][v] == 0.0 and dist[v][u] == 0.0:
            dist[u][v] = w
            dist[v][u] = w
        else:
            if w < dist[u][v]:
                dist[u][v] = w
                dist[v][u] = w

    for i in range(n):
        dist[i][i] = 0.0

    return dist


def prim_mst(dist: List[List[float]]) -> List[Tuple[int, int]]:
    """Prim's MST for complete graph given by distance matrix."""
    n = len(dist)
    in_mst = [False] * n
    key = [math.inf] * n
    parent = [-1] * n

    key[0] = 0.0

    for _ in range(n):
        u = -1
        best = math.inf
        for i in range(n):
            if not in_mst[i] and key[i] < best:
                best = key[i]
                u = i
        if u == -1:
            break
        in_mst[u] = True

        for v in range(n):
            w = dist[u][v]
            if not in_mst[v] and u != v and w < key[v]:
                key[v] = w
                parent[v] = u

    mst_edges = []
    for v in range(1, n):
        if parent[v] != -1:
            mst_edges.append((parent[v], v))
    return mst_edges


def find_odd_degree_vertices(n: int, edges: List[Tuple[int, int]]) -> List[int]:
    """Return vertices with odd degree in an undirected edge list."""
    degree = [0] * n
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    return [i for i in range(n) if degree[i] % 2 == 1]


def minimum_weight_perfect_matching(
    odd_vertices: List[int],
    dist: List[List[float]],
) -> List[Tuple[int, int]]:
    """Minimum-weight perfect matching on odd-degree vertices (NetworkX Blossom)."""
    G = nx.Graph()
    for u in odd_vertices:
        G.add_node(u)

    for i in range(len(odd_vertices)):
        u = odd_vertices[i]
        for j in range(i + 1, len(odd_vertices)):
            v = odd_vertices[j]
            w = dist[u][v]
            G.add_edge(u, v, weight=w)

    matching = nx.algorithms.matching.min_weight_matching(G, weight="weight")
    return [(u, v) for u, v in matching]


def build_multigraph(
    n: int,
    mst_edges: List[Tuple[int, int]],
    matching_edges: List[Tuple[int, int]],
) -> Dict[int, Dict[int, int]]:
    """Build Eulerian multigraph (adjacency counts) from MST + matching edges."""
    multi: Dict[int, Dict[int, int]] = {i: {} for i in range(n)}

    def add_edge(u: int, v: int) -> None:
        multi[u][v] = multi[u].get(v, 0) + 1
        multi[v][u] = multi[v].get(u, 0) + 1

    for (u, v) in mst_edges:
        add_edge(u, v)
    for (u, v) in matching_edges:
        add_edge(u, v)

    return multi


def find_eulerian_tour(multigraph: Dict[int, Dict[int, int]], start: int = 0) -> List[int]:
    """Eulerian tour of a connected Eulerian multigraph (Hierholzer's algorithm)."""
    local = {u: dict(neigh) for u, neigh in multigraph.items()}

    stack = [start]
    path: List[int] = []

    while stack:
        v = stack[-1]
        if local[v]:
            u, _ = next(iter(local[v].items()))
            # consume edge v-u
            local[v][u] -= 1
            if local[v][u] == 0:
                del local[v][u]
            local[u][v] -= 1
            if local[u][v] == 0:
                del local[u][v]
            stack.append(u)
        else:
            path.append(stack.pop())

    path.reverse()
    return path


def shortcut_eulerian_tour_to_hamiltonian(euler_tour: List[int]) -> List[int]:
    """Shortcut Eulerian tour to a Hamiltonian cycle by skipping repeated vertices."""
    visited = set()
    tour: List[int] = []
    for v in euler_tour:
        if v not in visited:
            visited.add(v)
            tour.append(v)
    return tour  # cycle is closed when computing length


def compute_tour_length(tour: List[int], dist: List[List[float]]) -> float:
    """Compute length of a TSP tour (cycle) for given vertex order."""
    n = len(tour)
    length = 0.0
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        length += dist[u][v]
    return length


def debug_print_tour_length(tour: List[int], dist: List[List[float]]) -> float:
    """Print edge-by-edge length breakdown for a tour (1-based indices)."""
    n = len(tour)
    print("\n=== Debug: Christofides tour edge breakdown ===")
    print("Format: step: (u -> v) [1-based], edge_weight, cumulative_sum\n")

    total = 0.0
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        w = dist[u][v]
        total += w
        print(f"Step {i+1}: ({u+1} -> {v+1}), weight = {w}, cumulative = {total}")
    print(f"\nTotal from edge breakdown = {total}")
    print("==============================================\n")
    return total


def christofides_tsp(dist: List[List[float]]) -> Tuple[List[int], float]:
    """Run Christofides algorithm on distance matrix dist. Returns (tour, length)."""
    n = len(dist)
    if n == 0:
        return [], 0.0

    mst_edges = prim_mst(dist)
    odd_vertices = find_odd_degree_vertices(n, mst_edges)
    if len(odd_vertices) % 2 != 0:
        raise RuntimeError("Odd number of odd-degree vertices (unexpected).")

    matching_edges = minimum_weight_perfect_matching(odd_vertices, dist)
    multi = build_multigraph(n, mst_edges, matching_edges)
    euler_tour = find_eulerian_tour(multi, start=0)
    tour = shortcut_eulerian_tour_to_hamiltonian(euler_tour)
    length = compute_tour_length(tour, dist)

    return tour, length


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python christofides_tsp.py input_graph.txt", file=sys.stderr)
        sys.exit(1)

    filename = sys.argv[1]
    dist = read_complete_graph(filename)

    t0 = time.perf_counter()
    tour, length = christofides_tsp(dist)
    t1 = time.perf_counter()
    elapsed_sec = t1 - t0

    if not tour:
        print("Best tour: (none)")
        print("Best tour cost: 0.00")
        print(f"Total runtime (seconds): {elapsed_sec:.6f}")
        sys.exit(1)

    # Match ACO-style output
    print("Best tour: ", end="")
    cycle = [v + 1 for v in tour] + [tour[0] + 1]
    print(", ".join(str(x) for x in cycle))

    print(f"Best tour cost: {length:.2f}")
    print(f"Total runtime (seconds): {elapsed_sec:.6f}")

    # Debug: edge-by-edge length breakdown
    # if tour:
    #     debug_len = debug_print_tour_length(tour, dist)
    #     print(f"Difference (debug_len - reported_len) = {debug_len - length}\n")


if __name__ == "__main__":
    main()