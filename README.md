# TSP Solvers: ACO (C++) and Christofides (Python)

This repository provides two Traveling Salesman Problem (TSP) solvers:

- **Christofides algorithm (Python)** for **Euclidean distance graphs** (Graph A).
- **Ant Colony Optimization (ACO, C++)** for **random distance graphs** (Graph B).

All source files are under the `src/` directory.

---

## 1. Repository Layout

```text
src/
  ├─ aco_tsp.cpp             # ACO-based TSP solver (C++)
  ├─ christofides_tsp.py     # Christofides-based TSP solver (Python)
```
## 2. Running the Code - Christofides Python Solver

```text
cd src
pip install networkx
python3 christofides_tsp.py TSP_1000_euclidianDistance.txt
```
## 3. Running the Code - ACO C++ Solver

```text
cd src
g++ -O2 -std=c++17 aco_tsp.cpp -o aco_tsp
./aco_tsp TSP_1000_randomDistance.txt
```
