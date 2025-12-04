// aco_tsp.cpp
// ACO-based TSP solver for complete graphs.

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>
#include <cmath>
#include <limits>
#include <algorithm>
#include <string>
#include <iomanip>
#include <chrono>

using namespace std;

const double INF = 1e18;

// ===== ACO parameters =====
const int NUM_ANTS = 50;
const int NUM_ITERATIONS = 175;
const double ALPHA = 1.0;    // pheromone importance
const double BETA = 3.0;     // heuristic (1/distance) importance
const double RHO = 0.001;    // evaporation rate
const double Q = 100.0;      // pheromone deposit factor
const double TAU0 = 1.0;     // initial pheromone on each edge

// Reads weighted complete graph from file.
bool read_graph(const string &filename, vector<vector<double>> &dist) {
    ifstream in(filename);
    if (!in) {
        cerr << "Error: cannot open file " << filename << "\n";
        return false;
    }

    string line;
    int n;

    // Read number of nodes
    if (!getline(in, line)) {
        cerr << "Error: empty file.\n";
        return false;
    }
    {
        istringstream iss(line);
        if (!(iss >> n)) {
            cerr << "Error: first line must contain number of nodes.\n";
            return false;
        }
    }

    // Initialize distance matrix
    dist.assign(n, vector<double>(n, 0.0));

    // Ignore second line
    if (!getline(in, line)) {
        cerr << "Warning: no edge lines found.\n";
        return true;
    }

    // Read edges
    while (getline(in, line)) {
        if (line.empty()) continue;
        istringstream iss(line);
        int u, v;
        double w;
        if (!(iss >> u >> v >> w)) continue;
        u--; v--; // convert to 0-based
        if (u < 0 || u >= n || v < 0 || v >= n) {
            cerr << "Warning: invalid edge " << (u + 1) << " " << (v + 1) << "\n";
            continue;
        }
        dist[u][v] = w;
        dist[v][u] = w;
    }

    // Zero out diagonal
    for (int i = 0; i < n; ++i) dist[i][i] = 0.0;

    return true;
}

// Compute total length of a tour (tour is permutation of 0..n-1)
double compute_tour_length(const vector<int> &tour, const vector<vector<double>> &dist) {
    int n = (int)tour.size();
    double len = 0.0;
    for (int i = 0; i < n; ++i) {
        int u = tour[i];
        int v = tour[(i + 1) % n];
        len += dist[u][v];
    }
    return len;
}

// Debug: print breakdown of how the tour length is summed from edges
double compute_tour_length_debug(const vector<int> &tour, const vector<vector<double>> &dist) {
    int n = (int)tour.size();
    double len = 0.0;

    cout << "\n=== Debug: Tour edge breakdown ===\n";
    cout << "Format: step: (u -> v), edge_weight, cumulative_sum\n\n";
    cout << fixed << setprecision(6);

    for (int i = 0; i < n; ++i) {
        int u = tour[i];
        int v = tour[(i + 1) % n];
        double w = dist[u][v];
        len += w;
        cout << "Step " << (i + 1) << ": (" << (u + 1) << " -> " << (v + 1) << "), "
             << "weight = " << w << ", cumulative = " << len << "\n";
    }

    cout << "\nTotal tour length from sum of edges = " << len << "\n";
    cout << "====================================\n\n";
    return len;
}

// Choose next city using pheromone^alpha * (1/distance)^beta roulette selection.
int choose_next_city(int current,
                     const vector<bool> &visited,
                     const vector<vector<double>> &dist,
                     const vector<vector<double>> &pheromone,
                     mt19937 &rng) {
    int n = (int)dist.size();
    vector<double> attractiveness(n, 0.0);
    double sum = 0.0;

    for (int j = 0; j < n; ++j) {
        if (!visited[j] && j != current) {
            double tau = pheromone[current][j];
            double eta = (dist[current][j] > 0.0) ? (1.0 / dist[current][j]) : 0.0;
            double val = pow(tau, ALPHA) * pow(eta, BETA);
            attractiveness[j] = val;
            sum += val;
        }
    }

    if (sum == 0.0) {
        // Fallback: nearest unvisited city
        double best_d = numeric_limits<double>::infinity();
        int best_j = -1;
        for (int j = 0; j < n; ++j) {
            if (!visited[j] && j != current) {
                double d = dist[current][j];
                if (d < best_d) { best_d = d; best_j = j; }
            }
        }
        return best_j;
    }

    uniform_real_distribution<double> u01(0.0, 1.0);
    double r = u01(rng) * sum;
    double acc = 0.0;
    for (int j = 0; j < n; ++j) {
        if (attractiveness[j] > 0.0) {
            acc += attractiveness[j];
            if (acc >= r) return j;
        }
    }

    // Numeric fallback
    for (int j = n - 1; j >= 0; --j)
        if (attractiveness[j] > 0.0) return j;

    return -1;
}

// Run ACO and return best tour and length.
void run_aco_tsp(const vector<vector<double>> &dist,
                 vector<int> &best_tour,
                 double &best_len,
                 long long &out_totalCyclesEvaluated,
                 long long &out_totalValidTours,
                 vector<long long> &out_per_iter_completed,
                 vector<long long> &out_per_iter_valid,
                 vector<double> &out_per_iter_bestlen) {
    int n = (int)dist.size();
    out_per_iter_completed.assign(NUM_ITERATIONS, 0);
    out_per_iter_valid.assign(NUM_ITERATIONS, 0);
    out_per_iter_bestlen.assign(NUM_ITERATIONS, numeric_limits<double>::infinity());

    random_device rd;
    mt19937 rng(rd());

    vector<vector<double>> pheromone(n, vector<double>(n, TAU0));
    best_len = numeric_limits<double>::infinity();
    out_totalCyclesEvaluated = 0;
    out_totalValidTours = 0;

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        vector<vector<int>> ant_tours(NUM_ANTS);
        vector<double> ant_lengths(NUM_ANTS, numeric_limits<double>::infinity());

        // Each ant constructs a full tour
        for (int k = 0; k < NUM_ANTS; ++k) {
            vector<bool> visited(n, false);
            ant_tours[k].reserve(n);

            uniform_int_distribution<int> start_dist(0, n - 1);
            int start = start_dist(rng);
            int current = start;
            visited[current] = true;
            ant_tours[k].push_back(current);

            for (int step = 1; step < n; ++step) {
                int next = choose_next_city(current, visited, dist, pheromone, rng);
                if (next == -1) {
                    // Failed to continue tour
                    ant_tours[k].clear();
                    ant_lengths[k] = numeric_limits<double>::infinity();
                    break;
                }
                ant_tours[k].push_back(next);
                visited[next] = true;
                current = next;
            }

            // If completed, evaluate
            if ((int)ant_tours[k].size() == n) {
                out_per_iter_completed[iter]++;
                out_totalCyclesEvaluated++;
                double L = compute_tour_length(ant_tours[k], dist);
                ant_lengths[k] = L;
                if (L < numeric_limits<double>::infinity()) {
                    out_per_iter_valid[iter]++;
                    out_totalValidTours++;
                }
                if (L < best_len) {
                    best_len = L;
                    best_tour = ant_tours[k];
                }
            }
        }

        out_per_iter_bestlen[iter] =
            (best_len < numeric_limits<double>::infinity()) ? best_len
                                                            : numeric_limits<double>::infinity();

        // Evaporation
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                pheromone[i][j] *= (1.0 - RHO);
                if (pheromone[i][j] < 1e-12) pheromone[i][j] = 1e-12;
            }

        // Deposit pheromone based on all valid tours
        for (int k = 0; k < NUM_ANTS; ++k) {
            if (ant_lengths[k] >= numeric_limits<double>::infinity()) continue;
            double delta = Q / ant_lengths[k];
            const auto &tour = ant_tours[k];
            for (int i = 0; i < n; ++i) {
                int u = tour[i];
                int v = tour[(i + 1) % n];
                pheromone[u][v] += delta;
                pheromone[v][u] += delta;
            }
        }
    }
}

// ===== Main =====
int main(int argc, char *argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " input_file.txt\n";
        return 1;
    }

    vector<vector<double>> dist;
    if (!read_graph(argv[1], dist)) return 1;

    int n = (int)dist.size();
    if (n == 0) {
        cerr << "Empty graph.\n";
        return 1;
    }

    vector<int> best_tour;
    double best_len;
    long long totalCyclesEvaluated = 0;
    long long totalValidTours = 0;
    vector<long long> per_iter_completed, per_iter_valid;
    vector<double> per_iter_bestlen;

    auto t0 = chrono::steady_clock::now();
    run_aco_tsp(dist, best_tour, best_len,
                totalCyclesEvaluated, totalValidTours,
                per_iter_completed, per_iter_valid, per_iter_bestlen);
    auto t1 = chrono::steady_clock::now();
    long long elapsed_ms = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
    double elapsed_sec = elapsed_ms / 1000.0;

#ifdef DEBUG
    cout << "Total attempted cycles = "
         << (long long)NUM_ANTS * (long long)NUM_ITERATIONS << "\n";
    cout << "Total completed cycles = " << totalCyclesEvaluated << "\n";
    cout << "Total valid tours = " << totalValidTours << "\n";
#endif

    if (best_tour.empty()) {
        cerr << "Failed to find a valid tour.\n";
        cout << "Total runtime (seconds): " << elapsed_sec << "\n";
        return 1;
    }

    // Final requested output: best tour, best cost, total runtime
    cout << "Best tour: ";
    for (size_t i = 0; i < best_tour.size(); ++i) {
        cout << (best_tour[i] + 1);  // 1-based indices
        if (i != best_tour.size() - 1)
            cout << ", ";
    }
    // Close the cycle by printing the first node again
    cout << ", " << (best_tour[0] + 1) << "\n";

    cout << "Best tour cost: " << fixed << setprecision(2) << best_len << "\n";
    cout << "Total runtime (seconds): " << elapsed_sec << "\n";

#ifdef DEBUG
    // Per-iteration summary
    cout << "\nPer-iteration summary (iteration: completed, valid, best_so_far):\n";
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        cout << (iter + 1) << ": "
             << per_iter_completed[iter] << ","
             << per_iter_valid[iter] << ",";
        if (per_iter_bestlen[iter] == numeric_limits<double>::infinity()) cout << "INF";
        else cout << fixed << setprecision(2) << per_iter_bestlen[iter];
        cout << "\n";
    }

    // Debug edge-sum check
    double debug_len = compute_tour_length_debug(best_tour, dist);
    cout << "Difference (debug_len - best_len) = " << (debug_len - best_len) << "\n";
#endif

    return 0;
}