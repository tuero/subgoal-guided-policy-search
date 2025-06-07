// File: louvain.h
// Louvain algorithm

#ifndef HPTS_CLUSTERING_LOUVAIN_H_
#define HPTS_CLUSTERING_LOUVAIN_H_

#include "clustering/graph.h"
#include "util/zip.h"

#include <absl/container/flat_hash_map.h>

#include <random>
#include <ranges>
#include <vector>

namespace hpts::clustering {

using Cluster = std::unordered_set<NodeId>;
using ClusterList = std::vector<Cluster>;

inline auto are_clusters_connected(const Graph &graph, const Cluster &lhs, const Cluster &rhs) -> bool {
    if (lhs == rhs) {
        return false;
    }
    for (const auto &u : lhs) {
        for (const auto &v : rhs) {
            if (graph.is_connected(u, v)) {
                return true;
            }
        }
    }
    return false;
}

inline auto modularity(const Graph &graph, const ClusterList &clusters, double rho) -> double {
    double m = 0;
    for (const auto &cluster : clusters) {
        double e = 0;    // Sum of edge weight between nodes within the cluster
        double a = 0;    // Sum of edge weight for nodes within cluster (including link to other clusters)
        for (const auto &node_id : cluster) {
            for (const auto &neighbour_id : graph.neighbour_list.at(static_cast<std::size_t>(node_id))) {
                const double cost = graph.edge_cost(node_id, neighbour_id);
                a += cost;
                if (cluster.find(neighbour_id) != cluster.end()) {
                    e += cost;
                }
            }
        }
        m += (e / graph.edge_weight_sum) - (rho * std::pow(a / graph.edge_weight_sum, 2));
    }

    return m;
}

inline auto modularity(
    const ClusterList &clusters,
    const std::vector<double> &inner_edge_sum,
    const std::vector<double> &outer_edge_sum,
    double rho
) -> double {
    double m = 0;
    // for (const auto &[cluster_idx, cluster] : std::views::enumerate(clusters)) {
    for (auto &&[cluster_idx, cluster] : enumerate(clusters)) {
        if (cluster.empty()) {
            continue;
        }
        m += inner_edge_sum.at(cluster_idx) - rho * std::pow(outer_edge_sum.at(cluster_idx), 2);
    }
    return m;
}

inline auto init_clusters(const Graph &graph) -> std::pair<ClusterList, std::vector<int>> {
    ClusterList clusters;
    std::vector<int> node_to_cluster_map;
    node_to_cluster_map.reserve(graph.nodes.size());
    clusters.reserve(graph.nodes.size());
    for (int i = 0; i < static_cast<int>(graph.nodes.size()); ++i) {
        clusters.push_back({i});
        node_to_cluster_map.push_back(i);
    }
    return {clusters, node_to_cluster_map};
}

constexpr double EPS = 1e-5;

inline auto find_maximal_neighbour(
    int node_id,
    int cluster_id_from,
    const Graph &graph,
    const ClusterList &clusters,
    const Graph &cluster_graph,
    const std::vector<double> &outer_edge_sum,
    double m,
    double rho
) -> std::pair<int, double> {
    int max_id = -1;
    double max_modularity = std::numeric_limits<double>::lowest();
    for (const auto &cluster_id_to : cluster_graph.neighbour_list.at(static_cast<std::size_t>(cluster_id_from))) {
        const auto NE = graph.edge_weight_sum;
        double m_curr = m;

        // Move node to neighbour cluster
        for (const auto &_node_id : clusters[static_cast<std::size_t>(cluster_id_from)]) {
            m_curr -= graph.edge_cost(node_id, _node_id) / NE;
        }
        // to cluster inner edge weight increased
        for (const auto &_node_id : clusters[static_cast<std::size_t>(cluster_id_to)]) {
            m_curr += graph.edge_cost(node_id, _node_id) / NE;
        }
        double outer_edge_sum_from = outer_edge_sum[static_cast<std::size_t>(cluster_id_from)];
        double outer_edge_sum_to = outer_edge_sum[static_cast<std::size_t>(cluster_id_to)];
        m_curr += rho * std::pow(outer_edge_sum_from, 2);
        m_curr += rho * std::pow(outer_edge_sum_to, 2);
        // outer edge sums
        for (const auto &_neighbour_id : graph.neighbour_list[static_cast<std::size_t>(node_id)]) {
            outer_edge_sum_from -= graph.edge_cost(node_id, _neighbour_id) / NE;
            outer_edge_sum_to += graph.edge_cost(node_id, _neighbour_id) / NE;
        }

        m_curr -= rho * std::pow(outer_edge_sum_from, 2);
        m_curr -= rho * std::pow(outer_edge_sum_to, 2);

        if (m_curr > max_modularity) {
            max_id = cluster_id_to;
            max_modularity = m_curr;
        }
    }
    return {max_id, max_modularity};
}

inline auto graph_from_clusters(const ClusterList &clusters, const Graph &g) -> Graph {
    Graph graph;
    graph.nodes.reserve(clusters.size());

    // Create node in graph for each cluster
    ClusterList pruned_clusters;
    pruned_clusters.reserve(clusters.size());
    for (const auto &cluster : clusters) {
        if (!cluster.empty()) {
            pruned_clusters.push_back(std::move(cluster));
        }
    }
    for (const auto &cluster : pruned_clusters) {
        if (cluster.empty()) {
            continue;
        }
        GraphNode node;
        for (const auto &node_id : cluster) {
            for (const auto &d : g.nodes.at(static_cast<std::size_t>(node_id)).data) {
                node.data.insert(d);
            }
        }
        graph.add_node(std::move(node));
    }

    // Join nodes in graph
    for (const auto &&[u_id, u] : enumerate(pruned_clusters)) {
        for (const auto &&[v_id, v] : enumerate(pruned_clusters)) {
            if (are_clusters_connected(g, u, v)) {
                // Get total edge weight between nodes in the cluster
                double edge_weight = 0;
                for (const auto &node_u : u) {
                    for (const auto &node_v : v) {
                        edge_weight += g.edge_cost(node_u, node_v);
                    }
                }
                graph.join(static_cast<int>(u_id), static_cast<int>(v_id), edge_weight);
            }
        }
    }

    return graph;
}

inline auto create_clique_abstraction(const Graph &g_old) {
    absl::flat_hash_set<int> nodes;
    // std::unordered_set<int> nodes;
    nodes.reserve(g_old.nodes.size());
    for (int i = 0; i < static_cast<int>(g_old.nodes.size()); ++i) {
        nodes.insert(i);
    }
    std::vector<absl::flat_hash_set<int>> _nodes_to_add;
    Graph g_new;
    while (!nodes.empty()) {
        _nodes_to_add.push_back({});
        int _node_id = *nodes.begin();
        _nodes_to_add.back().insert(_node_id);
        nodes.erase(_node_id);

        auto check = [&](int _id) -> bool {
            for (const auto &_i : _nodes_to_add) {
                if (!g_old.is_connected(_node_id, _id)) {
                    return false;
                }
            }
            return true;
        };

        for (const auto &n : nodes) {
            if (check(n)) {
                _nodes_to_add.back().insert(n);
            }
        }

        for (const auto &n : _nodes_to_add.back()) {
            nodes.erase(n);
        }
        GraphNode node;
        for (const auto &n : _nodes_to_add.back()) {
            for (const auto &d : g_old.nodes[static_cast<std::size_t>(n)].get_data()) {
                node.data.insert(d);
            }
        }
        g_new.add_node(std::move(node));
    }
    // connect
    for (int i = 0; i < static_cast<int>(_nodes_to_add.size()); ++i) {
        for (int j = 0; j < static_cast<int>(_nodes_to_add.size()); ++j) {
            auto do_join = [&]() {
                for (const auto &a : _nodes_to_add[static_cast<std::size_t>(i)]) {
                    for (const auto &b : _nodes_to_add[static_cast<std::size_t>(j)]) {
                        if (g_old.is_connected(a, b)) {
                            return true;
                        }
                    }
                }
                return false;
            };
            auto edge_count = [&]() {
                double edge_weight = 0;
                for (const auto &a : _nodes_to_add[static_cast<std::size_t>(i)]) {
                    for (const auto &b : _nodes_to_add[static_cast<std::size_t>(j)]) {
                        if (g_old.is_connected(a, b)) {
                            edge_weight += g_old.edge_cost(a, b);
                        }
                    }
                }
                return edge_weight;
            };
            if (do_join()) {
                g_new.join(i, j, edge_count());
            }
        }
    }
    return g_new;
}

inline auto louvain(const Graph &base_graph, double rho, int clique_iter) -> std::vector<Graph> {
    std::vector<Graph> graphs;
    std::mt19937 rng(0);
    graphs.push_back(base_graph);

    // Do a clique abstraction as first pass
    for (int i = 0; i < clique_iter; ++i) {
        graphs.push_back(create_clique_abstraction(graphs.back()));
        if (graphs.back().nodes.size() < 10) {
            break;
        }
    }

    // Outer loop over hierarchies
    while (true) {
        const Graph &g_i = graphs.back();
        auto [c_i, node_to_cluster_map] = init_clusters(g_i);
        auto c_i_old = c_i;

        // -------
        std::vector<double> inner_edge_sum(c_i.size(), 0);
        std::vector<double> outer_edge_sum(c_i.size(), 0);
        for (int i = 0; i < static_cast<int>(c_i.size()); ++i) {
            inner_edge_sum[static_cast<std::size_t>(i)] = 0;
            for (int j = 0; j < static_cast<int>(c_i.size()); ++j) {
                outer_edge_sum[static_cast<std::size_t>(i)] += g_i.edge_cost(i, j) / g_i.edge_weight_sum;
            }
        }
        Graph cluster_graph;
        cluster_graph.nodes.reserve(c_i.size());
        for (int i = 0; i < static_cast<int>(c_i.size()); ++i) {
            cluster_graph.add_node(i);
        }
        for (auto &&[i, cluster_i] : enumerate(c_i)) {
            for (auto &&[j, cluster_j] : enumerate(c_i)) {
                if (are_clusters_connected(g_i, cluster_i, cluster_j)) {
                    cluster_graph.join(static_cast<int>(i), static_cast<int>(j), 1);
                }
            }
        }
        // -------

        // Inner loop to continue while progress is being made
        double m_old = modularity(c_i, inner_edge_sum, outer_edge_sum, rho);
        double m_curr = m_old;
        std::vector<int> shuffled_node_ids;
        shuffled_node_ids.reserve(g_i.nodes.size());
        for (int i = 0; i < static_cast<int>(g_i.nodes.size()); ++i) {
            shuffled_node_ids.push_back(i);
        }
        int inner_counter = 0;

        while (true) {
            ++inner_counter;
            // Try to increase modularity by moving each node
            std::shuffle(std::begin(shuffled_node_ids), std::end(shuffled_node_ids), rng);
            for (const auto &node_id : shuffled_node_ids) {
                // Find modularity of each neighbouring cluster
                const int current_cluster_id = node_to_cluster_map[static_cast<std::size_t>(node_id)];
                Cluster &current_cluster = c_i[static_cast<std::size_t>(current_cluster_id)];
                auto [max_cluster_idx, max_modularity] = find_maximal_neighbour(
                    node_id,
                    current_cluster_id,
                    g_i,
                    c_i,
                    cluster_graph,
                    outer_edge_sum,
                    m_curr,
                    rho
                );

                // If modularity gain, move node into new cluster
                if (max_cluster_idx >= 0 && max_modularity > m_curr + EPS) {
                    current_cluster.erase(node_id);
                    c_i[static_cast<std::size_t>(max_cluster_idx)].insert(node_id);
                    node_to_cluster_map[static_cast<std::size_t>(node_id)] = max_cluster_idx;

                    // -------
                    cluster_graph.remove_edge(current_cluster_id, max_cluster_idx);
                    if (are_clusters_connected(g_i, current_cluster, c_i[static_cast<std::size_t>(max_cluster_idx)])) {
                        cluster_graph.join(current_cluster_id, max_cluster_idx, 1);
                    }
                    // from cluster inner edge weight reduced
                    const auto NE = g_i.edge_weight_sum;
                    for (const auto &_node_id : current_cluster) {
                        inner_edge_sum[static_cast<std::size_t>(current_cluster_id)] -=
                            g_i.edge_cost(node_id, _node_id) / NE;
                    }
                    // to cluster inner edge weight increased
                    for (const auto &_node_id : c_i.at(static_cast<std::size_t>(max_cluster_idx))) {
                        inner_edge_sum[static_cast<std::size_t>(max_cluster_idx)] +=
                            g_i.edge_cost(node_id, _node_id) / NE;
                    }
                    // outer edge sums
                    for (const auto &_neighbour_id : g_i.neighbour_list.at(static_cast<std::size_t>(node_id))) {
                        outer_edge_sum[static_cast<std::size_t>(current_cluster_id)] -=
                            g_i.edge_cost(node_id, _neighbour_id) / NE;
                        outer_edge_sum[static_cast<std::size_t>(max_cluster_idx)] +=
                            g_i.edge_cost(node_id, _neighbour_id) / NE;
                    }
                    m_curr = max_modularity;
                    // -------
                }
            }
            // After sweeping over all nodes, check if modularity has net gained
            if (m_curr < m_old + EPS || inner_counter >= 3) {
                break;
            }
            m_old = m_curr;
        }

        // If input graph equals output graph after a clustering pass,
        if (c_i == c_i_old) {
            graphs.push_back(create_clique_abstraction(graphs.back()));
        } else {
            graphs.push_back(graph_from_clusters(c_i, g_i));
        }

        // If resulting graph is a single node, break
        if (graphs.back().nodes.size() == 1) {
            break;
        }
    }
    return graphs;
}

}    // namespace hpts::clustering

#endif    // HPTS_CLUSTERING_LOUVAIN_H_
