// File: graph.h
// Graph structs

#ifndef HPTS_CLUSTERING_GRAPH_H_
#define HPTS_CLUSTERING_GRAPH_H_

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>

#include <cassert>
#include <utility>
#include <vector>

namespace std {
template <>
struct hash<std::pair<int, int>> {
    size_t operator()(const std::pair<int, int> &x) const {
        const auto hasher = std::hash<int>();
        auto h = hasher(x.first);
        h ^= hasher(x.second) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};
}    // namespace std

namespace hpts::clustering {

using NodeId = int;
using NodeData = absl::flat_hash_set<int>;
using Edge = std::pair<NodeId, NodeId>;

struct GraphNode {
    GraphNode() = default;
    GraphNode(int _data) {
        data.insert(_data);
    }
    GraphNode(NodeData _data)
        : data(std::move(_data)) {}
    GraphNode(const std::unordered_set<int> &_data) {
        set_data_std(_data);
    }
    [[nodiscard]] auto get_data() const -> const NodeData & {
        return data;
    }
    [[nodiscard]] auto get_data_std() const -> std::unordered_set<int> {
        std::unordered_set<int> _data;
        for (const auto &d : data) {
            _data.insert(d);
        }
        return _data;
    }
    void set_data(const NodeData &_data) {
        data = _data;
    }
    void set_data_std(const std::unordered_set<int> &_data) {
        for (const auto &d : _data) {
            data.insert(d);
        }
    }
    NodeData data;
};

struct Graph {
    using NeighbourCost = std::pair<int, double>;
    Graph() = default;

    void add_node(const GraphNode &node) {
        nodes.push_back(node);
        neighbour_list.emplace_back();
    }

    void add_node(GraphNode &&node) {
        nodes.push_back(std::move(node));
        neighbour_list.emplace_back();
    }

    void join(NodeId lhs, NodeId rhs, double cost) {
        assert(cost > 0);
        if (!is_connected(lhs, rhs)) {
            neighbour_list[static_cast<std::size_t>(lhs)].insert(rhs);
            neighbour_list[static_cast<std::size_t>(rhs)].insert(lhs);
            edge_costs[{lhs, rhs}] = cost;
            edge_costs[{rhs, lhs}] = cost;
            edge_weight_sum += 2 * cost;
        }
    }

    void remove_edge(NodeId lhs, NodeId rhs) {
        if (is_connected(lhs, rhs)) {
            edge_weight_sum -= 2 * edge_costs[{lhs, rhs}];
            edge_costs.erase({lhs, rhs});
            edge_costs.erase({rhs, lhs});
            neighbour_list[static_cast<std::size_t>(lhs)].erase(rhs);
            neighbour_list[static_cast<std::size_t>(rhs)].erase(lhs);
        }
    }

    [[nodiscard]] auto is_connected(NodeId lhs, NodeId rhs) const -> bool {
        return neighbour_list.at(static_cast<std::size_t>(lhs)).find(rhs)
               != neighbour_list.at(static_cast<std::size_t>(lhs)).end();
    }

    [[nodiscard]] auto edge_cost(NodeId lhs, NodeId rhs) const -> double {
        return is_connected(lhs, rhs) ? edge_costs.at({lhs, rhs}) : 0;
    }

    std::vector<GraphNode> nodes;
    absl::flat_hash_map<Edge, double> edge_costs;
    std::vector<absl::flat_hash_set<int>> neighbour_list;
    double edge_weight_sum = 0;
};

}    // namespace hpts::clustering

#endif    // HPTS_CLUSTERING_GRAPH_H_
