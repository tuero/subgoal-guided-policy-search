// File: sgps.h
// Description: SGPS with Louvain clustering implementation

#ifndef HPTS_ALGORITHM_SGPS_IMPL_H_
#define HPTS_ALGORITHM_SGPS_IMPL_H_

#include "clustering/graph.h"
#include "clustering/louvain.h"
#include "common/timer.h"
#include "model/twoheaded_convnet_wrapper.h"
#include "model/vsc_wrapper.h"
#include "util/block_allocator.h"
#include "util/concepts.h"
#include "util/stop_token.h"
#include "util/utility.h"

#include <spdlog/spdlog.h>

#include <absl/container/flat_hash_set.h>
#include <absl/hash/hash.h>

#include <cassert>
#include <cstdint>
#include <memory>
#include <random>
#include <ranges>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace hpts::algorithm::hphs_mix {

// clang-format has trouble formatting concepts:
// https://github.com/llvm/llvm-project/issues/48250
// clang-format off

// Concept for simple states for flat search
template <typename T>
concept HPHSEnv = IsSTDHashable<T> && requires(T t, const T ct, const std::string &s) {
    T(s);
    { t.apply_action(makeval<std::size_t>()) } -> std::same_as<void>;
    { ct.get_observation() } -> std::same_as<Observation>;
    { ct.get_hash() } -> std::same_as<uint64_t>;
    { ct.is_solution() } -> std::same_as<bool>;
    { ct.is_terminal() } -> std::same_as<bool>;
    *(&T::name) == makeval<std::string>();
    *(&T::num_actions) == makeval<int>();
};

// clang-format on

template <typename T>
concept VSCWrapper = IsAny<T, model::wrapper::VSCWrapper, model::wrapper::VSCFlatWrapper>;

// Input to PHS search algorithm
template <HPHSEnv EnvT, VSCWrapper VSCWrapperT>
struct SearchInput {
    using env_t = EnvT;
    using wrapper_t = VSCWrapperT;
    std::string puzzle_name;
    EnvT state;
    bool use_heuristic = true;
    int search_budget{};
    int inference_batch_size{};
    double mix_epsilon{};
    double mix_low_alpha{};
    double rho = 1;
    int num_cluster_samples = 1;
    int cluster_level = -1;
    double sample_mean = 1;    // Sample mean/std for solution sub-trajectory lengths
    double sample_stdev = 0;
    mutable unsigned long long int seed = 0;
    bool create_cluster_graph = true;    // No need to cluster at test time
    std::shared_ptr<StopToken> stop_token;
    std::shared_ptr<VSCWrapperT> model_vsc;
    std::shared_ptr<model::wrapper::TwoHeadedConvNetWrapper> model_low;
};

// Search algorithm output
struct SearchOutput {
    std::string puzzle_name;
    bool solution_found = false;
    double solution_cost = -1;
    int num_expanded = 0;
    int num_generated = 0;
    double solution_prob = 0;
    double solution_prob_raw = 0;
    double time = 0;
    // Solution path
    std::vector<Observation> solution_path_observations;
    std::vector<int> solution_path_actions;
    std::vector<double> solution_path_costs;
    // Lowlevel partial paths
    std::vector<Observation> low_partial_input_observations;
    std::vector<Observation> low_partial_target_observations;
    std::vector<int> low_partial_actions;
    // Subgoal solution path
    std::vector<Observation> subgoal_input_observations;
    std::vector<Observation> subgoal_target_observations;
    // VQVAE
    std::vector<Observation> vqvae_input_observations;
    std::vector<Observation> vqvae_target_observations;
    // Statistics for clustering
    std::vector<int> cluster_distances;
};

constexpr double EPS = 1e-8;
constexpr int BLOCK_ALLOCATION_SIZE = 2000;

namespace detail {
// Node used in search
template <HPHSEnv EnvT>
struct Node {
    Node() = delete;
    Node(const EnvT &state_)
        : state(state_) {
        neighbour_ids.reserve(EnvT::num_actions * 4);
    }

    void apply_action(const Node<EnvT> *current, double c, int a) {
        state.apply_action(a);
        log_p = current->log_p + current->action_log_prob[static_cast<std::size_t>(a)];
        g = current->g + c;
        action = a;
    }

    struct Hasher {
        using is_transparent = void;
        std::size_t operator()(const Node &node) const {
            return node.state.get_hash();
        }
        auto operator()(const Node *node) const -> std::size_t {
            return node->state.get_hash();
        }
    };
    struct CompareEqual {
        using is_transparent = void;
        bool operator()(const Node &lhs, const Node &rhs) const {
            return lhs.state == rhs.state;
        }
        auto operator()(const Node *lhs, const Node *rhs) const -> bool {
            return lhs->state == rhs->state;
        }
    };
    struct CompareOrderedLess {
        auto operator()(const Node *lhs, const Node *rhs) const -> bool {
            return lhs->cost < rhs->cost;
        }
    };
    struct CompareOrderedGreater {
        auto operator()(const Node *lhs, const Node *rhs) const -> bool {
            return lhs->cost > rhs->cost;
        }
    };

    // NOLINTBEGIN (misc-non-private-member-variables-in-classes)
    EnvT state;
    double log_p = 0;
    double g = 0;
    double h = 0;
    double cost = 0;
    const Node *parent = nullptr;
    int action = -1;
    int id{};
    std::vector<double> action_log_prob;
    mutable std::unordered_set<int> neighbour_ids;
    // NOLINTEND (misc-non-private-member-variables-in-classes)
};

// PHS cost
[[nodiscard]] inline constexpr auto phs_cost(double log_p, double g, double h) -> double {
    h = (h < 0) ? 0 : h;
    return g == 0 ? 0 : std::log(h + g + EPS) - (log_p * (1.0 + (h / g)));
}

template <HPHSEnv EnvT>
struct NodeLight {
    NodeLight(EnvT _state)
        : state(std::move(_state)) {}
    EnvT state;
    mutable std::vector<int> actions;

    struct Hasher {
        using is_transparent = void;
        std::size_t operator()(const NodeLight &node) const {
            return node.state.get_hash();
        }
    };
    struct CompareEqual {
        using is_transparent = void;
        bool operator()(const NodeLight &lhs, const NodeLight &rhs) const {
            return lhs.state == rhs.state;
        }
    };
};

template <HPHSEnv EnvT>
using Open = std::deque<NodeLight<EnvT>>;

template <HPHSEnv EnvT>
using Closed =
    std::unordered_set<NodeLight<EnvT>, typename NodeLight<EnvT>::Hasher, typename NodeLight<EnvT>::CompareEqual>;

template <typename EnvT>
bool bfs_iter(Open<EnvT> &open, Closed<EnvT> &closed, const EnvT &sol_state) {
    if (open.empty()) {
        return false;
    }
    auto current_iter = closed.find(open.front());
    assert(current_iter != closed.end());
    if (current_iter->state == sol_state) {
        return true;
    }
    // Don't remove immediately as we need to refer to this when we solveo
    open.pop_front();

    // If we are terminal but this wasn't state we are looking for, don't generate
    if (current_iter->state.is_terminal() || current_iter->state.is_solution()) {
        return false;
    }

    for (int i = 0; i < current_iter->state.num_actions; ++i) {
        NodeLight<EnvT> child_node(current_iter->state);
        child_node.actions = current_iter->actions;
        child_node.state.apply_action(i);
        child_node.actions.push_back(i);

        // Already seen this node, but new way to get there
        auto child_iter = closed.find(child_node);
        if (child_iter != closed.end()) {
            continue;
        }
        open.push_back(child_node);
        closed.insert(child_node);
    }

    return false;
}

}    // namespace detail

enum class Status {
    INIT,
    OK,
    ERROR,
    TIMEOUT,
    SOLVED,
};

template <HPHSEnv EnvT, VSCWrapper VSCWrapperT>
class HPHSMix {
    using NodeT = detail::Node<EnvT>;
    using VSCInferenceInputT = typename VSCWrapperT::InferenceRefInput;
    using VSCInferenceOutputT = typename VSCWrapperT::InferenceOutput;
    using VQVAEQuantizeInputT = typename VSCWrapperT::VQVAEQuantizeInput;
    using VQVAEQuantizeOutputT = typename VSCWrapperT::VQVAEQuantizeOutput;
    using LowInferenceInputT = model::wrapper::TwoHeadedConvNetWrapper::InferenceRefInput;
    using LowInferenceOutputT = model::wrapper::TwoHeadedConvNetWrapper::InferenceOutput;
    using OpenListT =
        std::priority_queue<const NodeT *, std::vector<const NodeT *>, typename NodeT::CompareOrderedGreater>;
    using ClosedListT = absl::flat_hash_set<const NodeT *, typename NodeT::Hasher, typename NodeT::CompareEqual>;
    using VecIntHash = absl::Hash<std::vector<int>>;

public:
    HPHSMix(const SearchInput<EnvT, VSCWrapperT> &input_)
        : input(input_),
          status(Status::INIT),
          model_vsc(input.model_vsc),
          model_low(input.model_low),
          node_allocator(BLOCK_ALLOCATION_SIZE, input.state) {
        reset();
    }

    // Initialize the search with root node inference output
    void init() {
        if (status != Status::INIT) {
            SPDLOG_ERROR("Coroutine needs to be reset() before calling init()");
            throw std::logic_error("Coroutine needs to be reset() before calling init()");
        }
        node_id_counter = -1;
        NodeT root_node(input.state);
        root_node.id = ++node_id_counter;
        inference_nodes.push_back(root_node);
        batch_predict();
        status = Status::OK;
    }

    void reset() {
        status = Status::INIT;
        timeout = false;
        search_output = SearchOutput{.puzzle_name = input.puzzle_name};
        inference_nodes.clear();
        closed_batched.clear();
        {
            decltype(open) empty;
            std::swap(open, empty);
        }
        node_allocator.clear();
        closed.clear();
        node_id_counter = -1;
    }

    void step() {
        if (open.empty()) {
            status = Status::ERROR;
            spdlog::error("Exhausted open list - name: {:s}, budget: {:d}.", input.puzzle_name, input.search_budget);
            return;
        }

        // Remove top node from open and put into closed
        const auto current = open.top();
        open.pop();
        closed.insert(current);
        ++search_output.num_expanded;

        // Timeout
        if (input.search_budget >= 0 && search_output.num_expanded >= input.search_budget) {
            timeout = true;
            spdlog::info(
                "Buget timeout - name: {:s}, exp: {:d}, gen: {:d}, budget: {:d}",
                input.puzzle_name,
                search_output.num_expanded,
                search_output.num_generated,
                input.search_budget
            );
            status = Status::TIMEOUT;
            batch_predict();
            return;
        }

        // Consider all children
        for (int a = 0; a < current->state.num_actions; ++a) {
            NodeT child_node = *current;
            child_node.neighbour_ids.clear();
            child_node.id = ++node_id_counter;
            child_node.parent = current;
            child_node.apply_action(current, 1, a);

            // Previously generated
            // Check global or current batched closed
            // If we did previously generate, we have another path through the node,
            // and thus we need to link underlying graph
            auto closed_iter = closed.find(&child_node);
            auto closed_batched_iter = closed_batched.find(child_node);
            if (closed_iter != closed.end()) {
                // Self loop check
                if ((*closed_iter)->id != current->id) {
                    current->neighbour_ids.insert((*closed_iter)->id);
                    (*closed_iter)->neighbour_ids.insert(current->id);
                }
                continue;
            }
            if (closed_batched_iter != closed_batched.end()) {
                // Self loop check
                if ((*closed_batched_iter).id != current->id) {
                    current->neighbour_ids.insert((*closed_batched_iter).id);
                    (*closed_batched_iter).neighbour_ids.insert(current->id);
                }
                continue;
            }
            closed_batched.insert(child_node);

            // Link underlying graph
            current->neighbour_ids.insert(child_node.id);
            child_node.neighbour_ids.insert(current->id);

            // State is not solution but has separate terminal condition, we don't geneate
            if (!child_node.state.is_solution() && child_node.state.is_terminal()) {
                continue;
            }

            // Solution found, no optimality guarantees so we return on generation instead of expansion
            if (child_node.state.is_solution()) {
                spdlog::info(
                    "Solved - name: {:s}, exp: {:d}, gen: {:d}, budget: {:d}, c: {:.0f}",
                    input.puzzle_name,
                    search_output.num_expanded,
                    search_output.num_generated,
                    input.search_budget,
                    child_node.g
                );
                set_solution(child_node);
                status = Status::SOLVED;
                return;
            }

            inference_nodes.push_back(std::move(child_node));
        }

        // Batch inference
        if (open.empty() || inference_nodes.size() >= static_cast<std::size_t>(input.inference_batch_size)) {
            batch_predict();
        }
    }

    [[nodiscard]] Status get_status() const {
        return status;
    }

    [[nodiscard]] SearchOutput get_search_output() const {
        // @NOTE: Delayed creation so we can batch with more threads
        if (input.create_cluster_graph && status == Status::TIMEOUT) {
            create_cluster_data();
        }
        return search_output;
    }

    void run() {
        TimerWall timer(-1);
        timer.start();
        init();
        while (get_status() == Status::OK && !input.stop_token->stop_requested()) {
            step();
        }
    }

private:
    void batch_predict() {
        if (inference_nodes.empty()) {
            return;
        }

        // Cache observations
        std::vector<Observation> observations;
        observations.reserve(inference_nodes.size());
        for (const auto &node : inference_nodes) {
            observations.push_back(node.state.get_observation());
        }

        // Shared subgoal and conditional low inference
        std::vector<VSCInferenceInputT> vsc_inputs;
        vsc_inputs.reserve(observations.size());
        for (auto &obs : observations) {
            vsc_inputs.emplace_back(obs);
        }
        std::vector<VSCInferenceOutputT> vsc_predictions = model_vsc->Inference(vsc_inputs);

        // Low inference
        std::vector<LowInferenceInputT> low_inference_inputs;
        low_inference_inputs.reserve(observations.size());
        for (auto &obs : observations) {
            low_inference_inputs.emplace_back(obs);
        }
        std::vector<LowInferenceOutputT> low_predictions = model_low->Inference(low_inference_inputs);

        // Mixing
        for (const auto &[node_idx, child_node, vsc_output, low_output] :
             std::views::zip(std::views::iota(0), inference_nodes, vsc_predictions, low_predictions))
        {
            std::vector<std::vector<double>> conditional_low_policies = std::move(vsc_output.conditional_low_policies);
            // Mixture conditional low policies + low policy by subgoal weights + mixture low alpha
            auto mixture_alphas = scalar_mul(std::move(vsc_output.subgoal_policy), 1.0 - input.mix_low_alpha);
            mixture_alphas.push_back(input.mix_low_alpha);
            conditional_low_policies.push_back(std::move(low_output.policy));
            auto mixture_policy =
                geo_mix_policy(conditional_low_policies, mixture_alphas, child_node.state.num_actions);

            child_node.h = input.use_heuristic ? low_output.heuristic : 0;
            child_node.action_log_prob = log_policy_noise(std::move(mixture_policy), input.mix_epsilon);
            child_node.cost = detail::phs_cost(child_node.log_p, child_node.g, child_node.h);

            const auto child_node_ptr = node_allocator.add(std::move(child_node));
            open.push(child_node_ptr);
            closed.insert(child_node_ptr);
            ++search_output.num_generated;
        }
        inference_nodes.clear();
        closed_batched.clear();
    }

    void create_cluster_data() const {
        // Need to set node id maping to be contiguous
        clustering::Graph base_graph;
        std::unordered_map<int, int> idx_to_id;
        std::unordered_map<int, int> id_to_idx;
        std::vector<const NodeT *> search_nodes;
        for (auto &node : closed) {
            search_nodes.push_back(node);
        }
        for (const auto &[i, node] : std::views::enumerate(search_nodes)) {
            id_to_idx[node->id] = static_cast<int>(i);
            idx_to_id[static_cast<int>(i)] = node->id;
            base_graph.add_node({static_cast<int>(i)});
        }

        // Edges
        for (const auto &[idx1, node] : std::views::enumerate(search_nodes)) {
            int id1 = idx_to_id.at(static_cast<int>(idx1));
            for (const auto &id2 : node->neighbour_ids) {
                int idx2 = id_to_idx.at(id2);
                assert(id1 != id2);
                base_graph.join(idx1, idx2, 1);
            }
        }

        // Create Louvain clusters
        // clamp the hierarchy into legal range
        auto num_clique_iter = static_cast<int>(std::log2(static_cast<double>(input.search_budget) / 4000)) + 2;
        auto graphs = clustering::louvain(base_graph, input.rho, num_clique_iter);
        auto cluster_level = input.cluster_level < 1 ? (static_cast<int>(graphs.size()) / 2) - 1 : input.cluster_level;
        cluster_level =
            static_cast<std::size_t>(std::max(0, std::min(static_cast<int>(graphs.size()) - 1, cluster_level)));
        auto &graph = graphs[cluster_level];

        // Sample pairs of states
        std::mt19937_64 rng(++input.seed);
        std::uniform_int_distribution<int> dist_nodes(0, static_cast<int>(graph.nodes.size()) - 1);
        int counter = input.num_cluster_samples;
        while (counter > 0) {
            // Sample node in graph and a neighbour
            int cluster_idx1 = dist_nodes(rng);
            std::uniform_int_distribution<int> dist_neighbours(
                0,
                static_cast<int>(graph.neighbour_list.at(cluster_idx1).size()) - 1
            );
            int cluster_idx2 = *std::next(std::begin(graph.neighbour_list.at(cluster_idx1)), dist_neighbours(rng));

            // Randomly sample state from each node
            const auto &cluster1 = graph.nodes.at(cluster_idx1);
            const auto &cluster2 = graph.nodes.at(cluster_idx2);
            std::uniform_int_distribution<int> state_dist1(0, static_cast<int>(cluster1.data.size()) - 1);
            std::uniform_int_distribution<int> state_dist2(0, static_cast<int>(cluster2.data.size()) - 1);
            int search_node_idx1 = *std::next(std::begin(cluster1.data), state_dist1(rng));
            int search_node_idx2 = *std::next(std::begin(cluster2.data), state_dist2(rng));

            const auto &state1 = search_nodes.at(search_node_idx1)->state;
            const auto &state2 = search_nodes.at(search_node_idx2)->state;

            // Try to find path length between two nodes in underlying graph
            // Here we just walk backwards up the tree
            auto traj_from_walk = [&](int node_id_from, int node_id_to, const std::vector<int> &actions) {
                EnvT current_state = search_nodes.at(node_id_from)->state;
                EnvT target_state = search_nodes.at(node_id_to)->state;

                search_output.vqvae_input_observations.push_back(current_state.get_observation());
                search_output.vqvae_target_observations.push_back(target_state.get_observation());
                for (const auto &a : actions) {
                    search_output.low_partial_input_observations.push_back(current_state.get_observation());
                    search_output.low_partial_target_observations.push_back(target_state.get_observation());
                    search_output.low_partial_actions.push_back(a);
                    current_state.apply_action(a);
                }
                if (current_state != target_state) {
                    SPDLOG_ERROR("Failed to walk to state");
                    std::exit(1);
                }
                search_output.cluster_distances.push_back(static_cast<int>(actions.size()));
            };
            auto walk = [&](int node_id_from, int node_id_to) {
                const NodeT *current = search_nodes.at(node_id_to);
                std::vector<int> actions;
                while (current->parent != nullptr) {
                    // We found a path in the tree, set the data for learning
                    actions.push_back(current->action);
                    if (current->parent->state == search_nodes.at(node_id_from)->state) {
                        std::ranges::reverse(actions);
                        traj_from_walk(node_id_from, node_id_to, actions);
                        --counter;    // Found a sample that works, decrement remaining count
                        return;
                    }
                    current = current->parent;
                }
            };

            // Try to walk from both ends in our search tree to find direction in directed graph
            walk(search_node_idx1, search_node_idx2);
            walk(search_node_idx2, search_node_idx1);
        }
    }

    void create_solution_data(const NodeT &node) {
        auto current = &node;
        std::vector<Observation> observations = {current->state.get_observation()};
        std::vector<int> actions;
        double solution_cost = 0;
        while (current->parent) {
            search_output.solution_path_observations.push_back(current->parent->state.get_observation());
            search_output.solution_path_actions.push_back(current->action);
            solution_cost += (current->g - current->parent->g);
            search_output.solution_path_costs.push_back(solution_cost);
            observations.push_back(current->parent->state.get_observation());
            actions.push_back(current->action);
            current = current->parent;
        }
        std::ranges::reverse(observations);
        std::ranges::reverse(actions);

        // Get pairs of observations separated by offset
        std::vector<int> indices = {0};
        std::mt19937_64 rd(++input.seed);
        std::normal_distribution<double> dist(input.sample_mean - 1, std::sqrt(input.sample_stdev));
        auto sample_from_dist = [&]() -> int {
            auto val = dist(rd);
            return static_cast<int>(std::max(0.0, val)) + 1;
        };
        int next_idx = sample_from_dist();
        for (int i = 1; i < static_cast<int>(observations.size()); ++i) {
            --next_idx;
            if (next_idx <= 0) {
                next_idx = sample_from_dist();
                indices.push_back(i);
            }
        }
        indices.push_back(static_cast<int>(observations.size()) - 1);

        // Get index pairs
        std::vector<std::pair<int, int>> idx_pairs;
        for (std::size_t i = 0; i < indices.size() - 1; ++i) {
            idx_pairs.emplace_back(indices[i], indices[i + 1]);
        }

        // Create VQVAE pairs
        for (const auto &[idx_from, idx_to] : idx_pairs) {
            search_output.vqvae_input_observations.push_back(observations[idx_from]);
            search_output.vqvae_target_observations.push_back(observations[idx_to]);
        }

        // Create conditional low inputs
        for (const auto &[idx_from, idx_to] : idx_pairs) {
            for (int i = idx_from; i < idx_to; ++i) {
                search_output.low_partial_input_observations.push_back(observations[i]);
                search_output.low_partial_target_observations.push_back(observations[idx_to]);
                search_output.low_partial_actions.push_back(actions[i]);
            }
        }

        search_output.subgoal_input_observations = search_output.low_partial_input_observations;
        search_output.subgoal_target_observations = search_output.low_partial_target_observations;
    }

    void set_solution(NodeT &node) {
        status = Status::SOLVED;
        search_output.solution_found = true;
        search_output.solution_cost = node.g;
        search_output.solution_prob = std::exp(node.log_p);
        search_output.solution_prob_raw = std::exp(node.log_p);
        create_solution_data(node);
    }

    SearchInput<EnvT, VSCWrapperT> input;
    Status status{};                                                       // Current search status
    bool timeout = false;                                                  // Timeout flag on budget
    int node_id_counter = 0;                                               // Node ID counter
    std::shared_ptr<VSCWrapperT> model_vsc;                                // VSC  model
    std::shared_ptr<model::wrapper::TwoHeadedConvNetWrapper> model_low;    // Low model
    mutable SearchOutput search_output;    // Output of the search algorithm, containing trajectory + stats
    std::vector<NodeT> inference_nodes;    // Nodes in queue for batch inference
    OpenListT open;                        // Open list
    ClosedListT closed;                    // Closed list
    std::unordered_set<NodeT, typename NodeT::Hasher, typename NodeT::CompareEqual> closed_batched;
    BlockAllocator<NodeT, typename NodeT::Hasher, typename NodeT::CompareEqual> node_allocator;
};

template <HPHSEnv EnvT, VSCWrapper VSCWrapperT>
auto search(const SearchInput<EnvT, VSCWrapperT> &input) -> SearchOutput {
    TimerWall timer(-1);
    HPHSMix<EnvT, VSCWrapperT> step_hphs(input);
    timer.start();
    step_hphs.init();
    // Iteratively search until status changes (solved or timeout)
    while (step_hphs.get_status() == Status::OK && !input.stop_token->stop_requested()) {
        step_hphs.step();
    }
    auto output = step_hphs.get_search_output();
    output.time = timer.get_duration();
    return output;
}

}    // namespace hpts::algorithm::hphs_mix

#endif    // HPTS_ALGORITHM_SGPS_IMPL_H_
