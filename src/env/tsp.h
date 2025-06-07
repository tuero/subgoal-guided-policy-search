// File: tsp.h
// Description: Wrapper around tsp_cpp standalone environment
#ifndef HPTS_ENV_TSP_BASE_H_
#define HPTS_ENV_TSP_BASE_H_

#include "common/observation.h"

#include <tsp/tsp.h>

#include <functional>
#include <string>

namespace hpts::env::tsp {

class TSPTreeState {
public:
    TSPTreeState(const std::string &board_str);
    ~TSPTreeState() = default;

    TSPTreeState(const TSPTreeState &) noexcept = default;
    TSPTreeState(TSPTreeState &&) noexcept = default;
    auto operator=(const TSPTreeState &) noexcept -> TSPTreeState & = default;
    auto operator=(TSPTreeState &&) noexcept -> TSPTreeState & = default;

    void apply_action(int action);
    [[nodiscard]] auto get_observation() const noexcept -> Observation;
    [[nodiscard]] auto observation_shape() const noexcept -> ObservationShape;
    [[nodiscard]] auto is_solution() const noexcept -> bool;
    [[nodiscard]] auto is_terminal() const noexcept -> bool;
    [[nodiscard]] auto get_heuristic() const noexcept -> double;
    [[nodiscard]] auto get_hash() const noexcept -> uint64_t;
    [[nodiscard]] auto operator==(const TSPTreeState &rhs) const -> bool;

    friend auto operator<<(std::ostream &os, const TSPTreeState &state) -> std::ostream & {
        return os << state.state;
    }

    inline static const std::string name{"tsp"};
    inline static const int num_actions = 4;

public:
    ::tsp::TSPGameState state;    // NOLINT
};

}    // namespace hpts::env::tsp

namespace std {
template <>
struct hash<hpts::env::tsp::TSPTreeState> {
    size_t operator()(const hpts::env::tsp::TSPTreeState &state) const {
        return state.get_hash();
    }
};
}    // namespace std

#endif    // HPTS_ENV_TSP_BASE_H_
