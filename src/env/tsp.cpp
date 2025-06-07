// File: tsp.cpp
// Description: Wrapper around tsp_cpp standalone environment

#include "env/tsp.h"

namespace hpts::env::tsp {

TSPTreeState::TSPTreeState(const std::string &board_str)
    : state(board_str) {}

void TSPTreeState::apply_action(int action) {
    state.apply_action(static_cast<::tsp::Action>(action));
}

auto TSPTreeState::get_observation() const noexcept -> Observation {
    return state.get_observation();
}

auto TSPTreeState::observation_shape() const noexcept -> ObservationShape {
    return state.observation_shape();
}

auto TSPTreeState::is_solution() const noexcept -> bool {
    return state.is_solution();
}

auto TSPTreeState::is_terminal() const noexcept -> bool {
    return state.is_solution();
}

auto TSPTreeState::get_heuristic() const noexcept -> double {
    return 0;
}

auto TSPTreeState::get_hash() const noexcept -> uint64_t {
    return state.get_hash();
}

auto TSPTreeState::operator==(const TSPTreeState &rhs) const -> bool {
    return state == rhs.state;
}

}    // namespace hpts::env::tsp
