// File: boudlerdash.cpp
// Description: Base wrapper around boulderdash_cpp standalone environment

#include "env/boulderdash.h"

namespace hpts::env::bd {

BoulderDashTreeState::BoulderDashTreeState(const std::string &board_str)
    : state(board_str) {}

void BoulderDashTreeState::apply_action(int action) {
    state.apply_action(static_cast<boulderdash::Action>(action));
}

auto BoulderDashTreeState::get_observation() const noexcept -> Observation {
    return state.get_observation();
}

auto BoulderDashTreeState::observation_shape() const noexcept -> ObservationShape {
    return state.observation_shape();
}

auto BoulderDashTreeState::is_solution() const noexcept -> bool {
    return state.is_solution();
}

auto BoulderDashTreeState::is_terminal() const noexcept -> bool {
    return state.is_terminal();
}

auto BoulderDashTreeState::get_heuristic() const noexcept -> double {
    return 0;
}

auto BoulderDashTreeState::get_hash() const noexcept -> uint64_t {
    return state.get_hash();
}

auto BoulderDashTreeState::operator==(const BoulderDashTreeState &rhs) const -> bool {
    return state == rhs.state;
}

}    // namespace hpts::env::bd
