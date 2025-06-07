// File: boxworld.cpp
// Description: Base wrapper around boxworld standalone environment

#include "env/boxworld.h"

namespace hpts::env::bw {

using namespace boxworld;

BoxWorldTreeState::BoxWorldTreeState(const std::string &board_str)
    : state(board_str) {}

void BoxWorldTreeState::apply_action(int action) {
    state.apply_action(static_cast<Action>(action));
}

auto BoxWorldTreeState::get_observation() const noexcept -> Observation {
    return state.get_observation();
}

auto BoxWorldTreeState::observation_shape() const noexcept -> ObservationShape {
    return state.observation_shape();
}

auto BoxWorldTreeState::is_solution() const noexcept -> bool {
    return state.is_solution();
}

auto BoxWorldTreeState::is_terminal() const noexcept -> bool {
    return state.is_solution();
}

auto BoxWorldTreeState::get_heuristic() const noexcept -> double {
    return 0;
}

auto BoxWorldTreeState::get_hash() const noexcept -> uint64_t {
    return state.get_hash();
}

auto BoxWorldTreeState::operator==(const BoxWorldTreeState &rhs) const -> bool {
    return state == rhs.state;
}

}    // namespace hpts::env::bw
