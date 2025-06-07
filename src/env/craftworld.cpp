// File: craftworld.cpp
// Description: Base wrapper around craftworld_cpp standalone environment

#include "env/craftworld.h"

namespace hpts::env::cw {

using namespace craftworld;

CraftWorldTreeState::CraftWorldTreeState(const std::string &board_str)
    : state(board_str) {}

void CraftWorldTreeState::apply_action(int action) {
    state.apply_action(static_cast<craftworld::Action>(action));
}

auto CraftWorldTreeState::get_observation() const noexcept -> Observation {
    return state.get_observation();
}

auto CraftWorldTreeState::observation_shape() const noexcept -> ObservationShape {
    return state.observation_shape();
}

auto CraftWorldTreeState::is_solution() const noexcept -> bool {
    return state.is_solution();
}

auto CraftWorldTreeState::is_terminal() const noexcept -> bool {
    return state.is_solution();
}

auto CraftWorldTreeState::get_heuristic() const noexcept -> double {
    return 0;
}

auto CraftWorldTreeState::get_hash() const noexcept -> uint64_t {
    return state.get_hash();
}

auto CraftWorldTreeState::operator==(const CraftWorldTreeState &rhs) const -> bool {
    return state == rhs.state;
}

}    // namespace hpts::env::cw
