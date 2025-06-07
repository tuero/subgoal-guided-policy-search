// File: sokoban.cpp
// Description: Wrapper around sokoban_cpp standalone environment

#include "env/sokoban.h"

namespace hpts::env::sokoban {

namespace soko = ::sokoban;

SokobanTreeState::SokobanTreeState(const std::string &board_str)
    : state(board_str) {}

void SokobanTreeState::apply_action(int action) {
    state.apply_action(static_cast<soko::Action>(action));
}

auto SokobanTreeState::get_observation() const noexcept -> Observation {
    return state.get_observation();
}

auto SokobanTreeState::observation_shape() const noexcept -> ObservationShape {
    return state.observation_shape();
}

auto SokobanTreeState::is_solution() const noexcept -> bool {
    return state.is_solution();
}

auto SokobanTreeState::is_terminal() const noexcept -> bool {
    return state.is_solution();
}

auto SokobanTreeState::get_heuristic() const noexcept -> double {
    return 0;
}

auto SokobanTreeState::get_hash() const noexcept -> uint64_t {
    return state.get_hash();
}

auto SokobanTreeState::operator==(const SokobanTreeState &rhs) const -> bool {
    return state == rhs.state;
}

}    // namespace hpts::env::sokoban
