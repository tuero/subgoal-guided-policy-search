// File: sokoban.h
// Description: Wrapper around sokoban_cpp standalone environment
#ifndef HPTS_ENV_SOKOBAN_BASE_H_
#define HPTS_ENV_SOKOBAN_BASE_H_

#include "common/observation.h"

#include <sokoban/sokoban.h>

#include <functional>
#include <string>

namespace hpts::env::sokoban {

class SokobanTreeState {
public:
    explicit SokobanTreeState(const std::string &board_str);
    ~SokobanTreeState() = default;

    SokobanTreeState(const SokobanTreeState &) noexcept = default;
    SokobanTreeState(SokobanTreeState &&) noexcept = default;
    auto operator=(const SokobanTreeState &) noexcept -> SokobanTreeState & = default;
    auto operator=(SokobanTreeState &&) noexcept -> SokobanTreeState & = default;

    void apply_action(int action);
    [[nodiscard]] auto get_observation() const noexcept -> Observation;
    [[nodiscard]] auto observation_shape() const noexcept -> ObservationShape;
    [[nodiscard]] auto is_solution() const noexcept -> bool;
    [[nodiscard]] auto is_terminal() const noexcept -> bool;
    [[nodiscard]] auto get_heuristic() const noexcept -> double;
    [[nodiscard]] auto get_hash() const noexcept -> uint64_t;
    [[nodiscard]] auto operator==(const SokobanTreeState &rhs) const -> bool;

    friend auto operator<<(std::ostream &os, const SokobanTreeState &state) -> std::ostream & {
        return os << state.state;
    }

    inline static const std::string name{"sokoban"};
    inline static const int num_actions = 4;

public:
    ::sokoban::SokobanGameState state;    // NOLINT
};

}    // namespace hpts::env::sokoban

namespace std {
template <>
struct hash<hpts::env::sokoban::SokobanTreeState> {
    size_t operator()(const hpts::env::sokoban::SokobanTreeState &state) const {
        return state.get_hash();
    }
};
}    // namespace std

#endif    // HPTS_ENV_SOKOBAN_BASE_H_
