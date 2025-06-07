// File: boxworld.h
// Description: Wrapper around boxworld standalone environment
#ifndef HPTS_ENV_BOXWORLD_H_
#define HPTS_ENV_BOXWORLD_H_

#include "common/observation.h"

#include <boxworld/boxworld.h>

#include <functional>
#include <ostream>
#include <string>
#include <vector>

namespace hpts::env::bw {

class BoxWorldTreeState {
public:
    explicit BoxWorldTreeState(const std::string &board_str);
    ~BoxWorldTreeState() = default;

    BoxWorldTreeState(const BoxWorldTreeState &) noexcept = default;
    BoxWorldTreeState(BoxWorldTreeState &&) noexcept = default;
    auto operator=(const BoxWorldTreeState &) noexcept -> BoxWorldTreeState & = default;
    auto operator=(BoxWorldTreeState &&) noexcept -> BoxWorldTreeState & = default;

    void apply_action(int action);
    [[nodiscard]] auto observation_shape() const noexcept -> ObservationShape;
    [[nodiscard]] auto get_observation() const noexcept -> std::vector<float>;
    [[nodiscard]] auto is_solution() const noexcept -> bool;
    [[nodiscard]] auto is_terminal() const noexcept -> bool;
    [[nodiscard]] auto get_heuristic() const noexcept -> double;
    [[nodiscard]] auto get_hash() const noexcept -> uint64_t;
    [[nodiscard]] auto operator==(const BoxWorldTreeState &rhs) const -> bool;

    friend auto operator<<(std::ostream &os, const BoxWorldTreeState &state) -> std::ostream & {
        return os << state.state;
    }

    inline static const std::string name{"boxworld"};
    inline static const int num_actions = 4;

protected:
    boxworld::BoxWorldGameState state;    // NOLINT
};

}    // namespace hpts::env::bw

namespace std {
template <>
struct hash<hpts::env::bw::BoxWorldTreeState> {
    size_t operator()(const hpts::env::bw::BoxWorldTreeState &state) const {
        return state.get_hash();
    }
};

}    // namespace std

#endif    // HPTS_ENV_BOXWORLD_H_
