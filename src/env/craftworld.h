// File: craftworld.h
// Description: Base wrapper around craftworld_cpp standalone environment
#ifndef HPTS_ENV_CRAFTWORLD_H_
#define HPTS_ENV_CRAFTWORLD_H_

#include "common/observation.h"

#include <craftworld/craftworld.h>

#include <functional>
#include <string>

namespace hpts::env::cw {

class CraftWorldTreeState {
public:
    explicit CraftWorldTreeState(const std::string &board_str);
    ~CraftWorldTreeState() = default;

    CraftWorldTreeState(const CraftWorldTreeState &) noexcept = default;
    CraftWorldTreeState(CraftWorldTreeState &&) noexcept = default;
    auto operator=(const CraftWorldTreeState &) noexcept -> CraftWorldTreeState & = default;
    auto operator=(CraftWorldTreeState &&) noexcept -> CraftWorldTreeState & = default;

    void apply_action(int action);
    [[nodiscard]] auto get_observation() const noexcept -> Observation;
    [[nodiscard]] auto observation_shape() const noexcept -> ObservationShape;
    [[nodiscard]] auto is_solution() const noexcept -> bool;
    [[nodiscard]] auto is_terminal() const noexcept -> bool;
    [[nodiscard]] auto get_heuristic() const noexcept -> double;
    [[nodiscard]] auto get_hash() const noexcept -> uint64_t;
    [[nodiscard]] auto operator==(const CraftWorldTreeState &rhs) const -> bool;

    friend auto operator<<(std::ostream &os, const CraftWorldTreeState &state) -> std::ostream & {
        return os << state.state;
    }

    inline static const std::string name{"craftworld"};
    inline static const int num_actions = craftworld::kNumActions;

protected:
    craftworld::CraftWorldGameState state;    // NOLINT
};

}    // namespace hpts::env::cw

namespace std {
template <>
struct hash<hpts::env::cw::CraftWorldTreeState> {
    size_t operator()(const hpts::env::cw::CraftWorldTreeState &state) const {
        return state.get_hash();
    }
};
}    // namespace std

#endif    // HPTS_ENV_CRAFTWORLD_H_
