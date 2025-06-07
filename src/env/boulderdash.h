// File: boudlerdash.h
// Description: Wrapper around boulderdash_cpp standalone environment
#ifndef HPTS_ENV_BOULDERDASH_H_
#define HPTS_ENV_BOULDERDASH_H_

#include "common/observation.h"

#include <boulderdash/boulderdash.h>

#include <functional>
#include <string>
#include <vector>

namespace hpts::env::bd {

class BoulderDashTreeState {
public:
    explicit BoulderDashTreeState(const std::string &board_str);
    virtual ~BoulderDashTreeState() = default;

    BoulderDashTreeState(const BoulderDashTreeState &) noexcept = default;
    BoulderDashTreeState(BoulderDashTreeState &&) noexcept = default;
    auto operator=(const BoulderDashTreeState &) noexcept -> BoulderDashTreeState & = default;
    auto operator=(BoulderDashTreeState &&) noexcept -> BoulderDashTreeState & = default;

    virtual void apply_action(int action);
    [[nodiscard]] auto observation_shape() const noexcept -> ObservationShape;
    [[nodiscard]] auto get_observation() const noexcept -> std::vector<float>;
    [[nodiscard]] auto is_solution() const noexcept -> bool;
    [[nodiscard]] auto is_terminal() const noexcept -> bool;
    [[nodiscard]] auto get_heuristic() const noexcept -> double;
    [[nodiscard]] auto get_hash() const noexcept -> uint64_t;
    [[nodiscard]] auto operator==(const BoulderDashTreeState &rhs) const -> bool;

    friend auto operator<<(std::ostream &os, const BoulderDashTreeState &state) -> std::ostream & {
        return os << state.state;
    }

    inline static const std::string name{"rnd"};
    inline static const int num_actions = 4;

protected:
    ::boulderdash::BoulderDashGameState state;    // NOLINT
};

}    // namespace hpts::env::bd

namespace std {
template <>
struct hash<hpts::env::bd::BoulderDashTreeState> {
    size_t operator()(const hpts::env::bd::BoulderDashTreeState &state) const {
        return state.get_hash();
    }
};
}    // namespace std

#endif    // HPTS_ENV_BOULDERDASH_H_
