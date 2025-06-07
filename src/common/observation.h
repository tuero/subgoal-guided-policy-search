// File: observation.h
// Description: A common observation type

#ifndef HPTS_COMMON_OBSERVATION_H_
#define HPTS_COMMON_OBSERVATION_H_

#include <array>
#include <vector>

namespace hpts {

using Observation = std::vector<float>;
struct ObservationShape {
    int c = 0;    // Number of channels NOLINT (misc-non-private-member-variable-in-classes)
    int h = 0;    // Height of observation NOLINT (misc-non-private-member-variable-in-classes)
    int w = 0;    // Width of observation NOLINT (misc-non-private-member-variable-in-classes)
    ObservationShape() = default;
    ~ObservationShape() = default;
    ObservationShape(int channels, int height, int width)
        : c(channels), h(height), w(width) {}
    ObservationShape(const std::array<int, 3> &shape)
        : c(shape[0]), h(shape[1]), w(shape[2]) {}
    ObservationShape(const std::array<std::size_t, 3> &shape)
        : c(static_cast<int>(shape[0])), h(static_cast<int>(shape[1])), w(static_cast<int>(shape[2])) {}
    ObservationShape(const ObservationShape &) = default;
    ObservationShape(ObservationShape &&) = default;
    ObservationShape &operator=(const ObservationShape &) = default;
    ObservationShape &operator=(ObservationShape &&) = default;
    auto operator==(const ObservationShape &rhs) const -> bool {
        return c == rhs.c && h == rhs.h && w == rhs.w;
    }
    auto operator!=(const ObservationShape &rhs) const -> bool {
        return c != rhs.c || h != rhs.h || w != rhs.w;
    }
    [[nodiscard]] auto flat_size() const -> int {
        return c * h * w;
    }
};

}    // namespace hpts

#endif    // HPTS_COMMON_OBSERVATION_H_
