// File: timer.h
// Description: Measures user space time and signals a timeout

#ifndef HPTS_UTIL_TIMER_H_
#define HPTS_UTIL_TIMER_H_

#include <chrono>
#include <ctime>

namespace hpts {

class TimerCPU {
    // Ensure clock_t is a 64 bit value
    // A 32 bit clock_t width will overflow after ~72 minutes which is longer than the expected runtime.
    // A 64 bit clock_t width will overflow after ~300,00 years
    constexpr static int BYTE_CHECK = 8;
    static_assert(sizeof(std::clock_t) == BYTE_CHECK);

public:
    TimerCPU() = delete;
    TimerCPU(double seconds_limit);

    void start() noexcept;

    [[nodiscard]] auto is_timeout() const noexcept -> bool;

    [[nodiscard]] auto get_duration() const noexcept -> double;

    [[nodiscard]] auto get_time_remaining() const noexcept -> double;

private:
    double _seconds_limit;
    std::clock_t _cpu_start_time = 0;
};

class TimerWall {
public:
    TimerWall() = delete;
    TimerWall(double seconds_limit);

    void start() noexcept;

    [[nodiscard]] auto is_timeout() const noexcept -> bool;

    [[nodiscard]] auto get_duration() const noexcept -> double;

    [[nodiscard]] auto get_time_remaining() const noexcept -> double;

private:
    double _seconds_limit;
    std::chrono::time_point<std::chrono::high_resolution_clock> _wall_start_time;
};

}    // namespace hpts

#endif    // HPTS_UTIL_TIMER_H_
