// File: timer.cpp
// Description: Measures user space time and signals a timeout

#include "common/timer.h"

namespace hpts {

TimerCPU::TimerCPU(double seconds_limit)
    : _seconds_limit(seconds_limit) {}

void TimerCPU::start() noexcept {
    _cpu_start_time = std::clock();
}

auto TimerCPU::is_timeout() const noexcept -> bool {
    const std::clock_t cpu_current_time = std::clock();
    const auto current_duration = static_cast<double>(cpu_current_time - _cpu_start_time) / CLOCKS_PER_SEC;
    return _seconds_limit > 0 && current_duration >= _seconds_limit;
}

auto TimerCPU::get_duration() const noexcept -> double {
    const std::clock_t cpu_current_time = std::clock();
    const auto current_duration = static_cast<double>(cpu_current_time - _cpu_start_time) / CLOCKS_PER_SEC;
    return current_duration;
}

auto TimerCPU::get_time_remaining() const noexcept -> double {
    const std::clock_t cpu_current_time = std::clock();
    const auto current_duration = static_cast<double>(cpu_current_time - _cpu_start_time) / CLOCKS_PER_SEC;
    return _seconds_limit - current_duration;
}

// -----------------------

constexpr int MILLISECONDS_PER_SECOND = 1000;

TimerWall::TimerWall(double seconds_limit)
    : _seconds_limit(seconds_limit) {}

void TimerWall::start() noexcept {
    _wall_start_time = std::chrono::high_resolution_clock::now();
}

auto TimerWall::is_timeout() const noexcept -> bool {
    const auto ellapsed = std::chrono::high_resolution_clock::now() - _wall_start_time;
    auto duration_count = std::chrono::duration_cast<std::chrono::milliseconds>(ellapsed).count();
    auto current_duration = static_cast<double>(duration_count) / MILLISECONDS_PER_SECOND;
    return _seconds_limit > 0 && current_duration >= _seconds_limit;
}

auto TimerWall::get_duration() const noexcept -> double {
    const auto ellapsed = std::chrono::high_resolution_clock::now() - _wall_start_time;
    auto duration_count = std::chrono::duration_cast<std::chrono::milliseconds>(ellapsed).count();
    auto current_duration = static_cast<double>(duration_count) / MILLISECONDS_PER_SECOND;
    return current_duration;
}

auto TimerWall::get_time_remaining() const noexcept -> double {
    const auto ellapsed = std::chrono::high_resolution_clock::now() - _wall_start_time;
    auto duration_count = std::chrono::duration_cast<std::chrono::milliseconds>(ellapsed).count();
    auto current_duration = static_cast<double>(duration_count) / MILLISECONDS_PER_SECOND;
    return _seconds_limit - current_duration;
}

}    // namespace hpts
