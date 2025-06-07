// File: utility.cpp
// Description: Utility helper functions

#include "util/utility.h"

#include "util/zip.h"

#include <sys/resource.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>

namespace hpts {

constexpr double SMALL_E = 1e-8;    // Ensure log(0) doesn't happen
const double SQRT_2 = std::sqrt(2.0);
constexpr double PI = 3.14159265358979;

long get_mem_usage() {
    struct rusage usage {};
    int ret{};
    ret = getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;    // in KB
}

auto log_policy_noise(const std::vector<double> &policy, double epsilon) -> std::vector<double> {
    std::vector<double> log_policy;
    log_policy.reserve(policy.size());
    const double noise = 1.0 / static_cast<double>(policy.size());
    for (const auto p : policy) {
        log_policy.push_back(std::log(((1.0 - epsilon) * p) + (epsilon * noise) + SMALL_E));
    }
    return log_policy;
}
auto log_policy_noise(std::vector<double> &&policy, double epsilon) -> std::vector<double> {
    if (epsilon == 0) {
        return policy;
    }
    const double noise = 1.0 / static_cast<double>(policy.size());
    for (auto &p : policy) {
        p = std::log(((1.0 - epsilon) * p) + (epsilon * noise) + SMALL_E);
    }
    return policy;
}

template <typename T>
auto vec_sum(const std::vector<T> &values) -> T {
    T sum{};
    for (const auto &value : values) {
        sum += value;
    }
    return sum;
}

auto scalar_mul(std::vector<double> &&values, double alpha) -> std::vector<double> {
    for (auto &value : values) {
        value *= alpha;
    }
    return values;
}
auto scalar_mul(const std::vector<double> &values, double alpha) -> std::vector<double> {
    std::vector<double> result = values;
    for (auto &r : result) {
        r *= alpha;
    }
    return result;
}

auto softmax(const std::vector<double> &values, double temperature) -> std::vector<double> {
    std::vector<double> new_values = values;
    for (double &v : new_values) {
        v *= temperature;
    }
    const double max_value = *std::max_element(std::begin(new_values), std::end(new_values));
    const double
        sum{std::accumulate(std::begin(new_values), std::end(new_values), 0.0, [&](double left_sum, double next) {
            return left_sum + std::exp(next - max_value);
        })};
    const double k = max_value + std::log(sum);
    std::vector<double> output;
    output.reserve(new_values.size());
    for (auto const &v : new_values) {
        output.push_back(std::exp(v - k));
    }
    return output;
}
void softmax(std::vector<double> &values, double temperature) {
    std::vector<double> new_values = values;
    for (double &v : new_values) {
        v *= temperature;
    }
    const double max_value = *std::max_element(std::begin(new_values), std::end(new_values));
    const double
        sum{std::accumulate(std::begin(new_values), std::end(new_values), 0.0, [&](double left_sum, double next) {
            return left_sum + std::exp(next - max_value);
        })};
    const double k = max_value + std::log(sum);
    for (std::size_t i = 0; i < new_values.size(); ++i) {
        values[i] = std::exp(new_values[i] - k);
    }
}

auto geo_mix_policy(
    const std::vector<std::vector<double>> &vs,
    const std::vector<double> &alphas,
    std::size_t policy_size,
    bool normalize
) -> std::vector<double> {
    assert(vs.size() == alphas.size());
    std::vector<double> result(policy_size, 0);
    for (std::size_t i = 0; i < policy_size; ++i) {
        for (std::size_t j = 0; j < alphas.size(); ++j) {
            result[i] += std::log(vs.at(j).at(i) + SMALL_E) * alphas.at(j);
        }
        result[i] = std::exp(result[i]);
    }
    return normalize ? scalar_mul(std::move(result), 1.0 / vec_sum(result)) : result;
}

}    // namespace hpts
