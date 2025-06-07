// File: utility.h
// Description: Utility helper functions

#ifndef HPTS_UTIL_UTILITY_H_
#define HPTS_UTIL_UTILITY_H_

#include "util/concepts.h"
#include "util/zip.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace hpts {

/**
 * Split a list of items into a train and validate set
 * @param items The entire list of items to split
 * @param num_train Number of items to place in the train set
 * @param num_validate Number of items to place into the validation set
 * @param seed The seed used on the data shuffling
 * @return Pair of train and validation sets
 */
template <typename T>
auto split_train_validate(std::vector<T> &items, std::size_t num_train, std::size_t num_validate, int seed) {
    if (items.size() < num_train + num_validate) {
        SPDLOG_ERROR(
            "Input items {:d} is less than num_train {:d} + num_validate {:d}",
            items.size(),
            num_train,
            num_validate
        );
        std::exit(1);
    }
    assert(items.size() >= num_train + num_validate);
    std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
    std::shuffle(items.begin(), items.end(), rng);
    return std::make_pair(
        std::vector<T>(items.begin(), items.begin() + num_train),
        std::vector<T>(items.begin() + num_train, items.begin() + num_train + num_validate)
    );
}

/**
 * Split a vector if items into batches
 * @param items The tiems to split
 * @param batch_size Size of each batch
 * @return Vector with each item containing batch_size number of items from the input vector
 */
template <typename T>
    requires IsSpecialization<typename std::decay_t<T>, std::vector>
auto split_to_batch(T &&items, std::size_t batch_size) {
    // Get underlying type and match with lvalue or rvalue from perfect forwarding vector
    using U = template_parameter_t<typename std::decay<T>::type>;
    using V = std::conditional_t<std::is_lvalue_reference<T>::value, U &, U &&>;

    std::vector<std::vector<U>> batches;
    for (int i = 0; i < std::ceil(static_cast<double>(items.size()) / static_cast<double>(batch_size)); ++i) {
        batches.push_back({});
        batches.back().reserve(batch_size);
    }

    for (auto &&[i, item] : hpts::enumerate(items)) {
        batches[i / batch_size].push_back(std::forward<V>(item));
    }
    return batches;
}

template <typename T>
auto vec_to_str(const std::vector<T> &vec) -> std::string {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i != 0) {
            ss << ",";
        }
        ss << vec[i];
    }
    ss << "]";
    return ss.str();
}

/**
 * Get maximum memory usage used thus far
 * @return memory usage in KB
 */
long get_mem_usage();

/**
 * Multiply a vector by a scalar result
 */
auto scalar_mul(const std::vector<double> &values, double alpha) -> std::vector<double>;
auto scalar_mul(const std::vector<double> &values, double alpha) -> std::vector<double>;

/**
 * Apply log + uniform mixture to policy
 * @param policy The policy
 * @param epislon Amount of mixing with uniform policy, between 0 and 1.
 * @return Vector of policy with log + uniform mixture applied
 */
auto log_policy_noise(const std::vector<double> &policy, double epsilon = 0) -> std::vector<double>;
auto log_policy_noise(std::vector<double> &&policy, double epsilon = 0) -> std::vector<double>;

/**
 * Apply softmax to vector of values
 * @param values The values
 * @return Softmax of given values
 */
auto softmax(const std::vector<double> &values, double temperature = 1) -> std::vector<double>;
void softmax(std::vector<double> &values, double temperature = 1);

/**
 * Geometric mixing of a collection of policies, weighted by the alphas
 * @param vs Vector containing the policies
 * @param alphas Mixing factor of same length as vs
 * @param policy_size Length of policy (should match each inner size of vs)
 * @param normalize Flag to normalize as a proper policy
 */
auto geo_mix_policy(
    const std::vector<std::vector<double>> &vs,
    const std::vector<double> &alphas,
    std::size_t policy_size,
    bool normalize = true
) -> std::vector<double>;

}    // namespace hpts

#endif    // HPTS_UTIL_UTILITY_H_
