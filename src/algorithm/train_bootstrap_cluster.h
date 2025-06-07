// File: train_bootstrap.h
// Description: Generic train runner based on the bootstrap method
// Shahab Jabbari Arfaee, Sandra Zilles, and Robert C. Holte. Learning heuristic functions for large state spaces, 2011.

#ifndef HPTS_ALGORITHM_TRAIN_BOOTSTRAP_H_
#define HPTS_ALGORITHM_TRAIN_BOOTSTRAP_H_

#include "algorithm/hphs_mix_cluster.h"
#include "common/metrics_tracker.h"
#include "common/timer.h"
#include "util/stop_token.h"
#include "util/thread_pool.h"
#include "util/utility.h"

#include <spdlog/spdlog.h>

#include <absl/strings/str_cat.h>

#include <random>
#include <unordered_set>

namespace hpts::algorithm {

constexpr int CHECKPOINT_INTERVAL = 10;
constexpr int KB_PER_MB = 1024;

// clang-format off
template <typename T>
concept IsTrainInput = requires(T t, const T ct) {
    { t.puzzle_name } -> std::same_as<std::string &>;
    { t.search_budget } -> std::same_as<int &>;
};

template <typename T>
concept IsTrainOutput = requires(T t, const T ct) {
    { t.puzzle_name } -> std::same_as<std::string &>;
    { t.solution_found } -> std::same_as<bool &>;
    { t.solution_cost } -> std::same_as<double &>;
    { t.num_expanded } -> std::same_as<int &>;
    { t.num_generated } -> std::same_as<int &>;
    { t.solution_prob } -> std::same_as<double &>;
    { t.solution_prob_raw } -> std::same_as<double &>;
};

// Concept for learning handler to satisfy requirements to interface with training
template <typename T, typename SearchInput, typename SearchOutput>
concept IsLearningHandler = requires(T t, std::vector<SearchInput> &batch, std::vector<SearchOutput> &&results, std::mt19937 &rng) {
    { t.preprocess(batch) } -> std::same_as<void>;
    { t.process_data(std::move(results)) } -> std::same_as<void>;
    { t.learning_step(rng) } -> std::same_as<void>;
    { t.checkpoint() } -> std::same_as<void>;
};
// clang-format on

enum class BootstrapPolicy {
    DOUBLE = 0,
    LTS_CM = 1
};

constexpr double default_bootstrap_factor = 0.1;

template <typename SearchInputT>
using AlgorithmOutputT = algorithm::hphs_mix::HPHSMix<typename SearchInputT::env_t, typename SearchInputT::wrapper_t>;
template <typename SearchInputT>
using AlgorithmOutputSharedT =
    std::shared_ptr<algorithm::hphs_mix::HPHSMix<typename SearchInputT::env_t, typename SearchInputT::wrapper_t>>;

template <typename SearchInputT>
auto run_alg(const SearchInputT &input) -> AlgorithmOutputSharedT<SearchInputT> {
    auto step_hphs = std::make_shared<AlgorithmOutputT<SearchInputT>>(input);
    step_hphs->run();
    return step_hphs;
}

template <typename SearchInputT>
auto get_output(const AlgorithmOutputSharedT<SearchInputT> &step_hphs) -> algorithm::hphs_mix::SearchOutput {
    return step_hphs->get_search_output();
}

template <typename SearchInputT, typename SearchOutputT, typename LearnerT>
    requires IsTrainInput<SearchInputT> && IsTrainOutput<SearchOutputT>
             && IsLearningHandler<LearnerT, SearchInputT, SearchOutputT>
void train_bootstrap(
    std::vector<SearchInputT> problems_train,
    std::vector<SearchInputT> problems_validate,
    std::function<SearchOutputT(const SearchInputT &)> algorithm,
    LearnerT &learner,
    const std::string &output_dir,
    std::mt19937 rng,
    std::shared_ptr<StopToken> stop_token,
    int initial_search_budget,
    double validation_solved_ratio,
    int num_threads,
    int num_problems_per_batch,
    int max_iterations,
    int max_budget,
    double time_budget,
    BootstrapPolicy bootstrap_policy = BootstrapPolicy::DOUBLE,
    double bootstrap_factor = default_bootstrap_factor,
    int extra_iterations = 0
) {
    int bootstrap_iter = 0;
    int search_budget = initial_search_budget;
    int validation_budget = initial_search_budget;
    int64_t total_expansions = 0;
    int n_validate_exit = static_cast<int>(static_cast<double>(problems_validate.size()) * validation_solved_ratio);
    std::unordered_set<std::string> solved_set_train;
    std::unordered_set<std::string> solved_set_validate;
    bool last_iteration = false;

    // If more threads than problems per batch, reduce thread usage to match
    if (num_problems_per_batch < num_threads) {
        num_threads = num_problems_per_batch;
    }

    // Create metrics logger + directory
    const std::string metrics_dir = absl::StrCat(output_dir, "/metrics");
    MetricsTracker<ProblemMetrics> metrics_tracker_train(metrics_dir, "train");
    MetricsTracker<ProblemMetrics> metrics_tracker_validate(metrics_dir, "validate");
    MetricsTracker<MemoryMetrics> memory_tracker(metrics_dir, "memory");
    MetricsTracker<OutstandingMetrics> outstanding_problems_tracker(metrics_dir, "outstanding_problems");
    MetricsTracker<TimeMetrics> time_tracker(metrics_dir, "train_time");

    // Initialize metrics
    outstanding_problems_tracker.add_row(
        {total_expansions, static_cast<int>(problems_train.size() - solved_set_train.size())}
    );

    ThreadPool<SearchInputT, AlgorithmOutputSharedT<SearchInputT>> pool_search(num_threads / 4);
    ThreadPool<AlgorithmOutputSharedT<SearchInputT>, SearchOutputT> pool_output(num_threads);
    learner.checkpoint();

    TimerCPU timer_cpu(time_budget);
    TimerWall timer_wall(time_budget);
    timer_cpu.start();
    timer_wall.start();

    while (!timer_cpu.is_timeout() && bootstrap_iter < max_iterations && search_budget <= max_budget) {
        ++bootstrap_iter;
        int n_outstanding_train = problems_train.size() - solved_set_train.size();
        int n_outstanding_validate = problems_validate.size() - solved_set_validate.size();
        std::size_t prev_n_solved_train = solved_set_train.size();
        long long int solved_expansions = 0;    // Expansions required for solved problems this iter
        int curr_n_solved_train = 0;
        SPDLOG_INFO("Bootstrap iteration: {:d} of {:d}", bootstrap_iter, max_iterations);
        SPDLOG_INFO(
            "Remaining unsolved problems: Train = {:d}, Validate = {:d}",
            n_outstanding_train,
            n_outstanding_validate
        );

        // Update problem instance budget
        for (auto &p : problems_train) {
            p.search_budget = search_budget;
        }
        for (auto &p : problems_validate) {
            p.search_budget = validation_budget;
        }

        // Shuffle training and iterate
        std::shuffle(problems_train.begin(), problems_train.end(), rng);
        auto batched_problems_train = split_to_batch(problems_train, num_problems_per_batch);
        for (auto &&[batch_idx, batch] : enumerate(batched_problems_train)) {
            SPDLOG_INFO(
                "Iteration {:d}, Batch: {:d} of {:d}, CPU time: {:.2f}, Wall time: {:.2f}",
                bootstrap_iter,
                batch_idx,
                batched_problems_train.size(),
                timer_cpu.get_duration(),
                timer_wall.get_duration()
            );

            if (stop_token->stop_requested()) {
                SPDLOG_INFO("Stop requested, exiting train batch loop");
                break;
            }
            if (timer_cpu.is_timeout()) {
                SPDLOG_INFO("Timeout, exiting train batch loop");
                break;
            }
            learner.preprocess(batch);
            // std::vector<SearchOutputT> results = pool.run(algorithm, batch);
            std::vector<AlgorithmOutputSharedT<SearchInputT>> algorithm_states =
                pool_search.run(run_alg<SearchInputT>, batch);
            std::vector<SearchOutputT> results = pool_output.run(get_output<SearchInputT>, algorithm_states);
            if (stop_token->stop_requested()) {
                SPDLOG_INFO("Stop requested, exiting train batch loop");
                break;
            }

            for (const auto &result : results) {
                metrics_tracker_train.add_row(
                    {bootstrap_iter,
                     result.puzzle_name,
                     result.solution_found,
                     result.solution_cost,
                     result.solution_prob,
                     result.solution_prob_raw,
                     result.num_expanded,
                     result.num_generated,
                     result.time,
                     search_budget}
                );
                total_expansions += result.num_expanded;
                if (result.solution_found) {
                    solved_set_train.insert(result.puzzle_name);
                    solved_expansions += result.num_expanded;
                    ++curr_n_solved_train;
                }
            }
            // Process results
            learner.process_data(std::move(results));

            // Update model
            learner.learning_step(rng);

            // Metrics for outstanding problems
            outstanding_problems_tracker.add_row(
                {total_expansions, static_cast<int>(problems_train.size() - solved_set_train.size())}
            );
            metrics_tracker_train.save();
            outstanding_problems_tracker.save();

            // Checkpoint model
            if (batch_idx % CHECKPOINT_INTERVAL == 0) {
                learner.checkpoint();
            }
        }

        // Track max memory usage in megabytes
        memory_tracker.add_row({bootstrap_iter, static_cast<double>(get_mem_usage()) / KB_PER_MB});
        memory_tracker.save();

        if (stop_token->stop_requested()) {
            SPDLOG_INFO("Stop requested, exiting iteration loop");
            break;
        }

        SPDLOG_INFO("Running Validation Iteration");
        auto batched_problems_validate = split_to_batch(problems_validate, num_problems_per_batch);
        for (auto &&[batch_idx, batch] : enumerate(batched_problems_validate)) {
            // std::vector<SearchOutputT> results = pool.run(algorithm, batch);
            std::vector<AlgorithmOutputSharedT<SearchInputT>> algorithm_states =
                pool_search.run(run_alg<SearchInputT>, batch);
            std::vector<SearchOutputT> results = pool_output.run(get_output<SearchInputT>, algorithm_states);

            if (stop_token->stop_requested()) {
                SPDLOG_INFO("Stop requested, exiting validation batch loop");
                break;
            }

            for (const auto &result : results) {
                metrics_tracker_validate.add_row(
                    {bootstrap_iter,
                     result.puzzle_name,
                     result.solution_found,
                     result.solution_cost,
                     result.solution_prob,
                     result.solution_prob_raw,
                     result.num_expanded,
                     result.num_generated,
                     result.time,
                     initial_search_budget}
                );
                if (result.solution_found) {
                    solved_set_validate.insert(result.puzzle_name);
                }
            }
        }
        // Break out if stop requested
        if (stop_token->stop_requested()) {
            SPDLOG_INFO("Stop requested, exiting bootstrap loop");
            break;
        }

        if (last_iteration) {
            if (extra_iterations <= 0) {
                break;
            }
            --extra_iterations;
        }

        if (solved_set_validate.size() >= n_validate_exit && !last_iteration) {
            SPDLOG_INFO("Solved validation set ratio exceeded.");
            SPDLOG_INFO("Running one more pass over training set.");
            last_iteration = true;
            // Don't check if we should modify budget, as this was enough to solve validation set
            continue;
        }

        switch (bootstrap_policy) {
        case BootstrapPolicy::DOUBLE:
            // If no new problems in the training solved, double budget
            if (prev_n_solved_train == solved_set_train.size() && n_outstanding_train > 0) {
                search_budget *= 2;
            }
            break;
        case BootstrapPolicy::LTS_CM:
            // If more than factor increase, reduce budget. Else double
            // if (curr_n_solved_train > static_cast<std::size_t>((1 + bootstrap_factor) * (double)prev_n_solved_train))
            // {
            // New solved problems is more than X percentage of train size
            bool flag1 = (solved_set_train.size() - prev_n_solved_train) > (bootstrap_factor * problems_train.size());
            // Current iteration solves 1.X more than previous iteration
            bool flag2 = curr_n_solved_train > ((1 + bootstrap_factor) * (double)prev_n_solved_train);
            if (flag1 && flag2) {
                search_budget = std::max(initial_search_budget, search_budget / 2);
            } else if (n_outstanding_train > 0) {
                search_budget =
                    2 * search_budget + solved_expansions / (problems_train.size() - solved_set_train.size());
            }
            validation_budget = search_budget;
            break;
        }
    }

    double total_time_cpu = timer_cpu.get_duration();
    double total_time_wall = timer_wall.get_duration();
    time_tracker.add_row({total_time_cpu, total_time_wall});

    // Export
    learner.checkpoint();
    metrics_tracker_train.save();
    metrics_tracker_validate.save();
    memory_tracker.save();
    time_tracker.save();

    SPDLOG_INFO("Total cpu time: {:.2f}, wall time: {:.2f}", total_time_cpu, total_time_wall);
    SPDLOG_INFO("Maximum resident usage: {:.2f}MB", static_cast<double>(get_mem_usage()) / 1024);
}

}    // namespace hpts::algorithm

#endif    // HPTS_ALGORITHM_TRAIN_BOOTSTRAP_H_
