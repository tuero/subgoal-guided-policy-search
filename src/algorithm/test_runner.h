// File: test_runner.h
// Description: Generic test runner

#ifndef HPTS_ALGORITHM_TEST_BOOTSTRAP_H_
#define HPTS_ALGORITHM_TEST_BOOTSTRAP_H_

#include "common/metrics_tracker.h"
#include "common/timer.h"
#include "util/stop_token.h"
#include "util/thread_pool.h"
#include "util/utility.h"
#include "util/zip.h"

#include <spdlog/spdlog.h>

#include <absl/strings/str_cat.h>

namespace hpts::algorithm {

constexpr int KB_PER_MB = 1024;

// clang-format off
template <typename T>
concept IsTestInput = requires(T t, const T ct) {
    { t.puzzle_name } -> std::same_as<std::string &>;
    { t.search_budget } -> std::same_as<int &>;
};

template <typename T>
concept IsTestOutput = requires(T t, const T ct) {
    { t.puzzle_name } -> std::same_as<std::string &>;
    { t.solution_found } -> std::same_as<bool &>;
    { t.solution_cost } -> std::same_as<double &>;
    { t.num_expanded } -> std::same_as<int &>;
    { t.num_generated } -> std::same_as<int &>;
    { t.solution_prob } -> std::same_as<double &>;
    { t.solution_prob_raw } -> std::same_as<double &>;
    { t.time } -> std::same_as<double &>;
};

// clang-format on

template <typename SearchInputT, typename SearchOutputT>
    requires IsTestInput<SearchInputT> && IsTestOutput<SearchOutputT>
void test_runner(
    std::vector<SearchInputT> problems,
    std::function<SearchOutputT(const SearchInputT &)> algorithm,
    const std::string &output_dir,
    std::shared_ptr<StopToken> stop_token,
    int initial_search_budget,
    int num_threads,
    int max_iterations,
    double time_budget,
    std::string export_suffix
) {
    int bootstrap_iter = 0;
    int64_t total_expansions = 0;
    int64_t total_generated = 0;
    double total_cost = 0;
    int search_budget = initial_search_budget;
    std::vector<SearchInputT> outstanding_problems = problems;

    // Create metrics logger + directory
    const std::string metrics_dir = absl::StrCat(output_dir, "/metrics");
    if (export_suffix != "") {
        export_suffix = absl::StrCat("_", export_suffix);
    }
    MetricsTracker<ProblemMetrics> metrics_tracker(metrics_dir, absl::StrCat("test", export_suffix));
    MetricsTracker<TimeMetrics> time_tracker(metrics_dir, absl::StrCat("test_time", export_suffix));
    MetricsTracker<MemoryMetrics> memory_tracker(metrics_dir, absl::StrCat("test_memory", export_suffix));

    ThreadPool<SearchInputT, SearchOutputT> pool(num_threads);

    TimerCPU timer_cpu(time_budget);
    TimerWall timer_wall(time_budget);
    timer_cpu.start();
    timer_wall.start();

    while (!timer_cpu.is_timeout() && bootstrap_iter < max_iterations && !outstanding_problems.empty()) {
        ++bootstrap_iter;
        SPDLOG_INFO("Bootstrap iteration: {:d} of {:d}", bootstrap_iter, max_iterations);
        SPDLOG_INFO("Remaining unsolved problems: {:d}", outstanding_problems.size());

        // Update problem instance budget
        for (auto &p : outstanding_problems) {
            p.search_budget = search_budget;
        }
        std::vector<SearchInputT> unsolved_problems;

        // Shuffle training and iterate
        auto batched_problems = split_to_batch(outstanding_problems, num_threads);
        for (auto &&[batch_idx, batch] : enumerate(batched_problems)) {
            if (stop_token->stop_requested()) {
                SPDLOG_INFO("Stop requested, exiting train batch loop");
                break;
            }
            std::vector<SearchOutputT> results = pool.run(algorithm, batch);
            if (stop_token->stop_requested()) {
                SPDLOG_INFO("Stop requested, exiting train batch loop");
                break;
            }

            for (auto &&[input_problem, result] : zip(batch, results)) {
                metrics_tracker.add_row(
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
                if (result.solution_found) {
                    total_generated += result.num_generated;
                    total_expansions += result.num_expanded;
                    total_cost += result.solution_cost;
                } else {
                    unsolved_problems.push_back(input_problem);
                }
            }
        }

        // Track max memory usage in megabytes
        memory_tracker.add_row({bootstrap_iter, static_cast<double>(get_mem_usage()) / KB_PER_MB});
        memory_tracker.save();

        outstanding_problems = unsolved_problems;

        // Unconditionally double budget
        search_budget *= 2;

        if (stop_token->stop_requested()) {
            SPDLOG_INFO("Stop requested, exiting test iteration");
            break;
        }
    }

    double total_time_cpu = timer_cpu.get_duration();
    double total_time_wall = timer_wall.get_duration();
    time_tracker.add_row({total_time_cpu, total_time_wall});

    // Export
    metrics_tracker.save();
    time_tracker.save();
    memory_tracker.save();

    SPDLOG_INFO(
        "Total time cpu: {:.2f}s, total time wall: {:.2f}s total exp: {:d}, total gen: {:d}, total cost: {:.2f}",
        total_time_cpu,
        total_time_wall,
        total_expansions,
        total_generated,
        total_cost
    );
    SPDLOG_INFO("Maximum resident usage: {:.2f}MB", static_cast<double>(get_mem_usage()) / 1024);
}

}    // namespace hpts::algorithm

#endif    // HPTS_ALGORITHM_TEST_BOOTSTRAP_H_
