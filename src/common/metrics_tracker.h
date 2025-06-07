// File: metrics_tracker.h
// Description: Holds metrics from search, and saves to file

#ifndef HPTS_UTIL_METRICS_TRACKER_H_
#define HPTS_UTIL_METRICS_TRACKER_H_

#include <spdlog/spdlog.h>

#include <absl/strings/str_format.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace hpts {

// Specific problem instance being solved
struct ProblemMetrics {
    int iter;
    std::string puzzle_name;
    bool solution_found;
    double solution_cost;
    double solution_prob;
    double solution_prob_raw;
    int expanded;
    int generated;
    double time;
    int budget;
    static auto make_from_str(const std::string &str) -> ProblemMetrics;
    static void dump_header(std::ostream &os);
    friend auto operator<<(std::ostream &os, const ProblemMetrics &metrics_item) -> std::ostream &;
};

// Total memory usage
struct MemoryMetrics {
    int iter;
    double max_rss;
    static auto make_from_str(const std::string &str) -> MemoryMetrics;
    static void dump_header(std::ostream &os);
    friend auto operator<<(std::ostream &os, const MemoryMetrics &metrics_item) -> std::ostream &;
};

// Outstanding problems by total expansions
struct OutstandingMetrics {
    int64_t expansions;
    int outstanding_problems;
    static auto make_from_str(const std::string &str) -> OutstandingMetrics;
    static void dump_header(std::ostream &os);
    friend auto operator<<(std::ostream &os, const OutstandingMetrics &metrics_item) -> std::ostream &;
};

// Total time metric
struct TimeMetrics {
    double total_time_cpu;
    double total_time_wall;
    int outstanding_problems;
    static auto make_from_str(const std::string &str) -> TimeMetrics;
    static void dump_header(std::ostream &os);
    friend auto operator<<(std::ostream &os, const TimeMetrics &metrics_item) -> std::ostream &;
};

// Generic tracker for the above metrics type
template <typename MetricsItem>
class MetricsTracker {
public:
    MetricsTracker() = delete;
    MetricsTracker(const std::string &export_path, const std::string &file_name, bool resume = false)
        : full_path((absl::StrFormat("%s/%s.csv", export_path, file_name))) {
        // create directory for metrics
        if (std::filesystem::exists(full_path)) {
            if (resume) {
                std::ifstream export_file(full_path);
                std::string line;
                // Read the header
                std::getline(export_file, line);
                // Load data
                while (std::getline(export_file, line)) {
                    rows.push_back(MetricsItem::make_from_str(line));
                }
                export_file.close();
            }
            std::filesystem::remove(full_path);
        }
        std::filesystem::create_directories(export_path);
        save_header();
    }

    void save_header() {
        std::ofstream export_file(full_path, std::ofstream::app | std::ofstream::out);
        MetricsItem::dump_header(export_file);
    }

    void add_row(MetricsItem &&metrics_item) noexcept {
        rows.push_back(std::forward<MetricsItem>(metrics_item));
    }

    void clear() noexcept {
        rows.clear();
    }

    void save() noexcept {
        if (rows.empty()) {
            return;
        }

        // We assume the export parent directory exists and can be written to
        std::ofstream export_file(full_path, std::ofstream::app | std::ofstream::out);

        SPDLOG_INFO("Exporting metrics to {:s}", full_path);
        for (auto const &row : rows) {
            export_file << row;
        }
        export_file.close();
        rows.clear();
    }

    std::vector<MetricsItem> rows;

private:
    std::string full_path;
};

}    // namespace hpts

#endif    // HPTS_UTIL_METRICS_TRACKER_H_
