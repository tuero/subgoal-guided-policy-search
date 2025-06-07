// File: metrics_tracker.cpp
// Description: Holds metrics from search, and saves to file

#include "common/metrics_tracker.h"

#include <absl/strings/str_split.h>

#include <stdexcept>

namespace hpts {

void ProblemMetrics::dump_header(std::ostream &os) {
    os << "iter,puzzle_name,solution_found,solution_cost,solution_prob,solution_prob_raw,expanded,generated,"
          "time,budget\n";
}

auto operator<<(std::ostream &os, const ProblemMetrics &metrics_item) -> std::ostream & {
    os << metrics_item.iter << "," << metrics_item.puzzle_name << "," << metrics_item.solution_found << ","
       << metrics_item.solution_cost << "," << metrics_item.solution_prob << "," << metrics_item.solution_prob_raw
       << "," << metrics_item.expanded << "," << metrics_item.generated << "," << metrics_item.time << ","
       << metrics_item.budget << "\n";
    return os;
}
auto ProblemMetrics::make_from_str(const std::string &str) -> ProblemMetrics {
    std::vector<std::string> strs = absl::StrSplit(str, ',');
    if (strs.size() != 10) {
        SPDLOG_ERROR("Error reading line {:s}, {:d}", str, strs.size());
        throw std::runtime_error("line does not contain valid data for this metric");
    }
    return {
        .iter = std::stoi(strs[0]),
        .puzzle_name = strs[1],
        .solution_found = static_cast<bool>(std::stoi(strs[2])),
        .solution_cost = std::stod(strs[3]),
        .solution_prob = std::stod(strs[4]),
        .solution_prob_raw = std::stod(strs[5]),
        .expanded = std::stoi(strs[6]),
        .generated = std::stoi(strs[7]),
        .time = std::stod(strs[8]),
        .budget = std::stoi(strs[9]),
    };
}

// ----------------------------

void MemoryMetrics::dump_header(std::ostream &os) {
    os << "iter,max_rss\n";
}

auto operator<<(std::ostream &os, const MemoryMetrics &metrics_item) -> std::ostream & {
    os << metrics_item.iter << "," << metrics_item.max_rss << "\n";
    return os;
}
auto MemoryMetrics::make_from_str(const std::string &str) -> MemoryMetrics {
    std::vector<std::string> strs = absl::StrSplit(str, ',');
    if (strs.size() != 2) {
        SPDLOG_ERROR("Error reading line {:s}, {:d}", str, strs.size());
        throw std::runtime_error("line does not contain valid data for this metric");
    }
    return {.iter = std::stoi(strs[0]), .max_rss = std::stod(strs[1])};
}

// ----------------------------

void OutstandingMetrics::dump_header(std::ostream &os) {
    os << "expansions,outstanding_problems\n";
}

auto operator<<(std::ostream &os, const OutstandingMetrics &metrics_item) -> std::ostream & {
    os << metrics_item.expansions << "," << metrics_item.outstanding_problems << "\n";
    return os;
}
auto OutstandingMetrics::make_from_str(const std::string &str) -> OutstandingMetrics {
    std::vector<std::string> strs = absl::StrSplit(str, ',');
    if (strs.size() != 2) {
        SPDLOG_ERROR("Error reading line {:s}, {:d}", str, strs.size());
        throw std::runtime_error("line does not contain valid data for this metric");
    }
    return {.expansions = std::stoi(strs[0]), .outstanding_problems = std::stoi(strs[1])};
}

// ----------------------------

void TimeMetrics::dump_header(std::ostream &os) {
    os << "total_time_cpu,total_time_wall,outstanding_problems\n";
}

auto operator<<(std::ostream &os, const TimeMetrics &metrics_item) -> std::ostream & {
    os << metrics_item.total_time_cpu << "," << metrics_item.total_time_wall << "," << metrics_item.outstanding_problems
       << "\n";
    return os;
}
auto TimeMetrics::make_from_str(const std::string &str) -> TimeMetrics {
    std::vector<std::string> strs = absl::StrSplit(str, ',');
    if (strs.size() != 3) {
        SPDLOG_ERROR("Error reading line {:s}, {:d}", str, strs.size());
        throw std::runtime_error("line does not contain valid data for this metric");
    }
    return {
        .total_time_cpu = std::stod(strs[0]),
        .total_time_wall = std::stod(strs[1]),
        .outstanding_problems = std::stoi(strs[2])
    };
}

}    // namespace hpts
