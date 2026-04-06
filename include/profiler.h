#pragma once
#include "workload.h"
#include <vector>
#include <string>
#include <chrono>

namespace acr {

class Profiler {
public:
    void record(const ExecutionResult& r);

    // Print summary table to stdout.
    void print_summary() const;

    // Dump CSV to file for offline Python analysis.
    void dump_csv(const std::string& path) const;

    const std::vector<ExecutionResult>& results() const { return results_; }

    // Aggregate stats helpers
    double avg_wall_ms() const;
    double avg_kernel_ms() const;
    double avg_h2d_ms() const;
    double avg_d2h_ms() const;

    // Per-path breakdown
    struct PathStats {
        ExecutionPath path;
        size_t count;
        double avg_wall_ms;
        double avg_kernel_ms;
    };
    std::vector<PathStats> per_path_stats() const;

private:
    std::vector<ExecutionResult> results_;
};

// RAII wall-clock timer returning milliseconds.
struct WallTimer {
    std::chrono::high_resolution_clock::time_point start;
    WallTimer() : start(std::chrono::high_resolution_clock::now()) {}
    double elapsed_ms() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start).count();
    }
};

} // namespace acr
