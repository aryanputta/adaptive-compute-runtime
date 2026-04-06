#include "profiler.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>

namespace acr {

static std::string path_name(ExecutionPath p) {
    switch (p) {
        case ExecutionPath::CPU_SERIAL:   return "CPU_SERIAL";
        case ExecutionPath::CPU_PARALLEL: return "CPU_PARALLEL";
        case ExecutionPath::GPU_DIRECT:   return "GPU_DIRECT";
        case ExecutionPath::GPU_BATCHED:  return "GPU_BATCHED";
        case ExecutionPath::HYBRID:       return "HYBRID";
        case ExecutionPath::DEFERRED:     return "DEFERRED";
        default:                          return "UNKNOWN";
    }
}

void Profiler::record(const ExecutionResult& r) {
    results_.push_back(r);
}

double Profiler::avg_wall_ms() const {
    if (results_.empty()) return 0.0;
    double sum = 0.0;
    for (const auto& r : results_) sum += r.wall_ms;
    return sum / static_cast<double>(results_.size());
}

double Profiler::avg_kernel_ms() const {
    if (results_.empty()) return 0.0;
    double sum = 0.0;
    for (const auto& r : results_) sum += r.kernel_ms;
    return sum / static_cast<double>(results_.size());
}

double Profiler::avg_h2d_ms() const {
    if (results_.empty()) return 0.0;
    double sum = 0.0;
    for (const auto& r : results_) sum += r.h2d_ms;
    return sum / static_cast<double>(results_.size());
}

double Profiler::avg_d2h_ms() const {
    if (results_.empty()) return 0.0;
    double sum = 0.0;
    for (const auto& r : results_) sum += r.d2h_ms;
    return sum / static_cast<double>(results_.size());
}

std::vector<Profiler::PathStats> Profiler::per_path_stats() const {
    std::map<ExecutionPath, std::vector<double>> wall_by_path;
    std::map<ExecutionPath, std::vector<double>> kern_by_path;

    for (const auto& r : results_) {
        wall_by_path[r.path_used].push_back(r.wall_ms);
        kern_by_path[r.path_used].push_back(r.kernel_ms);
    }

    std::vector<PathStats> out;
    for (auto& [path, walls] : wall_by_path) {
        PathStats ps{};
        ps.path  = path;
        ps.count = walls.size();
        ps.avg_wall_ms   = std::accumulate(walls.begin(), walls.end(), 0.0) / walls.size();
        auto& kerns      = kern_by_path[path];
        ps.avg_kernel_ms = std::accumulate(kerns.begin(), kerns.end(), 0.0) / kerns.size();
        out.push_back(ps);
    }
    return out;
}

void Profiler::print_summary() const {
    std::cout << "\n=== Adaptive Compute Runtime — Execution Summary ===\n";
    std::cout << std::left
              << std::setw(24) << "Workload ID"
              << std::setw(16) << "Path"
              << std::setw(12) << "Wall(ms)"
              << std::setw(12) << "Kernel(ms)"
              << std::setw(10) << "H2D(ms)"
              << std::setw(10) << "D2H(ms)"
              << std::setw(14) << "Mem(GB/s)"
              << "OK\n";
    std::cout << std::string(108, '-') << "\n";

    for (const auto& r : results_) {
        std::cout << std::left
                  << std::setw(24) << r.workload_id
                  << std::setw(16) << path_name(r.path_used)
                  << std::setw(12) << std::fixed << std::setprecision(3) << r.wall_ms
                  << std::setw(12) << r.kernel_ms
                  << std::setw(10) << r.h2d_ms
                  << std::setw(10) << r.d2h_ms
                  << std::setw(14) << r.memory_throughput_GBs
                  << (r.correct ? "YES" : "NO") << "\n";
    }

    std::cout << "\n--- Per-Path Averages ---\n";
    for (const auto& ps : per_path_stats()) {
        std::cout << std::setw(16) << path_name(ps.path)
                  << "  count=" << ps.count
                  << "  avg_wall=" << std::fixed << std::setprecision(3) << ps.avg_wall_ms << "ms"
                  << "  avg_kernel=" << ps.avg_kernel_ms << "ms\n";
    }
    std::cout << "\nOverall avg wall: " << avg_wall_ms() << " ms\n";
}

void Profiler::dump_csv(const std::string& path) const {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot open CSV output: " + path);

    f << "workload_id,path,wall_ms,kernel_ms,h2d_ms,d2h_ms,memory_throughput_GBs,correct\n";
    for (const auto& r : results_) {
        f << r.workload_id << ","
          << path_name(r.path_used) << ","
          << r.wall_ms << ","
          << r.kernel_ms << ","
          << r.h2d_ms << ","
          << r.d2h_ms << ","
          << r.memory_throughput_GBs << ","
          << (r.correct ? 1 : 0) << "\n";
    }
}

} // namespace acr
