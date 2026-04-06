#!/usr/bin/env bash
# Run the benchmark and open the CSV for offline analysis.
set -euo pipefail

BUILD_DIR="$(dirname "$(dirname "${BASH_SOURCE[0]}")")/build"

if [[ ! -f "$BUILD_DIR/acr_bench" ]]; then
  echo "Build not found. Run ./scripts/build.sh first."
  exit 1
fi

echo "==> Running adaptive compute runtime benchmark ..."
"$BUILD_DIR/acr_bench"

if [[ -f "benchmark_results.csv" ]]; then
  echo ""
  echo "CSV written to: $(pwd)/benchmark_results.csv"
  echo "Analyse with:   python3 scripts/plot_results.py"
fi
