#!/usr/bin/env bash
# Build the adaptive compute runtime.
# Usage:
#   ./scripts/build.sh          # Release, CUDA auto-detect
#   ./scripts/build.sh debug    # Debug build
#   ./scripts/build.sh nocuda   # Force CPU-only build

set -euo pipefail

BUILD_TYPE="Release"
CUDA_FLAG="-DACR_ENABLE_CUDA=ON"

case "${1:-}" in
  debug)   BUILD_TYPE="Debug" ;;
  nocuda)  CUDA_FLAG="-DACR_ENABLE_CUDA=OFF" ;;
  *)       ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$ROOT_DIR/build"

echo "==> Configuring (${BUILD_TYPE}) ..."
cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
      -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
      "$CUDA_FLAG" \
      -DACR_ENABLE_OPENMP=ON \
      -DACR_BUILD_TESTS=ON \
      -DACR_BUILD_BENCH=ON

echo "==> Building ..."
cmake --build "$BUILD_DIR" --parallel "$(nproc 2>/dev/null || sysctl -n hw.ncpu)"

echo ""
echo "Build complete. Artifacts:"
echo "  Benchmark:  $BUILD_DIR/acr_bench"
echo "  Tests:      $BUILD_DIR/acr_tests"
echo ""
echo "Run tests:      cd $BUILD_DIR && ctest --output-on-failure"
echo "Run benchmark:  $BUILD_DIR/acr_bench"
