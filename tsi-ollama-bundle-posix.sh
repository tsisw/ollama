#!/bin/bash
# NOTE:
# To ensure environment variables are set in the current shell,
# always run this script using:
#   source tsi-ollama-bundle-posix.sh
# or
#   ./tsi-ollama-bundle-posix.sh
set -e

BUILD_TYPE=""
CLEAN_ONLY=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        clean)
            CLEAN_ONLY=true
            shift
            ;;
        patch|release|debug)
            BUILD_TYPE="$1"
            shift
            ;;
        *)
            echo "Usage: $0 [clean|patch|release|debug]"
            exit 1
            ;;
    esac
done

# Clean build artifacts if requested
if [ "$CLEAN_ONLY" = true ]; then
    echo "Cleaning build artifacts..."

    # Clean posix kernel build directories
    if [ -d "llama/vendor/ggml-tsi-kernel/posix-kernel/build-posix" ]; then
        echo "Removing posix-kernel/build-posix..."
        rm -rf llama/vendor/ggml-tsi-kernel/posix-kernel/build-posix
    fi

    # Clean Python venv
    if [ -d "llama/vendor/ggml-tsi-kernel/blob-creation" ]; then
        echo "Removing blob-creation venv..."
        rm -rf llama/vendor/ggml-tsi-kernel/blob-creation
    fi

    # Clean main build directories
    if [ -d "build-posix" ]; then
        echo "Removing build-posix..."
        rm -rf build-posix
    fi

    # Clean Go binary
    if [ -f "ollama-x86_64" ]; then
        echo "Removing ollama-x86_64..."
        rm -f ollama-x86_64
    fi

    # Clean release artifacts
    if [ -d "ollama-x86_64-release" ]; then
        echo "Removing ollama-x86_64-release..."
        rm -rf ollama-x86_64-release
    fi

    if [ -f "ollama-x86_64-release.tar.gz" ]; then
        echo "Removing ollama-x86_64-release.tar.gz..."
        rm -f ollama-x86_64-release.tar.gz
    fi

    echo "Clean completed."
    exit 0
fi

# Apply patches if the patches have not been applied and the first argument is patch otherwise just build
if [ "$BUILD_TYPE" == "patch" ]
then
    make -f Makefile.sync checkout
    cd llama/vendor
    git apply ../patches/tsi-consolidated-patches.patch
    cd ../../
    make -f Makefile.sync ml/backend/ggml/ggml
fi

log_error() {
  echo "ERROR: $*" >&2
}

m="$(uname -m)"
case "$m" in
    x86_64|amd64) arch="x86_64" ;;
    aarch64|arm64) arch="aarch64" ;;
    *)
      log_error "Unsupported host arch from uname -m: $m"
      exit
      ;;
esac

cd llama/vendor
#Ensure prerequisites are met as follows
echo 'updating submodule'
git submodule update --recursive --init
cd ggml-tsi-kernel/
module load gcc/13.3.0
export MLIR_SDK_VERSION=/proj/rel/sw/sdk-r.0.2.8/${arch}

#export variable for FFM LUT lookup tabel to run in posix environment
export FAU_LOOKUP_TABLE_PATH=${MLIR_SDK_VERSION}/ffm/txe-ffm-cpp/third-party/FAU/include/

echo 'creating python virtual env'
/proj/local/Python-3.11.12/bin/python3 -m venv blob-creation
source blob-creation/bin/activate
echo 'installing mlir and python dependencies'
pip install --upgrade pip
pip install -r ${MLIR_SDK_VERSION}/compiler/python/requirements-common.txt
pip install ${MLIR_SDK_VERSION}/compiler/python/mlir_external_packages-*.whl
pip install onnxruntime-training

#build TSI kernels for the Tsavorite backend - POSIX only
echo 'creating posix kernel'
cd posix-kernel/
./create-all-kernels.sh

cd ../..
echo "$(pwd)"

#Change directory to top level ollama
cd ../../

#Compile for posix with build-posix as a target folder
echo 'building llama.cp, ggml for tsavorite and other binary for posix'
if [ "$(echo "$BUILD_TYPE" | tr '[:upper:]' '[:lower:]')" = "release" ];
then
  cmake -B build-posix -DGGML_TSAVORITE=ON -DGGML_TSAVORITE_TARGET=posix -DCMAKE_C_FLAGS="-DGGML_PERF_RELEASE -DGGML_TARGET_POSIX -DGGML_TSAVORITE" -DCMAKE_CXX_FLAGS="-DGGML_PERF_RELEASE -DGGML_TARGET_POSIX -DGGML_TSAVORITE"
elif [ "$(echo "$BUILD_TYPE" | tr '[:upper:]' '[:lower:]')" = "debug" ]; then
  cmake -B build-posix -DGGML_TSAVORITE=ON -DGGML_TSAVORITE_TARGET=posix -DCMAKE_C_FLAGS="-DGGML_PERF_DETAIL -DGGML_TARGET_POSIX -DGGML_TSAVORITE" -DCMAKE_CXX_FLAGS="-DGGML_PERF_DETAIL -DGGML_TARGET_POSIX -DGGML_TSAVORITE"
else
  cmake -B build-posix -DGGML_TSAVORITE=ON -DGGML_TSAVORITE_TARGET=posix -DCMAKE_C_FLAGS="-DGGML_PERF -DGGML_TARGET_POSIX -DGGML_TSAVORITE" -DCMAKE_CXX_FLAGS="-DGGML_PERF -DGGML_TARGET_POSIX -DGGML_TSAVORITE"
fi

cmake --build build-posix --config Release

# Build Go binaries for x86_64
echo "Building Go binaries..."
export PATH=$PATH:/proj/local/go/bin

# Build Go binary for x86_64
echo "Building Go binary for x86_64..."

# Use native x86_64 compiler (unset ARM cross-compiler, use system default)
unset CC
unset CXX

# Explicitly set to native compiler if available, otherwise let Go find it
if [ -f "/proj/local/gcc-13.3.0/bin/gcc" ]; then
    export CC=/proj/local/gcc-13.3.0/bin/gcc
    export CXX=/proj/local/gcc-13.3.0/bin/g++
fi

GOARCH=amd64 GOOS=linux CGO_ENABLED=1 go build -o ollama-x86_64 .

# Prepare x86_64 release directory
RELEASE_DIR_X86="ollama-x86_64-release"
TARBALL_X86="ollama-x86_64-release.tar.gz"

echo "Preparing x86_64 release directory..."

rm -rf $RELEASE_DIR_X86
mkdir -p $RELEASE_DIR_X86/bin
mkdir -p $RELEASE_DIR_X86/lib

cp ollama-x86_64 $RELEASE_DIR_X86/bin/ollama
cp llama/vendor/ggml-tsi-kernel/posix-kernel/build-posix/blobs ${RELEASE_DIR_X86}/ -r 2>/dev/null || echo "No posix blobs found"
cp build-posix/lib/ollama/libggml-*.so ${RELEASE_DIR_X86}/bin 2>/dev/null || echo "No posix libraries found"
cp build-posix/lib/ollama/libggml-*.so ${RELEASE_DIR_X86}/lib 2>/dev/null || echo "No posix libraries found"
cp -r lib $RELEASE_DIR_X86/ 2>/dev/null || echo "No lib directory to copy"
cp README.md $RELEASE_DIR_X86/ 2>/dev/null || echo "No README.md to copy"

# Create x86_64 tarball
echo "Creating x86_64 tarball..."
tar -czvf $TARBALL_X86 $RELEASE_DIR_X86

echo "x86_64 tarbundle created: $TARBALL_X86"

