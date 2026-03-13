
set -e

# Parse command line arguments
TOOLBOX_DIR_ARG=""
BUILD_TYPE=""
CLEAN_ONLY=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --toolbox-dir|-t)
            TOOLBOX_DIR_ARG="$2"
            shift 2
            ;;
        clean)
            CLEAN_ONLY=true
            shift
            ;;
        patch|release|debug)
            BUILD_TYPE="$1"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [clean|patch|release|debug] [--toolbox-dir|-t TOOLBOX_DIR]"
            exit 1
            ;;
    esac
done

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

export MLIR_SDK_VERSION=/proj/rel/sw/sdk-r.0.2.8/${arch}

#export variable for FFM FAU Lookup table
export FAU_LOOKUP_TABLE_PATH=${MLIR_SDK_VERSION}/ffm/txe-ffm-cpp/third-party/FAU/include/

# Set TOOLBOX_DIR with priority: command line argument > environment variable > default
# Priority: command line argument (--toolbox-dir) > environment variable (TOOLBOX_DIR) > default
if [ -n "$TOOLBOX_DIR_ARG" ]; then
    TOOLBOX_DIR="$TOOLBOX_DIR_ARG"
elif [ -n "${TOOLBOX_DIR}" ]; then
    # Use environment variable if set
    :
else
    TOOLBOX_DIR="${MLIR_SDK_VERSION}/toolbox/build/install-fpga"
fi

# Export TOOLBOX_DIR so CMake can see it
export TOOLBOX_DIR
echo "Using TOOLBOX_DIR: ${TOOLBOX_DIR}"

# Clean build artifacts if requested
if [ "$CLEAN_ONLY" = true ]; then
    echo "Cleaning build artifacts..."
    
    # Clean kernel build directories
    if [ -d "llama/vendor/ggml-tsi-kernel/fpga-kernel/build-fpga" ]; then
        echo "Removing fpga-kernel/build-fpga..."
        rm -rf llama/vendor/ggml-tsi-kernel/fpga-kernel/build-fpga
    fi
    
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
    
    if [ -d "build-fpga" ]; then
        echo "Removing build-fpga..."
        rm -rf build-fpga
    fi
    
    # Clean release artifacts
    if [ -d "ollama-arm64-release" ]; then
        echo "Removing ollama-arm64-release..."
        rm -rf ollama-arm64-release
    fi
    
    if [ -f "ollama-arm64-release.tar.gz" ]; then
        echo "Removing ollama-arm64-release.tar.gz..."
        rm -f ollama-arm64-release.tar.gz
    fi
    
    if [ -f "ollama" ]; then
        echo "Removing ollama binary..."
        rm -f ollama
    fi
    
    if [ -d "tsi-ggml" ]; then
        echo "Removing tsi-ggml bundle..."
        rm -rf tsi-ggml
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

cd llama/vendor
#Ensure prerequisites are met as follows
echo 'updating submodule'
git submodule update --recursive --init
cd ggml-tsi-kernel/
module load gcc/13.3.0

echo 'creating python virtual env'
# Use Python 3.11 to match MLIR bindings
/proj/local/Python-3.11.12/bin/python3 -m venv blob-creation
source blob-creation/bin/activate
echo 'installing mlir and python dependencies'
# Set LD_LIBRARY_PATH for MLIR native extensions
export LD_LIBRARY_PATH="${MLIR_SDK_VERSION}/compiler/lib:${LD_LIBRARY_PATH:-}"
pip install --upgrade pip
pip install -r ${MLIR_SDK_VERSION}/compiler/python/requirements-common.txt
pip install ${MLIR_SDK_VERSION}/compiler/python/mlir_external_packages-*.whl
pip install onnxruntime-training

#build TSI kernels for the Tsavorite backend
#First for FPGA

#echo 'creating fpga kernel'
cd fpga-kernel
cmake -B build-fpga -DTOOLBOX_DIR="${TOOLBOX_DIR}"
./create-all-kernels.sh
#The for Posix Use cases

echo 'creating posix kernel'
cd ../posix-kernel/
cmake -B build-posix -DTOOLBOX_DIR="${TOOLBOX_DIR}" 2>/dev/null || true
./create-all-kernels.sh

cd ../..
echo "$(pwd)"

#Change directory to top level ollama

cd ../../

#Compile for posix & fpga with build-posix as a target folder

echo 'building llama.cp, ggml for tsavorite  and other binary for posix'
if [ "$(echo "$BUILD_TYPE" | tr '[:upper:]' '[:lower:]')" = "release" ];
then
  cmake -B build-posix -DGGML_TSAVORITE=ON -DGGML_TSAVORITE_TARGET=posix -DTOOLBOX_DIR="${TOOLBOX_DIR}" -DCMAKE_C_FLAGS="-DGGML_PERF_RELEASE -DGGML_TARGET_POSIX -DGGML_TSAVORITE" -DCMAKE_CXX_FLAGS="-DGGML_PERF_RELEASE -DGGML_TARGET_POSIX -DGGML_TSAVORITE"
elif [ "$(echo "$BUILD_TYPE" | tr '[:upper:]' '[:lower:]')" = "debug" ]; then
  cmake -B build-posix -DGGML_TSAVORITE=ON -DGGML_TSAVORITE_TARGET=posix -DTOOLBOX_DIR="${TOOLBOX_DIR}" -DCMAKE_C_FLAGS="-DGGML_PERF_DETAIL -DGGML_TARGET_POSIX -DGGML_TSAVORITE" -DCMAKE_CXX_FLAGS="-DGGML_PERF_DETAIL -DGGML_TARGET_POSIX -DGGML_TSAVORITE"
else
  cmake -B build-posix -DGGML_TSAVORITE=ON -DGGML_TSAVORITE_TARGET=posix -DTOOLBOX_DIR="${TOOLBOX_DIR}" -DCMAKE_C_FLAGS="-DGGML_PERF -DGGML_TARGET_POSIX -DGGML_TSAVORITE" -DCMAKE_CXX_FLAGS="-DGGML_PERF -DGGML_TARGET_POSIX -DGGML_TSAVORITE"
fi

cmake --build build-posix --config Release

# Fix GLIBC compatibility for TSI binaries
#echo 'fixing GLIBC compatibility for TSI binaries'

# Fix simple-backend-tsi
#mkdir -p build-posix/bin/
#mv llama/vendor/build-posix/bin/simple-backend-tsi build-posix/bin/simple-backend-tsi-original
#cat > build-posix/bin/simple-backend-tsi << 'EOL'
#!/bin/bash
#export LD_LIBRARY_PATH="/proj/local/gcc-13.3.0/lib64:$LD_LIBRARY_PATH"
#exec "$(dirname "$0")/simple-backend-tsi-original" "$@"
#EOL
#chmod +x build-posix/bin/simple-backend-tsi

# Fix llama-cli
#mkdir -p build-posix/bin/
#mv llama/vendor/build-posix/bin/llama-cli build-posix/bin/llama-cli-original
#cat > build-posix/bin/llama-cli << 'EOL'
#!/bin/bash
#export LD_LIBRARY_PATH="/proj/local/gcc-13.3.0/lib64:$LD_LIBRARY_PATH"
#exec "$(dirname "$0")/llama-cli-original" "$@"
#EOL
#chmod +x build-posix/bin/llama-cli

#Compile for fpga with build-fpga as a target folder

echo 'building llama.cp, ggml for tsavorite  and other binary for fpga'
# Source toolbox ARM toolchain environment
echo "Using TOOLBOX_DIR: ${TOOLBOX_DIR}"
source "${TOOLBOX_DIR}/scripts/arm-toolchain-env.sh"
export CC="${ARM_COMPILER_PREFIX}gcc"
export CXX="${ARM_COMPILER_PREFIX}g++"

if [ "$(echo "$BUILD_TYPE" | tr '[:upper:]' '[:lower:]')" = "release" ];
then
 cmake -B build-fpga -DGGML_TSAVORITE=ON -DGGML_TSAVORITE_TARGET=fpga -DTOOLBOX_DIR="${TOOLBOX_DIR}" -DCMAKE_C_FLAGS="-DGGML_PERF_RELEASE -DGGML_TARGET_FPGA -DGGML_TSAVORITE" -DCMAKE_CXX_FLAGS="-DGGML_PERF_RELEASE -DGGML_TARGET_FPGA -DGGML_TSAVORITE" -DCURL_INCLUDE_DIR=${ARM_TOOLCHAIN_PATH}/include  -DCURL_LIBRARY=${ARM_TOOLCHAIN_PATH}/lib/libcurl.so
elif [ "$(echo "$BUILD_TYPE" | tr '[:upper:]' '[:lower:]')" = "debug" ]; then
  cmake -B build-fpga -DGGML_TSAVORITE=ON -DGGML_TSAVORITE_TARGET=fpga -DTOOLBOX_DIR="${TOOLBOX_DIR}" -DCMAKE_C_FLAGS="-DGGML_PERF_DETAIL -DGGML_TARGET_FPGA -DGGML_TSAVORITE" -DCMAKE_CXX_FLAGS="-DGGML_PERF_DETAIL -DGGML_TARGET_FPGA -DGGML_TSAVORITE" -DCURL_INCLUDE_DIR=${ARM_TOOLCHAIN_PATH}/include  -DCURL_LIBRARY=${ARM_TOOLCHAIN_PATH}/lib/libcurl.so
else
  cmake -B build-fpga -DGGML_TSAVORITE=ON -DGGML_TSAVORITE_TARGET=fpga -DTOOLBOX_DIR="${TOOLBOX_DIR}" -DCMAKE_C_FLAGS="-DGGML_PERF -DGGML_TARGET_FPGA -DGGML_TSAVORITE" -DCMAKE_CXX_FLAGS="-DGGML_PERF -DGGML_TARGET_FPGA -DGGML_TSAVORITE" -DCURL_INCLUDE_DIR=${ARM_TOOLCHAIN_PATH}/include  -DCURL_LIBRARY=${ARM_TOOLCHAIN_PATH}/lib/libcurl.so
fi

cmake --build build-fpga --config Release


#echo 'creating tar bundle for fpga'
TSI_GGML_VERSION=0.2.3
TSI_GGML_BUNDLE_INSTALL_DIR=tsi-ggml
GGML_TSI_INSTALL_DIR=llama/vendor/ggml-tsi-kernel
TSI_GGML_RELEASE_DIR=/proj/rel/sw/ggml
TSI_BLOB_INSTALL_DIR=$(pwd)/${GGML_TSI_INSTALL_DIR}/fpga-kernel/build-fpga

if [ -e ${TSI_GGML_BUNDLE_INSTALL_DIR} ]; then
   echo "${TSI_GGML_BUNDLE_INSTALL_DIR} exist"
else
   echo "creating ${TSI_GGML_BUNDLE_INSTALL_DIR}"
   mkdir ${TSI_GGML_BUNDLE_INSTALL_DIR}
fi
if [ -e ${TSI_GGML_BUNDLE_INSTALL_DIR}/ggml.sh ]; then
   rm -fr ${TSI_GGML_BUNDLE_INSTALL_DIR}/ggml.sh
fi

cat > ./${TSI_GGML_BUNDLE_INSTALL_DIR}/ggml.sh << EOL
#!/bin/bash
# Set up library paths for GCC 13.3.0 compatibility
export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:\$(pwd)

tsi_kernels=("add" "sub" "mult" "div" "abs" "inv" "neg" "sin" "sqrt" "sqr" "sigmoid" "silu" "rms_norm"  "swiglu" "add_16" "sub_16" "mult_16" "div_16" "abs_16" "inv_16" "neg_16" "sin_16" "sqrt_16" "sqr_16" "sigmoid_16" "silu_16" "rms_norm_16" "swiglu_16")

for kernel in "\${tsi_kernels[@]}"; do
    mkdir -p ${TSI_BLOB_INSTALL_DIR}/txe_\$kernel
    cp --parent blobs/txe_\$kernel*.blob ${TSI_BLOB_INSTALL_DIR}/txe_\$kernel/ -r
done
EOL
chmod +x ${TSI_GGML_BUNDLE_INSTALL_DIR}/ggml.sh
cp ${GGML_TSI_INSTALL_DIR}/fpga/blobs ${TSI_GGML_BUNDLE_INSTALL_DIR}/ -r

if [ "$(echo "$BUILD_TYPE" | tr '[:upper:]' '[:lower:]')" = "release" ];
then
    cp ${TSI_GGML_BUNDLE_INSTALL_DIR}-${TSI_GGML_VERSION}.tz ${TSI_GGML_RELEASE_DIR}/

    LATEST_TZ="${TSI_GGML_BUNDLE_INSTALL_DIR}-${TSI_GGML_VERSION}.tz"
    LATEST_FULL_PATH="${TSI_GGML_RELEASE_DIR}/$(basename "$LATEST_TZ")"

    # Remove old symlinks if they exist
    rm -f "$TSI_GGML_RELEASE_DIR/tsi-ggml-aws-latest.tz"
    rm -f "$TSI_GGML_RELEASE_DIR/tsi-ggml-latest.tz"
    # Create new symbolic links
    ln -s /aws"$LATEST_FULL_PATH" "$TSI_GGML_RELEASE_DIR/tsi-ggml-aws-latest.tz"
    ln -s "$LATEST_FULL_PATH" "$TSI_GGML_RELEASE_DIR/tsi-ggml-latest.tz"

    echo "Symlinks updated to point to $(basename "$LATEST_FULL_PATH")"
fi

RELEASE_DIR="ollama-arm64-release"
TARBALL="ollama-arm64-release.tar.gz"

# Build Go binary for ARM64
echo "Building Go binary for ARM64..."
export CGO_ENABLED=1
export PATH=$PATH:/proj/local/go/bin
GOARCH=arm64 GOOS=linux go build -o ollama .

# Prepare release directory
echo "Preparing release directory..."
rm -rf $RELEASE_DIR
mkdir -p $RELEASE_DIR/bin
mkdir -p $RELEASE_DIR/lib
cp ollama $RELEASE_DIR/bin/
cp llama/vendor/ggml-tsi-kernel/fpga/blobs ${RELEASE_DIR}/ -r
cp build-fpga/lib/ollama/libggml-*.so ${RELEASE_DIR}/bin
cp build-fpga/lib/ollama/libggml-*.so ${RELEASE_DIR}/lib

cp -r lib $RELEASE_DIR/ 2>/dev/null || echo "No lib directory to copy"
cp README.md $RELEASE_DIR/ 2>/dev/null || echo "No README.md to copy"
cp -r tsi-ggml $RELEASE_DIR/ 2>/dev/null || echo "No tsi-ggml-ollama*.tz to copy"

# Create tarball
echo "Creating tarball..."
tar -czvf $TARBALL $RELEASE_DIR

echo "ARM64 tarbundle created: $TARBALL"
