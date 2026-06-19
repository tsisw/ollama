#!/usr/bin/env bash
set -o pipefail

log_error(){ echo "ERROR: $*" >&2; }
log_info(){ echo "INFO: $*"; }

run() {
  "$@"
  local rc=$?
  if [ $rc -ne 0 ]; then
    log_error "cmd failed ($rc): $*"
    return $rc
  fi
  return 0
}

die() {
  log_error "$*"
  exit 1
}

tolower(){ echo "$1" | tr '[:upper:]' '[:lower:]'; }

select_arch() {
  local m
  m="$(uname -m)"
  case "$m" in
    x86_64|amd64) echo "x86_64" ;;
    aarch64|arm64) echo "aarch64" ;;
    *) die "Unsupported host arch from uname -m: $m" ;;
  esac
}

usage() {
  cat <<'EOF'
Usage:
  SDK_VERSION=<version> ./tsi-ollama-bundle.sh [patch] [build-mode] [flags...]

Examples:
  SDK_VERSION=0.4.9 ./tsi-ollama-bundle.sh
  SDK_VERSION=0.4.9 ./tsi-ollama-bundle.sh patch
  SDK_VERSION=0.4.9 ./tsi-ollama-bundle.sh debug build-posix
  SDK_VERSION=0.4.9 ./tsi-ollama-bundle.sh debug build-fpga
  SDK_VERSION=0.4.9 ./tsi-ollama-bundle.sh clean
  SDK_VERSION=0.4.9 ./tsi-ollama-bundle.sh clean-all

This script:
  1. Optionally syncs/patches llama/vendor using Makefile.sync
  2. Invokes llama/vendor/tsi-pkg-build.sh for llama.cpp/ggml/blob build
  3. Builds Ollama POSIX/FPGA binaries
  4. Creates Ollama x86_64 and/or arm64 release tarballs
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORIG_PWD="$(pwd)"

[ "${ORIG_PWD}" = "${SCRIPT_DIR}" ] || die "Run this script from Ollama root: ${SCRIPT_DIR}"

[ -n "${SDK_VERSION:-}" ] || die "SDK_VERSION not set. Usage: SDK_VERSION=<version> ./tsi-ollama-bundle.sh"

[ -f "Makefile.sync" ] || die "Makefile.sync not found. Please run from Ollama root."
[ -d "llama" ] || die "llama directory not found."
[ -f "CMakeLists.txt" ] || die "CMakeLists.txt not found. Please run from Ollama root."

ARCH="$(select_arch)"

BUILD_TYPE="debug"
DO_PATCH=0
DO_CLEAN=0
DO_CLEAN_ALL=0
DO_BUILD_POSIX=1
DO_BUILD_FPGA=1
USER_BUILD_SELECT=0
LLAMA_ARGS=()

for arg in "$@"; do
  case "$(tolower "$arg")" in
    help|-h|--help|-help)
      usage
      exit 0
      ;;
    patch)
      DO_PATCH=1
      ;;
    clean)
      DO_CLEAN=1
      LLAMA_ARGS+=("$arg")
      ;;
    clean-all)
      DO_CLEAN_ALL=1
      LLAMA_ARGS+=("$arg")
      ;;
    release|debug|debug-tmu|debug-tmu-detail)
      BUILD_TYPE="$arg"
      LLAMA_ARGS+=("$arg")
      ;;
    build-posix|posix)
      if [ "${USER_BUILD_SELECT}" -eq 0 ]; then
        DO_BUILD_POSIX=0
        DO_BUILD_FPGA=0
        USER_BUILD_SELECT=1
      fi
      DO_BUILD_POSIX=1
      LLAMA_ARGS+=("$arg")
      ;;
    build-fpga|fpga)
      if [ "${USER_BUILD_SELECT}" -eq 0 ]; then
        DO_BUILD_POSIX=0
        DO_BUILD_FPGA=0
        USER_BUILD_SELECT=1
      fi
      DO_BUILD_FPGA=1
      LLAMA_ARGS+=("$arg")
      ;;
    build-posix-tmu-only|build-posix-tmu-disable)
      if [ "${USER_BUILD_SELECT}" -eq 0 ]; then
        DO_BUILD_POSIX=0
        DO_BUILD_FPGA=0
        USER_BUILD_SELECT=1
      fi
      DO_BUILD_POSIX=1
      LLAMA_ARGS+=("$arg")
      ;;
    build-fpga-tmu-only|build-fpga-tmu-disable)
      if [ "${USER_BUILD_SELECT}" -eq 0 ]; then
        DO_BUILD_POSIX=0
        DO_BUILD_FPGA=0
        USER_BUILD_SELECT=1
      fi
      DO_BUILD_FPGA=1
      LLAMA_ARGS+=("$arg")
      ;;
    *)
      LLAMA_ARGS+=("$arg")
      ;;
  esac
done

resolve_sdk_paths() {
  local arch="$1"

  MLIR_SDK_VERSION="${MLIR_SDK_VERSION:-/proj/rel/sw/tsi-sw/staging/sdk/sdk-r.${SDK_VERSION}/${arch}}"
  MLIR_COMPILER_DIR="${MLIR_COMPILER_DIR:-${MLIR_SDK_VERSION}/compiler}"
  TOOLBOX_DIR="${TOOLBOX_DIR:-${MLIR_SDK_VERSION}/toolbox/build/install-fpga}"
  TSICommon_DIR="${TOOLBOX_DIR}/lib/cmake/TSICommon"

  [ -d "${MLIR_COMPILER_DIR}" ] || die "MLIR_COMPILER_DIR not found: ${MLIR_COMPILER_DIR}"
  [ -d "${TOOLBOX_DIR}" ] || die "TOOLBOX_DIR not found: ${TOOLBOX_DIR}"
  [ -d "${TSICommon_DIR}" ] || die "TSICommon_DIR not found: ${TSICommon_DIR}"

  export SDK_VERSION
  export MLIR_SDK_VERSION
  export MLIR_COMPILER_DIR
  export COMPILER_INSTALL_DIR="${MLIR_COMPILER_DIR}"
  export TOOLBOX_DIR
  export TSICommon_DIR
  export FAU_LOOKUP_TABLE_PATH="${MLIR_SDK_VERSION}/ffm/txe-ffm-cpp/third-party/FAU/include/"

  log_info "SDK_VERSION:        ${SDK_VERSION}"
  log_info "MLIR_SDK_VERSION:   ${MLIR_SDK_VERSION}"
  log_info "MLIR_COMPILER_DIR:  ${MLIR_COMPILER_DIR}"
  log_info "TOOLBOX_DIR:        ${TOOLBOX_DIR}"
  log_info "TSICommon_DIR:      ${TSICommon_DIR}"
}

setup_native_toolchain() {
  export CC="/proj/local/gcc-13.3.0/bin/gcc"
  export CXX="/proj/local/gcc-13.3.0/bin/g++"
  export LD_LIBRARY_PATH="/proj/local/gcc-13.3.0/lib64:${LD_LIBRARY_PATH:-}"
}

sync_and_patch_llama_vendor() {
  local force_patch="$1"

  if [ "${force_patch}" -eq 1 ]; then
    log_info "patch requested: syncing llama/vendor and applying TSI patch"
    run make -f Makefile.sync checkout || return 1
  elif [ ! -d "llama/vendor" ]; then
    log_info "llama/vendor missing: syncing llama/vendor and applying TSI patch"
    run make -f Makefile.sync checkout || return 1
  else
    log_info "llama/vendor already exists; skipping checkout"
  fi

  [ -d "llama/vendor" ] || die "llama/vendor still missing after checkout"

  if [ -f "llama/patches/tsi-consolidated-patches.patch" ]; then
    if git -C llama/vendor apply --check ../patches/tsi-consolidated-patches.patch >/dev/null 2>&1; then
      log_info "applying llama/vendor TSI patch"
      run git -C llama/vendor apply ../patches/tsi-consolidated-patches.patch || return 1
    else
      log_info "TSI patch already applied or not applicable; skipping git apply"
    fi
  else
    log_info "WARNING: llama/patches/tsi-consolidated-patches.patch not found; skipping patch"
  fi

  log_info "syncing ml/backend/ggml/ggml"
  run make -f Makefile.sync ml/backend/ggml/ggml || return 1

  return 0
}

invoke_llama_cpp_build() {
  [ -f "llama/vendor/tsi-pkg-build.sh" ] || die "llama/vendor/tsi-pkg-build.sh not found"

  log_info "invoking llama.cpp build through llama/vendor/tsi-pkg-build.sh"

  (
    cd llama/vendor || exit 1
    SDK_VERSION="${SDK_VERSION}" source ./tsi-pkg-build.sh "${LLAMA_ARGS[@]}"
  )
}

compute_perf_defs() {
  local target="$1"
  local bt
  bt="$(tolower "${BUILD_TYPE}")"

  PERF_DEF="-DGGML_PERF"
  DBG_DEFS=""

  if [ "$bt" = "release" ]; then
    PERF_DEF="-DGGML_PERF_RELEASE"
    DBG_DEFS=""
    return 0
  fi

  if [ "$bt" = "debug" ]; then
    if [ "$target" = "fpga" ]; then
      PERF_DEF="-DGGML_PERF"
    else
      PERF_DEF="-DGGML_PERF_DETAIL"
    fi
    DBG_DEFS=""
    return 0
  fi

  if [ "$bt" = "debug-tmu" ]; then
    PERF_DEF="-DGGML_PERF_DETAIL"
    DBG_DEFS="-DTMU_DEBUG"
    return 0
  fi

  if [ "$bt" = "debug-tmu-detail" ]; then
    PERF_DEF="-DGGML_PERF_DETAIL"
    DBG_DEFS="-DTMU_DEBUG -DTMU_DEBUG_VALIDATE"
    return 0
  fi
}

build_ollama_posix() {
  log_info "building Ollama POSIX"

  setup_native_toolchain
  compute_perf_defs "posix"

  rm -rf build-posix

  local common="-DGGML_TSAVORITE=ON -DGGML_TSAVORITE_TARGET=posix -DGGML_NATIVE=ON -DGGML_AMX_TILE=OFF -DGGML_AMX_INT8=OFF -DGGML_AMX_BF16=OFF -DGGML_AVX512_BF16=OFF -DGGML_AVX_VNNI=OFF -DOLLAMA=ON"
  local cflags_base="-DGGML_TARGET_POSIX -DGGML_TSAVORITE -DTMU_SUPPORTED -DTVU_SUPPORTED -DOLLAMA=ON -mno-amx-tile -mno-amx-int8 -mno-amx-bf16 -mno-avx512bf16 -mno-avxvnni"

  run cmake -B build-posix ${common} \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_C_FLAGS="${PERF_DEF} ${DBG_DEFS} ${cflags_base}" \
    -DCMAKE_CXX_FLAGS="${PERF_DEF} ${DBG_DEFS} ${cflags_base}" || return 1

  run cmake --build build-posix --config Release || return 1

  [ -f "build-posix/ollama" ] || die "build-posix/ollama not found after build"

  return 0
}

build_ollama_fpga() {
  log_info "building Ollama FPGA/ARM64"

  compute_perf_defs "fpga"

  rm -rf build-fpga

  local ARM_TOOLCHAIN_FILE="${TOOLBOX_DIR}/lib/cmake/toolchains/arm.cmake"
  [ -f "${ARM_TOOLCHAIN_FILE}" ] || die "ARM toolchain file not found: ${ARM_TOOLCHAIN_FILE}"

  run cmake -B build-fpga \
    -DCMAKE_TOOLCHAIN_FILE="${ARM_TOOLCHAIN_FILE}" \
    -DTOOLBOX_DIR="${TOOLBOX_DIR}" \
    -DGGML_TSAVORITE=ON \
    -DGGML_TSAVORITE_TARGET=fpga \
    -DLLAMA_CURL=OFF \
    -DOLLAMA=ON \
    -DCMAKE_C_FLAGS="${PERF_DEF} ${DBG_DEFS} -DGGML_TSAVORITE -DTMU_SUPPORTED -DTVU_SUPPORTED -DOLLAMA=ON" \
    -DCMAKE_CXX_FLAGS="${PERF_DEF} ${DBG_DEFS} -DGGML_TSAVORITE -DTMU_SUPPORTED -DTVU_SUPPORTED -DOLLAMA=ON" || return 1

  run cmake --build build-fpga --config Release || return 1

  [ -f "build-fpga/ollama" ] || die "build-fpga/ollama not found after build"

  return 0
}

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [ -e "${src}" ]; then
    cp -r "${src}" "${dst}"
  fi
}

create_ollama_tsi_ggml_runtime_dir() {
  local out_dir="$1"
  local tsi_ggml_dir="${out_dir}/tsi-ggml"
  local ggml_tsi_kernel_dir="${SCRIPT_DIR}/llama/vendor/ggml-tsi-kernel"
  local ml_backend_ggml_dir="${SCRIPT_DIR}/ml/backend/ggml"
  local blob_install_dir="${ggml_tsi_kernel_dir}/fpga-kernel/build-fpga"

  rm -rf "${tsi_ggml_dir}"
  mkdir -p "${tsi_ggml_dir}"

  if [ -d "${ggml_tsi_kernel_dir}/fpga/blobs" ]; then
    cp -r "${ggml_tsi_kernel_dir}/fpga/blobs" "${tsi_ggml_dir}/"
  fi

  cat > "${tsi_ggml_dir}/ggml.sh" <<EOF
#!/bin/bash
export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:\$(pwd)

TSI_BLOB_INSTALL_DIR="${blob_install_dir}"
ML_BACKEND_GGML_DIR="${ml_backend_ggml_dir}"
GGML_TSI_KERNEL_DIR="${ggml_tsi_kernel_dir}"

tsi_kernels=("add" "sub" "mult" "div" "abs" "inv" "neg" "sin" "sqrt" "sqr" "sigmoid" "silu" "rms_norm" "swiglu" \\
"add_16" "sub_16" "mult_16" "div_16" "abs_16" "inv_16" "neg_16" "sin_16" "sqrt_16" "sqr_16" "sigmoid_16" "silu_16" "rms_norm_16" "swiglu_16" \\
"mul_mat_tile_f32_k32" "mul_mat_tile_f32_k64" "mul_mat_tile_f32_k128")

for kernel in "\${tsi_kernels[@]}"; do
  dst="\${TSI_BLOB_INSTALL_DIR}/txe_\${kernel}/blobs"
  rm -rf "\${dst}"
  mkdir -p "\${dst}"
  if [ -f "blobs/txe_\${kernel}.blob" ]; then
    cp "blobs/txe_\${kernel}.blob" "\${dst}/txe_\${kernel}.blob"
  fi
done

# Triton ADD
dst="\${TSI_BLOB_INSTALL_DIR}/txe_triton_add/blobs"
rm -rf "\${dst}"
mkdir -p "\${dst}"
if [ -f "blobs/txe_triton_add/txe_blob_0.blob" ]; then
  cp "blobs/txe_triton_add/txe_blob_0.blob" "\${dst}/txe_blob_0.blob"
fi

mkdir -p "\${ML_BACKEND_GGML_DIR}"
rm -f "\${ML_BACKEND_GGML_DIR}/ggml-tsi-kernel"
ln -s "\${GGML_TSI_KERNEL_DIR}" "\${ML_BACKEND_GGML_DIR}/ggml-tsi-kernel"
EOF

  chmod +x "${tsi_ggml_dir}/ggml.sh"
}

package_ollama_posix() {
  log_info "packaging Ollama x86_64 release"

  local release_dir="ollama-x86_64-release"
  local tarball="ollama-x86_64-release.tar.gz"

  rm -rf "${release_dir}" "${tarball}"
  mkdir -p "${release_dir}/bin" "${release_dir}/lib"

  cp build-posix/ollama "${release_dir}/bin/ollama"

  copy_if_exists "llama/vendor/ggml-tsi-kernel/posix-kernel/build-posix/blobs" "${release_dir}/"

  cp build-posix/lib/ollama/libggml-*.so "${release_dir}/bin/" 2>/dev/null || true
  cp build-posix/lib/ollama/libggml-*.so "${release_dir}/lib/" 2>/dev/null || true

  copy_if_exists "lib" "${release_dir}/"
  copy_if_exists "README.md" "${release_dir}/"

  if [ -f "llama/vendor/tsavorite-model-deployment.yaml" ]; then
    cp "llama/vendor/tsavorite-model-deployment.yaml" "${release_dir}/bin/"
  fi

  tar -czvf "${tarball}" "${release_dir}"
}

package_ollama_fpga() {
  log_info "packaging Ollama arm64 release"

  local release_dir="ollama-arm64-release"
  local tarball="ollama-arm64-release.tar.gz"

  rm -rf "${release_dir}" "${tarball}"
  mkdir -p "${release_dir}/bin" "${release_dir}/lib"

  cp build-fpga/ollama "${release_dir}/bin/ollama"

  copy_if_exists "llama/vendor/ggml-tsi-kernel/fpga/blobs" "${release_dir}/"

  cp build-fpga/lib/ollama/libggml-*.so "${release_dir}/bin/" 2>/dev/null || true
  cp build-fpga/lib/ollama/libggml-*.so "${release_dir}/lib/" 2>/dev/null || true

  copy_if_exists "lib" "${release_dir}/"
  copy_if_exists "README.md" "${release_dir}/"

  if [ -f "llama/vendor/tsavorite-model-deployment.yaml" ]; then
    cp "llama/vendor/tsavorite-model-deployment.yaml" "${release_dir}/bin/"
  fi

  create_ollama_tsi_ggml_runtime_dir "${release_dir}"

  tar -czvf "${tarball}" "${release_dir}"
}

clean_ollama_outputs() {
  log_info "cleaning Ollama build/package outputs"

  rm -rf \
    build-posix build-fpga \
    ollama-x86_64-release ollama-x86_64-release.tar.gz \
    ollama-arm64-release ollama-arm64-release.tar.gz \
    tsi-ggml 2>/dev/null || true
}

main() {
  if [ "${DO_CLEAN}" -eq 1 ] || [ "${DO_CLEAN_ALL}" -eq 1 ]; then
    clean_ollama_outputs
  fi

  sync_and_patch_llama_vendor "${DO_PATCH}" || return 1

  resolve_sdk_paths "${ARCH}" || return 1

  invoke_llama_cpp_build || return 1

  if [ "${DO_CLEAN}" -eq 1 ] || [ "${DO_CLEAN_ALL}" -eq 1 ]; then
    return 0
  fi

  if [ "${DO_BUILD_POSIX}" -eq 1 ]; then
    build_ollama_posix || return 1
    package_ollama_posix || return 1
  fi

  if [ "${DO_BUILD_FPGA}" -eq 1 ]; then
    build_ollama_fpga || return 1
    package_ollama_fpga || return 1
  fi

  log_info "Done"
  return 0
}

main "$@"

