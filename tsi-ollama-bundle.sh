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

# TOOLBOX_POSIX and TOOLBOX_FPGA (below, consulted directly inside
# resolve_toolbox_dir_for_target()) let a caller override each target's
# toolbox directory independently -- useful for internal testing, or to
# substitute a custom toolbox build when the SDK's own toolbox is broken.
# Neither needs capturing here since the script never reassigns them.
#
# TOOLBOX_DIR (bare, no suffix) is the older, single-variable form. It is
# captured once here, before resolve_toolbox_dir_for_target() starts
# reassigning/exporting the plain TOOLBOX_DIR variable per build step --
# otherwise, by the time that function runs a second time (posix then
# fpga), the caller's original value would already be overwritten by
# whatever the first call left there. It is kept as a backward-compatible
# alias for TOOLBOX_FPGA only, matching its historical, exclusively-fpga
# meaning (see resolve_toolbox_dir_for_target()'s header comment below).
TOOLBOX_DIR_LEGACY="${TOOLBOX_DIR:-}"

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

# Resolves MLIR_COMPILER_DIR/MLIR_SDK_VERSION only -- the SDK's compiler
# install is the same directory regardless of build target (posix vs fpga).
# Toolbox is NOT resolved here: see resolve_toolbox_dir_for_target() below,
# which each build step calls with its own already-known target. A single
# unconditional install-fpga default here would be wrong to hand to a
# posix-specific consumer -- mirrors tsi-pkg-build.sh's resolve_paths()/
# resolve_toolbox_dir_for_target() split.
resolve_sdk_paths() {
  local arch="$1"

  MLIR_SDK_VERSION="${MLIR_SDK_VERSION:-/proj/rel/sw/tsi-sw/staging/sdk/sdk-r.${SDK_VERSION}/${arch}}"
  MLIR_COMPILER_DIR="${MLIR_COMPILER_DIR:-${MLIR_SDK_VERSION}/compiler}"

  [ -d "${MLIR_COMPILER_DIR}" ] || die "MLIR_COMPILER_DIR not found: ${MLIR_COMPILER_DIR}"

  export SDK_VERSION
  export MLIR_SDK_VERSION
  export MLIR_COMPILER_DIR
  export COMPILER_INSTALL_DIR="${MLIR_COMPILER_DIR}"
  export FAU_LOOKUP_TABLE_PATH="${MLIR_SDK_VERSION}/ffm/txe-ffm-cpp/third-party/FAU/include/"

  log_info "SDK_VERSION:        ${SDK_VERSION}"
  log_info "MLIR_SDK_VERSION:   ${MLIR_SDK_VERSION}"
  log_info "MLIR_COMPILER_DIR:  ${MLIR_COMPILER_DIR}"
}

# Resolves TOOLBOX_DIR for a single, explicitly-named build target ("posix"
# or "fpga") -- called by build_ollama_posix() (posix) and
# build_ollama_fpga() (fpga), each already knowing its own target
# unambiguously.
#
# Each target has its own dedicated override variable, consulted BY NAME
# only in that target's own case branch below: TOOLBOX_POSIX can only ever
# affect the posix build, TOOLBOX_FPGA can only ever affect the fpga build.
# There is no variable shared between the two branches, so there is nothing
# for a caller to set once and have it apply to (or accidentally leak into)
# the wrong target -- e.g. pointing TOOLBOX_FPGA at a custom install-fpga
# can never cause that (aarch64) libomp.so to be linked into the x86_64
# posix binary, because build_ollama_posix()'s call into this function with
# target=posix never reads TOOLBOX_FPGA at all.
#
# TOOLBOX_DIR (bare, no suffix) is kept as a backward-compatible alias for
# TOOLBOX_FPGA only, matching its historical, exclusively-fpga meaning --
# README.md's "Specify toolbox directory" section has only ever documented
# it pointing at a custom install-fpga, and posix's toolbox resolution
# never consumed it (posix_libomp_dir was always independently derived
# from install-posix). If both TOOLBOX_FPGA and the legacy TOOLBOX_DIR are
# set, TOOLBOX_FPGA (the newer, explicit name) wins. Because a bare
# TOOLBOX_DIR is still easy to misread as "applies to everything" (see PR
# #72 review discussion), if it's set but the target being resolved here is
# posix, that's logged explicitly below rather than silently doing nothing.
resolve_toolbox_dir_for_target() {
  local target="$1" # posix|fpga
  local dir=""
  local override="" override_source=""

  case "${target}" in
    posix)
      if [ -n "${TOOLBOX_POSIX:-}" ]; then
        override="${TOOLBOX_POSIX}"
        override_source="TOOLBOX_POSIX"
      elif [ -n "${TOOLBOX_DIR_LEGACY:-}" ]; then
        log_info "NOTE: TOOLBOX_DIR is set in the environment (${TOOLBOX_DIR_LEGACY}), but it is a legacy alias for TOOLBOX_FPGA only and does not affect the posix build. Set TOOLBOX_POSIX to override posix's toolbox instead."
      fi
      ;;
    fpga)
      if [ -n "${TOOLBOX_FPGA:-}" ]; then
        override="${TOOLBOX_FPGA}"
        override_source="TOOLBOX_FPGA"
      elif [ -n "${TOOLBOX_DIR_LEGACY:-}" ]; then
        override="${TOOLBOX_DIR_LEGACY}"
        override_source="TOOLBOX_DIR (legacy alias for TOOLBOX_FPGA)"
      fi
      ;;
    *)
      die "resolve_toolbox_dir_for_target: invalid target '${target}' (expected posix or fpga)"
      ;;
  esac

  if [ -n "${override}" ]; then
    dir="${override}"
    log_info "NOTE: explicit toolbox override in use for the ${target} build step via ${override_source} (see README.md)."
  else
    dir="${MLIR_SDK_VERSION}/toolbox/build/install-${target}"
  fi

  [ -d "${dir}" ] || die "TOOLBOX_DIR (${target}) not found: ${dir}"
  [ -d "${dir}/lib/cmake/TSICommon" ] || die "TOOLBOX_DIR (${target}) doesn't look like a toolbox install (missing lib/cmake/TSICommon): ${dir}"

  if [ "${target}" = "fpga" ]; then
    [ -f "${dir}/lib/cmake/toolchains/arm.cmake" ] || die "TOOLBOX_DIR (fpga) is missing lib/cmake/toolchains/arm.cmake: ${dir}"
  fi

  TOOLBOX_DIR="${dir}"
  export TOOLBOX_DIR
  log_info "TOOLBOX_DIR (${target}):  ${TOOLBOX_DIR}"
}

setup_native_toolchain() {
  # Host toolchain for native (posix) builds. Not SDK/toolbox-derived -- this
  # is the build host's own local GCC install, kept in one place and
  # overridable via env var rather than repeated as a literal at each use
  # site (here and build_ollama_posix()'s linker flags). Mirrors
  # tsi-pkg-build.sh's HOST_GCC_DIR.
  export HOST_GCC_DIR="${HOST_GCC_DIR:-/proj/local/gcc-13.3.0}"
  export CC="${HOST_GCC_DIR}/bin/gcc"
  export CXX="${HOST_GCC_DIR}/bin/g++"

  export CGO_ENABLED=1
  export CGO_CC="${CC}"
  export CGO_CXX="${CXX}"

  export PATH="${HOST_GCC_DIR}/bin:${PATH}"
  export LD_LIBRARY_PATH="${HOST_GCC_DIR}/lib64:${LD_LIBRARY_PATH:-}"

  export CGO_LDFLAGS="${CGO_LDFLAGS:-} -L${HOST_GCC_DIR}/lib64 -Wl,-rpath,${HOST_GCC_DIR}/lib64 -lstdc++fs"
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
  resolve_toolbox_dir_for_target posix || return 1

  rm -rf build-posix

  local triton_defs="-DTRITON_ADD=1 -DTRITON_MAT_MUL=1 -DTRITON_DEBUG=0"
  local common="-DGGML_TSAVORITE=ON -DGGML_TSAVORITE_TARGET=posix -DGGML_NATIVE=ON -DGGML_AMX_TILE=OFF -DGGML_AMX_INT8=OFF -DGGML_AMX_BF16=OFF -DGGML_AVX512_BF16=OFF -DGGML_AVX_VNNI=OFF -DOLLAMA=ON"
  local cflags_base="-DGGML_TARGET_POSIX -DGGML_TSAVORITE -DTMU_SUPPORTED -DTVU_SUPPORTED -DOLLAMA=ON ${triton_defs}-mno-amx-tile -mno-amx-int8 -mno-amx-bf16 -mno-avx512bf16 -mno-avxvnni"
  local posix_libomp_dir="${TOOLBOX_DIR}/lib"
  [ -f "${posix_libomp_dir}/libomp.so" ] || die "POSIX libomp.so not found: ${posix_libomp_dir}/libomp.so"

  run cmake -B build-posix ${common} \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_C_FLAGS="${PERF_DEF} ${DBG_DEFS} ${cflags_base}" \
    -DCMAKE_CXX_FLAGS="${PERF_DEF} ${DBG_DEFS} ${cflags_base}" \
    -DCMAKE_EXE_LINKER_FLAGS="-L${posix_libomp_dir} -Wl,-rpath,${posix_libomp_dir} -Wl,-rpath-link,${posix_libomp_dir} -L${HOST_GCC_DIR}/lib64 -Wl,-rpath-link,${HOST_GCC_DIR}/lib64 -Wl,-rpath,${HOST_GCC_DIR}/lib64 -lomp -lgcc_s" \
    -DCMAKE_SHARED_LINKER_FLAGS="-L${posix_libomp_dir} -Wl,-rpath,${posix_libomp_dir} -Wl,-rpath-link,${posix_libomp_dir} -L${HOST_GCC_DIR}/lib64 -Wl,-rpath-link,${HOST_GCC_DIR}/lib64 -Wl,-rpath,${HOST_GCC_DIR}/lib64 -lomp -lgcc_s" || return 1

  run cmake --build build-posix --config Release || return 1

  [ -f "build-posix/ollama" ] || die "build-posix/ollama not found after build"

  return 0
}

build_ollama_fpga() {
  log_info "building Ollama FPGA/ARM64"

  compute_perf_defs "fpga"
  resolve_toolbox_dir_for_target fpga || return 1

  rm -rf build-fpga

  local ARM_TOOLCHAIN_FILE="${TOOLBOX_DIR}/lib/cmake/toolchains/arm.cmake"
  [ -f "${ARM_TOOLCHAIN_FILE}" ] || die "ARM toolchain file not found: ${ARM_TOOLCHAIN_FILE}"

  local triton_defs="-DTRITON_ADD=1 -DTRITON_MAT_MUL=1 -DTRITON_DEBUG=0"
  local fpga_libomp_dir="${TOOLBOX_DIR}/lib"
  [ -f "${fpga_libomp_dir}/libomp.so" ] || die "FPGA/aarch64 libomp.so not found: ${fpga_libomp_dir}/libomp.so"

  run cmake -B build-fpga \
    -DCMAKE_TOOLCHAIN_FILE="${ARM_TOOLCHAIN_FILE}" \
    -DTOOLBOX_DIR="${TOOLBOX_DIR}" \
    -DGGML_TSAVORITE=ON \
    -DGGML_TSAVORITE_TARGET=fpga \
    -DLLAMA_CURL=OFF \
    -DOLLAMA=ON \
    -DCMAKE_C_FLAGS="${PERF_DEF} ${DBG_DEFS} ${triton_defs} -DGGML_TSAVORITE -DTMU_SUPPORTED -DTVU_SUPPORTED -DOLLAMA=ON" \
    -DCMAKE_CXX_FLAGS="${PERF_DEF} ${DBG_DEFS} ${triton_defs} -DGGML_TSAVORITE -DTMU_SUPPORTED -DTVU_SUPPORTED -DOLLAMA=ON" \
    -DCMAKE_EXE_LINKER_FLAGS="-L${fpga_libomp_dir} -Wl,-rpath,${fpga_libomp_dir} -Wl,-rpath-link,${fpga_libomp_dir} -lomp" \
    -DCMAKE_SHARED_LINKER_FLAGS="-L${fpga_libomp_dir} -Wl,-rpath,${fpga_libomp_dir} -Wl,-rpath-link,${fpga_libomp_dir} -lomp" || return 1

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

copy_libomp_files() {
  local src_dir="$1"
  local dst_dir="$2"

  mkdir -p "${dst_dir}"

  local f
  for f in "${src_dir}"/libomp.so*; do
    if [ -e "${f}" ]; then
      cp -P "${f}" "${dst_dir}/"
    fi
  done
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

  if [ ! -f "llama/vendor/tsavorite-model-deployment.yaml" ]; then
    die "required llama/vendor/tsavorite-model-deployment.yaml not found for Ollama FPGA package"
  fi

  cp "llama/vendor/tsavorite-model-deployment.yaml" "${tsi_ggml_dir}/tsavorite-model-deployment.yaml" || return 1
  log_info "included llama/vendor/tsavorite-model-deployment.yaml in Ollama tsi-ggml runtime"

  cat > "${tsi_ggml_dir}/ggml.sh" <<'EOF'
#!/bin/bash
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(pwd)

TAOS_CONFIG_PATH="/etc/taos/taos.json"

update_one_tsavorite_deployment_yaml() {
  local deployment_yaml_path="$1"
  local txe_count="$2"
  local advanced_matmul_shape_offload="false"
  local advanced_matmul_broadcast_offload="false"
  local triton_matmul_small_n_transpose_opt="false"
  local user_dram_size_gb="8"

  mkdir -p "$(dirname "${deployment_yaml_path}")" || return 1

  if [ -f "${deployment_yaml_path}" ]; then
    local existing_advanced
    local existing_broadcast
    local existing_small_n_opt
    local existing_user_dram_size_gb

    existing_advanced="$(awk -F: '
      /^[[:space:]]*advanced_matmul_shape_offload[[:space:]]*:/ {
        v=$2
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", v)
        print v
        exit
      }
    ' "${deployment_yaml_path}")"

    existing_broadcast="$(awk -F: '
      /^[[:space:]]*advanced_matmul_broadcast_offload[[:space:]]*:/ {
        v=$2
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", v)
        print v
        exit
      }
    ' "${deployment_yaml_path}")"

    existing_small_n_opt="$(awk -F: '
      /^[[:space:]]*triton_matmul_small_n_transpose_opt[[:space:]]*:/ {
        v=$2
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", v)
        print v
        exit
      }
    ' "${deployment_yaml_path}")"

    existing_user_dram_size_gb="$(awk -F: '
      /^[[:space:]]*user_dram_size_gb[[:space:]]*:/ {
        v=$2
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", v)
        print v
        exit
      }
    ' "${deployment_yaml_path}")"

    if [ -n "${existing_advanced}" ]; then
      advanced_matmul_shape_offload="${existing_advanced}"
    fi

    if [ -n "${existing_broadcast}" ]; then
      advanced_matmul_broadcast_offload="${existing_broadcast}"
    fi

    if [ -n "${existing_small_n_opt}" ]; then
      triton_matmul_small_n_transpose_opt="${existing_small_n_opt}"
    fi

    if [ -n "${existing_user_dram_size_gb}" ]; then
      user_dram_size_gb="${existing_user_dram_size_gb}"
    fi
  fi

  cat > "${deployment_yaml_path}" <<YAML_EOF
# Tsavorite deployment config
txe_count: ${txe_count}
multi_thread_enable: true
# Runtime user DRAM size in GiB.
user_dram_size_gb: ${user_dram_size_gb}
# Enable additional Triton MAT_MUL shapes beyond stable baseline.
# false = old behavior
# true  = new offload shapes
advanced_matmul_shape_offload: ${advanced_matmul_shape_offload}

## Enable Triton MAT_MUL broadcast/batched D2/D3 offload.
## false = keep broadcast MAT_MUL on fallback path
## true  = allow advanced MAT_MUL helper to offload supported broadcast shapes
advanced_matmul_broadcast_offload: ${advanced_matmul_broadcast_offload}

# Enable Triton MAT_MUL small-N transpose optimization.
# false = old behavior
# true  = for M >> N, compute swapped [N x M] and transpose copyback to [M x N]
triton_matmul_small_n_transpose_opt: ${triton_matmul_small_n_transpose_opt}
YAML_EOF

  echo "INFO: updated ${deployment_yaml_path} with txe_count:${txe_count}, multi_thread_enable:true; preserved advanced_matmul_shape_offload:${advanced_matmul_shape_offload}, advanced_matmul_broadcast_offload:${advanced_matmul_broadcast_offload}, triton_matmul_small_n_transpose_opt:${triton_matmul_small_n_transpose_opt}, user_dram_size_gb:${user_dram_size_gb}"
  return 0
}

read_txe_count_from_taos_json() {
  if [ ! -f "${TAOS_CONFIG_PATH}" ]; then
    echo "WARNING: ${TAOS_CONFIG_PATH} not found; using default txe_count=1" >&2
    echo "1"
    return 0
  fi

  if ! command -v python3 >/dev/null 2>&1; then
    echo "WARNING: python3 not found; cannot parse ${TAOS_CONFIG_PATH}; using default txe_count=1" >&2
    echo "1"
    return 0
  fi

  local txe_count
  txe_count="$(python3 - <<'PY'
import json
import sys

path = "/etc/taos/taos.json"

try:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("top-level JSON must be an object")

    txe_count = data.get("txe_count", 1)

    if isinstance(txe_count, bool) or not isinstance(txe_count, int) or txe_count < 1:
        raise ValueError("txe_count must be an integer >= 1")

    print(txe_count)

except Exception as e:
    print(f"WARNING: failed to parse {path}: {e}; using default txe_count=1", file=sys.stderr)
    print(1)
PY
)"

  if [ -z "${txe_count}" ]; then
    echo "WARNING: empty txe_count parsed from ${TAOS_CONFIG_PATH}; using default txe_count=1" >&2
    echo "1"
    return 0
  fi

  echo "${txe_count}"
  return 0
}

update_tsavorite_deployment_yaml_from_taos() {
  local txe_count=""
  local script_dir=""

  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
  txe_count="$(read_txe_count_from_taos_json)"

  update_one_tsavorite_deployment_yaml "${script_dir}/tsavorite-model-deployment.yaml" "${txe_count}" || return 1

  if [ -d "${script_dir}/../bin" ] || [ -f "${script_dir}/../bin/tsavorite-model-deployment.yaml" ]; then
    update_one_tsavorite_deployment_yaml "${script_dir}/../bin/tsavorite-model-deployment.yaml" "${txe_count}" || return 1
  fi

  return 0
}

update_tsavorite_deployment_yaml_from_taos || exit 1

TSI_BLOB_INSTALL_DIR="__TSI_BLOB_INSTALL_DIR__"
ML_BACKEND_GGML_DIR="__ML_BACKEND_GGML_DIR__"
GGML_TSI_KERNEL_DIR="__GGML_TSI_KERNEL_DIR__"

tsi_kernels=(
  "add" "sub" "mult" "div" "abs" "inv" "neg" "sin" "sqrt" "sqr" "sigmoid" "silu" "rms_norm" "swiglu"
  "add_16" "sub_16" "mult_16" "div_16" "abs_16" "inv_16" "neg_16" "sin_16" "sqrt_16" "sqr_16" "sigmoid_16" "silu_16" "rms_norm_16" "swiglu_16"
  "mul_mat_tile_f32_k32" "mul_mat_tile_f32_k64" "mul_mat_tile_f32_k128"
)

for kernel in "${tsi_kernels[@]}"; do
  dst="${TSI_BLOB_INSTALL_DIR}/txe_${kernel}/blobs"
  rm -rf "${dst}"
  mkdir -p "${dst}"
  if [ -f "blobs/txe_${kernel}.blob" ]; then
    cp "blobs/txe_${kernel}.blob" "${dst}/txe_${kernel}.blob"
  fi
done

triton_kernels=(
  "triton_add"
  "triton_mat_mul_1x8"
  "triton_mat_mul_2x4"
)

for kernel in "${triton_kernels[@]}"; do
  dst="${TSI_BLOB_INSTALL_DIR}/txe_${kernel}/blobs"
  rm -rf "${dst}"
  mkdir -p "${dst}"

  if [ -f "blobs/txe_${kernel}/txe_blob_0.blob" ]; then
    cp "blobs/txe_${kernel}/txe_blob_0.blob" "${dst}/txe_blob_0.blob"
  else
    echo "WARNING: Triton blob not found for ${kernel}: blobs/txe_${kernel}/txe_blob_0.blob" >&2
  fi
done

mkdir -p "${ML_BACKEND_GGML_DIR}"
rm -f "${ML_BACKEND_GGML_DIR}/ggml-tsi-kernel"
ln -s "${GGML_TSI_KERNEL_DIR}" "${ML_BACKEND_GGML_DIR}/ggml-tsi-kernel"
EOF

  sed -i "s|__TSI_BLOB_INSTALL_DIR__|${blob_install_dir}|g" "${tsi_ggml_dir}/ggml.sh"
  sed -i "s|__ML_BACKEND_GGML_DIR__|${ml_backend_ggml_dir}|g" "${tsi_ggml_dir}/ggml.sh"
  sed -i "s|__GGML_TSI_KERNEL_DIR__|${ggml_tsi_kernel_dir}|g" "${tsi_ggml_dir}/ggml.sh"

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
  # TOOLBOX_DIR is install-posix here: build_ollama_posix() (called immediately
  # before this in main()) already resolved it via resolve_toolbox_dir_for_target.
  local posix_libomp_dir="${TOOLBOX_DIR}/lib"
  copy_libomp_files "${posix_libomp_dir}" "${release_dir}/bin"
  copy_libomp_files "${posix_libomp_dir}" "${release_dir}/lib"

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
  # TOOLBOX_DIR is install-fpga here: build_ollama_fpga() (called immediately
  # before this in main()) already resolved it via resolve_toolbox_dir_for_target.
  local fpga_libomp_dir="${TOOLBOX_DIR}/lib"
  copy_libomp_files "${fpga_libomp_dir}" "${release_dir}/bin"
  copy_libomp_files "${fpga_libomp_dir}" "${release_dir}/lib"

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

