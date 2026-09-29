// -----------------------------------------------------------------------------n
// Copyright (c) 2023 Tsavorite Scalable Intelligence, Inc . All rights reserved.
//
//
// This file is the confidential and proprietary property of
// Tsavorite Scalable Intelligence, Inc
//
// Possession or use of this file requires a written license from
// Tsavorite Scalable Intelligence, Inc

/******************************************************************************
 * File: ggml-tsavorite.cpp
 * Author TSI Inc
 *
 * Description:
 * ***************************************************************************/

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <execinfo.h>
#include <signal.h>

#include "ggml-tsavorite.h"
#include <unistd.h>
#include <inttypes.h>
#include <math.h>
#include <iostream>
#include <filesystem>
#include <fcntl.h>
#include <sys/mman.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <fstream>
#include <cctype>
#include <cstdlib>
#include <dlfcn.h>
#include <magic_enum/magic_enum.hpp>
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "vec.h"
#include "ops.h"
#include "tsi-rt/TXEDeviceConfig.h"
#include "tsi-rt/host/BlobDescriptor.h"
#include "tsi-rt/queues/Command.h"
#include "HostShimCAPI.h"
#include "tsi-rt/utils/Profiler.h"
#ifdef GGML_TARGET_POSIX
#include "device/posix/PosixSimDeviceConfig.h"
using TsavoriteDeviceConfig = tsi::runtime::PosixDeviceConfig;
#else
#include "device/fpga/FPGADeviceConfig.h"
using TsavoriteDeviceConfig = tsi::runtime::FPGADeviceConfig;
#endif

#include <thread>
#include <atomic>
#include <vector>
#include  <mutex>
#include <condition_variable>
#include <algorithm>
#include <map>

using namespace tsi::runtime;

// The application currently supports up to 20 TXEs. Initialization will fail if txe_count
// in the deployment configuration exceeds this limit.
#define MAX_TXES_SUPPORTED 20

#define TSAV_DIMS_STR_LEN 128
#define TSAV_TYPE_NAME_LEN 32
#define TSAV_PROFILE_KEY_LEN 512
#define TSAV_MATMUL_ALIGN_N 64
#define TSAV_MATMUL_ALIGN_N_MASK (TSAV_MATMUL_ALIGN_N - 1)


// ggml-tsavorite.cpp
namespace {

struct TsavoriteRuntimeState {
    // device / threading
    uint32_t num_of_txes = 1;
    bool *device_free = nullptr;
    // Mirrors whether device_free/packed_args/scalar_*_args are currently
    // allocated. Must be reset to false everywhere device_free is freed
    // (tsi_cleanup, ggml_tsavorite_free, tsi_log_profile_info) or the next
    // tsi_init_per_txe_state_once() call will wrongly skip reallocating
    // them, leaving dangling pointers behind.
    std::atomic<bool> per_txe_state_initialized{false};
    bool multi_thread_enable = false;
    // one packed-args buffer per TXE
    std::vector<void *> packed_args;

    std::vector<void *> scalar_loop_args;

    std::vector<void *> scalar_m_args;
    std::vector<void *> scalar_n_args;
    std::vector<void *> scalar_k_args;

    std::vector<void *> scalar_grid1_args;
    std::vector<void *> scalar_grid2_args;
    std::vector<void *> scalar_grid3_args;

    std::vector<std::thread> workers;
    std::mutex workers_mutex;
    std::mutex device_mutex;
    std::mutex tsi_pack_mutex;
    std::mutex tsi_init_mutex;
    std::condition_variable device_cv;
    // blobs
    BlobDescriptor **blobDescriptor_add = nullptr;
    BlobDescriptor **blobDescriptor_mult = nullptr;
    BlobDescriptor **blobDescriptor_rms_norm = nullptr;
#if TRITON_ADD
    BlobDescriptor **blobDescriptor_triton_add = nullptr;
#endif
#if TRITON_MAT_MUL
    BlobDescriptor **blobDescriptor_matmul_1x8 = nullptr;
    BlobDescriptor **blobDescriptor_matmul_2x4 = nullptr;
#endif


    void **loadResult_add = nullptr;
    void **loadResult_mult = nullptr;
    void **loadResult_rms_norm = nullptr;
#if TRITON_ADD
    void **loadResult_triton_add = nullptr;
#endif
#if TRITON_MAT_MUL
    void **loadResult_matmul_1x8 = nullptr;
    void **loadResult_matmul_2x4 = nullptr;
    bool advanced_matmul_shape_offload = false;
    bool advanced_matmul_broadcast_offload = false;
    bool triton_matmul_small_n_transpose_opt = false;
#endif

    // blob lifetime state machine
    enum BlobState : uint8_t {
        BLOB_UNINITIALIZED = 0,   // no tables, no blobs
        BLOB_TABLES_ALLOCATED,    // tables allocated, blobs may be null
        BLOB_BLOBS_LOADED         // all blobs loaded successfully
    };

    BlobState blob_state = BLOB_UNINITIALIZED;
    uint32_t blob_tables_txes = 0;   // tracks num_of_txes used to size the tables
};

static TsavoriteRuntimeState g_rt;

// aliases (USE THESE EVERYWHERE)
auto &num_of_txes = g_rt.num_of_txes;
auto &device_free = g_rt.device_free;
auto &per_txe_state_initialized = g_rt.per_txe_state_initialized;
auto &multi_thread_enable     = g_rt.multi_thread_enable;
auto &packed_args     = g_rt.packed_args;

auto &scalar_loop_args        = g_rt.scalar_loop_args;
auto &scalar_m_args           = g_rt.scalar_m_args;
auto &scalar_n_args           = g_rt.scalar_n_args;
auto &scalar_k_args           = g_rt.scalar_k_args;

auto &scalar_grid1_args       = g_rt.scalar_grid1_args;
auto &scalar_grid2_args       = g_rt.scalar_grid2_args;
auto &scalar_grid3_args       = g_rt.scalar_grid3_args;

auto &workers = g_rt.workers;
auto &workers_mutex = g_rt.workers_mutex;
auto &device_mutex = g_rt.device_mutex;
auto &tsi_pack_mutex = g_rt.tsi_pack_mutex;
auto &tsi_init_mutex = g_rt.tsi_init_mutex;
auto &device_cv = g_rt.device_cv;

auto &blobDescriptor_add      = g_rt.blobDescriptor_add;
auto &blobDescriptor_mult     = g_rt.blobDescriptor_mult;
auto &blobDescriptor_rms_norm = g_rt.blobDescriptor_rms_norm;
#if TRITON_ADD
auto &blobDescriptor_triton_add = g_rt.blobDescriptor_triton_add;
#endif
#if TRITON_MAT_MUL
auto &blobDescriptor_matmul_1x8 = g_rt.blobDescriptor_matmul_1x8;
auto &blobDescriptor_matmul_2x4 = g_rt.blobDescriptor_matmul_2x4;
#endif


auto &loadResult_add          = g_rt.loadResult_add;
auto &loadResult_mult         = g_rt.loadResult_mult;
auto &loadResult_rms_norm     = g_rt.loadResult_rms_norm;
#if TRITON_ADD
auto &loadResult_triton_add = g_rt.loadResult_triton_add;
#endif
#if TRITON_MAT_MUL
auto &loadResult_matmul_1x8     = g_rt.loadResult_matmul_1x8;
auto &loadResult_matmul_2x4     = g_rt.loadResult_matmul_2x4;
auto &advanced_matmul_shape_offload = g_rt.advanced_matmul_shape_offload;
auto &advanced_matmul_broadcast_offload = g_rt.advanced_matmul_broadcast_offload;
auto &triton_matmul_small_n_transpose_opt = g_rt.triton_matmul_small_n_transpose_opt;
#endif
} // anonymous namespace

constexpr int kMaxBacktraceFrames = 64;
constexpr int kSignalExitBase     = 128;

static void tsavorite_sig_handler(int sig) {
    void *array[kMaxBacktraceFrames];
    int size = backtrace(array, kMaxBacktraceFrames);

    fprintf(stderr, "\n\n=== TSAVORITE FATAL SIGNAL %d ===\n", sig);
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    fprintf(stderr, "=== END BACKTRACE ===\n");

    _exit(kSignalExitBase + sig); // hard exit, no cleanup
}

static void tsavorite_install_signal_handlers() {
    signal(SIGSEGV, tsavorite_sig_handler);
    signal(SIGABRT, tsavorite_sig_handler);
    signal(SIGBUS,  tsavorite_sig_handler);
    signal(SIGILL,  tsavorite_sig_handler);
    signal(SIGFPE,  tsavorite_sig_handler);
}

// =============================================================================
// YAML deployment parsing (no external yaml lib)
// Supports:
// txe_count: 2
// multi_thread_enable: true\false\1\0\yes\no\on\off
// Optional env:
// TSAVORITE_MODEL_DEPLOYMENT_YAML=/path/to/tsavorite-model-deployment.yaml
// Notes:
// - txe_count is CLAMPED to MAX_TXES_SUPPORTED (fixed-size arrays in this file)
// =============================================================================
static inline std::string tsi_trim_copy(const std::string &s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace((unsigned char)s[b])) b++;
   while (e > b && std::isspace((unsigned char)s[e - 1])) e--;
    return s.substr(b, e - b);
}

static inline bool tsi_starts_with(const std::string &s, const char *pfx) {
    return s.rfind(pfx, 0) == 0;
}

static inline std::string tsi_to_lower(std::string v) {
    for (char &c : v) c = (char)std::tolower((unsigned char)c);
    return v;
}

static int tsi_parse_int_after_colon(const std::string &line) {
    size_t c = line.find(':');
    if (c == std::string::npos) return -1;
    std::string rhs = tsi_trim_copy(line.substr(c + 1));
    if (rhs.empty()) return -1;
    // allow quotes
    if (rhs.front() == '"' || rhs.front() == '\'') rhs.erase(0, 1);
    // parse leading integer
    int sign = 1;
    size_t i = 0;
    if (i < rhs.size() && rhs[i] == '-') { sign = -1; i++; }
    long v = 0;
    bool any = false;
    for (; i < rhs.size() && std::isdigit((unsigned char)rhs[i]); i++) {
        any = true;
        v = v * 10 + (rhs[i] - '0');
    }
    return any ? (int)(sign * v) : -1;
}

static bool tsi_parse_bool_after_colon(const std::string &line, bool *out) {
    if (!out) return false;
    size_t c = line.find(':');
    if (c == std::string::npos) return false;
    std::string rhs = tsi_trim_copy(line.substr(c + 1));
    if (rhs.empty()) return false;
    // allow quotes
    if (rhs.front() == '"' || rhs.front() == '\'') rhs.erase(0, 1);
    rhs = tsi_to_lower(tsi_trim_copy(rhs));
    // accept common yaml-ish bools
    if (rhs == "true" || rhs == "1" || rhs == "yes" || rhs == "y" || rhs == "on")  { *out = true;  return true; }
    if (rhs == "false"|| rhs == "0" || rhs == "no"  || rhs == "n" || rhs == "off") { *out = false; return true; }
    return false;
}

struct tsi_deploy_cfg_t {
    int  txe_count = -1;
    bool mt_enable = false;
    bool has_mt    = false;

    int  user_dram_size_gb = -1;
    bool has_user_dram_size_gb = false;

#if TRITON_MAT_MUL
    bool advanced_matmul_shape_offload = false;
    bool has_advanced_matmul_shape_offload = false;
    bool advanced_matmul_broadcast_offload = false;
    bool has_advanced_matmul_broadcast_offload = false;
    bool triton_matmul_small_n_transpose_opt = false;
    bool has_triton_matmul_small_n_transpose_opt = false;
#endif
};

// Heuristics supported for txe_count:
// (A) explicit scalar: txe_count: 4 OR num_txe: 4 OR txeCount: 4
// (B) list form: txes: \n - ... \n - ...  => count list items under "txes:"
static tsi_deploy_cfg_t tsi_read_deploy_yaml(const std::string &path) {
    tsi_deploy_cfg_t cfg;
    std::ifstream in(path);
    if (!in.is_open()) return cfg;

    std::string line;
    bool in_txes_list = false;
    int txes_list_indent = -1;
    int txes_count = 0;

    while (std::getline(in, line)) {
        // strip comments
        size_t hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);

        std::string raw = line;
        std::string t = tsi_trim_copy(line);
        if (t.empty()) continue;

        // txe_count scalar keys
        if (t.find("txe_count") != std::string::npos && t.find(':') != std::string::npos) {
            int v = tsi_parse_int_after_colon(t);
            if (v > 0) cfg.txe_count = v;
        } else if (t.find("num_txe") != std::string::npos && t.find(':') != std::string::npos) {
            int v = tsi_parse_int_after_colon(t);
            if (v > 0) cfg.txe_count = v;
        } else if (t.find("txeCount") != std::string::npos && t.find(':') != std::string::npos) {
            int v = tsi_parse_int_after_colon(t);
            if (v > 0) cfg.txe_count = v;
        }

        // multi-thread enable keys (accept a few spellings)
        if (t.find("multi_thread_enable") != std::string::npos && t.find(':') != std::string::npos) {
            bool b = false;
            if (tsi_parse_bool_after_colon(t, &b)) { cfg.mt_enable = b; cfg.has_mt = true; }
        } else if (t.find("multi_threading_enable") != std::string::npos && t.find(':') != std::string::npos) {
            bool b = false;
            if (tsi_parse_bool_after_colon(t, &b)) { cfg.mt_enable = b; cfg.has_mt = true; }
        } else if (t.find("multithreading_enable") != std::string::npos && t.find(':') != std::string::npos) {
            bool b = false;
            if (tsi_parse_bool_after_colon(t, &b)) { cfg.mt_enable = b; cfg.has_mt = true; }
        } else if (t.find("multiThreadEnable") != std::string::npos && t.find(':') != std::string::npos) {
            bool b = false;
            if (tsi_parse_bool_after_colon(t, &b)) { cfg.mt_enable = b; cfg.has_mt = true; }
        }

        const size_t user_dram_colon = t.find(':');

        if (user_dram_colon != std::string::npos &&
            tsi_trim_copy(t.substr(0, user_dram_colon)) == "user_dram_size_gb") {

            const int v = tsi_parse_int_after_colon(t);

            if (v > 0) {
                cfg.user_dram_size_gb = v;
                cfg.has_user_dram_size_gb = true;
            }
        }


#if TRITON_MAT_MUL
        if (t.find("advanced_matmul_shape_offload") != std::string::npos &&
            t.find(':') != std::string::npos) {
            bool b = false;
            if (tsi_parse_bool_after_colon(t, &b)) {
                cfg.advanced_matmul_shape_offload = b;
                cfg.has_advanced_matmul_shape_offload = true;
            }
        }
        if (t.find("advanced_matmul_broadcast_offload") != std::string::npos &&
            t.find(':') != std::string::npos) {
            bool b = false;
            if (tsi_parse_bool_after_colon(t, &b)) {
                cfg.advanced_matmul_broadcast_offload = b;
                cfg.has_advanced_matmul_broadcast_offload = true;
            }
        }

        if (t.find("triton_matmul_small_n_transpose_opt") != std::string::npos &&
            t.find(':') != std::string::npos) {
            bool b = false;
            if (tsi_parse_bool_after_colon(t, &b)) {
                cfg.triton_matmul_small_n_transpose_opt = b;
                cfg.has_triton_matmul_small_n_transpose_opt = true;
            }
        }
#endif

        // list counting under "txes:"
        if (tsi_starts_with(t, "txes:")) {
            in_txes_list = true;
            txes_count = 0;
            int indent = 0;
            while (indent < (int)raw.size() && std::isspace((unsigned char)raw[indent])) indent++;
            txes_list_indent = indent;
            continue;
        }

        if (in_txes_list) {
            int indent = 0;
            while (indent < (int)raw.size() && std::isspace((unsigned char)raw[indent])) indent++;
            if (txes_list_indent >= 0 && indent <= txes_list_indent) {
                if (txes_count > 0 && cfg.txe_count <= 0) cfg.txe_count = txes_count;
                in_txes_list = false;
                txes_list_indent = -1;
            } else {
                std::string tt = tsi_trim_copy(raw);
                if (tsi_starts_with(tt, "-")) txes_count++;
            }
        }
    }

    if (txes_count > 0 && cfg.txe_count <= 0) cfg.txe_count = txes_count;
    return cfg;
}

static inline size_t tsi_user_dram_size_bytes_from_cfg(const tsi_deploy_cfg_t &cfg) {
    if (!cfg.has_user_dram_size_gb || cfg.user_dram_size_gb <= 0) {
        fprintf(stderr,
               "WARNING: user_dram_size_gb=%d is invalid. Using runtime default DRAM size.\n",
                                                                              cfg.user_dram_size_gb);
        return 0;
    }

    const uint64_t gib = 1024ull * 1024ull * 1024ull;
    const uint64_t gb = static_cast<uint64_t>(cfg.user_dram_size_gb);

    if (gb > static_cast<uint64_t>(SIZE_MAX) / gib) {
        fprintf(stderr,
                "ERROR: user_dram_size_gb=%d is too large for size_t\n",
                cfg.user_dram_size_gb);
        abort();
    }

    return static_cast<size_t>(gb * gib);
}

// ============================================================================
// DEPLOYMENT YAML PATH RESOLUTION
// Supports both:
//  - Dev/posix: YAML in current working directory (./tsavorite-model-deployment.yaml)
//  - FPGA package: YAML next to the loaded .so (same dir as libggml*.so)
// Priority:
//   1) TSAVORITE_MODEL_DEPLOYMENT_YAML (explicit override)
//   2) <dir-of-loaded-so>/tsavorite-model-deployment.yaml
//   3) ./tsavorite-model-deployment.yaml (current working dir; current behavior)
// ============================================================================
static inline std::string tsi_resolve_deployment_yaml_path() {
    // 1) env override
    if (const char *p = std::getenv("TSAVORITE_MODEL_DEPLOYMENT_YAML")) {
        if (access(p, R_OK) == 0) {
            return std::string(p);
        }
    }

    // 2) next to this loaded shared object
    Dl_info info;
    if (dladdr((void *)&tsi_resolve_deployment_yaml_path, &info) && info.dli_fname) {
        std::string so_path(info.dli_fname);
        size_t pos = so_path.find_last_of('/');
        if (pos != std::string::npos) {
            std::string yaml = so_path.substr(0, pos + 1) + "tsavorite-model-deployment.yaml";
            if (access(yaml.c_str(), R_OK) == 0) {
                return yaml;
            }
        }
    }

    // 3) dev/posix: current working dir (existing behavior)
    return std::string("tsavorite-model-deployment.yaml");
}

#ifdef TMU_DEBUG_VALIDATE

// CPU reference GEMM for TMU packed tiles using the SAME MemRefDescriptor<4>
// struct used by your MLIR ciface wrappers (base/data/offset/shape/strides). 
// Interprets shapes as you set them in init_memref_4d():
//   A: [1,1,M,K]  -> shape[2]=M, shape[3]=K
//   B: [1,1,N,K]  -> shape[2]=N, shape[3]=K   (NOTE: your B_pack is stored as rows=N, cols=K)
//   C: [1,1,M,N]  -> shape[2]=M, shape[3]=N
//
// Strides are in ELEMENTS (not bytes), consistent with init_memref_4d(). 
//
static void cpu_ref_mul_mat_f32(
    const MemRefDescriptor<4> *A_desc,
    const MemRefDescriptor<4> *B_desc,
    MemRefDescriptor<4>       *C_desc
) {
    if (!A_desc || !B_desc || !C_desc) return;
    if (!A_desc->data || !B_desc->data || !C_desc->data) return;

    const int64_t M = A_desc->shape[2];
    const int64_t K = A_desc->shape[3];
    const int64_t N = B_desc->shape[2];

    if (M <= 0 || N <= 0 || K <= 0) return;

    // Reduction dims must match
    if (B_desc->shape[3] != K) return;

    // Output dims must match
    if (C_desc->shape[2] != M) return;
    if (C_desc->shape[3] != N) return;

    const float *A = (const float *) A_desc->data;
    const float *B = (const float *) B_desc->data;
    float       *C = (float       *) C_desc->data;

    // Strides in elements
    const int64_t a_s2 = A_desc->strides[2];
    const int64_t a_s3 = A_desc->strides[3];
    const int64_t b_s2 = B_desc->strides[2];
    const int64_t b_s3 = B_desc->strides[3];
    const int64_t c_s2 = C_desc->strides[2];
    const int64_t c_s3 = C_desc->strides[3];

    const int64_t a_off = A_desc->offset;
    const int64_t b_off = B_desc->offset;
    const int64_t c_off = C_desc->offset;

    for (int64_t r = 0; r < M; ++r) {
        const int64_t a_row = a_off + r * a_s2;
        const int64_t c_row = c_off + r * c_s2;

        for (int64_t n = 0; n < N; ++n) {
            const int64_t b_row = b_off + n * b_s2;   // B is [N,K] packed

            float acc = 0.0f;
            for (int64_t kk = 0; kk < K; ++kk) {
                acc += A[a_row + kk * a_s3] * B[b_row + kk * b_s3];
            }
            C[c_row + n * c_s3] = acc;
        }
    }
}

#endif // TMU_DEBUG_VALIDATE


enum ggml_tsavorite_kernel_mode ggml_tsavorite_kernel_mode_flag = GGML_TSAVORITE_KERNEL_MODE_MLIR;
enum ggml_tsavorite_log_type ggml_tsavorite_log_type_val        = GGML_TSAVORITE_LOG_ALL;

using namespace std;
namespace tsirt = ::tsi::runtime;
typedef struct _txe_device_t *txe_device_s;
typedef struct _txe_compute_pipeline_state_t *txe_compute_pipeline_state_s;
FILE *tsi_op_log_file;
bool runtime_initialized = false;
uint64_t num_of_op;
#define TSI_RUN_TIME_INSTANCE 1

// ============================================================================
// Tsavorite MAT_MUL detailed profiling
//
// Output files are written once at shutdown:
//   1) ggml_tmu_matmul_profile_summary.tsv
//      Small table intended for Confluence / quick review.
//   2) ggml_tmu_matmul_profile_detail.txt
//      Human-readable section report, one section per matrix shape.
//
// Hot path policy:
//   MAT_MUL only updates global in-memory counters. No per-run file writes.
// ============================================================================
struct tsavorite_matmul_profile_sample_t {
    int64_t matrix_total_us = 0;
    int64_t pack_a_us = 0;
    int64_t pack_b_us = 0;
    int64_t padding_memset_us = 0;
    int64_t launch_us = 0;
    int64_t txe_wait_critical_us = 0;
    int64_t txe_wait_sum_us = 0;
    int64_t copyback_us = 0;
    int64_t postprocess_us = 0;
    int64_t kernel_calls = 0;
};

static inline int64_t tsavorite_now_us() {
    return ggml_time_us();
}

static inline int64_t tsavorite_elapsed_us(int64_t start_us) {
    const int64_t end_us = tsavorite_now_us();
    if (end_us >= start_us) {
        return end_us - start_us;
    }
    return 0;
}

#ifdef TMU_DEBUG

struct tsavorite_matmul_profile_bucket_t {
    int64_t runs = 0;
    int64_t kernel_calls = 0;

    int64_t matrix_total_us = 0;
    int64_t pack_a_us = 0;
    int64_t pack_b_us = 0;
    int64_t padding_memset_us = 0;
    int64_t launch_us = 0;
    int64_t txe_wait_critical_us = 0;
    int64_t txe_wait_sum_us = 0;
    int64_t copyback_us = 0;
    int64_t postprocess_us = 0;

    char op_dims[TSAV_DIMS_STR_LEN]    = {0};
    char src0_dims[TSAV_DIMS_STR_LEN]  = {0};
    char src1_dims[TSAV_DIMS_STR_LEN]  = {0};
    char type_name[TSAV_TYPE_NAME_LEN] = {0};
};

static std::mutex g_tsavorite_matmul_profile_mutex;
static std::map<std::string, tsavorite_matmul_profile_bucket_t> g_tsavorite_matmul_profile;

static inline double tsavorite_us_to_ms(int64_t us) {
    return (double)us / 1000.0;
}

static inline double tsavorite_pct(double numerator, double denominator) {
    if (denominator <= 0.0) {
        return 0.0;
    }
    return (numerator * 100.0) / denominator;
}

static inline void tsavorite_dims_to_string(const struct ggml_tensor *t, char *buf, size_t buf_size) {
    if (!buf || buf_size == 0) {
        return;
    }

    if (!t) {
        snprintf(buf, buf_size, "[0,0,0,0]");
        return;
    }

    snprintf(buf, buf_size,
             "[%ld,%ld,%ld,%ld]",
             (long)t->ne[0],
             (long)t->ne[1],
             (long)t->ne[2],
             (long)t->ne[3]);
}

static void tsavorite_matmul_profile_record(
    const struct ggml_tensor *node,
    const tsavorite_matmul_profile_sample_t &sample) {

    if (!node || !node->src[0] || !node->src[1]) {
        return;
    }

    char op_dims[TSAV_DIMS_STR_LEN];
    char src0_dims[TSAV_DIMS_STR_LEN];
    char src1_dims[TSAV_DIMS_STR_LEN];

    tsavorite_dims_to_string(node, op_dims, sizeof(op_dims));
    tsavorite_dims_to_string(node->src[0], src0_dims, sizeof(src0_dims));
    tsavorite_dims_to_string(node->src[1], src1_dims, sizeof(src1_dims));

    char key[TSAV_PROFILE_KEY_LEN];
    snprintf(key, sizeof(key),
             "op=%s|src0=%s|src1=%s|type=%s",
             op_dims,
             src0_dims,
             src1_dims,
             ggml_type_name(node->type));

    std::lock_guard<std::mutex> lock(g_tsavorite_matmul_profile_mutex);

    tsavorite_matmul_profile_bucket_t &b = g_tsavorite_matmul_profile[std::string(key)];

    if (b.runs == 0) {
        snprintf(b.op_dims, sizeof(b.op_dims), "%s", op_dims);
        snprintf(b.src0_dims, sizeof(b.src0_dims), "%s", src0_dims);
        snprintf(b.src1_dims, sizeof(b.src1_dims), "%s", src1_dims);
        snprintf(b.type_name, sizeof(b.type_name), "%s", ggml_type_name(node->type));
    }

    b.runs += 1;
    b.kernel_calls += sample.kernel_calls;

    b.matrix_total_us += sample.matrix_total_us;
    b.pack_a_us += sample.pack_a_us;
    b.pack_b_us += sample.pack_b_us;
    b.padding_memset_us += sample.padding_memset_us;
    b.launch_us += sample.launch_us;
    b.txe_wait_critical_us += sample.txe_wait_critical_us;
    b.txe_wait_sum_us += sample.txe_wait_sum_us;
    b.copyback_us += sample.copyback_us;
    b.postprocess_us += sample.postprocess_us;
}


static void tsavorite_matmul_profile_dump_summary_locked() {
    FILE *f = fopen("ggml_tmu_matmul_profile_summary.tsv", "w");
    if (!f) {
        fprintf(stderr, "ERROR: failed to open ggml_tmu_matmul_profile_summary.tsv for write\n");
        return;
    }

    fprintf(f,
            "Backend\tOp\tType\tRuns\tTSI_KERNEL\t"
            "MatrixTotal_ms\tMatrixAvg_ms\t"
            "TXEWaitCritical_ms\tTXEWaitSum_ms\t"
            "HostWallResidual_ms\tHostWallResidualPct\t"
            "HostAccumulatedAcrossTXE_ms\tHostAccumulatedPct\t"
            "Dimensions\tSrc0\tSrc1\n");

    for (const auto &kv : g_tsavorite_matmul_profile) {
        const tsavorite_matmul_profile_bucket_t &b = kv.second;

        const double matrix_total_ms = tsavorite_us_to_ms(b.matrix_total_us);
        const double matrix_avg_ms = b.runs > 0 ? matrix_total_ms / (double)b.runs : 0.0;
        const double wait_critical_ms = tsavorite_us_to_ms(b.txe_wait_critical_us);
        const double wait_sum_ms = tsavorite_us_to_ms(b.txe_wait_sum_us);

        double host_wall_residual_ms = matrix_total_ms - wait_critical_ms;
        if (host_wall_residual_ms < 0.0) {
            host_wall_residual_ms = 0.0;
        }

        const double host_accumulated_ms =
            tsavorite_us_to_ms(b.pack_a_us) +
            tsavorite_us_to_ms(b.pack_b_us) +
            tsavorite_us_to_ms(b.padding_memset_us) +
            tsavorite_us_to_ms(b.launch_us) +
            tsavorite_us_to_ms(b.copyback_us) +
            tsavorite_us_to_ms(b.postprocess_us);

        fprintf(f,
                "OPU\tMUL_MAT\t%s\t%ld\t%ld\t"
                "%.3f\t%.3f\t"
                "%.3f\t%.3f\t"
                "%.3f\t%.2f\t"
                "%.3f\t%.2f\t"
                "%s\t%s\t%s\n",
                b.type_name,
                (long)b.runs,
                (long)b.kernel_calls,
                matrix_total_ms,
                matrix_avg_ms,
                wait_critical_ms,
                wait_sum_ms,
                host_wall_residual_ms,
                tsavorite_pct(host_wall_residual_ms, matrix_total_ms),
                host_accumulated_ms,
                tsavorite_pct(host_accumulated_ms, matrix_total_ms),
                b.op_dims,
                b.src0_dims,
                b.src1_dims);
    }

    fclose(f);
}

static void tsavorite_matmul_profile_dump_detail_locked() {
    FILE *f = fopen("ggml_tmu_matmul_profile_detail.txt", "w");
    if (!f) {
        fprintf(stderr, "ERROR: failed to open ggml_tmu_matmul_profile_detail.txt for write\n");
        return;
    }

    fprintf(f, "Tsavorite MAT_MUL Detailed Profile Report\n");
    fprintf(f, "=========================================\n\n");
    fprintf(f, "This report is aggregated in memory and written once at shutdown.\n");
    fprintf(f, "MatrixTotal is wall-clock time for the OPU MUL_MAT path.\n");
    fprintf(f, "TXEWaitCritical approximates parallel TXE wait wall time.\n");
    fprintf(f, "TXEWaitSum is accumulated wait across TXEs and may exceed wall time.\n");
    fprintf(f, "HostWallResidual is MatrixTotal minus TXEWaitCritical.\n");
    fprintf(f, "HostAccumulatedAcrossTXE is the sum of host-side worker timings across TXEs.\n");
    fprintf(f, "In multi-TXE mode, HostAccumulatedAcrossTXE is not expected to equal HostWallResidual.\n");
    fprintf(f, "tsi_finalize_command_list time is included in MatrixTotal and HostWallResidual, not in TXEWaitCritical.\n\n");

    int shape_index = 1;
    for (const auto &kv : g_tsavorite_matmul_profile) {
        const tsavorite_matmul_profile_bucket_t &b = kv.second;

        const double matrix_total_ms = tsavorite_us_to_ms(b.matrix_total_us);
        const double matrix_avg_ms = b.runs > 0 ? matrix_total_ms / (double)b.runs : 0.0;
        const double pack_a_ms = tsavorite_us_to_ms(b.pack_a_us);
        const double pack_b_ms = tsavorite_us_to_ms(b.pack_b_us);
        const double padding_ms = tsavorite_us_to_ms(b.padding_memset_us);
        const double launch_ms = tsavorite_us_to_ms(b.launch_us);
        const double wait_critical_ms = tsavorite_us_to_ms(b.txe_wait_critical_us);
        const double wait_sum_ms = tsavorite_us_to_ms(b.txe_wait_sum_us);
        const double copyback_ms = tsavorite_us_to_ms(b.copyback_us);
        const double post_ms = tsavorite_us_to_ms(b.postprocess_us);

        double host_wall_residual_ms = matrix_total_ms - wait_critical_ms;
        if (host_wall_residual_ms < 0.0) {
            host_wall_residual_ms = 0.0;
        }

        const double host_accumulated_ms =
            pack_a_ms +
            pack_b_ms +
            padding_ms +
            launch_ms +
            copyback_ms +
            post_ms;

        fprintf(f, "============================================================\n");
        fprintf(f, "Matrix Shape #%d\n", shape_index++);
        fprintf(f, "============================================================\n");
        fprintf(f, "Backend              : OPU\n");
        fprintf(f, "Op                   : MUL_MAT\n");
        fprintf(f, "Type                 : %s\n", b.type_name);
        fprintf(f, "Runs                 : %ld\n", (long)b.runs);
        fprintf(f, "TSI_KERNEL           : %ld\n\n", (long)b.kernel_calls);

        fprintf(f, "Shape\n");
        fprintf(f, "-----\n");
        fprintf(f, "Dimensions           : %s\n", b.op_dims);
        fprintf(f, "Src0                 : %s\n", b.src0_dims);
        fprintf(f, "Src1                 : %s\n\n", b.src1_dims);

        fprintf(f, "Timing Summary\n");
        fprintf(f, "--------------\n");
        fprintf(f, "MatrixTotal_ms              : %.3f\n", matrix_total_ms);
        fprintf(f, "MatrixAvg_ms                : %.3f\n", matrix_avg_ms);
        fprintf(f, "TXEWaitCritical_ms          : %.3f  %.2f%%\n",
                wait_critical_ms,
                tsavorite_pct(wait_critical_ms, matrix_total_ms));
        fprintf(f, "TXEWaitSum_ms               : %.3f\n", wait_sum_ms);
        fprintf(f, "HostWallResidual_ms         : %.3f  %.2f%%\n",
                host_wall_residual_ms,
                tsavorite_pct(host_wall_residual_ms, matrix_total_ms));
        fprintf(f, "HostAccumulatedAcrossTXE_ms : %.3f  %.2f%%\n\n",
                host_accumulated_ms,
                tsavorite_pct(host_accumulated_ms, matrix_total_ms));

        fprintf(f, "Accumulated Per-TXE Breakdown\n");
        fprintf(f, "-----------------------------\n");
        fprintf(f, "PackA_ms             : %.3f  %.2f%%\n",
                pack_a_ms,
                tsavorite_pct(pack_a_ms, matrix_total_ms));
        fprintf(f, "PackB_ms             : %.3f  %.2f%%\n",
                pack_b_ms,
                tsavorite_pct(pack_b_ms, matrix_total_ms));
        fprintf(f, "PaddingMemset_ms     : %.3f  %.2f%%\n",
                padding_ms,
                tsavorite_pct(padding_ms, matrix_total_ms));
        fprintf(f, "LaunchOrWrapper_ms   : %.3f  %.2f%%\n",
                launch_ms,
                tsavorite_pct(launch_ms, matrix_total_ms));
        fprintf(f, "CopyBack_ms          : %.3f  %.2f%%\n",
                copyback_ms,
                tsavorite_pct(copyback_ms, matrix_total_ms));
        fprintf(f, "PostProcess_ms       : %.3f  %.2f%%\n\n",
                post_ms,
                tsavorite_pct(post_ms, matrix_total_ms));

        fprintf(f, "Accounting Note\n");
        fprintf(f, "---------------\n");
        fprintf(f, "HostWallResidual_ms is wall-clock residual time: MatrixTotal_ms - TXEWaitCritical_ms.\n");
        fprintf(f, "The accumulated per-TXE breakdown above is summed across worker TXEs in multi-TXE mode.\n");
        fprintf(f, "Therefore PackA_ms + PackB_ms + PaddingMemset_ms + LaunchOrWrapper_ms + CopyBack_ms + PostProcess_ms is not expected to equal HostWallResidual_ms when multiple TXEs run in parallel.\n\n");

        fprintf(f, "Interpretation\n");
        fprintf(f, "--------------\n");
        if (matrix_total_ms <= 0.0) {
            fprintf(f, "No measurable matrix time was recorded for this shape.\n");
        } else if (wait_critical_ms > 0.0 && tsavorite_pct(wait_critical_ms, matrix_total_ms) >= 70.0) {
            fprintf(f, "Most wall-clock matrix time is in the TXE/blob wait critical path. Check TXE/blob execution or simulation time.\n");
        } else if (launch_ms > 0.0 && tsavorite_pct(launch_ms, matrix_total_ms) >= 70.0 && wait_critical_ms <= 0.0) {
            fprintf(f, "Most matrix time is charged to LaunchOrWrapper_ms while TXE wait is zero. For the single-TXE/generated-wrapper path, LaunchOrWrapper_ms includes blocking generated wrapper execution, so TXE wait is not separately visible in this path.\n");
        } else if (tsavorite_pct(host_wall_residual_ms, matrix_total_ms) >= 70.0) {
            fprintf(f, "Most wall-clock matrix time is outside TXE critical wait. Check packing, padding, launch wrapper, finalize, copy-back, and post-processing.\n");
        } else {
            fprintf(f, "Time is split across host wall-clock residual and TXE wait buckets. Review the timing summary and accumulated per-TXE breakdown above.\n");
        }

        fprintf(f, "\n\n");
    }

    fclose(f);
}

static void tsavorite_matmul_profile_dump() {
    std::lock_guard<std::mutex> lock(g_tsavorite_matmul_profile_mutex);
    tsavorite_matmul_profile_dump_summary_locked();
    tsavorite_matmul_profile_dump_detail_locked();
}

#else

static void tsavorite_matmul_profile_record(
    const struct ggml_tensor *node,
    const tsavorite_matmul_profile_sample_t &sample) {
    (void)node;
    (void)sample;
}

static void tsavorite_matmul_profile_dump() {
}

#endif /* TMU_DEBUG */

#if defined(GGML_PERF) || defined(GGML_PERF_DETAIL)

// ============================================================================
// Tsavorite op shape/dtype catalog profiling
//
// Output file is written once at shutdown:
//   ggml_op_shape_dtype_catalog.tsv
//
// Purpose:
//   Capture model-level op patterns, including result/src0/src1 dtype,
//   tensor shapes, count, support decision, and reject/support reason.
//
// This is intentionally not gated by TMU_DEBUG because it is not MAT_MUL-only.
// It is enabled for debug/perf builds and disabled for release builds.
// ============================================================================

struct tsavorite_op_shape_dtype_bucket_t {
    int64_t count = 0;

    char op_name[TSAV_TYPE_NAME_LEN] = {0};
    char decision[TSAV_TYPE_NAME_LEN] = {0};
    char reason[TSAV_PROFILE_KEY_LEN] = {0};

    char result_type[TSAV_TYPE_NAME_LEN] = {0};
    char result_dims[TSAV_DIMS_STR_LEN] = {0};

    char src0_type[TSAV_TYPE_NAME_LEN] = {0};
    char src0_dims[TSAV_DIMS_STR_LEN] = {0};

    char src1_type[TSAV_TYPE_NAME_LEN] = {0};
    char src1_dims[TSAV_DIMS_STR_LEN] = {0};
};

static std::mutex g_tsavorite_op_shape_dtype_catalog_mutex;
static std::map<std::string, tsavorite_op_shape_dtype_bucket_t> g_tsavorite_op_shape_dtype_catalog;
static std::map<std::string, bool> g_tsavorite_op_shape_dtype_catalog_seen;

static inline void tsavorite_catalog_dims_to_string(
    const struct ggml_tensor *t,
    char *buf,
    size_t buf_size) {

    if (!buf || buf_size == 0) {
        return;
    }

    if (!t) {
        snprintf(buf, buf_size, "[0,0,0,0]");
        return;
    }

    snprintf(buf, buf_size,
             "[%ld,%ld,%ld,%ld]",
             (long)t->ne[0],
             (long)t->ne[1],
             (long)t->ne[2],
             (long)t->ne[3]);
}

static inline const char *tsavorite_catalog_type_name_safe(const struct ggml_tensor *t) {
    if (!t) {
        return "none";
    }

    return ggml_type_name(t->type);
}

static void tsavorite_op_shape_dtype_catalog_record(
    const struct ggml_tensor *op,
    const char *decision,
    const char *reason) {

    if (!op) {
        return;
    }

    if (!decision) {
        decision = "UNKNOWN";
    }

    if (!reason) {
        reason = "none";
    }

    char result_dims[TSAV_DIMS_STR_LEN];
    char src0_dims[TSAV_DIMS_STR_LEN];
    char src1_dims[TSAV_DIMS_STR_LEN];

    tsavorite_catalog_dims_to_string(op, result_dims, sizeof(result_dims));
    tsavorite_catalog_dims_to_string(op->src[0], src0_dims, sizeof(src0_dims));
    tsavorite_catalog_dims_to_string(op->src[1], src1_dims, sizeof(src1_dims));

    const char *op_name = ggml_op_name(op->op);
    const char *result_type = tsavorite_catalog_type_name_safe(op);
    const char *src0_type = tsavorite_catalog_type_name_safe(op->src[0]);
    const char *src1_type = tsavorite_catalog_type_name_safe(op->src[1]);

    char key[TSAV_PROFILE_KEY_LEN * 2];
    snprintf(key, sizeof(key),
             "op=%s|decision=%s|reason=%s|result=%s:%s|src0=%s:%s|src1=%s:%s",
             op_name,
             decision,
             reason,
             result_type,
             result_dims,
             src0_type,
             src0_dims,
             src1_type,
             src1_dims);

    std::lock_guard<std::mutex> lock(g_tsavorite_op_shape_dtype_catalog_mutex);

    char seen_key[TSAV_PROFILE_KEY_LEN];
    snprintf(seen_key, sizeof(seen_key),
             "tensor=%p|decision=%s|reason=%s",
             (const void *)op,
             decision,
             reason);

    const std::string seen_key_str(seen_key);
    if (g_tsavorite_op_shape_dtype_catalog_seen.find(seen_key_str) !=
        g_tsavorite_op_shape_dtype_catalog_seen.end()) {
        return;
    }

    g_tsavorite_op_shape_dtype_catalog_seen[seen_key_str] = true;

    tsavorite_op_shape_dtype_bucket_t &b =
        g_tsavorite_op_shape_dtype_catalog[std::string(key)];

    if (b.count == 0) {
        snprintf(b.op_name, sizeof(b.op_name), "%s", op_name);
        snprintf(b.decision, sizeof(b.decision), "%s", decision);
        snprintf(b.reason, sizeof(b.reason), "%s", reason);

        snprintf(b.result_type, sizeof(b.result_type), "%s", result_type);
        snprintf(b.result_dims, sizeof(b.result_dims), "%s", result_dims);

        snprintf(b.src0_type, sizeof(b.src0_type), "%s", src0_type);
        snprintf(b.src0_dims, sizeof(b.src0_dims), "%s", src0_dims);

        snprintf(b.src1_type, sizeof(b.src1_type), "%s", src1_type);
        snprintf(b.src1_dims, sizeof(b.src1_dims), "%s", src1_dims);
    }

    b.count += 1;
}

static void tsavorite_op_shape_dtype_catalog_dump() {
    std::lock_guard<std::mutex> lock(g_tsavorite_op_shape_dtype_catalog_mutex);

    FILE *f = fopen("ggml_op_shape_dtype_catalog.tsv", "w");
    if (!f) {
        fprintf(stderr, "ERROR: failed to open ggml_op_shape_dtype_catalog.tsv for write\n");
        return;
    }

    fprintf(f,
            "%-18s %-10s %8s %-32s %-32s %-32s\n",
            "Op",
            "Decision",
            "Count",
            "Result",
            "Src0",
            "Src1");

    fprintf(f,
            "%-18s %-10s %8s %-32s %-32s %-32s\n",
            "------------------",
            "----------",
            "--------",
            "--------------------------------",
            "--------------------------------",
            "--------------------------------");


    for (const auto &kv : g_tsavorite_op_shape_dtype_catalog) {
        const tsavorite_op_shape_dtype_bucket_t &b = kv.second;

        char result[TSAV_PROFILE_KEY_LEN];
        char src0[TSAV_PROFILE_KEY_LEN];
        char src1[TSAV_PROFILE_KEY_LEN];

        snprintf(result, sizeof(result), "%s %s", b.result_type, b.result_dims);
        snprintf(src0, sizeof(src0), "%s %s", b.src0_type, b.src0_dims);
        snprintf(src1, sizeof(src1), "%s %s", b.src1_type, b.src1_dims);

        fprintf(f,
                "%-18s %-10s %8ld %-32s %-32s %-32s\n",
                b.op_name,
                b.decision,
                (long)b.count,
                result,
                src0,
                src1);
    }

    fclose(f);
}

#else

static void tsavorite_op_shape_dtype_catalog_record(
    const struct ggml_tensor *op,
    const char *decision,
    const char *reason) {
    (void)op;
    (void)decision;
    (void)reason;
}

static void tsavorite_op_shape_dtype_catalog_dump() {
}

#endif /* GGML_PERF || GGML_PERF_DETAIL */

// ============================================================
// (makes blob names unique per device to avoid collisions)
//  - tsi_load_blob() expects a FILE PREFIX, not a directory.
//  - It will append ".blob" internally.
//  - Therefore we must pass ".../blobs/txe_xxx" (NOT ".../blobs").
//  - This code reconstructs the SAME prefix paths that worked before.
// =======================================================================

static std::string tsavorite_llama_root() {
    // __FILE__ =
    // /proj/work/.../llama.cpp/ggml/src/ggml-tsavorite/ggml-tsavorite.cpp
    std::string f(__FILE__);
    const std::string tag = "/ggml/src/ggml-tsavorite/ggml-tsavorite.cpp";
    size_t pos = f.find(tag);
    if (pos == std::string::npos) {
        // Hard fail — path layout assumption broken
        return "";
    }
    // Result:
    // /proj/work/.../llama.cpp
    return f.substr(0, pos);
}

static std::string blob_prefix(const char *rel) {
    return tsavorite_llama_root() + rel;
}

#ifdef GGML_TARGET_POSIX
#define TSAVORITE_BLOB_BUILD_ROOT "/ggml-tsi-kernel/posix-kernel/build-posix"
#else
#define TSAVORITE_BLOB_BUILD_ROOT "/ggml-tsi-kernel/fpga-kernel/build-fpga"
#endif

static inline void tsi_blob_free_tables() {
    // free pointer tables only (does NOT unload blobs)
    if (loadResult_add) {
        free(loadResult_add);
        loadResult_add = nullptr;
    }
    if (loadResult_mult) {
        free(loadResult_mult);
        loadResult_mult = nullptr;
    }
    if (loadResult_rms_norm) {
        free(loadResult_rms_norm);
        loadResult_rms_norm = nullptr;
    }

    if (blobDescriptor_add) {
        free(blobDescriptor_add);
        blobDescriptor_add = nullptr;
    }
    if (blobDescriptor_mult) {
        free(blobDescriptor_mult);
        blobDescriptor_mult = nullptr;
    }
    if (blobDescriptor_rms_norm) {
        free(blobDescriptor_rms_norm);
        blobDescriptor_rms_norm = nullptr;
    }

#if TRITON_MAT_MUL
    if (loadResult_matmul_1x8) {
        free(loadResult_matmul_1x8);
        loadResult_matmul_1x8 = nullptr;
    }
    if (loadResult_matmul_2x4) {
        free(loadResult_matmul_2x4);
        loadResult_matmul_2x4 = nullptr;
    }

    if (blobDescriptor_matmul_1x8) {
        free(blobDescriptor_matmul_1x8);
        blobDescriptor_matmul_1x8 = nullptr;
    }
    if (blobDescriptor_matmul_2x4) {
        free(blobDescriptor_matmul_2x4);
        blobDescriptor_matmul_2x4 = nullptr;
    }
#endif

#if TRITON_ADD
    if (loadResult_triton_add) {
        free(loadResult_triton_add);
        loadResult_triton_add = nullptr;
    }
    if (blobDescriptor_triton_add) {
        free(blobDescriptor_triton_add);
        blobDescriptor_triton_add = nullptr;
    }
#endif

    g_rt.blob_tables_txes = 0;
    g_rt.blob_state = TsavoriteRuntimeState::BLOB_UNINITIALIZED;
}

static inline void tsi_blob_unload_only() {
    // unload blobs if present, keep tables allocated
    if (blobDescriptor_add) {
        for (uint32_t i = 0; i < TSI_RUN_TIME_INSTANCE; ++i) {
            if (blobDescriptor_add[i]) {
                tsi_unload_blob(blobDescriptor_add[i]);
                blobDescriptor_add[i] = nullptr;
            }
        }
    }
#if TRITON_ADD
    if (blobDescriptor_triton_add) {
        for (uint32_t i = 0; i < TSI_RUN_TIME_INSTANCE; ++i) {
            if (blobDescriptor_triton_add[i]) {
                tsi_unload_blob(blobDescriptor_triton_add[i]);
                blobDescriptor_triton_add[i] = nullptr;
            }
        }
    }
#endif

    if (blobDescriptor_mult) {
        for (uint32_t i = 0; i < TSI_RUN_TIME_INSTANCE; ++i) {
            if (blobDescriptor_mult[i]) {
                tsi_unload_blob(blobDescriptor_mult[i]);
                blobDescriptor_mult[i] = nullptr;
            }
        }
    }
    if (blobDescriptor_rms_norm) {
        for (uint32_t i = 0; i < TSI_RUN_TIME_INSTANCE; ++i) {
            if (blobDescriptor_rms_norm[i]) {
                tsi_unload_blob(blobDescriptor_rms_norm[i]);
                blobDescriptor_rms_norm[i] = nullptr;
            }
        }
    }
#if TRITON_ADD
    if (loadResult_triton_add) {
        memset(loadResult_triton_add, 0, TSI_RUN_TIME_INSTANCE * sizeof(void *));
    }
#endif
#if TRITON_MAT_MUL
    if (blobDescriptor_matmul_1x8) {
        for (uint32_t i = 0; i < TSI_RUN_TIME_INSTANCE; ++i) {
            if (blobDescriptor_matmul_1x8[i]) {
                tsi_unload_blob(blobDescriptor_matmul_1x8[i]);
                blobDescriptor_matmul_1x8[i] = nullptr;
            }
        }
    }
    if (blobDescriptor_matmul_2x4) {
        for (uint32_t i = 0; i < TSI_RUN_TIME_INSTANCE; ++i) {
            if (blobDescriptor_matmul_2x4[i]) {
                tsi_unload_blob(blobDescriptor_matmul_2x4[i]);
                blobDescriptor_matmul_2x4[i] = nullptr;
            }
        }
    }
#endif

    // best-effort: clear loadResult_* entries too
    if (loadResult_add)      memset(loadResult_add,      0, TSI_RUN_TIME_INSTANCE * sizeof(void *));
    if (loadResult_mult)     memset(loadResult_mult,     0, TSI_RUN_TIME_INSTANCE * sizeof(void *));
    if (loadResult_rms_norm) memset(loadResult_rms_norm, 0, TSI_RUN_TIME_INSTANCE * sizeof(void *));
#if TRITON_MAT_MUL
    if (loadResult_matmul_1x8) {
        memset(loadResult_matmul_1x8, 0, TSI_RUN_TIME_INSTANCE * sizeof(void *));
    }
    if (loadResult_matmul_2x4) {
        memset(loadResult_matmul_2x4, 0, TSI_RUN_TIME_INSTANCE * sizeof(void *));
    }
#endif

    g_rt.blob_state = TsavoriteRuntimeState::BLOB_TABLES_ALLOCATED;
}

static inline void tsi_blob_ensure_tables_allocated() {
    if (g_rt.blob_state != TsavoriteRuntimeState::BLOB_UNINITIALIZED) {
        // if sized for a different txe count, reset hard (future-proof)
        if (g_rt.blob_tables_txes != num_of_txes) {
            tsi_blob_unload_only();
            tsi_blob_free_tables();
        } else {
            return;
        }
    }

    loadResult_add      = (void **)calloc(TSI_RUN_TIME_INSTANCE, sizeof(void *));
    loadResult_mult     = (void **)calloc(TSI_RUN_TIME_INSTANCE, sizeof(void *));
    loadResult_rms_norm = (void **)calloc(TSI_RUN_TIME_INSTANCE, sizeof(void *));
#if TRITON_ADD
    loadResult_triton_add = (void **)calloc(TSI_RUN_TIME_INSTANCE, sizeof(void *));
#endif
#if TRITON_MAT_MUL
    loadResult_matmul_1x8 = (void **)calloc(TSI_RUN_TIME_INSTANCE, sizeof(void *));
    loadResult_matmul_2x4 = (void **)calloc(TSI_RUN_TIME_INSTANCE, sizeof(void *));
#endif

    blobDescriptor_add      = (BlobDescriptor **)calloc(TSI_RUN_TIME_INSTANCE, sizeof(BlobDescriptor *));
    blobDescriptor_mult     = (BlobDescriptor **)calloc(TSI_RUN_TIME_INSTANCE, sizeof(BlobDescriptor *));
    blobDescriptor_rms_norm = (BlobDescriptor **)calloc(TSI_RUN_TIME_INSTANCE, sizeof(BlobDescriptor *));
#if TRITON_ADD
    blobDescriptor_triton_add = (BlobDescriptor **)calloc(TSI_RUN_TIME_INSTANCE, sizeof(BlobDescriptor *));
    if (!loadResult_triton_add || !blobDescriptor_triton_add) {
        tsi_blob_free_tables();
        fprintf(stderr, "Failed to allocate Triton ADD blob tables\n");
        abort();
    }
#endif
#if TRITON_MAT_MUL
    blobDescriptor_matmul_1x8 = (BlobDescriptor **)calloc(TSI_RUN_TIME_INSTANCE, sizeof(BlobDescriptor *));
    blobDescriptor_matmul_2x4 = (BlobDescriptor **)calloc(TSI_RUN_TIME_INSTANCE, sizeof(BlobDescriptor *));
#endif

    if (!loadResult_add || !loadResult_mult || !loadResult_rms_norm ||
#if TRITON_MAT_MUL
        !loadResult_matmul_1x8 || !loadResult_matmul_2x4 ||
#endif
        !blobDescriptor_add || !blobDescriptor_mult || !blobDescriptor_rms_norm
#if TRITON_MAT_MUL
        || !blobDescriptor_matmul_1x8 || !blobDescriptor_matmul_2x4
#endif
        ) {
        // free any partial allocations before abort
        tsi_blob_free_tables();
        fprintf(stderr, "Failed to allocate blob tables (num_of_txes=%u)\n", (unsigned)num_of_txes);
        abort();
    }

    g_rt.blob_tables_txes = num_of_txes;
    g_rt.blob_state = TsavoriteRuntimeState::BLOB_TABLES_ALLOCATED;
}

static void tsi_load_all_blobs() {
    char blob_name[64];
    uint32_t failed_txe = 0;

    // already loaded
    if (g_rt.blob_state == TsavoriteRuntimeState::BLOB_BLOBS_LOADED) {
        return;
    }

    // ensure tables exist (allocates if needed)
    tsi_blob_ensure_tables_allocated();
    
    // size matches runtime txe_count
    //packed_args.resize(num_of_txes, nullptr);

    for (uint32_t i = 0; i < TSI_RUN_TIME_INSTANCE; ++i) {
        char name_add[64];
        char name_mult[64];
        char name_rms[64];
#if TRITON_MAT_MUL
        char name_matmul[64];
        char name_matmul_2x4[64];
#endif
#if TRITON_ADD
        char name_triton_add[64];
#endif


#ifdef GGML_TARGET_POSIX
        snprintf(name_add,  sizeof(name_add),  "txe_add");
        snprintf(name_mult, sizeof(name_mult), "txe_mult");
        snprintf(name_rms,  sizeof(name_rms),  "txe_rms_norm");
#if TRITON_MAT_MUL
        snprintf(name_matmul, sizeof(name_matmul), "txe_blob_0");
        snprintf(name_matmul_2x4, sizeof(name_matmul_2x4), "txe_blob_0");
#endif
#if TRITON_ADD
        snprintf(name_triton_add, sizeof(name_triton_add), "txe_blob_0");
#endif
#else
        snprintf(name_add,  sizeof(name_add),  "txe_add_dev%u",  i);
        snprintf(name_mult, sizeof(name_mult), "txe_mult_dev%u", i);
        snprintf(name_rms,  sizeof(name_rms),  "txe_rms_norm_dev%u", i);
#if TRITON_MAT_MUL
        snprintf(name_matmul, sizeof(name_matmul), "txe_triton_mat_mul_1x8_dev%u", i);
        snprintf(name_matmul_2x4, sizeof(name_matmul_2x4), "txe_triton_mat_mul_2x4_dev%u", i);
#endif
#if TRITON_ADD
        snprintf(name_triton_add, sizeof(name_triton_add), "txe_blob_0");
#endif
#endif
        failed_txe = i;

        // ADD
        loadResult_add[i] = tsi_load_blob(
            i,
            name_add,
            blob_prefix(
                TSAVORITE_BLOB_BUILD_ROOT "/txe_add/blobs/txe_add"
            ).c_str()
        );
        if (!loadResult_add[i]) {
            strcpy(blob_name, name_add);
            goto error;
        }
        blobDescriptor_add[i] =
            static_cast<BlobDescriptor *>(loadResult_add[i]);

        // MULT
        loadResult_mult[i] = tsi_load_blob(
            i,
            name_mult,
            blob_prefix(
                TSAVORITE_BLOB_BUILD_ROOT "/txe_mult/blobs/txe_mult"
            ).c_str()
        );
        if (!loadResult_mult[i]) {
            strcpy(blob_name, name_mult);
            goto error;
        }
        blobDescriptor_mult[i] =
            static_cast<BlobDescriptor *>(loadResult_mult[i]);

        // RMS NORM
        loadResult_rms_norm[i] = tsi_load_blob(
            i,
            name_rms,
            blob_prefix(
                TSAVORITE_BLOB_BUILD_ROOT "/txe_rms_norm/blobs/txe_rms_norm"
            ).c_str()
        );
        if (!loadResult_rms_norm[i]) {
            strcpy(blob_name, name_rms);
            goto error;
        }
        blobDescriptor_rms_norm[i] =
            static_cast<BlobDescriptor *>(loadResult_rms_norm[i]);


    #if TRITON_MAT_MUL
        // Triton MAT_MUL 1x8
        loadResult_matmul_1x8[i] = tsi_load_blob(
            i,
            name_matmul,
            blob_prefix(
                TSAVORITE_BLOB_BUILD_ROOT "/txe_triton_mat_mul_1x8/blobs/txe_blob_0"
            ).c_str()
        );

        if (!loadResult_matmul_1x8[i]) {
            strcpy(blob_name, name_matmul);
            goto error;
        }

        blobDescriptor_matmul_1x8[i] =
            static_cast<BlobDescriptor *>(loadResult_matmul_1x8[i]);

        // Triton MAT_MUL 2x4. Same packed-args ABI as 1x8; only blob/wrapper differs.
        if (advanced_matmul_shape_offload) {
            loadResult_matmul_2x4[i] = tsi_load_blob(
                i,
                name_matmul_2x4,
                blob_prefix(
                    TSAVORITE_BLOB_BUILD_ROOT "/txe_triton_mat_mul_2x4/blobs/txe_blob_0"
                ).c_str()
            );

            if (!loadResult_matmul_2x4[i]) {
                strcpy(blob_name, name_matmul_2x4);
                goto error;
            }

            blobDescriptor_matmul_2x4[i] =
                static_cast<BlobDescriptor *>(loadResult_matmul_2x4[i]);
        }
    #endif
#if TRITON_ADD
        // Triton ADD
        loadResult_triton_add[i] = tsi_load_blob(
            i,
            name_triton_add,
            blob_prefix(
                TSAVORITE_BLOB_BUILD_ROOT "/txe_triton_add/blobs/txe_blob_0"
            ).c_str()
        );

        if (!loadResult_triton_add[i]) {
            strcpy(blob_name, name_triton_add);
            goto error;
        }

        blobDescriptor_triton_add[i] =
            static_cast<BlobDescriptor *>(loadResult_triton_add[i]);
    #endif
    }

    // success
    g_rt.blob_state = TsavoriteRuntimeState::BLOB_BLOBS_LOADED;
    return;

error:
    fprintf(stderr,
        "Failed to load blob (txe=%u, name=%s)\n",
        failed_txe, blob_name
    );

    // Cleanup: unload any blobs that did load + free tables before abort
    tsi_blob_unload_only();   // unload any blobs that succeeded
    tsi_blob_free_tables();   // free calloc’d tables

    // preserve existing hard‑fail behavior
    tsi_cleanup();
    abort();
}

static void tsi_unload_all_blobs() {
    if (g_rt.blob_state == TsavoriteRuntimeState::BLOB_UNINITIALIZED) {
        return;
    }

    // if blobs were loaded (or partially loaded), unload them
    if (g_rt.blob_state == TsavoriteRuntimeState::BLOB_BLOBS_LOADED ||
        g_rt.blob_state == TsavoriteRuntimeState::BLOB_TABLES_ALLOCATED) {
        tsi_blob_unload_only();
    }

    // always free tables after unload
    tsi_blob_free_tables();
}

// Call at every teardown site that frees device_free (tsi_cleanup,
// ggml_tsavorite_free, tsi_log_profile_info). tsi_finalize() invalidates
// the per-TXE tsi_alloc buffers, but packed_args/scalar_*_args keep their
// old (now dangling) pointers -- tsi_init_per_txe_state_once() only
// reallocates when a vector's size changes, so leaving them at their
// current size would make the next dispatch use stale buffers. Clearing
// them (not just resetting per_txe_state_initialized) forces a full
// reallocation on the next init.
static inline void tsi_reset_per_txe_state_after_teardown() {
    packed_args.clear();
    scalar_loop_args.clear();
    scalar_m_args.clear();
    scalar_n_args.clear();
    scalar_k_args.clear();
    scalar_grid1_args.clear();
    scalar_grid2_args.clear();
    scalar_grid3_args.clear();
    per_txe_state_initialized.store(false, std::memory_order_release);
}

static inline void tsi_init_per_txe_state_once() {
    // This is called unconditionally at the top of every op-dispatch entry
    // point, so once initialization is done, skip the lock entirely rather
    // than paying a mutex acquisition on every single dispatch. The flag
    // lives in shared runtime state (not a function-local static) because
    // tsi_cleanup()/ggml_tsavorite_free()/tsi_log_profile_info() free
    // device_free and must reset this alongside it, or a later dispatch
    // would skip reallocation and dereference the freed pointer.
    if (per_txe_state_initialized.load(std::memory_order_acquire)) {
        return;
    }

    // Guards device_free/packed_args/scalar_*_args lazy allocation below.
    // Concurrent worker threads can race into the check-then-act allocation
    // (a non-atomic std::vector mutation) on first use, corrupting
    // packed_args and causing an intermittent, hard-to-repro crash later
    // inside the SDK.
    std::lock_guard<std::mutex> lock(tsi_init_mutex);
    if (per_txe_state_initialized.load(std::memory_order_relaxed)) {
        return; // another thread finished initializing while we waited for the lock
    }

    // allocate device_free[]
    if (!device_free) {
        device_free = (bool*)calloc(num_of_txes, sizeof(bool));
        if (!device_free) {
            fprintf(stderr, "ERROR: failed to allocate device_free array (calloc failed)\n");
            tsi_cleanup();
            abort();
        }
        for (uint32_t i = 0; i < num_of_txes; ++i) device_free[i] = true;
    }

    // allocate per-TXE packed args buffer (device-visible)
    constexpr size_t kPackedArgsBytesMax = 2048;
    constexpr size_t scalarLoopBytesMax  = 2048;

    if (packed_args.size() != num_of_txes) {
        packed_args.assign(num_of_txes, nullptr);

        scalar_loop_args.assign(num_of_txes, nullptr);
        scalar_m_args.assign(num_of_txes, nullptr);
        scalar_n_args.assign(num_of_txes, nullptr);
        scalar_k_args.assign(num_of_txes, nullptr);

        scalar_grid1_args.assign(num_of_txes, nullptr);
        scalar_grid2_args.assign(num_of_txes, nullptr);
        scalar_grid3_args.assign(num_of_txes, nullptr);
        for (uint32_t i = 0; i < num_of_txes; ++i) {
            if (!packed_args[i]) {
                packed_args[i] = tsi_alloc(kPackedArgsBytesMax);
                if (!packed_args[i]) {
                    fprintf(stderr, "tsi_alloc failed for packed_args[%u]\n", i);
                    abort();
                }
            }

            if (!scalar_loop_args[i]) {
                scalar_loop_args[i] = tsi_alloc(scalarLoopBytesMax);
                if (!scalar_loop_args[i]) {
                    fprintf(stderr, "tsi_alloc failed for scalar_loop_args[%u]\n", i);
                    abort();
                }
            }
            if (!scalar_m_args[i]) {
                scalar_m_args[i] = tsi_alloc(scalarLoopBytesMax);
                if (!scalar_m_args[i]) {
                    fprintf(stderr, "tsi_alloc failed for scalar_m_args[%u]\n", i);
                    abort();
                }
            }

            if (!scalar_n_args[i]) {
                scalar_n_args[i] = tsi_alloc(scalarLoopBytesMax);
                if (!scalar_n_args[i]) {
                    fprintf(stderr, "tsi_alloc failed for scalar_n_args[%u]\n", i);
                    abort();
                }
            }

            if (!scalar_k_args[i]) {
                scalar_k_args[i] = tsi_alloc(scalarLoopBytesMax);
                if (!scalar_k_args[i]) {
                    fprintf(stderr, "tsi_alloc failed for scalar_k_args[%u]\n", i);
                    abort();
                }
            }


            if (!scalar_grid1_args[i]) {
                scalar_grid1_args[i] = tsi_alloc(scalarLoopBytesMax);
                if (!scalar_grid1_args[i]) {
                    fprintf(stderr, "tsi_alloc failed for scalar_grid1_args[%u]\n", i);
                    abort();
                }
            }
            if (!scalar_grid2_args[i]) {
                scalar_grid2_args[i] = tsi_alloc(scalarLoopBytesMax);
                if (!scalar_grid2_args[i]) {
                    fprintf(stderr, "tsi_alloc failed for scalar_grid2_args[%u]\n", i);
                    abort();
                }
            }
            if (!scalar_grid3_args[i]) {
                scalar_grid3_args[i] = tsi_alloc(scalarLoopBytesMax);
                if (!scalar_grid3_args[i]) {
                    fprintf(stderr, "tsi_alloc failed for scalar_grid3_args[%u]\n", i);
                    abort();
                }
            }
        }
    }

    per_txe_state_initialized.store(true, std::memory_order_release);
}

// Centralized TSI runtime initialization - called once globally
//
static void ensure_tsi_runtime_initialized() {
    if (runtime_initialized) {
        GGML_TSAVORITE_LOG_INFO("\n tsavorite backend already initialized \n");
        return;
    }

    tsi_blob_free_tables();

    std::string mainProfilerName = "OPU ";
    tsirt::utils::TSIProfiler::initialize();

    std::string yaml_path = tsi_resolve_deployment_yaml_path();
    tsi_deploy_cfg_t cfg = tsi_read_deploy_yaml(yaml_path);

    int txe = (cfg.txe_count > 0) ? cfg.txe_count : 1;

    if (txe <= 0) {
        txe = 1;
    }

    if (txe > MAX_TXES_SUPPORTED) {
        fprintf(stderr,
                "ERROR: deployment txe_count=%d exceeds MAX_TXES_SUPPORTED=%d. "
                "Increase MAX_TXES_SUPPORTED or reduce txe_count in %s\n",
                txe,
                MAX_TXES_SUPPORTED,
                yaml_path.c_str());
        fflush(stderr);
        abort();
    }

    num_of_txes = (uint32_t)txe;
    multi_thread_enable = cfg.has_mt ? cfg.mt_enable : false;

#if TRITON_MAT_MUL
    advanced_matmul_shape_offload =
        cfg.has_advanced_matmul_shape_offload ?
        cfg.advanced_matmul_shape_offload :
        false;

    advanced_matmul_broadcast_offload =
        cfg.has_advanced_matmul_broadcast_offload ?
        cfg.advanced_matmul_broadcast_offload :
        false;
    triton_matmul_small_n_transpose_opt =
        cfg.has_triton_matmul_small_n_transpose_opt ?
        cfg.triton_matmul_small_n_transpose_opt :
        false;
#endif

    static TsavoriteDeviceConfig deviceConfig{};
    const size_t requested_user_dram_size =
        tsi_user_dram_size_bytes_from_cfg(cfg);

    TsavoriteDeviceConfig *deviceConfigPtr = NULL;
    if (requested_user_dram_size > 0) {
        deviceConfig.setUserDRAMSize(requested_user_dram_size);
        deviceConfigPtr = &deviceConfig;
    }

    printf("\n TSI deploy yaml=%s txe_count=%u multi_thread_enable=%d",
           yaml_path.c_str(),
           (unsigned)num_of_txes,
           (int)multi_thread_enable);

    if (requested_user_dram_size > 0) {
        printf(" user_dram_size_gb=%d user_dram_size_bytes=%zu",
               cfg.user_dram_size_gb,
               requested_user_dram_size);
    } else {
        printf(" user_dram_size_gb=default");
    }

#if TRITON_MAT_MUL
    printf(" advanced_matmul_shape_offload=%d",
           (int)advanced_matmul_shape_offload);
    printf(" advanced_matmul_broadcast_offload=%d",
           (int)advanced_matmul_broadcast_offload);
    printf(" triton_matmul_small_n_transpose_opt=%d",
           (int)triton_matmul_small_n_transpose_opt);
#endif

    printf("\n");

    tsi_initialize(num_of_txes, deviceConfigPtr);
    tsavorite_install_signal_handlers();

    if (multi_thread_enable) {
        tsi_load_all_blobs();
    } else {
#if NEW_HOST_CODE
        tsi_load_all_blobs();
#endif
    }

    tsi_init_per_txe_state_once();

    if (!device_free) {
        fprintf(stderr, "Failed to allocate device_free\n");
        tsi_unload_all_blobs();
        tsi_finalize();
        abort();
    }

    workers.reserve(num_of_txes);
    runtime_initialized = true;

    GGML_TSAVORITE_LOG_INFO("Profiler and TSI runtime initialized early in registration\n");
}

#ifdef USE_COMMAND_BUFFERS
typedef struct _txe_command_queue_t *txe_command_queue_s;
typedef struct _txe_dispatch_queue_t *txe_dispatch_queue_s;
typedef struct _txe_command_buffer_t *txe_command_buffer_s;
#endif /* USE_COMMAND_BUFFERS */
typedef struct ggml_backend_tsavorite_buffer ggml_backend_tsavorite_buffer_s;

const int Rank = MEM_REF_DESCRIPTOR_RANK;
const int Rank_Triton = MEM_REF_DESCRIPTOR_RANK_TRITON;
MemRefDescriptor<Rank>* glob_buf;

template<int Rank>
// Assumes tsi_alloc is available and returns a pointer to allocated memory
static MemRefDescriptor<Rank>* create_mlir_buf(int K) {
    // TVU load size (e.g., 32 for 1024-bit vector with 32-bit elements)
    const int32_t mem_align = TSI_TVU_MEM_ALIGN;
    // we are supporting only float or F32
    int data_type_len = 4;
    // MemRef Header also added
    int total_bytes = (sizeof(MemRefDescriptor<Rank>) + 4*K);

    // Round up K to the next multiple of tvu_size
    int32_t total_align_bytes = ((total_bytes % mem_align) != 0) ? ((total_bytes / mem_align) + 1) * mem_align : total_bytes;

    // Allocate memory dynamically: space for header + data
    MemRefDescriptor<Rank>* header = (MemRefDescriptor<Rank>*) tsi_alloc(total_align_bytes);

    if (!header) {
        return header;
    }
    // Advance pointer to skip header and get to data
    int32_t* data = (int32_t*)(header + 1);

    for (int32_t i = 0; i < K; ++i) {
        data[i] = 0;
    }
    return header;
}


struct _txe_device_t {
  char name[100];
  uint32_t max_buf_len;
  size_t recommended_max_working_set_size;
  size_t current_allocated_size;
  int reserved;
  struct _stats {
    struct _op_run_count {
      // Each Kernel operation belong to one tensor. Below count will increment for each Node Tensor
      uint64_t total_tensor_count;
      // This counter increment whenever kernel call are  made
      uint64_t num_of_kernel_call;
      // For Any application below field maintain smallest tensor num of elem
      uint64_t min_num_of_elem;
      // For Any application below field maintain largest tensor num of elem
      uint64_t max_num_of_elem;
    } op_run_count[GGML_TSAVORITE_KERNEL_TYPE_COUNT];
  } stats;
};

struct _txe_compute_pipeline_state_t {
  void (*_mlir_fptr_3_input[DATA_TYPE_MAX_INDEX])(void *, void *, void *, void *);
  void (*_mlir_fptr_2_input[DATA_TYPE_MAX_INDEX])(void *, void *, void *);
  void (*_mlir_fptr_1_input[DATA_TYPE_MAX_INDEX])(void *, void *);
  std::string kernel_name;
  int reserved;
};

#ifdef USE_COMMAND_BUFFERS
struct _txe_command_queue_t {
  int reserved;
};

struct _txe_dispatch_queue_t {
  int reserved;
};

struct _txe_command_buffer_t {
  int reserved;
};
#endif /* USE_COMMAND_BUFFERS */

static txe_device_s tsi_system_default_device_create();

// kernels

struct ggml_tsavorite_kernel {
  txe_compute_pipeline_state_s pipeline;
};

struct ggml_backend_tsavorite_context {
#ifdef USE_COMMAND_BUFFERS
  txe_command_queue_s queue;

  txe_dispatch_queue_s d_queue;
#endif /* USE_COMMAND_BUFFERS */

  struct ggml_tsavorite_kernel kernels[GGML_TSAVORITE_KERNEL_TYPE_COUNT];

  // capture state
  bool capture_next_compute;
  bool capture_started;

  // command buffer state
  int n_cb;       // number of extra threads used to submit the command buffers
  int n_nodes_0;  // number of nodes submitted by the main thread
  int n_nodes_1;  // remaining number of nodes submitted by the n_cb threads
  int n_nodes_per_cb;

  struct ggml_cgraph *gf;

  // the callback given to the thread pool
  // void (^encode_async)(size_t ith);

#ifdef USE_COMMAND_BUFFERS
  // n_cb command buffers + 1 used by the main thread
  txe_command_buffer_s command_buffers[GGML_TSAVORITE_MAX_COMMAND_BUFFERS + 1];
#endif /* USE_COMMAND_BUFFERS */

  // abort ggml_tsavorite_graph_compute if callback returns true
  ggml_abort_callback abort_callback;
  void *abort_callback_data;

  // picking CPU compute example
  int n_threads;
  ggml_threadpool_t threadpool;

  uint8_t *work_data;
  size_t work_size;
};

// global
ggml_threadpool_t global_threadpool = NULL;

// initialized in ggml_backend_tsavorite_reg
static struct ggml_backend_reg g_ggml_backend_tsavorite_reg;
static struct ggml_backend_device g_ggml_backend_tsavorite_device;

// information about a tSavorite device
// note: assumes single GPU device - the default one
// Need to Add Support for multiple GPU devices
static struct ggml_backend_tsavorite_device_context {
  txe_device_s device;
  int ref_count;

  char name[128];
} g_ggml_ctx_dev_main = {
    /*.device                  =*/tsi_nil,
    /*.ref_count               =*/0,
    /*.name                    =*/"",
};

// temporarily defined here for compatibility between ggml-backend and the old API

struct ggml_backend_tsavorite_buffer {
  void *data;
  size_t size;
};

struct ggml_backend_tsavorite_buffer_context {
  void *all_data;
  size_t all_size;
  bool owned;

  // multiple buffers are used only to avoid the maximum buffer size limitation when using mmap
  int n_buffers;
  ggml_backend_tsavorite_buffer_s buffers[GGML_TSAVORITE_MAX_BUFFERS];
};

static txe_device_s tsi_system_default_device_create() {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  txe_device_s device = (txe_device_s)malloc(sizeof(struct _txe_device_t));
  device->max_buf_len = TSAVORITE_DEVICE_MAX_BUF_LEN;
  device->recommended_max_working_set_size = TSAVORITE_DEVICE_MAX_BUF_LEN;
  device->current_allocated_size = 0;
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return device;
}

static void tsi_device_free(txe_device_s device) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  free(device);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return;
}

#ifdef USE_COMMAND_BUFFERS
static txe_command_queue_s tsi_command_queue_create() {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  txe_command_queue_s cqueue = (txe_command_queue_s)malloc(sizeof(struct _txe_command_queue_t));
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return cqueue;
}

static txe_dispatch_queue_s tsi_dispatch_queue_create() {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  txe_dispatch_queue_s dqueue = (txe_dispatch_queue_s)malloc(sizeof(struct _txe_dispatch_queue_t));
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return dqueue;
}

static void tsi_command_queue_free(txe_command_queue_s cqueue) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  if (cqueue)
    free(cqueue);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return;
}

static void tsi_dispatch_queue_free(txe_dispatch_queue_s dqueue) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  if (dqueue)
    free(dqueue);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return;
}
#endif /* USE_COMMAND_BUFFERS */

static void tsi_buffer_free(void *data) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  if (data)
    free(data);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return;
}

static bool tsi_log_setup() {
  tsi_op_log_file = fopen("tsi-op.txt", "w+");
  if (tsi_op_log_file == NULL) {
    printf("Error Creating or opening log file\n");
    return false;
  }
  return true;
}


void ggml_tsi_log_tensor_data(tensor_log log_data) {
  if (!log_data.log_file) {
    GGML_TSAVORITE_LOG_ERROR("%s: error: log file Cant be NULL\n", __func__);
    return;
  }

  switch (log_data.data_type) {
  case GGML_TSAVORITE_TENSOR_HEADER:
    fprintf(log_data.log_file, "\n\n");
    fprintf(log_data.log_file, "#############################################################\n");
    fprintf(log_data.log_file,
            "Tensor Number %ld and Type %s \n leaf1  len %d, leaf2 len %d, Node len %d\n",
            log_data.num_of_op, ggml_op_name(log_data.kernel_type), log_data.leaf1_len, log_data.leaf2_len,
            log_data.node_len);
    fprintf(log_data.log_file, "############################################################\n");
    fprintf(log_data.log_file, "\n\n");
    fflush(log_data.log_file);
    return;
  case GGML_TSAVORITE_TENSOR_LEAF1:
    fprintf(log_data.log_file, "\n---------------------------------------------------\n");
    fprintf(log_data.log_file, "leaf1 Detail:\n");
    break;
  case GGML_TSAVORITE_TENSOR_LEAF2:
    fprintf(log_data.log_file, "\n---------------------------------------------------\n");
    fprintf(log_data.log_file, "leaf2 Detail:\n");
    break;
  case GGML_TSAVORITE_TENSOR_NODE:
    fprintf(log_data.log_file, "\n---------------------------------------------------\n");
    fprintf(log_data.log_file, "Node Detail:\n");
    break;
  case GGML_TSAVORITE_TENSOR_END_DATA:
    fprintf(log_data.log_file, "DONE WITH THIS OPERATION %ld\n", log_data.num_of_op);
    fprintf(log_data.log_file, "############################################################\n");
    fprintf(log_data.log_file, "\n\n");
    fflush(log_data.log_file);
    return;
  default:
    GGML_TSAVORITE_LOG_ERROR("%s: error: Invalid Data Type Passed\n", __func__);
    return;
  }
  if (!log_data.tensor) {
    GGML_TSAVORITE_LOG_ERROR("%s: error: tensor pointer is  NULL\n", __func__);
    return;
  }
  float *p;
  int64_t count = (log_data.tensor->ne[0]) * (log_data.tensor->ne[1]) * (log_data.tensor->ne[2]) *
                  (log_data.tensor->ne[3]);
  p = (float *)log_data.tensor->data;
  if ((!p) || (count == 0)) {
    fprintf(log_data.log_file, "\n\n");
    fprintf(log_data.log_file, "Tensor Data is Empty");
    fprintf(log_data.log_file, "\n---------------------------------------------------\n");
    fprintf(log_data.log_file, "\n\n");
    fflush(log_data.log_file);
    return;
  }
  fprintf(tsi_op_log_file, "%.16f ", p[0]);
  for (int64_t ii = 1; ii < count; ++ii) {
    if (!(ii % 4))
      fprintf(log_data.log_file, "\n");
    fprintf(log_data.log_file, "%.16f ", p[ii]);
  }
  fprintf(log_data.log_file, "\n\n");
  fprintf(log_data.log_file, "\n---------------------------------------------------\n");
  fflush(log_data.log_file);
  return;
}

static void ggml_tsavorite_disp_stats(struct ggml_backend_tsavorite_context *ctx,
                                      txe_device_s device) {
  if (!ctx || !device) {
    GGML_TSAVORITE_LOG_ERROR(
        "At %s Either backend context or device or both are NULL, hence cant display Stats",
        __func__);
    return;
  }
  for (int i = 0; i < GGML_TSAVORITE_KERNEL_TYPE_COUNT; ++i) {
    if (!ctx->kernels[i].pipeline)
      continue;
    GGML_TSAVORITE_LOG_CONT(
        "\n %s Operation, total tensor: %lu  Number of Kernel Call: %lu  Number of tensor got "
        "Min Num of Elem %lu Max Num of Elem %lu \n",
        ctx->kernels[i].pipeline->kernel_name.c_str(),
        device->stats.op_run_count[i].total_tensor_count,
        device->stats.op_run_count[i].num_of_kernel_call,
        device->stats.op_run_count[i].min_num_of_elem,
        device->stats.op_run_count[i].max_num_of_elem);
  }
  return;
}

static void _mlir_ciface_txe_add_test (void *src0, void *src1, void *res)
{
    // MemRefDescriptor
    if (!src0 || !src1 || !res)
        return;

    MemRefDescriptor<Rank> *srcP0, *srcP1, *nodeP;
    srcP0 = (MemRefDescriptor<Rank> *)src0;
    srcP1 = (MemRefDescriptor<Rank> *)src1;
    nodeP = (MemRefDescriptor<Rank> *)res;

    // TVU kernels operate using a single dimension for the TVU add operation.
    uint32_t count = srcP0->shape[0];

    float *s0      = (float*)srcP0->data;
    float *s1      = (float*)srcP1->data;
    float *n       = (float*)nodeP->data;

    for(uint32_t i=0; i < count; ++i)
        n[i] = s0[i] + s1[i];
    //printf("\n Calling mlir_add cpu function-5 \n");
    return;
}

static void _mlir_ciface_txe_mult_test (void *src0, void *src1, void *res)
{
    // MemRefDescriptor
    if (!src0 || !src1 || !res)
        return;

    MemRefDescriptor<Rank> *srcP0, *srcP1, *nodeP;
    srcP0 = (MemRefDescriptor<Rank> *)src0;
    srcP1 = (MemRefDescriptor<Rank> *)src1;
    nodeP = (MemRefDescriptor<Rank> *)res;

    // TVU kernels operate using a single dimension for the TVU mul operation.
    uint32_t count = srcP0->shape[0];

    float *s0      = (float*)srcP0->data;
    float *s1      = (float*)srcP1->data;
    float *n       = (float*)nodeP->data;

    for(uint32_t i=0; i < count; ++i)
        n[i] = s0[i]*s1[i];
    return;
}


// Packed args layout for 3x memref<?xf32, strided<[1], offset: ?>, 1>
// Per TXE_PackArgsOp: group per-arg as (handle, offset, sizes, strides),
// and only dynamic metadata is packed. For this type: offset is dynamic, size(0) is dynamic, stride(0)=1 is static.
// So each arg contributes: (handle, offset, size0) => 3 int64s per arg.
// Total = 3 args * 3 int64 = 9 int64 = 72 bytes.)


// ============================================================
// DEVICE ACQUIRE / RELEASE
// ============================================================

static inline int acquire_device_blocking() {
    std::unique_lock<std::mutex> lock(device_mutex);

    device_cv.wait(lock, []() {
        if (!device_free) return false;
        for (uint32_t i = 0; i < num_of_txes; ++i) {
            if (device_free[i]) return true;
        }
        return false;
    });

    for (uint32_t i = 0; i < num_of_txes; ++i) {
        if (device_free[i]) {
            device_free[i] = false;
            return (int)i;
        }
    }

    // Should be unreachable because wait predicate ensures availability
    return -1;
}

static inline void release_device(int deviceId) {
    std::lock_guard<std::mutex> lock(device_mutex);
    device_free[deviceId] = true;
    device_cv.notify_one();
}

// ============================================================
// Final join — invoke at the end of each node’s execution
// within the ggml_tsavorite_graph_compute() subgraph loop.
// ============================================================

static inline void join_all_workers() {
    std::vector<std::thread> local;
    {
        std::lock_guard<std::mutex> lk(workers_mutex);
        if (workers.empty()) return;
        local.swap(workers);   // take ownership, release lock early
    }

    for (auto &t : local) {
        if (t.joinable()) t.join();
    }
}


static int64_t tsi_blob_execution_internal(void *commandList) {
    if (!commandList) {
        return 0;
    }

    tsi_finalize_command_list(commandList);

    const int64_t wait_start_us = tsavorite_now_us();
    tsi_wait(commandList);
    return tsavorite_elapsed_us(wait_start_us);
}


// NOTE:
// Triton ADD kernel supports a host_wrapper-based invocation path today.
// For Multi-TXE support, we plan to bypass the generated host_wrapper and
// directly invoke the runtime shim API with manually packed arguments.
//
// However, the exact argument packing (ABI/layout) used by Triton-generated
// kernels is currently not well understood. Due to this, the direct
// pack-args + runtime shim path is temporarily disabled
//
// This will be revisited once we fully reverse-engineer or document the
// Triton argument packing format through experiments and validation.
//
// Follow-up work tracked here:
// https://tsavoritesi.atlassian.net/browse/FIR-1984
#if TRITON_MULTI_TXE
//lock goes out of scope
// <--- function scope ends here mutex will be released
//tsi_pack_mutex.unlock() is called automatically
//std::lock_guard releases the mutex automatically when it goes out of scope.
static void *_mlir_ciface_txe_add_host_internal(void *a, void *b, void *res, TSI_DeviceIdType deviceId) {
    constexpr int64_t kPackedArgsI64   = 9;
    constexpr int64_t kPackedArgsBytes = kPackedArgsI64 * 8;
    
    // Lock to protect packed_args usage
    std::lock_guard<std::mutex> lock(tsi_pack_mutex);

    void *commandList = tsi_create_command_list(deviceId);

    if ((uint32_t)deviceId >= num_of_txes) {
        fprintf(stderr, "ERROR: deviceId=%d out of range num_of_txes=%u\n", deviceId, num_of_txes);
        tsi_cleanup();
        abort();
    }
    if (packed_args.size() != num_of_txes || !packed_args[deviceId]) {
        fprintf(stderr, "ERROR: packed_args not initialized for deviceId=%d (size=%zu, num_of_txes=%u)\n",
           deviceId, packed_args.size(), num_of_txes);
        tsi_cleanup();
        abort();
    }

    auto *p = static_cast<int64_t *>(packed_args[deviceId]);

    MemRefDescriptor<Rank> *A = (MemRefDescriptor<Rank> *)a;
    MemRefDescriptor<Rank> *B = (MemRefDescriptor<Rank> *)b;
    MemRefDescriptor<Rank> *C = (MemRefDescriptor<Rank> *)res;

    int idx = 0;
    p[idx++] = tsi_shmem_handle_from_ptr(A->data);
    p[idx++] = (int64_t)A->offset;
    p[idx++] = (int64_t)A->shape[0];

    p[idx++] = tsi_shmem_handle_from_ptr(B->data);
    p[idx++] = (int64_t)B->offset;
    p[idx++] = (int64_t)B->shape[0];

    p[idx++] = tsi_shmem_handle_from_ptr(C->data);
    p[idx++] = (int64_t)C->offset;
    p[idx++] = (int64_t)C->shape[0];

    if (idx != kPackedArgsI64) {
        fprintf(stderr, "ERROR: packed-args idx=%d expected=%ld\n", idx, (long)kPackedArgsI64);
        tsi_cleanup();
        abort();
    }

    const int64_t packedHandle = tsi_shmem_handle_from_ptr(packed_args[deviceId]);
    void *blobExecuteCmd = tsi_launch_blob(blobDescriptor_add[0], packedHandle, kPackedArgsBytes);

    if (!blobExecuteCmd) {
        fprintf(stderr, "tsi_launch_blob failed for device %lu and blobDescriptor %s\n",
                                     (unsigned long)deviceId, (char *)blobDescriptor_add[0]);
        tsi_cleanup();
        abort();
    }

    tsi_add_command_to_list(commandList, blobExecuteCmd);

    return commandList;
}

static void _mlir_ciface_txe_add_host_new(void *a, void *b, void *res) {
    tsi_init_per_txe_state_once();

    if (!multi_thread_enable) {
      // Temporarily disabled; will be enabled in the next release to avoid collateral impact
       #if NEW_HOST_CODE
           void *commandList = _mlir_ciface_txe_add_host_internal(a, b, res, 0);
           if (!commandList) {
                fprintf(stderr, "Command List Empt for ADD OPERATION on device 0\n");
                tsi_cleanup();
                abort();
            }
            tsi_blob_execution_internal(commandList);
       #else
              _mlir_ciface_txe_add_host(a, b, res);
       #endif  /* NEW_HOST_CODE */
        return;
    }

    const int deviceId = acquire_device_blocking();
    
    if (deviceId < 0) {
        fprintf(stderr, "Failed to acquire device for ADD\n");
        tsi_cleanup();
        abort();
    }

   // IMPORTANT: pack args NOW while MemRefDescriptor fields are still correct
   void *commandList = _mlir_ciface_txe_add_host_internal(a, b, res, deviceId);
   if (!commandList) {
       fprintf(stderr, "Command List Empt for ADD on device %d\n", deviceId);
       release_device(deviceId);
       tsi_cleanup();
       abort();
    }
    {
       std::lock_guard<std::mutex> lk(workers_mutex);
       workers.emplace_back([=]() {
           tsi_blob_execution_internal(commandList);
           release_device(deviceId);
       });
    }
}
#endif /* TRITON_MULTI_TXE */

static void *_mlir_ciface_txe_mult_host_internal(void *a, void *b, void *res, TSI_DeviceIdType deviceId) {
    constexpr int64_t kPackedArgsI64   = 9;
    constexpr int64_t kPackedArgsBytes = kPackedArgsI64 * 8;

    // Lock to protect packed_args usage
    std::lock_guard<std::mutex> lock(tsi_pack_mutex);

    void *commandList = tsi_create_command_list(deviceId);

    if ((uint32_t)deviceId >= num_of_txes) {
        fprintf(stderr, "ERROR: deviceId=%d out of range num_of_txes=%u\n", deviceId, num_of_txes);
        tsi_cleanup();
        abort();
    }

    if (packed_args.size() != num_of_txes || !packed_args[deviceId]) {
        fprintf(stderr, "ERROR: packed_args not initialized for deviceId=%d (size=%zu, num_of_txes=%u)\n",
           deviceId, packed_args.size(), num_of_txes);
        tsi_cleanup();
        abort();
    }

    auto *p = static_cast<int64_t *>(packed_args[deviceId]);

    MemRefDescriptor<Rank> *A = (MemRefDescriptor<Rank> *)a;
    MemRefDescriptor<Rank> *B = (MemRefDescriptor<Rank> *)b;
    MemRefDescriptor<Rank> *C = (MemRefDescriptor<Rank> *)res;

    int idx = 0;
    p[idx++] = tsi_shmem_handle_from_ptr(A->data);
    p[idx++] = (int64_t)A->offset;
    p[idx++] = (int64_t)A->shape[0];

    p[idx++] = tsi_shmem_handle_from_ptr(B->data);
    p[idx++] = (int64_t)B->offset;
    p[idx++] = (int64_t)B->shape[0];
    
    p[idx++] = tsi_shmem_handle_from_ptr(C->data);
    p[idx++] = (int64_t)C->offset;
    p[idx++] = (int64_t)C->shape[0];

    if (idx != kPackedArgsI64) {
        fprintf(stderr, "ERROR: packed-args idx=%d expected=%ld\n", idx, (long)kPackedArgsI64);
        tsi_cleanup();
        abort();
    }

    const int64_t packedHandle = tsi_shmem_handle_from_ptr(packed_args[deviceId]);
    void *blobExecuteCmd = tsi_launch_blob(blobDescriptor_mult[0], packedHandle, kPackedArgsBytes);
    if (!blobExecuteCmd) {
        fprintf(stderr, "tsi_launch_blob failed for device %lu and blobDescriptor %s\n",
                                     (unsigned long)deviceId, (char *)blobDescriptor_mult[0]);
        tsi_cleanup();
        abort();
    }
    tsi_add_command_to_list(commandList, blobExecuteCmd);

    return commandList;
}

static void _mlir_ciface_txe_mult_host_new(void *a, void *b, void *res) {
    tsi_init_per_txe_state_once();

    if (!multi_thread_enable) {
        // Temporarily disabled; will be enabled in the next release to avoid collateral impact
        #if NEW_HOST_CODE
            void *commandList = _mlir_ciface_txe_mult_host_internal(a, b, res, 1);
            if (!commandList) {
                fprintf(stderr, "Command List Empt for MUL OPERATION on device 0\n");
                tsi_cleanup();
                abort();
            }
            tsi_blob_execution_internal(commandList);
        #else
            _mlir_ciface_txe_mult_host(a, b, res);
        #endif /* NEW_HOST_CODE */
        return;
    }

    int deviceId = acquire_device_blocking();

   // IMPORTANT: pack args NOW while MemRefDescriptor fields are still correct
   void *commandList = _mlir_ciface_txe_mult_host_internal(a, b, res, deviceId);
   if (!commandList) {
        fprintf(stderr, "Command List Empt for MUL OPERATION on device %d\n", deviceId);
        release_device(deviceId);
        tsi_cleanup();
        abort();
   }
   {
       std::lock_guard<std::mutex> lk(workers_mutex);
       workers.emplace_back([=]() {
           tsi_blob_execution_internal(commandList);
           release_device(deviceId);
       });
   }
}

static void *_mlir_ciface_txe_rms_norm_host_internal(void *a, void *b, void *buf, TSI_DeviceIdType deviceId) {
    constexpr int64_t kPackedArgsI64   = 20;
    constexpr int64_t kPackedArgsBytes = kPackedArgsI64 * 8;

    // Lock to protect packed_args usage
    std::lock_guard<std::mutex> lock(tsi_pack_mutex);

    void *commandList = tsi_create_command_list(deviceId);

    if ((uint32_t)deviceId >= num_of_txes) {
        fprintf(stderr, "ERROR: deviceId=%d out of range num_of_txes=%u\n", deviceId, num_of_txes);
        tsi_cleanup();
        abort();
    }

    if (packed_args.size() != num_of_txes || !packed_args[deviceId]) {
        fprintf(stderr, "ERROR: packed_args not initialized for deviceId=%d (size=%zu, num_of_txes=%u)\n",
           deviceId, packed_args.size(), num_of_txes);
        tsi_cleanup();
        abort();
    }

    auto *p = static_cast<int64_t *>(packed_args[deviceId]);

    MemRefDescriptor<Rank> *A = (MemRefDescriptor<Rank> *)a;
    MemRefDescriptor<Rank> *B = (MemRefDescriptor<Rank> *)b;
    MemRefDescriptor<Rank> *C = (MemRefDescriptor<Rank> *)buf;

    int idx = 0;

    p[idx++] = tsi_shmem_handle_from_ptr(A->data);
    p[idx++] = (int64_t)A->offset;
    for (int i = 0; i <= 3; ++i) p[idx++] = (int64_t)A->shape[i];
    for (int i = 0; i <= 2; ++i) p[idx++] = (int64_t)A->strides[i];

    p[idx++] = tsi_shmem_handle_from_ptr(B->data);
    p[idx++] = (int64_t)B->offset;
    for (int i = 0; i <= 3; ++i) p[idx++] = (int64_t)B->shape[i];
    for (int i = 0; i <= 2; ++i) p[idx++] = (int64_t)B->strides[i];

    p[idx++] = tsi_shmem_handle_from_ptr(C->data);
    p[idx++] = (int64_t)C->offset;

    if (idx != kPackedArgsI64) {
        fprintf(stderr, "ERROR: packed-args idx=%d expected=%ld\n", idx, (long)kPackedArgsI64);
        tsi_cleanup();
        abort();
    }

    const int64_t packedHandle = tsi_shmem_handle_from_ptr(packed_args[deviceId]);
    void *blobExecuteCmd = tsi_launch_blob(blobDescriptor_rms_norm[0], packedHandle, kPackedArgsBytes);
    if (!blobExecuteCmd) {
        fprintf(stderr, "tsi_launch_blob failed for device %lu and blobDescriptor %s\n",
                                     (unsigned long)deviceId, (char *)blobDescriptor_rms_norm[0]);
        tsi_cleanup();
        abort();
    }
    tsi_add_command_to_list(commandList, blobExecuteCmd);

    return commandList;
} 

static void _mlir_ciface_txe_rms_norm_host_new(void *a, void *b, void *buf) {
    tsi_init_per_txe_state_once();

    if (!multi_thread_enable) {
      // Temporarily disabled; will be enabled in the next release to avoid collateral impact
        #if NEW_HOST_CODE
            void *commandList = _mlir_ciface_txe_rms_norm_host_internal(a, b, buf, 0);
            if (!commandList) {
                fprintf(stderr, "Command List Empt for RMS OPERATION  on device 0\n");
                tsi_cleanup();
                abort();
            }
            tsi_blob_execution_internal(commandList);
        #else
            _mlir_ciface_txe_rms_norm_host(a, b, buf);
        #endif  /* NEW_HOST_CODE */
        return;
    }

    int deviceId = acquire_device_blocking();

    // IMPORTANT: pack args NOW while MemRefDescriptor fields are still correct
    void *commandList = _mlir_ciface_txe_rms_norm_host_internal(a, b, buf, deviceId);
    if (!commandList) {
        fprintf(stderr, "Command List Empt for RMS OPERATION on device %d\n", deviceId);
        release_device(deviceId);
        tsi_cleanup();
        abort();
    }
    {
       std::lock_guard<std::mutex> lk(workers_mutex);
       workers.emplace_back([=]() {
           tsi_blob_execution_internal(commandList);
           release_device(deviceId);
       });
    }
}


static txe_compute_pipeline_state_s tsi_kernel_setup(enum ggml_tsavorite_kernel_type kernel_type) {
  txe_compute_pipeline_state_s kernel_pipeline =
      (txe_compute_pipeline_state_s)calloc(1, sizeof(struct _txe_compute_pipeline_state_t));
  bool flag = false;
  if (!kernel_pipeline) {
    GGML_TSAVORITE_LOG_ERROR("Calloc failing while setting up kernel");
    return NULL;
  }
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);


  switch (kernel_type) {
      case GGML_TSAVORITE_KERNEL_TYPE_ADD:
          if (ggml_tsavorite_kernel_mode_flag == GGML_TSAVORITE_KERNEL_MODE_CPU)
              kernel_pipeline->_mlir_fptr_2_input[DATA_TYPE_F32_INDEX] = &_mlir_ciface_txe_add_test;
          else {
// TODO(FIR-1984): Will be addressed as part of Triton multi-TXE packed-args support
#if TRITON_MULTI_TXE
              #ifdef GGML_TARGET_POSIX
                  kernel_pipeline->_mlir_fptr_2_input[DATA_TYPE_F32_INDEX] = &_mlir_ciface_txe_add_host;
              #else
                  kernel_pipeline->_mlir_fptr_2_input[DATA_TYPE_F32_INDEX] = &_mlir_ciface_txe_add_host_new;
              #endif /* GGML_TARGET_POSIX */
#endif /* TRITON_MULTI_TXE */
              kernel_pipeline->_mlir_fptr_2_input[DATA_TYPE_F32_INDEX] = &_mlir_ciface_txe_add_host;
              kernel_pipeline->_mlir_fptr_2_input[DATA_TYPE_F16_INDEX] = &_mlir_ciface_txe_add_16_host;
	  }
          kernel_pipeline->kernel_name = "TXE_ADD";
          flag = true;
          break;
      case GGML_TSAVORITE_KERNEL_TYPE_SUB:
          kernel_pipeline->_mlir_fptr_2_input[DATA_TYPE_F32_INDEX] = &_mlir_ciface_txe_sub_host;
          kernel_pipeline->_mlir_fptr_2_input[DATA_TYPE_F16_INDEX] = &_mlir_ciface_txe_sub_16_host;
          kernel_pipeline->kernel_name = "TXE_SUB";
          flag = true;
          break;
      case GGML_TSAVORITE_KERNEL_TYPE_MULT:
          if (ggml_tsavorite_kernel_mode_flag == GGML_TSAVORITE_KERNEL_MODE_CPU)
              kernel_pipeline->_mlir_fptr_2_input[DATA_TYPE_F32_INDEX] = &_mlir_ciface_txe_mult_test;
          else {
              #ifdef GGML_TARGET_POSIX
                  kernel_pipeline->_mlir_fptr_2_input[DATA_TYPE_F32_INDEX] = &_mlir_ciface_txe_mult_host;
              #else
                  kernel_pipeline->_mlir_fptr_2_input[DATA_TYPE_F32_INDEX] = &_mlir_ciface_txe_mult_host_new;
              #endif /* GGML_TARGET_POSIX */
              kernel_pipeline->_mlir_fptr_2_input[DATA_TYPE_F16_INDEX] = &_mlir_ciface_txe_mult_16_host;
	  }
          kernel_pipeline->kernel_name = "TXE_MULT";
          flag = true;
          break;
      case GGML_TSAVORITE_KERNEL_TYPE_DIV:
          kernel_pipeline->_mlir_fptr_2_input[DATA_TYPE_F32_INDEX] = &_mlir_ciface_txe_div_host;
          kernel_pipeline->_mlir_fptr_2_input[DATA_TYPE_F16_INDEX] = &_mlir_ciface_txe_div_16_host;
          kernel_pipeline->kernel_name = "TXE_DIV";
          flag = true;
          break;
      case GGML_TSAVORITE_KERNEL_TYPE_SQRT:
          kernel_pipeline->_mlir_fptr_1_input[DATA_TYPE_F32_INDEX] = &_mlir_ciface_txe_sqrt_host;
          kernel_pipeline->_mlir_fptr_1_input[DATA_TYPE_F16_INDEX] = &_mlir_ciface_txe_sqrt_16_host;
          kernel_pipeline->kernel_name = "TXE_SQRT";
          flag = true;
          break;
      case GGML_TSAVORITE_KERNEL_TYPE_SQR:
          kernel_pipeline->_mlir_fptr_1_input[DATA_TYPE_F32_INDEX] = &_mlir_ciface_txe_sqr_host;
          kernel_pipeline->_mlir_fptr_1_input[DATA_TYPE_F16_INDEX] = &_mlir_ciface_txe_sqr_16_host;
          kernel_pipeline->kernel_name = "TXE_SQR";
          flag = true;
          break;
      case GGML_TSAVORITE_KERNEL_TYPE_NEG:
          kernel_pipeline->_mlir_fptr_1_input[DATA_TYPE_F32_INDEX] = &_mlir_ciface_txe_neg_host;
          kernel_pipeline->_mlir_fptr_1_input[DATA_TYPE_F16_INDEX] = &_mlir_ciface_txe_neg_16_host;
          kernel_pipeline->kernel_name = "TXE_NEG";
          flag = true;
          break;
      case GGML_TSAVORITE_KERNEL_TYPE_ABS:
          kernel_pipeline->_mlir_fptr_1_input[DATA_TYPE_F32_INDEX] = &_mlir_ciface_txe_abs_host;
          kernel_pipeline->_mlir_fptr_1_input[DATA_TYPE_F16_INDEX] = &_mlir_ciface_txe_abs_16_host;
          kernel_pipeline->kernel_name = "TXE_ABS";
          flag = true;
          break;
      case GGML_TSAVORITE_KERNEL_TYPE_SIN:
          kernel_pipeline->_mlir_fptr_1_input[DATA_TYPE_F32_INDEX] = &_mlir_ciface_txe_sin_host;
          kernel_pipeline->_mlir_fptr_1_input[DATA_TYPE_F16_INDEX] = &_mlir_ciface_txe_sin_16_host;
          kernel_pipeline->kernel_name = "TXE_SIN";
          flag = true;
          break;
      case GGML_TSAVORITE_KERNEL_TYPE_SIGMOID:
          kernel_pipeline->_mlir_fptr_1_input[DATA_TYPE_F32_INDEX] = &_mlir_ciface_txe_sigmoid_host;
          kernel_pipeline->_mlir_fptr_1_input[DATA_TYPE_F16_INDEX] = &_mlir_ciface_txe_sigmoid_16_host;
          kernel_pipeline->kernel_name = "TXE_SIGMOID";
          flag = true;
          break;
      case GGML_TSAVORITE_KERNEL_TYPE_SILU:
          kernel_pipeline->_mlir_fptr_1_input[DATA_TYPE_F32_INDEX] = &_mlir_ciface_txe_silu_host;
          kernel_pipeline->_mlir_fptr_1_input[DATA_TYPE_F16_INDEX] = &_mlir_ciface_txe_silu_16_host;
          kernel_pipeline->kernel_name = "TXE_SILU";
          flag = true;
          break;
      case GGML_TSAVORITE_KERNEL_TYPE_RMS_NORM:
          #ifdef GGML_TARGET_POSIX
              kernel_pipeline->_mlir_fptr_2_input[DATA_TYPE_F32_INDEX] = &_mlir_ciface_txe_rms_norm_host;
          #else
              kernel_pipeline->_mlir_fptr_2_input[DATA_TYPE_F32_INDEX] = &_mlir_ciface_txe_rms_norm_host_new;
          #endif /* GGML_TARGET_POSIX */
          kernel_pipeline->_mlir_fptr_2_input[DATA_TYPE_F16_INDEX] = &_mlir_ciface_txe_rms_norm_16_host;
          kernel_pipeline->kernel_name = "TXE_RMS_NORM";
          flag = true;
          break;
      case GGML_TSAVORITE_KERNEL_TYPE_SWIGLU:
	  {
          kernel_pipeline->_mlir_fptr_2_input[DATA_TYPE_F32_INDEX] = &_mlir_ciface_txe_swiglu_host;
          kernel_pipeline->_mlir_fptr_2_input[DATA_TYPE_F16_INDEX] = &_mlir_ciface_txe_swiglu_16_host;
          kernel_pipeline->kernel_name = "TXE_SWI_GLU";
          flag = true;
          break;
	  }
      case GGML_TSAVORITE_KERNEL_TYPE_SOFT_MAX:
	  {
          kernel_pipeline->_mlir_fptr_3_input[DATA_TYPE_F32_INDEX] = &_mlir_ciface_txe_soft_max_host;
          //kernel_pipeline->_mlir_fptr_2_input[DATA_TYPE_F16_INDEX] = &_mlir_ciface_txe_soft_max_16_host;
          kernel_pipeline->kernel_name = "TXE_SOFTMAX";
          flag = true;
          break;
	  }
      case GGML_TSAVORITE_KERNEL_TYPE_MUL_MAT:
          {
              // IMPORTANT:
              // Real TMU blob entrypoint is selected at sub graph compute time in
              // ggml_tsavorite_run_tmu_mul_mat() based on K.
              // This pointer must be non-null only to pass support checks.
              kernel_pipeline->_mlir_fptr_2_input[DATA_TYPE_F32_INDEX] =
                  (void (*)(void*,void*,void*))1; // dummy non-null
              kernel_pipeline->kernel_name = "TXE_MUL_MAT";
              flag = true;
              break;
          }
      default:
          break;
  }
  if (!flag) {
    GGML_TSAVORITE_LOG_INFO("Kernel %d not supported \n", kernel_type);
    if (kernel_pipeline) {
      free(kernel_pipeline);
      kernel_pipeline = NULL;
    }
  }
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return kernel_pipeline;
}

static void tsi_kernel_release(txe_compute_pipeline_state_s kernel_pipeline) {
  // clear kernel_pipeline
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  if (kernel_pipeline) {
    free(kernel_pipeline);
  }
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return;
}

// acquire
static txe_device_s
ggml_backend_tsavorite_device_acq(struct ggml_backend_tsavorite_device_context *ctx) {
  assert(ctx != NULL);
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  if (ctx->device == tsi_nil) {
    ctx->device = tsi_system_default_device_create();
    snprintf(ctx->name, sizeof("txe"), "txe");
  }

  ctx->ref_count++;
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);

  return ctx->device;
}

// release
static void ggml_backend_tsavorite_device_rel(struct ggml_backend_tsavorite_device_context *ctx) {
  assert(ctx != NULL);
  assert(ctx->ref_count > 0);
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  ctx->ref_count--;

  // Need to define function txe_device_free
  if (ctx->ref_count == 0) {
    tsi_device_free(ctx->device);
    ctx->device = tsi_nil;
  }
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
}

// We will use Unified Memory this memory is used for buffer
static void *ggml_tsavorite_host_malloc(size_t n) {
  void *data = NULL;
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  GGML_TSAVORITE_LOG_INFO("\n Allocating memory from tsi_alloc with size  %ld \n", n);

  const int32_t mem_align = TSI_TVU_MEM_ALIGN;
  int total_align_bytes = (n/mem_align +1)*mem_align;
  data = tsi_alloc(total_align_bytes);

  GGML_TSAVORITE_LOG_CONT("\n Allocating memory from tsi_alloc with size  %ld starting memory %p\n",
                          n, data);

  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);

  return data;
}

#ifdef GGML_MUL_MAT_CPU_OPS
void ggml_backend_tsavorite_set_threadpool(ggml_backend_t backend_tsavorite, ggml_threadpool_t threadpool) {

    struct ggml_backend_tsavorite_context * ctx = (struct ggml_backend_tsavorite_context *)backend_tsavorite->context;

    if (ctx->threadpool && ctx->threadpool != threadpool) {
        // already had a different threadpool, pause/suspend it before switching
        ggml_threadpool_pause(ctx->threadpool);
    }
    ctx->threadpool = threadpool;
    global_threadpool = threadpool;
}
#endif

static struct ggml_backend_tsavorite_context *ggml_tsavorite_init(ggml_backend_dev_t dev) {
  GGML_TSAVORITE_LOG_INFO("%s: Start\n", __func__);
  // Open a file named "tsi-op.txt" in the current directory for writing
  num_of_op = 0;

  if (tsi_log_setup() == false)
    return NULL;

  
  std::string mainProfilerName = "OPU ";
  tsirt::utils::TSIScopedProfiler mainProfiler(mainProfilerName);

  // init context
  struct ggml_backend_tsavorite_context *ctx = (struct ggml_backend_tsavorite_context *)calloc(
      1, sizeof(struct ggml_backend_tsavorite_context));
  struct ggml_backend_tsavorite_device_context *ctx_dev =
      (struct ggml_backend_tsavorite_device_context *)dev->context;

  // setup the devie context
  txe_device_s device = ggml_backend_tsavorite_device_acq(ctx_dev);
  GGML_TSAVORITE_LOG_INFO("%s: picking default device: %s\n", __func__, device->name);
  for (uint32_t op = GGML_TSAVORITE_KERNEL_TYPE_ADD; op < GGML_TSAVORITE_KERNEL_TYPE_COUNT; ++op) {
    device->stats.op_run_count[op].total_tensor_count = 0;
    device->stats.op_run_count[op].num_of_kernel_call = 0;
    device->stats.op_run_count[op].min_num_of_elem = 0;
    device->stats.op_run_count[op].max_num_of_elem = 0;
  }
  ctx->n_threads = GGML_DEFAULT_N_THREADS;
  ctx->threadpool = NULL;
  ctx->work_data = NULL;
  ctx->work_size = 0;
  ctx->abort_callback = NULL;
  ctx->abort_callback_data = NULL;

  // We dont need it for now, we will revisit
#ifdef USE_COMMAND_BUFFERS
  // setting up backend context
  ctx->queue = tsi_command_queue_create();
  ctx->d_queue = tsi_dispatch_queue_create();
#endif /* USE_COMMAND_BUFFERS */

  ctx->capture_next_compute = false;
  ctx->capture_started = false;

  ctx->gf = tsi_nil;
  // ctx->encode_async = tsi_nil;

#ifdef USE_COMMAND_BUFFERS
  for (int i = 0; i < GGML_TSAVORITE_MAX_COMMAND_BUFFERS; ++i) {
    ctx->command_buffers[i] = tsi_nil;
  }
#endif /* USE_COMMAND_BUFFERS */

  // load TSavorite kernels
  {
    for (int i = 0; i < GGML_TSAVORITE_KERNEL_TYPE_COUNT; ++i) {
      ctx->kernels[i].pipeline = tsi_nil;
    }

#define GGML_TSAVORITE_KERNEL(e, supported)                                                        \
  if (supported) {                                                                                 \
    ctx->kernels[e].pipeline = tsi_kernel_setup(e);                                                \
    GGML_TSAVORITE_LOG_INFO(" TSAVORITE SUPPORTED KERNEL ");                                       \
  } else {                                                                                         \
    GGML_TSAVORITE_LOG_WARN("%s: skipping %-40s (not supported)\n", __func__, "kernel_" #e);       \
  }

    GGML_TSAVORITE_KERNEL(GGML_TSAVORITE_KERNEL_TYPE_ADD,                true);
    GGML_TSAVORITE_KERNEL(GGML_TSAVORITE_KERNEL_TYPE_SUB,                true);
    GGML_TSAVORITE_KERNEL(GGML_TSAVORITE_KERNEL_TYPE_MULT,               true);
    GGML_TSAVORITE_KERNEL(GGML_TSAVORITE_KERNEL_TYPE_DIV,                true);
    GGML_TSAVORITE_KERNEL(GGML_TSAVORITE_KERNEL_TYPE_SQRT,               true);
    GGML_TSAVORITE_KERNEL(GGML_TSAVORITE_KERNEL_TYPE_SQR,                true);
    GGML_TSAVORITE_KERNEL(GGML_TSAVORITE_KERNEL_TYPE_NEG,                true);
    GGML_TSAVORITE_KERNEL(GGML_TSAVORITE_KERNEL_TYPE_ABS,                true);
    GGML_TSAVORITE_KERNEL(GGML_TSAVORITE_KERNEL_TYPE_SIN,                true);
    GGML_TSAVORITE_KERNEL(GGML_TSAVORITE_KERNEL_TYPE_SIGMOID,            true);
    GGML_TSAVORITE_KERNEL(GGML_TSAVORITE_KERNEL_TYPE_SILU,               true);
    GGML_TSAVORITE_KERNEL(GGML_TSAVORITE_KERNEL_TYPE_RMS_NORM,           true);
    GGML_TSAVORITE_KERNEL(GGML_TSAVORITE_KERNEL_TYPE_SWIGLU,             true);
    GGML_TSAVORITE_KERNEL(GGML_TSAVORITE_KERNEL_TYPE_SOFT_MAX,           true);
    GGML_TSAVORITE_KERNEL(GGML_TSAVORITE_KERNEL_TYPE_MUL_MAT,            true);
  }
  glob_buf = create_mlir_buf<Rank>(96);
  if (!glob_buf) {
      GGML_TSAVORITE_LOG_ERROR("tsi_alloc failied for creating memory for buf \n");
      free(ctx);
      return NULL;
  }

  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return ctx;
}

static void ggml_tsavorite_free(struct ggml_backend_tsavorite_context *ctx) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);

  if (!ctx)
      return;

  for (int i = 0; i < GGML_TSAVORITE_KERNEL_TYPE_COUNT; ++i) {
    if (ctx->kernels[i].pipeline) {
      tsi_kernel_release(ctx->kernels[i].pipeline);
      ctx->kernels[i].pipeline = tsi_nil;
    }
  }

  // Block_release(ctx->encode_async);
  //
#ifdef USE_COMMAND_BUFFERS
  tsi_command_queue_free(ctx->queue);

  tsi_dispatch_queue_free(ctx->d_queue);
#endif /* USE_COMMAND_BUFFERS */

  free(ctx);

  // TSI run time free
  GGML_TSAVORITE_LOG_INFO("\n Calling tsi_finalize \n");
  // delay to allow any file operations to complete for runtime

  GGML_TSAVORITE_LOG_INFO("Delaying tsi_finalize for 2 sec");
  if (runtime_initialized == true) {
      runtime_initialized = false;
      tsi_unload_all_blobs();

      if(device_free) {
          free(device_free);
         device_free = NULL;
      }
      tsi_reset_per_txe_state_after_teardown();
      sleep(2);
      tsi_finalize();
      tsirt::utils::TSIProfiler::finalize();
      sleep(2);
  }
  tsavorite_matmul_profile_dump();
  tsavorite_op_shape_dtype_catalog_dump();

  std::cout << "\nOPU Profiling Results:" << std::endl;
  std::cout << tsirt::utils::TSIProfiler::getFormattedResults(
                 /*truncateFuncNames*/ true)
          << std::endl;
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return;
}

void
tsi_cleanup() {
    fflush(stderr);
    fflush(stdout);
    if (runtime_initialized != true)
        return;
    runtime_initialized = false;
    tsi_unload_all_blobs();
    if(device_free) {
        free(device_free);
        device_free = NULL;
    }
    tsi_reset_per_txe_state_after_teardown();
    sleep(2);
    tsi_finalize();
    GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
    tsirt::utils::TSIProfiler::finalize();
    // Profiling results already printed during first cleanup
    // std::cout << "\nOPU Profiling Results:" << std::endl;
    // std::cout << tsirt::utils::TSIProfiler::getFormattedResults(
    //              /*truncateFuncNames*/ true)
    //           << std::endl;
    sleep(2);
    GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
    return;
}

#if 0
// finds the tSavorite buffer that contains the tensor data on the TXE device unified memory
// the assumption is that there is 1-to-1 mapping between the host and device memory buffers, so we can find the
// tSavorite buffer based on the host memory pointer
//
static ggml_backend_tsavorite_buffer_s ggml_tsavorite_get_buffer(struct ggml_tensor * t, size_t * offs) {
    // GGML_TSAVORITE_LOG_INFO("%s: data tensor '%16s', offs_data = %8ld, offs_eval = %8ld, offs_cach = %8ld\n", __func__, t->name, offs_data, offs_eval, offs_cach);
    GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);


    const int64_t tsize = ggml_nbytes(t);

    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    struct ggml_backend_tsavorite_buffer_context * buf_ctx = (struct ggml_backend_tsavorite_buffer_context *) buffer->context;

    // find the view that contains the tensor fully
    for (int i = 0; i < buf_ctx->n_buffers; ++i) {
        const int64_t ioffs = (int64_t) t->data - (int64_t) buf_ctx->buffers[i].data;

        // GGML_TSAVORITE_LOG_INFO("ioffs = %10ld, tsize = %10ld, sum = %10ld, buf_ctx->buffers[%d].size = %10ld\n", ioffs, tsize, ioffs + tsize, i, buf_ctx->buffers[i].size);
        if (ioffs >= 0 && ioffs + tsize <= (int64_t) buf_ctx->buffers[i].size) {
            *offs = (size_t) ioffs;

            // GGML_TSAVORITE_LOG_INFO("%s: tensor '%16s', offs = %8ld\n", __func__, t->name, *offs);
            GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);

            return buf_ctx->buffers[i];
        }
    }

    GGML_TSAVORITE_LOG_ERROR("%s: error: tensor '%s' buffer is tsi_nil\n", __func__, t->name);
    GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);

    return tsi_nil;
}
#endif
static bool is_op_dtype_consistent_with_src(const struct ggml_tensor *op) {
  uint32_t tensor_data_type = op->type;
  for (size_t i = 0; i < GGML_MAX_DIMS; ++i) {
    if (op->src[i] != NULL) {
        if(tensor_data_type != op->src[i]->type)
          return false;
    }
  }
  return true;
}

/*
 * Return true if a tensor type can be converted to F32 during MAT_MUL packing.
 *
 * F32/F16/BF16 have explicit fast paths. Quantized / packed GGML types
 * such as Q8_0, Q5_0, Q4_0, etc. are supported only when GGML provides
 * a type-traits to_float() converter for that type. The Triton MAT_MUL
 * kernel still consumes packed F32 buffers.
 * GGML maintains a type-traits table for each ggml_type. For quantized types
 * such as Q8_0, Q5_0, Q4_0, and others, that table may provide a to_float() function.
 * This function knows how to dequantize the compressed block format into standard F32 values.
 * As a result, our code does not need separate conversion logic for every quantized format.
 * Instead, it queries GGML for the appropriate converter and uses it when available.
 */
static inline bool tsavorite_tensor_type_can_pack_to_f32(enum ggml_type type) {
    if (type == GGML_TYPE_F32 ||
        type == GGML_TYPE_F16 ||
        type == GGML_TYPE_BF16) {
        return true;
    }

    const struct ggml_type_traits *traits = ggml_get_type_traits(type);
    return traits && traits->to_float;
}

static inline int64_t tsavorite_tensor_nb0_or_type_size(const struct ggml_tensor *t) {
    if (!t) {
        return 0;
    }

    if (t->nb[0] != 0) {
        return (int64_t)t->nb[0];
    }

    return (int64_t)ggml_type_size(t->type);
}

static inline bool tsavorite_tensor_type_can_pack_to_f32_k(
    enum ggml_type type,
    int64_t K,
    int64_t nb0) {
    if (K <= 0 || nb0 <= 0) {
        return false;
    }

    if (type == GGML_TYPE_F32 ||
        type == GGML_TYPE_F16 ||
        type == GGML_TYPE_BF16) {
        return true;
    }

    const struct ggml_type_traits *traits = ggml_get_type_traits(type);
    if (!traits || !traits->to_float) {
        return false;
    }

    const int64_t bs = (int64_t)ggml_blck_size(type);
    if (bs <= 0 || (K % bs) != 0) {
        return false;
    }

    const int64_t expected_nb0 = (int64_t)ggml_type_size(type);
    return nb0 == expected_nb0;
}

static inline void tsavorite_tensor_copy_k_to_f32(
    const struct ggml_tensor *t,
    const char *base,
    float *dst,
    int64_t K,
    int64_t nb0) {
    if (!t || !base || !dst || K <= 0) {
        fprintf(stderr, "ERROR: invalid args in tsavorite_tensor_copy_k_to_f32\n");
        fflush(stderr);
        tsi_cleanup();
        abort();
    }

    if (t->type == GGML_TYPE_F32) {
        if (nb0 == (int64_t)sizeof(float)) {
            memcpy(dst, base, (size_t)K * sizeof(float));
        } else {
            for (int64_t k = 0; k < K; ++k) {
                dst[k] = *(const float *)(base + k * nb0);
            }
        }
        return;
    }

    if (t->type == GGML_TYPE_F16) {
        for (int64_t k = 0; k < K; ++k) {
            dst[k] = GGML_FP16_TO_FP32(*(const ggml_fp16_t *)(base + k * nb0));
        }
        return;
    }

    if (t->type == GGML_TYPE_BF16) {
        for (int64_t k = 0; k < K; ++k) {
            dst[k] = GGML_BF16_TO_FP32(*(const ggml_bf16_t *)(base + k * nb0));
        }
        return;
    }

    const struct ggml_type_traits *traits = ggml_get_type_traits(t->type);
    if (traits && traits->to_float) {
        const int64_t bs = (int64_t)ggml_blck_size(t->type);
        if (bs <= 0 || (K % bs) != 0) {
            fprintf(stderr,
                    "ERROR: unsupported quantized K block alignment type=%d K=%ld block_size=%ld\n",
                    (int)t->type,
                    (long)K,
                    (long)bs);
            fflush(stderr);
            tsi_cleanup();
            abort();
        }

        const int64_t expected_nb0 = (int64_t)ggml_type_size(t->type);
        if (nb0 != expected_nb0) {
            fprintf(stderr,
                    "ERROR: unsupported non-contiguous quantized K stride type=%d nb0=%ld expected=%ld\n",
                    (int)t->type,
                    (long)nb0,
                    (long)expected_nb0);
            fflush(stderr);
            tsi_cleanup();
            abort();
        }

        traits->to_float(base, dst, K);
        return;
    }

    fprintf(stderr,
            "ERROR: ggml type %d cannot be converted to F32 for Triton MAT_MUL packing\n",
            (int)t->type);
    fflush(stderr);
    tsi_cleanup();
    abort();
}

/*
 * Convert one logical K-row from the source tensor into F32 and scatter it
 * into a strided destination layout.
 *
 * F32, F16, and BF16 are converted directly. Quantized or packed GGML types
 * are first converted to a temporary contiguous F32 row through
 * tsavorite_tensor_copy_k_to_f32(), then scattered into Triton's physical
 * B layout [K x N_pad].
 */
static inline void tsavorite_tensor_scatter_k_to_f32_strided(
    const struct ggml_tensor *t,
    const char *base,
    float *dst,
    int64_t dst_stride,
    int64_t K,
    int64_t nb0,
    std::vector<float> &scratch) {
    if (!t || !base || !dst || dst_stride <= 0 || K <= 0) {
        fprintf(stderr, "ERROR: invalid args in tsavorite_tensor_scatter_k_to_f32_strided\n");
        fflush(stderr);
        tsi_cleanup();
        abort();
    }

    switch (t->type) {
    case GGML_TYPE_F32:
        for (int64_t k = 0; k < K; ++k) {
            dst[k * dst_stride] = *(const float *)(base + k * nb0);
        }
        return;

    case GGML_TYPE_F16:
        for (int64_t k = 0; k < K; ++k) {
            dst[k * dst_stride] = GGML_FP16_TO_FP32(*(const ggml_fp16_t *)(base + k * nb0));
        }
        return;

    case GGML_TYPE_BF16:
        for (int64_t k = 0; k < K; ++k) {
            dst[k * dst_stride] = GGML_BF16_TO_FP32(*(const ggml_bf16_t *)(base + k * nb0));
        }
        return;

    default:
        break;
    }

    if (scratch.size() < (size_t)K) {
        scratch.resize((size_t)K);
    }

    /*
     * Quantized / packed GGML types are handled through the generic
     * traits->to_float() path in tsavorite_tensor_copy_k_to_f32().
     * That conversion produces a contiguous temporary F32 K-row in scratch.
     * The loop below scatters that F32 row into the Triton B layout:
     * physical B is [K x N_pad], so each K element is written with dst_stride.
     */
    tsavorite_tensor_copy_k_to_f32(t, base, scratch.data(), K, nb0);
    for (int64_t k = 0; k < K; ++k) {
        dst[k * dst_stride] = scratch[(size_t)k];
    }
}

#if TRITON_MAT_MUL
static inline bool tsavorite_triton_matmul_dims_within_caps(
    int64_t K,
    int64_t M,
    int64_t N);

static inline bool tsavorite_mul_mat_advanced_shape_ok(
    const struct ggml_tensor * op);

static bool mul_mat_supported_size(const struct ggml_tensor *op) {
    if (!op) return false;

    if (advanced_matmul_shape_offload && tsavorite_mul_mat_advanced_shape_ok(op)) {
#if TRITON_DEBUG
        const struct ggml_tensor * a_dbg = op->src[0];
        const struct ggml_tensor * b_dbg = op->src[1];
        fprintf(stderr,
                "MUL_MAT_ADVANCED_SHAPE_ENABLE: a_type=%d b_type=%d op_type=%d "
                "a=[%ld,%ld,%ld,%ld] b=[%ld,%ld,%ld,%ld] op=[%ld,%ld,%ld,%ld]\n",
                (int)a_dbg->type, (int)b_dbg->type, (int)op->type,
                (long)a_dbg->ne[0], (long)a_dbg->ne[1], (long)a_dbg->ne[2], (long)a_dbg->ne[3],
                (long)b_dbg->ne[0], (long)b_dbg->ne[1], (long)b_dbg->ne[2], (long)b_dbg->ne[3],
                (long)op->ne[0], (long)op->ne[1], (long)op->ne[2], (long)op->ne[3]);
#endif
        return true;
    }


    const struct ggml_tensor *a = op->src[0];
    const struct ggml_tensor *b = op->src[1];

    if (!a || !b) return false;

    /*
     * Triton MAT_MUL kernel consumes packed F32 buffers.
     * Allow GGML source types that can be converted to F32 during host packing.
     * Quantized types must have block-aligned K and contiguous K stride.
     * Result remains F32 for this first mixed-precision path.
     */
    const int64_t K_dtype = a->ne[0];
    const int64_t a_nb0_dtype = tsavorite_tensor_nb0_or_type_size(a);
    const int64_t b_nb0_dtype = tsavorite_tensor_nb0_or_type_size(b);

    const bool a_dtype_ok = tsavorite_tensor_type_can_pack_to_f32_k(a->type, K_dtype, a_nb0_dtype);
    const bool b_dtype_ok = tsavorite_tensor_type_can_pack_to_f32_k(b->type, K_dtype, b_nb0_dtype);
    const bool op_dtype_ok = (op->type == GGML_TYPE_F32);

    if (!a_dtype_ok || !b_dtype_ok || !op_dtype_ok) {
#if TRITON_DEBUG
        fprintf(stderr,
                "MUL_MAT_REJECT_DTYPE: a_type=%d b_type=%d op_type=%d "
                "a=[%ld,%ld,%ld,%ld] b=[%ld,%ld,%ld,%ld] op=[%ld,%ld,%ld,%ld]\n",
                (int)a->type, (int)b->type, (int)op->type,
                (long)a->ne[0],  (long)a->ne[1],  (long)a->ne[2],  (long)a->ne[3],
                (long)b->ne[0],  (long)b->ne[1],  (long)b->ne[2],  (long)b->ne[3],
                (long)op->ne[0], (long)op->ne[1], (long)op->ne[2], (long)op->ne[3]);
#endif
        return false;
    }

    /* mul_mat_dtype_supported_generic_inputs_f32_result */

    /*
     * GGML MUL_MAT layout:
     *   a/src0 : [K, M, A2, A3]
     *   b/src1 : [K, N, B2, B3]
     *   op/dst : [M, N, D2, D3]
     */
    const int64_t K = a->ne[0];
    const int64_t M = a->ne[1];
    const int64_t N = b->ne[1];

    if (!tsavorite_triton_matmul_dims_within_caps(K, M, N)) {
        return false;
    }

    /*
     * Basic shape consistency.
     */
    if (b->ne[0] != K) {
        return false;
    }

    if (op->ne[0] != M) {
        return false;
    }

    if (op->ne[1] != N) {
        return false;
    }

    /*
     * Triton F32 MAT_MUL K alignment requirement.
     */
    if ((K % 32) != 0) {
#if TRITON_DEBUG
        fprintf(stderr,
                "MUL_MAT_REJECT_K_ALIGN: K=%ld "
                "a=[%ld,%ld,%ld,%ld] b=[%ld,%ld,%ld,%ld] op=[%ld,%ld,%ld,%ld]\n",
                (long)K,
                (long)a->ne[0],  (long)a->ne[1],  (long)a->ne[2],  (long)a->ne[3],
                (long)b->ne[0],  (long)b->ne[1],  (long)b->ne[2],  (long)b->ne[3],
                (long)op->ne[0], (long)op->ne[1], (long)op->ne[2], (long)op->ne[3]);
#endif
        return false;
    }

    /*
     * Validate 3D/4D broadcast semantics.
     */
    const int64_t A2 = a->ne[2] > 0 ? a->ne[2] : 1;
    const int64_t A3 = a->ne[3] > 0 ? a->ne[3] : 1;
    const int64_t B2 = b->ne[2] > 0 ? b->ne[2] : 1;
    const int64_t B3 = b->ne[3] > 0 ? b->ne[3] : 1;

    const int64_t D2 = (A2 > B2) ? A2 : B2;
    const int64_t D3 = (A3 > B3) ? A3 : B3;

    if (op->ne[2] != D2) {
        return false;
    }

    if (op->ne[3] != D3) {
        return false;
    }

    if (!(A2 == 1 || A2 == D2)) {
        return false;
    }

    if (!(B2 == 1 || B2 == D2)) {
        return false;
    }

    if (!(A3 == 1 || A3 == D3)) {
        return false;
    }

    if (!(B3 == 1 || B3 == D3)) {
        return false;
    }

    const bool is_baseline_2d =
        A2 == 1 && A3 == 1 &&
        B2 == 1 && B3 == 1 &&
        D2 == 1 && D3 == 1;

    /*
     * advanced_matmul_shape_offload == false:
     *   Preserve original July-13 behavior:
     *     - strict logical 2D only
     *     - reject N == 1
     *
     * advanced_matmul_shape_offload == true:
     *   Allow advanced 3D/4D/broadcast shapes.
     *   Do NOT reject N == 1 here, because many advanced shapes are N==1.
     */
    if (!advanced_matmul_shape_offload) {
        if (!is_baseline_2d) {
#if TRITON_DEBUG
            fprintf(stderr,
                    "MUL_MAT_REJECT_ADVANCED_FLAG_OFF: "
                    "K=%ld M=%ld N=%ld D2=%ld D3=%ld "
                    "a=[%ld,%ld,%ld,%ld] b=[%ld,%ld,%ld,%ld] op=[%ld,%ld,%ld,%ld]\n",
                    (long)K, (long)M, (long)N, (long)D2, (long)D3,
                    (long)a->ne[0],  (long)a->ne[1],  (long)a->ne[2],  (long)a->ne[3],
                    (long)b->ne[0],  (long)b->ne[1],  (long)b->ne[2],  (long)b->ne[3],
                    (long)op->ne[0], (long)op->ne[1], (long)op->ne[2], (long)op->ne[3]);
#endif
            return false;
        }

        if (N == 1) {
#if TRITON_DEBUG
            fprintf(stderr,
                    "MUL_MAT_REJECT_N_EQ_1_BASELINE: K=%ld M=%ld N=%ld "
                    "a=[%ld,%ld,%ld,%ld] b=[%ld,%ld,%ld,%ld] op=[%ld,%ld,%ld,%ld]\n",
                    (long)K, (long)M, (long)N,
                    (long)a->ne[0],  (long)a->ne[1],  (long)a->ne[2],  (long)a->ne[3],
                    (long)b->ne[0],  (long)b->ne[1],  (long)b->ne[2],  (long)b->ne[3],
                    (long)op->ne[0], (long)op->ne[1], (long)op->ne[2], (long)op->ne[3]);
#endif
            return false;
        }

        return true;
    }

    /*
     * Current 1x8 Triton shape padding.
     */
    const int64_t M_pad = ((M + 7)  / 8)  * 8;

    const int64_t elems_A = M_pad * K;
    const int64_t N_pad = ((N + 63) / 64) * 64;
    const int64_t elems_B = K * N_pad;
    const int64_t elems_C = M_pad * N_pad;

    if (elems_A <= 0 || elems_B <= 0 || elems_C <= 0) {
        return false;
    }

    const int64_t total_bytes =
        (elems_A + elems_B + elems_C) * (int64_t)sizeof(float);

#if TRITON_DEBUG
    fprintf(stderr,
            "MUL_MAT_TRITON_ENABLE: K=%ld M=%ld N=%ld D2=%ld D3=%ld "
            "baseline_2d=%d advanced_flag=%d "
            "M_pad=%ld N_pad=%ld total_bytes=%ld "
            "a=[%ld,%ld,%ld,%ld] b=[%ld,%ld,%ld,%ld] op=[%ld,%ld,%ld,%ld]\n",
            (long)K, (long)M, (long)N,
            (long)D2, (long)D3,
            (int)is_baseline_2d,
            (int)advanced_matmul_shape_offload,
            (long)M_pad,
            (long)N_pad,
            (long)total_bytes,
            (long)a->ne[0],  (long)a->ne[1],  (long)a->ne[2],  (long)a->ne[3],
            (long)b->ne[0],  (long)b->ne[1],  (long)b->ne[2],  (long)b->ne[3],
            (long)op->ne[0], (long)op->ne[1], (long)op->ne[2], (long)op->ne[3]);
#endif

    return true;
}

#else

static bool mul_mat_supported_size(const struct ggml_tensor *op) {
    const struct ggml_tensor *a = op->src[0];
    const struct ggml_tensor *b = op->src[1];


    if (!a || !b) return false;

    // GGML MUL_MAT:
    //   K = a->ne[0]
    //   out = [N = op->ne[0], M = op->ne[1]]
    const int64_t K = a->ne[0];
    const int64_t N = op->ne[0];
    const int64_t M = op->ne[1];

    if (K <= 0 || N <= 0 || M <= 0) return false;

    if ((K % TMU_K_MULTIPLE) != 0) return false;

    // avoid GEMV (you can remove this if you later implement GEMV path)
    if (M == 1) return false;

    // reduction dims must match
    if (b->ne[0] != K) return false;

    if (K == 64 || K == 576 || K == 1536 || K == 256) {
	    return true;
    }
    // Disable MAT_MUL offloading to Tsavorite for the Tiny‑Llama‑v0.3‑FP32‑1.1B model
    return false;

    // (optional but usually correct for ggml mul_mat wiring)
    // If this blocks valid cases in your build, comment it out.
    if (a->ne[1] != N) return false;

    // Disable MAT_MUL offloading to Tsavorite for the Tiny‑Llama‑v0.3‑FP32‑1.1B model
    return false;

    // -------------------------------------------------------------------------
    // Tiny-Llama-v0.3-FP32-1.1B-F32.gguf shapes (from your static-shape list)
    // Most frequent inference shapes are M=7 with K=2048 and N in {256,2048,5632}
    // -------------------------------------------------------------------------

    // Common token-batch matmuls (M=7)
    if (M == 7) {
        // src0: (2048,  256)  src1: (2048, 7)  result: ( 256, 7)
        if (K == 2048 && N == 256)  return true;

        // src0: (2048, 2048)  src1: (2048, 7)  result: (2048, 7)
        if (K == 2048 && N == 2048) return true;

        // src0: (2048, 5632)  src1: (2048, 7)  result: (5632, 7)
        if (K == 2048 && N == 5632) return true;

        // src0: (5632, 2048)  src1: (5632, 7)  result: (2048, 7)
        // Depending on how ggml wires A/B, this may appear as K=5632, N=2048, M=7.
        if (K == 5632 && N == 2048) return true;
    }

    // Packed multi-dim cases you listed:
    // Count=22: src0 (256,64,4,1) x src1 (256,7,32,1) => result (64,7,32,1)
    // Count=22: src0 (64,256,4,1) x src1 (64,7,32,1) => result (256,7,32,1)
    //
    // Note: In ggml, op->ne[2], op->ne[3] carry the higher dims.
    // Only enable these if your TMU path truly supports them.
    if (M == 7 && op->ne[2] == 32 && op->ne[3] == 1) {
        // result (64, 7, 32, 1)  with K=256 (from src1 ne[0]=256)
        if (N == 64  && K == 256) return true;

        // result (256, 7, 32, 1) with K=64
        if (N == 256 && K == 64)  return true;
    }

    // Default: not supported (prevents CMA from trying to hold most of the model)
    return false;
}
#endif /* TRITON_MAT_MUL */

static bool ggml_tsavorite_internal_supports_op(const struct ggml_tensor *op) {

  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);

  if (op->op == GGML_OP_NONE && tsavorite_tensor_type_can_pack_to_f32(op->type)) {
    tsavorite_op_shape_dtype_catalog_record(
        op,
        "SUPPORTED",
        "none_tensor_dtype_supported");
    return true;
  }

  if (op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_F16) {
    tsavorite_op_shape_dtype_catalog_record(
        op,
        "REJECTED",
        "result_dtype_not_f32_or_f16");
    return false;
  }

  switch (op->op) {
      case GGML_OP_SET_ROWS:
          tsavorite_op_shape_dtype_catalog_record(op, "SUPPORTED", "special_set_rows");
          return true;
    case GGML_OP_GET_ROWS:
          tsavorite_op_shape_dtype_catalog_record(op, "SUPPORTED", "special_get_rows");
          return true;
    case GGML_OP_FLASH_ATTN_EXT:
          tsavorite_op_shape_dtype_catalog_record(op, "REJECTED", "flash_attn_ext_not_supported");
	  return false;
    case GGML_OP_SOFT_MAX:
          tsavorite_op_shape_dtype_catalog_record(op, "SUPPORTED", "special_soft_max");
          return true;
    case GGML_OP_GET_ROWS_BACK:
          tsavorite_op_shape_dtype_catalog_record(op, "SUPPORTED", "special_get_rows_back");
          return true;
    case GGML_OP_ROPE:
          tsavorite_op_shape_dtype_catalog_record(op, "SUPPORTED", "special_rope");
          return true;
    case GGML_OP_ROPE_BACK:
          tsavorite_op_shape_dtype_catalog_record(op, "SUPPORTED", "special_rope_back");
          return true;
    case GGML_OP_RESHAPE:
          tsavorite_op_shape_dtype_catalog_record(op, "SUPPORTED", "special_reshape");
          return true;
    case GGML_OP_VIEW:
          tsavorite_op_shape_dtype_catalog_record(op, "SUPPORTED", "special_view");
          return true;
    case GGML_OP_TRANSPOSE:
          tsavorite_op_shape_dtype_catalog_record(op, "SUPPORTED", "special_transpose");
          return true;
    case GGML_OP_CPY:
          tsavorite_op_shape_dtype_catalog_record(op, "SUPPORTED", "special_cpy");
          return true;
    case GGML_OP_SET:
          tsavorite_op_shape_dtype_catalog_record(op, "SUPPORTED", "special_set");
          return true;
    case GGML_OP_CONT:
          tsavorite_op_shape_dtype_catalog_record(op, "SUPPORTED", "special_cont");
          return true;
    default:
	  break;
  }

#ifdef TMU_SUPPORTED
  if (op->op == GGML_OP_MUL_MAT) {
    if (!mul_mat_supported_size(op)) {
      tsavorite_op_shape_dtype_catalog_record(
          op,
          "REJECTED",
          "mul_mat_shape_or_dtype_not_supported");
      return false;
    }

    tsavorite_op_shape_dtype_catalog_record(
        op,
        "SUPPORTED",
        "mul_mat_shape_dtype_supported");
    return true;
  }
#endif

  if (!is_op_dtype_consistent_with_src(op)) {
    tsavorite_op_shape_dtype_catalog_record(
        op,
        "REJECTED",
        "mixed_dtype_rejected_by_dtype_consistency");
    return false;
  }

  switch (op->op) {
  case GGML_OP_NONE:
	  break;
#ifdef TVU_SUPPORTED
  case GGML_OP_ADD:
  case GGML_OP_SUB:
  case GGML_OP_MUL:
  case GGML_OP_DIV:
  case GGML_OP_SQRT:
  case GGML_OP_SQR:
  case GGML_OP_SIN:
  case GGML_OP_RESHAPE:
  case GGML_OP_VIEW:
  case GGML_OP_PERMUTE:
  case GGML_OP_TRANSPOSE:
  case GGML_OP_RMS_NORM:
#ifdef GGML_TARGET_POSIX_DEBUG
  case GGML_OP_SOFT_MAX:
#endif /* GGML_TARGET_POSIX_DEBUG */
    break;

  case GGML_OP_GLU:
    {
        const ggml_glu_op op_ext = ggml_get_glu_op(op);
        if (op_ext != GGML_GLU_OP_SWIGLU) {
            tsavorite_op_shape_dtype_catalog_record(
                op,
                "REJECTED",
                "glu_subtype_not_supported");
            return false;
        }
        break;
    }
  case GGML_OP_UNARY:
    switch (ggml_get_unary_op(op)) {
    case GGML_UNARY_OP_NEG:
    case GGML_UNARY_OP_ABS:
    case GGML_UNARY_OP_SIGMOID:
    case GGML_UNARY_OP_SILU:
      break;
    default:
      tsavorite_op_shape_dtype_catalog_record(
          op,
          "REJECTED",
          "unary_subtype_not_supported");
      return false;
    }
    break;
#endif /* TVU_SUPPORTED */
  default:
    tsavorite_op_shape_dtype_catalog_record(
        op,
        "REJECTED",
        "op_not_supported");
    return false;
  }
  tsavorite_op_shape_dtype_catalog_record(
      op,
      "SUPPORTED",
      "op_supported");

  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return true;
}

static bool ggml_tsavorite_supports_op(const struct ggml_backend_tsavorite_device_context *ctx_dev,
                                       const struct ggml_tensor *op) {
  bool return_value = false;
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);

  if (!ctx_dev)
    return return_value;

  return_value = ggml_tsavorite_internal_supports_op(op);

  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return return_value;
}

/*
static void ggml_tsavorite_encode_node(
                        ggml_backend_t   backend,
                                   int   idx,
          tsi_command_encoder   encoder) {
}
*/

static void ggml_tsavorite_decompose_unary_kernel_sin(uint32_t num_elem, ggml_tensor *src) {
  float *p = (float *)(src->data);
  for (uint32_t i = 0; i < num_elem; ++i) {
    *p = (*p) / (2 * M_PI);
    ++p;
  }
  return;
}

static void ggml_tsavorite_decompose_unary_kernel(uint32_t num_elem, ggml_tensor *src,
                                                  ggml_tensor *node) {
  switch (node->op) {
  case GGML_OP_SIN:
    ggml_tsavorite_decompose_unary_kernel_sin(num_elem, src);
    break;
  default:
    break;
  }
  return;
}


static enum ggml_tsavorite_kernel_type tsi_glu_kernel_type(struct ggml_tensor *node) {
    const ggml_glu_op op = ggml_get_glu_op(node);
    enum ggml_tsavorite_kernel_type kernel_type;

    switch (op) {
        case GGML_GLU_OP_REGLU:
		kernel_type = GGML_TSAVORITE_KERNEL_TYPE_REGLU;
            break;
        case GGML_GLU_OP_GEGLU:
		kernel_type = GGML_TSAVORITE_KERNEL_TYPE_GEGLU;
            break;
        case GGML_GLU_OP_SWIGLU:
		kernel_type = GGML_TSAVORITE_KERNEL_TYPE_SWIGLU;
            break;
        case GGML_GLU_OP_SWIGLU_OAI:
		kernel_type = GGML_TSAVORITE_KERNEL_TYPE_SWIGLU_OAI;
            break;
        case GGML_GLU_OP_GEGLU_ERF:
		kernel_type = GGML_TSAVORITE_KERNEL_TYPE_GEGLU_ERF;
            break;
        case GGML_GLU_OP_GEGLU_QUICK:
		kernel_type = GGML_TSAVORITE_KERNEL_TYPE_GEGLU_QUICK;
            break;
        default:
		kernel_type = GGML_TSAVORITE_KERNEL_TYPE_COUNT;
	    break;
    }
    return kernel_type;
}


// TMU CODE
// ============================================================================
// SINGLE-PHASE TMU K-TILING (ONE BLOB PER K) + ABI-SAFE WRAPPERS
// -----------------------------------------------------------------------------
// Blob semantics assumed:
//   - Each call reloads PP from C_tile scratchpad
//   - Runs: PP = A*B + PP across K in this blob
//   - Last internal op materializes C = A*B + PP
//   - Stores back to SAME C_tile (scratchpad updated)
//
// IMPORTANT: The MLIR-exported symbols (_mlir_ciface_*_host) expect MemRefDescriptor
// pointers, NOT raw float pointers. We therefore provide C ABI wrappers
//   tmu_mul_mat_k{K}(A_raw, B_raw, C_raw)
// that build MemRefDescriptor<4> and call the MLIR entrypoint safely.
//
// App responsibilities:
//   - pack A/B per K_chunk (contiguous stride=K_chunk)
//   - zero-pad A rows beyond m_tile
//   - zero-pad B cols beyond n_valid
//   - memset(C_tile,0) once per output tile
//   - call buckets in K decomposition order
//   - copy valid region from C_tile back to ggml output
// ============================================================================

#include <mutex>            // for std::once_flag / std::call_once

static inline int64_t nb_or_default(const struct ggml_tensor *t, int i) {
    if (t->nb[i] != 0) return t->nb[i];
    if (i == 0) return (int64_t) ggml_type_size(t->type);
    return nb_or_default(t, i - 1) * t->ne[i - 1];
}

static inline int64_t map_repeat_i64(int64_t out_idx, int64_t in_dim) {
    if (in_dim <= 1) return 0;
    int64_t r = out_idx % in_dim;
    return (r < 0) ? (r + in_dim) : r;
}

/*
 * Map an output D2/D3 batch index to the corresponding input batch index for
 * grouped broadcast MAT_MUL layouts.
 *
 * This is intentionally not generic GGML_OP_REPEAT modulo mapping. For grouped
 * head layouts such as A2=4, D2=32, output heads 0..7 map to input head 0,
 * 8..15 map to input head 1, etc. This matches the grouped-head MAT_MUL
 * packing requirement used by the Triton offload path.
 */
static inline int64_t map_repeat_dim_i64(int64_t out_idx, int64_t out_dim, int64_t in_dim) {
    if (in_dim <= 1) return 0;
    if (out_dim <= 0) return 0;
    if (in_dim == out_dim) return out_idx;

    if ((out_dim % in_dim) == 0) {
        const int64_t group = out_dim / in_dim;
        int64_t r = out_idx / group;
        if (r < 0) r = 0;
        if (r >= in_dim) r = in_dim - 1;
        return r;
    }

    return map_repeat_i64(out_idx, in_dim);
}


// ============================================================================
// ABI WRAPPERS: raw pointers -> MemRefDescriptor<4> -> call MLIR ciface
// This DEFINES the symbols your .h/.cpp want: tmu_mul_mat_k{K}.
// ============================================================================

static inline void init_memref_4d(MemRefDescriptor<4> &m,
                                 void *ptr,
                                 int64_t d0, int64_t d1, int64_t d2, int64_t d3) {
    m.base   = ptr;
    m.data   = ptr;
    m.offset = 0;

    m.shape[0] = d0;
    m.shape[1] = d1;
    m.shape[2] = d2;
    m.shape[3] = d3;

    // Strides in ELEMENTS (standard MLIR memref convention)
    m.strides[3] = 1;
    m.strides[2] = d3;
    m.strides[1] = d2 * d3;
    m.strides[0] = d1 * d2 * d3;
}


template<int K>
static inline void call_tmu_blob(
    const void *A_tile,
    const void *B_tile,
    void *C_tile,
    void (*fn)(void*, void*, void*)
) {
    // Allocate ABI descriptors in tsi_alloc (device-visible) ONCE.
    static MemRefDescriptor<4> *A = nullptr;
    static MemRefDescriptor<4> *B = nullptr;
    static MemRefDescriptor<4> *C = nullptr;
    static bool inited = false;

    if (!inited) {
        A = (MemRefDescriptor<4> *)tsi_alloc(sizeof(MemRefDescriptor<4>));
        B = (MemRefDescriptor<4> *)tsi_alloc(sizeof(MemRefDescriptor<4>));
        C = (MemRefDescriptor<4> *)tsi_alloc(sizeof(MemRefDescriptor<4>));
        TSAVORITE_GGML_ASSERT(A && B && C);
        inited = true;
    }

    // Fill descriptors (strides in ELEMENTS)
    init_memref_4d(*A, (void*)A_tile, 1, 1, TMU_M_TILE_MAX, K);
    init_memref_4d(*B, (void*)B_tile, 1, 1, TMU_N_BLOCK,   K);
    init_memref_4d(*C, (void*)C_tile, 1, 1, TMU_M_TILE_MAX, TMU_N_BLOCK);

    fn((void*)A, (void*)B, (void*)C);
}

// TMU blob host functions are temporarily stubbed out.
// The corresponding blobs exceed the 64 KB size limit, which is not supported today.
// As a result, these larger blobs are not generated by the compiler AOT infrastructure.
// Generating the host functions alone can take more than 20 minutes.
// Until large-blob support is available, dummy implementations are provided.
void _mlir_ciface_txe_mul_mat_tile_f32_k256_host  (void *A_tile, void *B_tile, void *C_tile) {
	return;
}
void _mlir_ciface_txe_mul_mat_tile_f32_k512_host  (void *A_tile, void *B_tile, void *C_tile) {
	return;
}
void _mlir_ciface_txe_mul_mat_tile_f32_k1024_host  (void *A_tile, void *B_tile, void *C_tile) {
	return;
}
void _mlir_ciface_txe_mul_mat_tile_f32_k2048_host  (void *A_tile, void *B_tile, void *C_tile) {
	return;
}

// -----------------------------------------------------------------------------
// Triton MAT_MUL implementation for dynamic matrix shapes.
//
// Current implementation targets the Triton 1x8 TXE shape. Supported matrix
// shapes are handled by padding M and N to the Triton kernel alignment
// requirements, then packing the full matrices into Triton-compatible buffers.
//
// K is dynamic and is passed at runtime. The only current K constraint is that
// it must satisfy the Triton MAT_MUL alignment requirement.
//
// Packed buffers:
//   A full : [M_pad x K]
//   B full : [K x N_pad]
//   C full : [M_pad x N_pad]
//
// Scalars:
//   M_pad, N_pad, K, grid1, grid2, grid3 are passed through packed_args.
//
// Note:
//   This implementation currently uses the Triton 1x8 TXE shape. Future TXE
//   configurations such as 8x1, 4x2, and 2x4 can be added through
//   shape-specific alignment, packing, and dispatch logic.
// -----------------------------------------------------------------------------


#if TRITON_MAT_MUL

// TXE shape descriptor for Triton MAT_MUL.
//
// Current PR scope:
//   - Uses only the 1x8 TXE shape.
//   - 1x8 requires M to be aligned to 8 rows and N to be aligned to 64 columns.
//   - When triton_matmul_small_n_transpose_opt is enabled, we may compute the
//     mathematically equivalent transposed problem for M >> N:
//       original: C[M,N] = A[K,M] x B[K,N]
//       swapped : T[N,M] = B[K,N] x A[K,M], then copy back C[m,n] = T[n,m]
//
// Future extension:
//   - Add another descriptor such as TRITON_MATMUL_SHAPE_2X4 when the 2x4 blob
//     is introduced.
//   - Reuse the same padding-cost heuristic and transpose decision logic by
//     passing that shape descriptor instead of hard-coding dimensions.
//   - The packing/copyback logic can stay common as long as the selected blob
//     uses the same flattened A[M,K], B[K,N], C[M,N] contract.
#define TRITON_MATMUL_1X8_M_DIM      8
#define TRITON_MATMUL_1X8_N_DIM     64
#define TRITON_MATMUL_2X4_M_DIM     16
#define TRITON_MATMUL_2X4_N_DIM     32

enum triton_matmul_shape_kind_t {
    TRITON_MATMUL_KIND_1X8 = 0,
    TRITON_MATMUL_KIND_2X4 = 1,
};

struct triton_matmul_txe_shape_t {
    int64_t m_dim;
    int64_t n_dim;
    triton_matmul_shape_kind_t kind;
    const char *name;
};

static constexpr triton_matmul_txe_shape_t TRITON_MATMUL_SHAPE_1X8 = {
    TRITON_MATMUL_1X8_M_DIM,
    TRITON_MATMUL_1X8_N_DIM,
    TRITON_MATMUL_KIND_1X8,
    "1x8",
};

static constexpr triton_matmul_txe_shape_t TRITON_MATMUL_SHAPE_2X4 = {
    TRITON_MATMUL_2X4_M_DIM,
    TRITON_MATMUL_2X4_N_DIM,
    TRITON_MATMUL_KIND_2X4,
    "2x4",
};

static inline int64_t triton_matmul_padding_cost(
    const triton_matmul_txe_shape_t &shape,
    int64_t M,
    int64_t N) {
    if (M <= 0 || N <= 0 || shape.m_dim <= 0 || shape.n_dim <= 0) {
        return INT64_MAX;
    }
    const int64_t M_pad = ((M + shape.m_dim - 1) / shape.m_dim) * shape.m_dim;
    const int64_t N_pad = ((N + shape.n_dim - 1) / shape.n_dim) * shape.n_dim;
    return M_pad * N_pad;
}

static inline const triton_matmul_txe_shape_t &triton_matmul_select_shape(
    int64_t M,
    int64_t N) {
    if (!advanced_matmul_shape_offload) {
        return TRITON_MATMUL_SHAPE_1X8;
    }

    const int64_t cost_1x8 = triton_matmul_padding_cost(TRITON_MATMUL_SHAPE_1X8, M, N);
    const int64_t cost_2x4 = triton_matmul_padding_cost(TRITON_MATMUL_SHAPE_2X4, M, N);
    return (cost_2x4 < cost_1x8) ? TRITON_MATMUL_SHAPE_2X4 : TRITON_MATMUL_SHAPE_1X8;
}


// Data type specific
#define TRITON_MATMUL_F32_K_DIM      32
#define TSAV_TRITON_MATMUL_MAX_K 12288
#define TSAV_TRITON_MATMUL_MAX_M 32768
#define TSAV_TRITON_MATMUL_MAX_N 4096

static inline bool tsavorite_triton_matmul_dims_within_caps(
    int64_t K,
    int64_t M,
    int64_t N) {
    return K > 0 && M > 0 && N > 0 &&
           K <= TSAV_TRITON_MATMUL_MAX_K &&
           M <= TSAV_TRITON_MATMUL_MAX_M &&
           N <= TSAV_TRITON_MATMUL_MAX_N;
}

#define TRITON_MATMUL_ALIGNMENT_BYTES      128
#define TRITON_MATMUL_ALIGNMENT_MASK      (TRITON_MATMUL_ALIGNMENT_BYTES - 1)

static inline bool tsavorite_mul_mat_advanced_shape_ok(const struct ggml_tensor * op) {
    if (!op || !op->src[0] || !op->src[1]) {
        return false;
    }

    const struct ggml_tensor * a = op->src[0];
    const struct ggml_tensor * b = op->src[1];

    if (op->type != GGML_TYPE_F32) {
        return false;
    }

    const int64_t K  = a->ne[0];
    const int64_t M  = a->ne[1];
    const int64_t N  = b->ne[1];

    if (!tsavorite_triton_matmul_dims_within_caps(K, M, N)) {
        return false;
    }

    const int64_t a_nb0 = tsavorite_tensor_nb0_or_type_size(a);
    const int64_t b_nb0 = tsavorite_tensor_nb0_or_type_size(b);

    if (!tsavorite_tensor_type_can_pack_to_f32_k(a->type, K, a_nb0) ||
        !tsavorite_tensor_type_can_pack_to_f32_k(b->type, K, b_nb0)) {
        return false;
    }
    const int64_t A2 = a->ne[2];
    const int64_t A3 = a->ne[3];
    const int64_t B2 = b->ne[2];
    const int64_t B3 = b->ne[3];
    const int64_t D2 = op->ne[2];
    const int64_t D3 = op->ne[3];

    /*
     * Only require the broadcast flag for actual broadcast layouts where an
     * input batch dimension differs from the output batch dimension. Batched
     * non-broadcast layouts such as A2=B2=D2=2 should remain controlled by
     * advanced_matmul_shape_offload and must not require the broadcast flag.
     */
    const bool has_broadcast_dims =
        (A2 != D2 || B2 != D2 || A3 != D3 || B3 != D3);

    if (has_broadcast_dims && !advanced_matmul_broadcast_offload) {
        return false;
    }

    if (b->ne[0] != K || op->ne[0] != M || op->ne[1] != N) {
        return false;
    }

    if ((K % TRITON_MATMUL_F32_K_DIM) != 0) {
        return false;
    }

    if (A2 <= 0 || A3 <= 0 || B2 <= 0 || B3 <= 0 || D2 <= 0 || D3 <= 0) {
        return false;
    }

    if (D2 != ((A2 > B2) ? A2 : B2) || D3 != ((A3 > B3) ? A3 : B3)) {
        return false;
    }

    /*
     * Allow GGML repeat/broadcast layouts where an input batch dimension divides
     * the output batch dimension. Runtime packing uses map_repeat_dim_i64(), so
     * shapes such as A2=4, B2=32, D2=32 are valid.
     */
    if ((D2 % A2) != 0 || (D2 % B2) != 0) {
        return false;
    }

    if ((D3 % A3) != 0 || (D3 % B3) != 0) {
        return false;
    }

    const triton_matmul_txe_shape_t & shape = triton_matmul_select_shape(M, N);
    const int64_t M_pad = ((M + shape.m_dim - 1) / shape.m_dim) * shape.m_dim;
    const int64_t N_pad = ((N + shape.n_dim - 1) / shape.n_dim) * shape.n_dim;
    const int64_t total_bytes = (M_pad * K + K * N_pad + M_pad * N_pad) * (int64_t)sizeof(float);

    if (M_pad <= 0 || N_pad <= 0 || total_bytes <= 0) {
        return false;
    }

    return true;
}


static int32_t g_triton_cur_M_tile = TMU_M_TILE_MAX;
static int32_t g_triton_cur_N_tile = TMU_N_BLOCK;


extern "C" void _mlir_ciface_add_kernel_device_wrapper(
    void *A,
    void *B,
    void *C,
    void *n_elements_scalar,
    void *grid_x_scalar,
    void *grid_y_scalar,
    void *grid_z_scalar,
    void *max_txes_scalar);

extern "C" void _mlir_ciface_matmul_kernel_1x8_device_wrapper(
    void *A,
    void *B,
    void *C,
    void *M_scalar,
    void *N_scalar,
    void *K_scalar,
    void *grid_x_scalar,
    void *grid_y_scalar,
    void *grid_z_scalar,
    void *max_txes_scalar);

extern "C" void _mlir_ciface_matmul_kernel_2x4_device_wrapper(
    void *A,
    void *B,
    void *C,
    void *M_scalar,
    void *N_scalar,
    void *K_scalar,
    void *grid_x_scalar,
    void *grid_y_scalar,
    void *grid_z_scalar,
    void *max_txes_scalar);

// -----------------------------------------------------------------------------
// Triton MAT_MUL manual memory wrapper
//
// TSISIM blob path:
//   fpga-kernel/build-fpga/txe_triton_mat_mul_1x8/blobs/txe_blob_0.blob
//
// tsi_load_blob() expects prefix WITHOUT ".blob":
//   .../txe_triton_mat_mul_1x8/blobs/txe_blob_0
//
// Blob packed-args ABI:
//   7 args:
//     A, B, C, M, N, K, program_id
//
// Each arg is packed as 16 bytes:
//   p[idx++] = tsi_shmem_handle_from_ptr(desc->data);
//   p[idx++] = desc->shape[0];
//
// Total:
//   7 args * 2 int64 = 14 int64 = 112 bytes
//
// Note:
//   grid1/grid2/grid3/max_txes are wrapper-level arguments for the generated
//   device wrapper path. They are not packed into the direct blob path.
//   The direct blob path appends only program_id after the user kernel args.
// -----------------------------------------------------------------------------



// Triton ADD direct blob pack helper.
// Direct blob ABI packs only kernel args plus program_id.
// Generated-wrapper grid args are not packed here.
static inline void tsi_pack_triton_add_arg(
    int64_t *p,
    int &idx,
    MemRefDescriptor<Rank_Triton> *d,
    const char *name) {
    if (!p || !d || !d->data) {
        fprintf(stderr,
                "ERROR: Triton ADD arg %s NULL p/desc/data p=%p desc=%p data=%p\n",
                name,
                (void *)p,
                (void *)d,
                d ? d->data : nullptr);
        fflush(stderr);
        tsi_cleanup();
        abort();
    }

    p[idx++] = tsi_shmem_handle_from_ptr(d->data);
    p[idx++] = (int64_t)d->shape[0];

#if TRITON_DEBUG
    fprintf(stderr,
            "TRITON_ADD_PACK_ARG: %s data=%p handle=%ld shape0=%ld offset=%ld stride0=%ld\n",
            name,
            d->data,
            (long)p[idx - 2],
            (long)d->shape[0],
            (long)d->offset,
            (long)d->strides[0]);
#endif
}

static inline void tsi_pack_triton_matmul_arg(
    int64_t *p,
    int &idx,
    MemRefDescriptor<Rank_Triton> *d,
    const char *name) {
    if (!d || !d->data) {
        fprintf(stderr,
                "ERROR: Triton MAT_MUL arg %s NULL desc/data desc=%p data=%p\n",
                name, (const void *)d, d ? d->data : nullptr);
        tsi_cleanup();
        abort();
    }

    p[idx++] = tsi_shmem_handle_from_ptr(d->data);
    p[idx++] = (int64_t)d->shape[0];

#if TRITON_DEBUG
    fprintf(stderr,
            "TRITON_MATMUL_PACK_ARG: %s data=%p handle=%ld shape0=%ld offset=%ld stride0=%ld\n",
            name,
            d->data,
            (long)p[idx - 2],
            (long)d->shape[0],
            (long)d->offset,
            (long)d->strides[0]);
#endif
}

// Triton MAT_MUL wrapper (F32-only)
//
// Inputs:
//   A_desc_v     : MemRefDescriptor<F32> for matrix A
//   B_desc_v     : MemRefDescriptor<F32> for matrix B
//   C_desc_v     : MemRefDescriptor<F32> for output matrix C
//   M_desc_v     : MemRefDescriptor<F32> scalar M
//   N_desc_v     : MemRefDescriptor<F32> scalar N
//   K_desc_v     : MemRefDescriptor<F32> scalar K
//   program_id_desc_v : MemRefDescriptor<F32> dim-encoded program_id scratch
//
// Note:
//   Current implementation supports F32 only. BF16/F16 and mixed-precision
//   are not supported in ggml-tsavorite.cpp.


// Triton ADD manual direct-blob path.
//
// Host-wrapper path uses:
//   A, B, C, n_elements, grid_x, grid_y, grid_z, max_txes
//
// Direct blob path packs only:
//   A, B, C, n_elements, program_id
//
// Each packed arg is:
//   handle, shape0
//
// Total:
//   5 args * 2 int64 = 10 int64 = 80 bytes
static void *_mlir_ciface_add_kernel_device_wrapper_triton_manual_internal(
    void *A_desc_v,
    void *B_desc_v,
    void *C_desc_v,
    void *n_elements_desc_v,
    void *program_id_desc_v,
    int32_t program_id,
    TSI_DeviceIdType deviceId) {

    constexpr int64_t kPackedArgsI64   = 10;
    constexpr int64_t kPackedArgsBytes = kPackedArgsI64 * (int64_t)sizeof(int64_t);

    std::lock_guard<std::mutex> lock(tsi_pack_mutex);

    if ((uint32_t)deviceId >= num_of_txes) {
        fprintf(stderr,
                "ERROR: Triton ADD deviceId=%d out of range num_of_txes=%u\n",
                (int)deviceId,
                (unsigned)num_of_txes);
        fflush(stderr);
        tsi_cleanup();
        abort();
    }

    if (packed_args.size() != num_of_txes || !packed_args[deviceId]) {
        fprintf(stderr,
                "ERROR: Triton ADD packed_args not initialized deviceId=%d size=%zu num_of_txes=%u\n",
                (int)deviceId,
                packed_args.size(),
                (unsigned)num_of_txes);
        fflush(stderr);
        tsi_cleanup();
        abort();
    }

    auto *A_desc = (MemRefDescriptor<Rank_Triton> *)A_desc_v;
    auto *B_desc = (MemRefDescriptor<Rank_Triton> *)B_desc_v;
    auto *C_desc = (MemRefDescriptor<Rank_Triton> *)C_desc_v;
    auto *n_elements_desc = (MemRefDescriptor<Rank_Triton> *)n_elements_desc_v;
    auto *program_id_desc = (MemRefDescriptor<Rank_Triton> *)program_id_desc_v;

    if (!program_id_desc || !program_id_desc->data) {
        fprintf(stderr, "ERROR: Triton ADD program_id descriptor/data is NULL\n");
        fflush(stderr);
        tsi_cleanup();
        abort();
    }

    int64_t *p = static_cast<int64_t *>(packed_args[deviceId]);
    memset(p, 0, (size_t)kPackedArgsBytes);

    A_desc->offset = 0;
    B_desc->offset = 0;
    C_desc->offset = 0;
    n_elements_desc->offset = 0;
    program_id_desc->offset = 0;
    program_id_desc->shape[0] = (int64_t)program_id + 1;
    program_id_desc->strides[0] = 1;
    *((int32_t *)program_id_desc->data) = program_id;

    int idx = 0;

    tsi_pack_triton_add_arg(p, idx, A_desc, "A");
    tsi_pack_triton_add_arg(p, idx, B_desc, "B");
    tsi_pack_triton_add_arg(p, idx, C_desc, "C");
    tsi_pack_triton_add_arg(p, idx, n_elements_desc, "n_elements");
    tsi_pack_triton_add_arg(p, idx, program_id_desc, "program_id");

    if (idx != kPackedArgsI64) {
        fprintf(stderr,
                "ERROR: Triton ADD packed idx=%d expected=%ld\n",
                idx,
                (long)kPackedArgsI64);
        fflush(stderr);
        tsi_cleanup();
        abort();
    }

    void *commandList = tsi_create_command_list(deviceId);
    if (!commandList) {
        fprintf(stderr,
                "ERROR: tsi_create_command_list failed for Triton ADD device=%d\n",
                (int)deviceId);
        fflush(stderr);
        tsi_cleanup();
        abort();
    }

    const int64_t packedHandle =
        tsi_shmem_handle_from_ptr(packed_args[deviceId]);

    if (!blobDescriptor_triton_add || !blobDescriptor_triton_add[0]) {
        fprintf(stderr, "ERROR: Triton ADD blob descriptor is not loaded\n");
        fflush(stderr);
        tsi_cleanup();
        abort();
    }

    void *blobExecuteCmd = tsi_launch_blob(
        blobDescriptor_triton_add[0],
        packedHandle,
        kPackedArgsBytes);

    if (!blobExecuteCmd) {
        fprintf(stderr,
                "ERROR: tsi_launch_blob failed for Triton ADD device=%d blob_desc=%p packedHandle=%ld bytes=%ld\n",
                (int)deviceId,
                (void *)blobDescriptor_triton_add[0],
                (long)packedHandle,
                (long)kPackedArgsBytes);
        fflush(stderr);
        tsi_cleanup();
        abort();
    }

    tsi_add_command_to_list(commandList, blobExecuteCmd);
    return commandList;
}


static void _mlir_ciface_add_kernel_device_wrapper_triton_dispatch(
    void *A_desc_v,
    void *B_desc_v,
    void *C_desc_v,
    void *n_elements_desc_v,
    void *grid1_desc_v,
    void *grid2_desc_v,
    void *grid3_desc_v,
    void *max_txes_desc_v) {

    tsi_init_per_txe_state_once();

    if (!multi_thread_enable) {
        _mlir_ciface_add_kernel_device_wrapper(
            A_desc_v,
            B_desc_v,
            C_desc_v,
            n_elements_desc_v,
            grid1_desc_v,
            grid2_desc_v,
            grid3_desc_v,
            max_txes_desc_v);
        return;
    }

    static MemRefDescriptor<Rank_Triton> *triton_add_program_id_desc = nullptr;
    static int32_t *triton_add_program_id_payload = nullptr;

    if (!triton_add_program_id_desc || !triton_add_program_id_payload) {
        triton_add_program_id_desc = (MemRefDescriptor<Rank_Triton> *)tsi_alloc(TRITON_MATMUL_ALIGNMENT_BYTES);
        triton_add_program_id_payload = (int32_t *)tsi_alloc(TRITON_MATMUL_ALIGNMENT_BYTES);
        TSAVORITE_GGML_ASSERT(triton_add_program_id_desc);
        TSAVORITE_GGML_ASSERT(triton_add_program_id_payload);
    }

    memset(triton_add_program_id_desc, 0, sizeof(MemRefDescriptor<Rank_Triton>));
    triton_add_program_id_desc->base = triton_add_program_id_payload;
    triton_add_program_id_desc->data = triton_add_program_id_payload;
    triton_add_program_id_desc->offset = 0;
    triton_add_program_id_desc->shape[0] = 1;
    triton_add_program_id_desc->strides[0] = 1;
    *triton_add_program_id_payload = 0;

    int deviceId = acquire_device_blocking();

    void *commandList =
        _mlir_ciface_add_kernel_device_wrapper_triton_manual_internal(
            A_desc_v,
            B_desc_v,
            C_desc_v,
            n_elements_desc_v,
            triton_add_program_id_desc,
            0,
            deviceId);

    if (!commandList) {
        fprintf(stderr,
                "Command List Empty for Triton ADD on device %d\n",
                deviceId);
        fflush(stderr);
        release_device(deviceId);
        tsi_cleanup();
        abort();
    }

    {
        std::lock_guard<std::mutex> lk(workers_mutex);
        workers.emplace_back([=]() {
            tsi_blob_execution_internal(commandList);
            release_device(deviceId);
        });
    }
}


static void *triton_matmul_kernel_device_wrapper_triton_manual_internal(
    void *A_desc_v,
    void *B_desc_v,
    void *C_desc_v,
    void *M_desc_v,
    void *N_desc_v,
    void *K_desc_v,
    void *program_id_desc_v,
    BlobDescriptor *blobDescriptor_matmul_selected,
    TSI_DeviceIdType deviceId) {

    constexpr int64_t kPackedArgsI64   = 14;
    constexpr int64_t kPackedArgsBytes = kPackedArgsI64 * (int64_t)sizeof(int64_t);

    std::lock_guard<std::mutex> lock(tsi_pack_mutex);

    if ((uint32_t)deviceId >= num_of_txes) {
        fprintf(stderr,
                "ERROR: Triton MAT_MUL deviceId=%d out of range num_of_txes=%u\n",
                (int)deviceId, (unsigned)num_of_txes);
        fflush(stderr);
        tsi_cleanup();
        abort();
    }

    if (packed_args.size() != num_of_txes || !packed_args[deviceId]) {
        fprintf(stderr,
                "ERROR: Triton MAT_MUL packed_args not initialized deviceId=%d size=%zu num_of_txes=%u\n",
                (int)deviceId,
                packed_args.size(),
                (unsigned)num_of_txes);
        fflush(stderr);
        tsi_cleanup();
        abort();
    }

    auto *program_id_desc = (MemRefDescriptor<Rank_Triton> *)program_id_desc_v;
    auto *A_desc = (MemRefDescriptor<Rank_Triton> *)A_desc_v;
    auto *B_desc = (MemRefDescriptor<Rank_Triton> *)B_desc_v;
    auto *C_desc = (MemRefDescriptor<Rank_Triton> *)C_desc_v;

    auto *M_desc = (MemRefDescriptor<Rank_Triton> *)M_desc_v;
    auto *N_desc = (MemRefDescriptor<Rank_Triton> *)N_desc_v;
    auto *K_desc = (MemRefDescriptor<Rank_Triton> *)K_desc_v;

    int64_t *p = static_cast<int64_t *>(packed_args[deviceId]);
    memset(p, 0, (size_t)kPackedArgsBytes);

    M_desc->offset = 0;
    N_desc->offset = 0;
    K_desc->offset = 0;
    A_desc->offset = 0;
    B_desc->offset = 0;
    C_desc->offset = 0;
    program_id_desc->offset = 0;
    program_id_desc->shape[0] = 1;

    int idx = 0;

    // A,B,C,M,N,K,program_id
    tsi_pack_triton_matmul_arg(p, idx, A_desc,     "A");
    tsi_pack_triton_matmul_arg(p, idx, B_desc,     "B");
    tsi_pack_triton_matmul_arg(p, idx, C_desc,     "C");
    tsi_pack_triton_matmul_arg(p, idx, M_desc,     "M");
    tsi_pack_triton_matmul_arg(p, idx, N_desc,     "N");
    tsi_pack_triton_matmul_arg(p, idx, K_desc,     "K");
    tsi_pack_triton_matmul_arg(p, idx, program_id_desc, "program_id");

    if (idx != kPackedArgsI64) {
        fprintf(stderr,
                "ERROR: Triton MAT_MUL packed idx=%d expected=%ld\n",
                idx, (long)kPackedArgsI64);
        fflush(stderr);
        tsi_cleanup();
        abort();
    }

    void *commandList = tsi_create_command_list(deviceId);
    if (!commandList) {
        fprintf(stderr,
                "ERROR: tsi_create_command_list failed for Triton MAT_MUL device=%d\n",
                (int)deviceId);
        fflush(stderr);
        tsi_cleanup();
        abort();
    }

    const int64_t packedHandle =
        tsi_shmem_handle_from_ptr(packed_args[deviceId]);

    if (!blobDescriptor_matmul_selected) {
        fprintf(stderr,
                "ERROR: missing Triton MAT_MUL blob descriptor for selected shape device=%d\n",
                (int)deviceId);
        fflush(stderr);
        tsi_cleanup();
        abort();
    }

    void *blobExecuteCmd = tsi_launch_blob(
        blobDescriptor_matmul_selected,
        packedHandle,
        kPackedArgsBytes);

    if (!blobExecuteCmd) {
        fprintf(stderr,
                "ERROR: tsi_launch_blob failed for Triton MAT_MUL device=%d blob_desc=%p packedHandle=%ld bytes=%ld\n",
                (int)deviceId,
                (void *)blobDescriptor_matmul_selected,
                (long)packedHandle,
                (long)kPackedArgsBytes);
        fflush(stderr);
        tsi_cleanup();
        abort();
    }

    tsi_add_command_to_list(commandList, blobExecuteCmd);
    return commandList;
}


static inline BlobDescriptor *triton_matmul_blob_descriptor_for_shape(
    const triton_matmul_txe_shape_t &shape) {
    if (shape.kind == TRITON_MATMUL_KIND_2X4) {
        return blobDescriptor_matmul_2x4 ? blobDescriptor_matmul_2x4[0] : nullptr;
    }
    return blobDescriptor_matmul_1x8 ? blobDescriptor_matmul_1x8[0] : nullptr;
}

static inline void triton_matmul_generated_wrapper_for_shape(
    const triton_matmul_txe_shape_t &shape,
    void *A_desc_v,
    void *B_desc_v,
    void *C_desc_v,
    void *M_desc_v,
    void *N_desc_v,
    void *K_desc_v,
    void *grid1_desc_v,
    void *grid2_desc_v,
    void *grid3_desc_v,
    void *max_txes_desc_v) {
    if (shape.kind == TRITON_MATMUL_KIND_2X4) {
        _mlir_ciface_matmul_kernel_2x4_device_wrapper(
            A_desc_v, B_desc_v, C_desc_v,
            M_desc_v, N_desc_v, K_desc_v,
            grid1_desc_v, grid2_desc_v, grid3_desc_v,
            max_txes_desc_v);
        return;
    }

    _mlir_ciface_matmul_kernel_1x8_device_wrapper(
        A_desc_v, B_desc_v, C_desc_v,
        M_desc_v, N_desc_v, K_desc_v,
        grid1_desc_v, grid2_desc_v, grid3_desc_v,
        max_txes_desc_v);
}

static void triton_matmul_kernel_device_wrapper_triton_dispatch(
    const triton_matmul_txe_shape_t &txe_shape,
    void *A_desc_v,
    void *B_desc_v,
    void *C_desc_v,
    void *M_desc_v,
    void *N_desc_v,
    void *K_desc_v,
    void *grid1_desc_v,
    void *grid2_desc_v,
    void *grid3_desc_v,
    void *max_txes_desc_v) {

    tsi_init_per_txe_state_once();

    if (!multi_thread_enable) {
        triton_matmul_generated_wrapper_for_shape(
            txe_shape,
            A_desc_v,
            B_desc_v,
            C_desc_v,
            M_desc_v,
            N_desc_v,
            K_desc_v,
            grid1_desc_v,
            grid2_desc_v,
            grid3_desc_v,
            max_txes_desc_v);
        return;
    }

    int deviceId = acquire_device_blocking();

    void *program_id_desc_v = grid1_desc_v;

    void *commandList =
        triton_matmul_kernel_device_wrapper_triton_manual_internal(
            A_desc_v,
            B_desc_v,
            C_desc_v,
            M_desc_v,
            N_desc_v,
            K_desc_v,
            program_id_desc_v,
            triton_matmul_blob_descriptor_for_shape(txe_shape),
            deviceId);

    if (!commandList) {
        fprintf(stderr,
                "Command List Empty for Triton MAT_MUL on device %d\n",
                deviceId);
        fflush(stderr);
        release_device(deviceId);
        tsi_cleanup();
        abort();
    }

    {
        std::lock_guard<std::mutex> lk(workers_mutex);
        workers.emplace_back([=]() {
       tsi_blob_execution_internal(commandList);
            release_device(deviceId);
        });
    }
}


// -----------------------------------------------------------------------------
// Triton MAT_MUL ABI helpers
// IMPORTANT:
// - Triton matmul wrapper wants flattened rank-1 memrefs for A/B/C
// - M/N/K are scalar memrefs; direct blob launch appends program_id only
// - Descriptor and scalar payload must both be device-visible and 128B aligned
// -----------------------------------------------------------------------------

template<int N>
static inline void init_rank1_memref_flat(
    MemRefDescriptor<N> *d,
    void *ptr,
    int64_t len
) {
    memset(d, 0, sizeof(MemRefDescriptor<N>));
    d->base       = ptr;
    d->data       = ptr;
    d->offset     = 0;
    d->shape[0]   = len;
    d->strides[0] = 1;
}

template<int N>
static inline void init_scalar_i32_memref_aligned(
    MemRefDescriptor<N> *d,
    void *payload_ptr,
    int32_t v
) {
    memset(d, 0, sizeof(MemRefDescriptor<N>));
    d->base       = payload_ptr;
    d->data       = payload_ptr;
    d->offset     = 0;
    //We bypass Triton generated host_wrapper and pack blob args manually.
    //d->shape[0]   = 1;
    d->shape[0]   = v + 1;
    d->strides[0] = 1;
    *((int32_t *) payload_ptr) = v;
}

static inline int64_t tsi_round_up_i64(int64_t v, int64_t a) {
    if (v <= 0 || a <= 0) {
        return 0;
    }

    const unsigned __int128 uv = (unsigned __int128)(uint64_t)v;
    const unsigned __int128 ua = (unsigned __int128)(uint64_t)a;
    const unsigned __int128 rounded = ((uv + ua - 1) / ua) * ua;

    if (rounded > (unsigned __int128)INT64_MAX) {
        return INT64_MAX;
    }

    return (int64_t)rounded;
}

static inline bool triton_matmul_should_use_small_n_transpose(
    const triton_matmul_txe_shape_t &shape,
    int64_t M,
    int64_t N,
    int64_t K) {
    // Current PR scope: small-N transpose is implemented for the single-TXE
    // path only. Keep the flag explicit for multi-TXE deployments so enabling
    // triton_matmul_small_n_transpose_opt does not silently imply multi-TXE
    // transpose support.
    if (!triton_matmul_small_n_transpose_opt ||
        (multi_thread_enable && num_of_txes > 1)) {
        return false;
    }

    if (M <= 0 || N <= 0 || K <= 0) {
        return false;
    }

    if (shape.m_dim <= 0 || shape.n_dim <= 0) {
        return false;
    }

    if (N >= M) {
        return false;
    }

    // Estimate total physical work for the selected TXE shape.
    // This is intentionally shape-driven so the same decision path can be reused
    // when a 2x4 shape is added in a later PR.
    const int64_t orig_M_pad = tsi_round_up_i64(M, shape.m_dim);
    const int64_t orig_N_pad = tsi_round_up_i64(N, shape.n_dim);
    const int64_t swap_M_pad = tsi_round_up_i64(N, shape.m_dim);
    const int64_t swap_N_pad = tsi_round_up_i64(M, shape.n_dim);

    if (orig_M_pad == INT64_MAX || orig_N_pad == INT64_MAX ||
        swap_M_pad == INT64_MAX || swap_N_pad == INT64_MAX) {
        return false;
    }

    const unsigned __int128 orig_work =
        (unsigned __int128)(uint64_t)orig_M_pad *
        (unsigned __int128)(uint64_t)orig_N_pad *
        (unsigned __int128)(uint64_t)K;

    const unsigned __int128 swap_work =
        (unsigned __int128)(uint64_t)swap_M_pad *
        (unsigned __int128)(uint64_t)swap_N_pad *
        (unsigned __int128)(uint64_t)K;

    if (orig_work > (unsigned __int128)INT64_MAX ||
        swap_work > (unsigned __int128)INT64_MAX) {
        return false;
    }

    if (swap_work >= orig_work) {
        return false;
    }

    // Use the transpose path only for clear small-N cases. This avoids changing
    // behavior for square matrices or cases where N is already large enough.
    return (M / 4 >= N) || (N <= 8);
}

static float *g_triton_A_full = nullptr; // [M_cap x K_cap]
static float *g_triton_B_full = nullptr; // [K_cap x N_cap]
static float *g_triton_C_full = nullptr; // [M_cap x N_cap]

static int64_t g_triton_M_cap = 0;
static int64_t g_triton_N_cap = 0;
static int64_t g_triton_K_cap = 0;


static inline void ensure_triton_full_buffers(
    int64_t M_pad,
    int64_t N_pad,
    int64_t K) {

    TSAVORITE_GGML_ASSERT(M_pad > 0);
    TSAVORITE_GGML_ASSERT(N_pad > 0);
    TSAVORITE_GGML_ASSERT(K > 0);

    const int64_t need_M = M_pad;
    const int64_t need_N = N_pad;
    const int64_t need_K =
        tsi_round_up_i64(K, TRITON_MATMUL_F32_K_DIM);

    if (g_triton_A_full &&
        g_triton_B_full &&
        g_triton_C_full &&
        need_M <= g_triton_M_cap &&
        need_N <= g_triton_N_cap &&
        need_K <= g_triton_K_cap) {
        return;
    }

    float *old_A = g_triton_A_full;
    float *old_B = g_triton_B_full;
    float *old_C = g_triton_C_full;

    int64_t new_M =
        std::max<int64_t>(need_M,
                          std::max<int64_t>(g_triton_M_cap, 1024));

    int64_t new_N =
        std::max<int64_t>(need_N,
                          std::max<int64_t>(g_triton_N_cap, 4096));

    int64_t new_K =
        std::max<int64_t>(need_K,
                          std::max<int64_t>(g_triton_K_cap, 4096));

    float *new_A = (float *) tsi_alloc(
        (size_t)new_M *
        (size_t)new_K *
        sizeof(float));

    float *new_B = (float *) tsi_alloc(
        (size_t)new_K *
        (size_t)new_N *
        sizeof(float));

    float *new_C = (float *) tsi_alloc(
        (size_t)new_M *
        (size_t)new_N *
        sizeof(float));

    TSAVORITE_GGML_ASSERT(new_A);
    TSAVORITE_GGML_ASSERT(new_B);
    TSAVORITE_GGML_ASSERT(new_C);

    g_triton_A_full = new_A;
    g_triton_B_full = new_B;
    g_triton_C_full = new_C;

    g_triton_M_cap = new_M;
    g_triton_N_cap = new_N;
    g_triton_K_cap = new_K;

    if (old_A) {
        tsi_dealloc(old_A);
    }
    if (old_B) {
        tsi_dealloc(old_B);
    }
    if (old_C) {
        tsi_dealloc(old_C);
    }

#if TRITON_DEBUG
    fprintf(stderr,
            "TRITON_GROW_BUFFER: M=%ld N=%ld K=%ld\n",
            (long)new_M,
            (long)new_N,
            (long)new_K);
#endif
}

// Guards the static descriptor/payload buffers below: they are shared,
// process-wide storage reused across every call (not per-device, unlike
// call_triton_matmul_full_packed_on_device()'s g_triton_desc_mt). With
// multi_thread_enable=true, concurrent calls into this function race on
// populating those buffers (init_rank1_memref_flat/init_scalar_i32_memref_aligned
// write them with no synchronization) before ever reaching the
// tsi_pack_mutex-protected dispatch call -- one caller's in-flight descriptor
// data can be overwritten by another's mid-dispatch, corrupting the handle
// tsi_shmem_handle_from_ptr() resolves it to. Serializing the whole
// populate+dispatch sequence here closes that race.
static std::mutex g_full_packed_static_mutex;

static inline void call_triton_matmul_full_packed(
    const triton_matmul_txe_shape_t &txe_shape,
    float *A_full,     // physical [M_pad x K]
    float *B_full,     // physical [K x N_pad]
    float *C_full,     // physical [M_pad x N_pad]
    int32_t M_pad,
    int32_t N_pad,
    int32_t K) {

    std::lock_guard<std::mutex> lock(g_full_packed_static_mutex);

    static MemRefDescriptor<Rank_Triton> *A_desc = nullptr;
    static MemRefDescriptor<Rank_Triton> *B_desc = nullptr;
    static MemRefDescriptor<Rank_Triton> *C_desc = nullptr;

    static MemRefDescriptor<Rank_Triton> *M_desc     = nullptr;
    static MemRefDescriptor<Rank_Triton> *N_desc     = nullptr;
    static MemRefDescriptor<Rank_Triton> *K_desc     = nullptr;
    static MemRefDescriptor<Rank_Triton> *grid1_desc = nullptr;
    static MemRefDescriptor<Rank_Triton> *grid2_desc = nullptr;
    static MemRefDescriptor<Rank_Triton> *grid3_desc = nullptr;
    static MemRefDescriptor<Rank_Triton> *max_txes_desc = nullptr;

    static int32_t *M_payload     = nullptr;
    static int32_t *N_payload     = nullptr;
    static int32_t *K_payload     = nullptr;
    static int32_t *grid1_payload = nullptr;
    static int32_t *grid2_payload = nullptr;
    static int32_t *grid3_payload = nullptr;
    static int32_t *max_txes_payload = nullptr;

    static bool inited = false;

    if (!inited) {
        A_desc = (MemRefDescriptor<Rank_Triton> *) tsi_alloc(sizeof(MemRefDescriptor<Rank_Triton>));
        B_desc = (MemRefDescriptor<Rank_Triton> *) tsi_alloc(sizeof(MemRefDescriptor<Rank_Triton>));
        C_desc = (MemRefDescriptor<Rank_Triton> *) tsi_alloc(sizeof(MemRefDescriptor<Rank_Triton>));

        M_desc     = (MemRefDescriptor<Rank_Triton> *) tsi_alloc(sizeof(MemRefDescriptor<Rank_Triton>));
        N_desc     = (MemRefDescriptor<Rank_Triton> *) tsi_alloc(sizeof(MemRefDescriptor<Rank_Triton>));
        K_desc     = (MemRefDescriptor<Rank_Triton> *) tsi_alloc(sizeof(MemRefDescriptor<Rank_Triton>));
        grid1_desc = (MemRefDescriptor<Rank_Triton> *) tsi_alloc(sizeof(MemRefDescriptor<Rank_Triton>));
        grid2_desc = (MemRefDescriptor<Rank_Triton> *) tsi_alloc(sizeof(MemRefDescriptor<Rank_Triton>));
        grid3_desc = (MemRefDescriptor<Rank_Triton> *) tsi_alloc(sizeof(MemRefDescriptor<Rank_Triton>));
        max_txes_desc = (MemRefDescriptor<Rank_Triton> *) tsi_alloc(sizeof(MemRefDescriptor<Rank_Triton>));

        M_payload     = (int32_t *) tsi_alloc(TRITON_MATMUL_ALIGNMENT_BYTES);
        N_payload     = (int32_t *) tsi_alloc(TRITON_MATMUL_ALIGNMENT_BYTES);
        K_payload     = (int32_t *) tsi_alloc(TRITON_MATMUL_ALIGNMENT_BYTES);
        grid1_payload = (int32_t *) tsi_alloc(TRITON_MATMUL_ALIGNMENT_BYTES);
        grid2_payload = (int32_t *) tsi_alloc(TRITON_MATMUL_ALIGNMENT_BYTES);
        grid3_payload = (int32_t *) tsi_alloc(TRITON_MATMUL_ALIGNMENT_BYTES);
        max_txes_payload = (int32_t *) tsi_alloc(TRITON_MATMUL_ALIGNMENT_BYTES);

        TSAVORITE_GGML_ASSERT(A_desc && B_desc && C_desc);
        TSAVORITE_GGML_ASSERT(M_desc && N_desc && K_desc);
        TSAVORITE_GGML_ASSERT(grid1_desc && grid2_desc && grid3_desc && max_txes_desc);
        TSAVORITE_GGML_ASSERT(M_payload && N_payload && K_payload);
        TSAVORITE_GGML_ASSERT(grid1_payload && grid2_payload && grid3_payload && max_txes_payload);

        inited = true;
    }

    TSAVORITE_GGML_ASSERT((M_pad % txe_shape.m_dim) == 0);
    TSAVORITE_GGML_ASSERT((N_pad % txe_shape.n_dim) == 0);
    TSAVORITE_GGML_ASSERT((K % TRITON_MATMUL_F32_K_DIM) == 0);

    init_rank1_memref_flat(
        A_desc,
        (void *) A_full,
        (int64_t) M_pad * (int64_t) K);

    init_rank1_memref_flat(
        B_desc,
        (void *) B_full,
        (int64_t) K * (int64_t) N_pad);

    init_rank1_memref_flat(
        C_desc,
        (void *) C_full,
        (int64_t) M_pad * (int64_t) N_pad);

    init_scalar_i32_memref_aligned(M_desc,     M_payload,     M_pad);
    init_scalar_i32_memref_aligned(N_desc,     N_payload,     N_pad);
    init_scalar_i32_memref_aligned(K_desc,     K_payload,     K);
    init_scalar_i32_memref_aligned(grid1_desc, grid1_payload, 1);
    init_scalar_i32_memref_aligned(grid2_desc, grid2_payload, 1);
    init_scalar_i32_memref_aligned(grid3_desc, grid3_payload, 1);
    init_scalar_i32_memref_aligned(max_txes_desc, max_txes_payload, (int32_t)num_of_txes);


    triton_matmul_kernel_device_wrapper_triton_dispatch(
        txe_shape,
        A_desc, B_desc, C_desc,
        M_desc, N_desc, K_desc,
        grid1_desc, grid2_desc, grid3_desc,
        max_txes_desc);
}


// ============================================================================
// Triton MAT_MUL Multi-TXE M-split support
// ============================================================================

static float *g_triton_A_full_mt[MAX_TXES_SUPPORTED] = { nullptr };
static float *g_triton_B_full_mt[MAX_TXES_SUPPORTED] = { nullptr };
static float *g_triton_C_full_mt[MAX_TXES_SUPPORTED] = { nullptr };

static int64_t g_triton_M_cap_mt[MAX_TXES_SUPPORTED] = { 0 };
static int64_t g_triton_N_cap_mt[MAX_TXES_SUPPORTED] = { 0 };
static int64_t g_triton_K_cap_mt[MAX_TXES_SUPPORTED] = { 0 };

static std::mutex g_triton_mt_alloc_mutex;

static std::vector<float> g_triton_B_packed_cache;
static size_t g_triton_B_packed_cache_capacity = 0;

static inline void ensure_triton_full_buffers_for_device(
    int deviceId,
    int64_t M_pad,
    int64_t N_pad,
    int64_t K) {

    TSAVORITE_GGML_ASSERT(deviceId >= 0);
    TSAVORITE_GGML_ASSERT(deviceId < MAX_TXES_SUPPORTED);
    TSAVORITE_GGML_ASSERT((uint32_t)deviceId < num_of_txes);
    TSAVORITE_GGML_ASSERT(M_pad > 0);
    TSAVORITE_GGML_ASSERT(N_pad > 0);
    TSAVORITE_GGML_ASSERT(K > 0);
    TSAVORITE_GGML_ASSERT((K % TRITON_MATMUL_F32_K_DIM) == 0);

    std::lock_guard<std::mutex> lk(g_triton_mt_alloc_mutex);

    const int64_t need_M = M_pad;
    const int64_t need_N = N_pad;
    const int64_t need_K = tsi_round_up_i64(K, TRITON_MATMUL_F32_K_DIM);

    if (g_triton_A_full_mt[deviceId] &&
        g_triton_B_full_mt[deviceId] &&
        g_triton_C_full_mt[deviceId] &&
        need_M <= g_triton_M_cap_mt[deviceId] &&
        need_N <= g_triton_N_cap_mt[deviceId] &&
        need_K <= g_triton_K_cap_mt[deviceId]) {
        return;
    }

    float *old_A = g_triton_A_full_mt[deviceId];
    float *old_B = g_triton_B_full_mt[deviceId];
    float *old_C = g_triton_C_full_mt[deviceId];

    const int64_t new_M =
        std::max<int64_t>(need_M,
        std::max<int64_t>(g_triton_M_cap_mt[deviceId], 64));

    const int64_t new_N =
        std::max<int64_t>(need_N,
        std::max<int64_t>(g_triton_N_cap_mt[deviceId], 4096));

    const int64_t new_K =
        std::max<int64_t>(need_K,
        std::max<int64_t>(g_triton_K_cap_mt[deviceId], 4096));

    float *new_A = (float *)tsi_alloc(
        (size_t)new_M * (size_t)new_K * sizeof(float));

    float *new_B = (float *)tsi_alloc(
        (size_t)new_K * (size_t)new_N * sizeof(float));

    float *new_C = (float *)tsi_alloc(
        (size_t)new_M * (size_t)new_N * sizeof(float));

    TSAVORITE_GGML_ASSERT(new_A);
    TSAVORITE_GGML_ASSERT(new_B);
    TSAVORITE_GGML_ASSERT(new_C);

    TSAVORITE_GGML_ASSERT((((uintptr_t)new_A) & TRITON_MATMUL_ALIGNMENT_MASK) == 0);
    TSAVORITE_GGML_ASSERT((((uintptr_t)new_B) & TRITON_MATMUL_ALIGNMENT_MASK) == 0);
    TSAVORITE_GGML_ASSERT((((uintptr_t)new_C) & TRITON_MATMUL_ALIGNMENT_MASK) == 0);

    g_triton_A_full_mt[deviceId] = new_A;
    g_triton_B_full_mt[deviceId] = new_B;
    g_triton_C_full_mt[deviceId] = new_C;

    g_triton_M_cap_mt[deviceId] = new_M;
    g_triton_N_cap_mt[deviceId] = new_N;
    g_triton_K_cap_mt[deviceId] = new_K;

    if (old_A) {
        tsi_dealloc(old_A);
    }
    if (old_B) {
        tsi_dealloc(old_B);
    }
    if (old_C) {
        tsi_dealloc(old_C);
    }

#if TRITON_DEBUG
    fprintf(stderr,
            "TRITON_MT_GROW_BUFFER: device=%d M_cap=%ld N_cap=%ld K_cap=%ld\n",
            deviceId,
            (long)new_M,
            (long)new_N,
            (long)new_K);
#endif
}

static float * ensure_triton_B_packed_cache(size_t elems)
{
    if (g_triton_B_packed_cache_capacity < elems) {
        g_triton_B_packed_cache.resize(elems);
        g_triton_B_packed_cache_capacity = elems;
    }

    return g_triton_B_packed_cache.data();
}


struct triton_matmul_desc_set_t {
    MemRefDescriptor<Rank_Triton> *A_desc = nullptr;
    MemRefDescriptor<Rank_Triton> *B_desc = nullptr;
    MemRefDescriptor<Rank_Triton> *C_desc = nullptr;
    MemRefDescriptor<Rank_Triton> *M_desc = nullptr;
    MemRefDescriptor<Rank_Triton> *N_desc = nullptr;
    MemRefDescriptor<Rank_Triton> *K_desc = nullptr;
    MemRefDescriptor<Rank_Triton> *grid1_desc = nullptr;
    MemRefDescriptor<Rank_Triton> *grid2_desc = nullptr;
    MemRefDescriptor<Rank_Triton> *grid3_desc = nullptr;
    MemRefDescriptor<Rank_Triton> *max_txes_desc = nullptr;

    int32_t *M_payload = nullptr;
    int32_t *N_payload = nullptr;
    int32_t *K_payload = nullptr;
    int32_t *grid1_payload = nullptr;
    int32_t *grid2_payload = nullptr;
    int32_t *grid3_payload = nullptr;
    int32_t *max_txes_payload = nullptr;
};

static std::vector<triton_matmul_desc_set_t> g_triton_desc_mt;
static std::mutex g_triton_desc_mt_mutex;

static inline triton_matmul_desc_set_t *ensure_triton_desc_for_device(int deviceId) {
    TSAVORITE_GGML_ASSERT(deviceId >= 0);
    TSAVORITE_GGML_ASSERT((uint32_t)deviceId < num_of_txes);

    std::lock_guard<std::mutex> lk(g_triton_desc_mt_mutex);

    if (g_triton_desc_mt.size() != num_of_txes) {
        g_triton_desc_mt.resize(num_of_txes);
    }

    triton_matmul_desc_set_t &s = g_triton_desc_mt[deviceId];

    if (!s.A_desc) {
        s.A_desc = (MemRefDescriptor<Rank_Triton> *) tsi_alloc(sizeof(MemRefDescriptor<Rank_Triton>));
        s.B_desc = (MemRefDescriptor<Rank_Triton> *) tsi_alloc(sizeof(MemRefDescriptor<Rank_Triton>));
        s.C_desc = (MemRefDescriptor<Rank_Triton> *) tsi_alloc(sizeof(MemRefDescriptor<Rank_Triton>));

        s.M_desc     = (MemRefDescriptor<Rank_Triton> *) tsi_alloc(sizeof(MemRefDescriptor<Rank_Triton>));
        s.N_desc     = (MemRefDescriptor<Rank_Triton> *) tsi_alloc(sizeof(MemRefDescriptor<Rank_Triton>));
        s.K_desc     = (MemRefDescriptor<Rank_Triton> *) tsi_alloc(sizeof(MemRefDescriptor<Rank_Triton>));
        s.grid1_desc = (MemRefDescriptor<Rank_Triton> *) tsi_alloc(sizeof(MemRefDescriptor<Rank_Triton>));
        s.grid2_desc = (MemRefDescriptor<Rank_Triton> *) tsi_alloc(sizeof(MemRefDescriptor<Rank_Triton>));
        s.grid3_desc = (MemRefDescriptor<Rank_Triton> *) tsi_alloc(sizeof(MemRefDescriptor<Rank_Triton>));
        s.max_txes_desc = (MemRefDescriptor<Rank_Triton> *) tsi_alloc(sizeof(MemRefDescriptor<Rank_Triton>));

        s.M_payload     = (int32_t *) tsi_alloc(TRITON_MATMUL_ALIGNMENT_BYTES);
        s.N_payload     = (int32_t *) tsi_alloc(TRITON_MATMUL_ALIGNMENT_BYTES);
        s.K_payload     = (int32_t *) tsi_alloc(TRITON_MATMUL_ALIGNMENT_BYTES);
        s.grid1_payload = (int32_t *) tsi_alloc(TRITON_MATMUL_ALIGNMENT_BYTES);
        s.grid2_payload = (int32_t *) tsi_alloc(TRITON_MATMUL_ALIGNMENT_BYTES);
        s.grid3_payload = (int32_t *) tsi_alloc(TRITON_MATMUL_ALIGNMENT_BYTES);
        s.max_txes_payload = (int32_t *) tsi_alloc(TRITON_MATMUL_ALIGNMENT_BYTES);

        TSAVORITE_GGML_ASSERT(s.A_desc && s.B_desc && s.C_desc);
        TSAVORITE_GGML_ASSERT(s.M_desc && s.N_desc && s.K_desc);
        TSAVORITE_GGML_ASSERT(s.grid1_desc && s.grid2_desc && s.grid3_desc && s.max_txes_desc);
        TSAVORITE_GGML_ASSERT(s.M_payload && s.N_payload && s.K_payload);
        TSAVORITE_GGML_ASSERT(s.grid1_payload && s.grid2_payload && s.grid3_payload && s.max_txes_payload);
    }

    return &s;
}

struct triton_matmul_dispatch_profile_t {
    int64_t launch_us = 0;
    int64_t wait_us = 0;
};

static inline triton_matmul_dispatch_profile_t call_triton_matmul_full_packed_on_device(
    const triton_matmul_txe_shape_t &txe_shape,
    int deviceId,
    float *A_full,
    float *B_full,
    float *C_full,
    int32_t M_pad,
    int32_t N_pad,
    int32_t K) {
    triton_matmul_dispatch_profile_t prof;

    TSAVORITE_GGML_ASSERT((M_pad % txe_shape.m_dim) == 0);
    TSAVORITE_GGML_ASSERT((N_pad % txe_shape.n_dim) == 0);
    TSAVORITE_GGML_ASSERT((K % TRITON_MATMUL_F32_K_DIM) == 0);

    TSAVORITE_GGML_ASSERT(deviceId >= 0);
    TSAVORITE_GGML_ASSERT((uint32_t)deviceId < num_of_txes);
    TSAVORITE_GGML_ASSERT((K % TRITON_MATMUL_F32_K_DIM) == 0);

    triton_matmul_desc_set_t *s = ensure_triton_desc_for_device(deviceId);

    init_rank1_memref_flat(s->A_desc, (void *)A_full, (int64_t)M_pad * (int64_t)K);
    init_rank1_memref_flat(s->B_desc, (void *)B_full, (int64_t)K * (int64_t)N_pad);
    init_rank1_memref_flat(s->C_desc, (void *)C_full, (int64_t)M_pad * (int64_t)N_pad);

    init_scalar_i32_memref_aligned(s->M_desc,     s->M_payload,     M_pad);
    init_scalar_i32_memref_aligned(s->N_desc,     s->N_payload,     N_pad);
    init_scalar_i32_memref_aligned(s->K_desc,     s->K_payload,     K);
    init_scalar_i32_memref_aligned(s->grid1_desc, s->grid1_payload, 1);
    init_scalar_i32_memref_aligned(s->grid2_desc, s->grid2_payload, 1);
    init_scalar_i32_memref_aligned(s->grid3_desc, s->grid3_payload, 1);
    init_scalar_i32_memref_aligned(s->max_txes_desc, s->max_txes_payload, (int32_t)num_of_txes);

    const int64_t launch_start_us = tsavorite_now_us();

    auto *program_id_desc = s->grid1_desc;

    void *commandList =
        triton_matmul_kernel_device_wrapper_triton_manual_internal(
            s->A_desc,
            s->B_desc,
            s->C_desc,
            s->M_desc,
            s->N_desc,
            s->K_desc,
            program_id_desc,
            triton_matmul_blob_descriptor_for_shape(txe_shape),
            (TSI_DeviceIdType)deviceId);

    prof.launch_us = tsavorite_elapsed_us(launch_start_us);

    if (!commandList) {
        fprintf(stderr,
                "Command List Empty for Triton MAT_MUL on device %d\n",
                deviceId);
        fflush(stderr);
        tsi_cleanup();
        abort();
    }

    prof.wait_us = tsi_blob_execution_internal(commandList);
    return prof;
}


// -----------------------------------------------------------------------------
// TMU MUL_MAT runner (called from ggml_tsavorite_graph_compute)
// FIXES:
//  - correct B packing (no memcpy across N)
//  - meaningful validation (pack correctness + full tile reference)
//  - increments node->tsi_kernel_runs and device stats for MUL_MAT
// -----------------------------------------------------------------------------

static inline void triton_matmul_log_offloaded_shape_once(
    const struct ggml_tensor *A,
    const struct ggml_tensor *B,
    const struct ggml_tensor *node) {
#if TRITON_DEBUG
    if (!A || !B || !node) {
        return;
    }

    static std::mutex s_log_mutex;
    static std::vector<std::string> s_seen;

    char key[512];
    snprintf(key, sizeof(key),
             "A=[%ld,%ld,%ld,%ld] B=[%ld,%ld,%ld,%ld] node=[%ld,%ld,%ld,%ld]",
             (long)A->ne[0],    (long)A->ne[1],    (long)A->ne[2],    (long)A->ne[3],
             (long)B->ne[0],    (long)B->ne[1],    (long)B->ne[2],    (long)B->ne[3],
             (long)node->ne[0], (long)node->ne[1], (long)node->ne[2], (long)node->ne[3]);

    bool already_seen = false;
    {
        std::lock_guard<std::mutex> lk(s_log_mutex);
        for (const std::string &s : s_seen) {
            if (s == key) {
                already_seen = true;
                break;
            }
        }
        if (!already_seen) {
            s_seen.push_back(std::string(key));
        }
    }

    if (!already_seen) {
        fprintf(stderr,
                "TRITON_MATMUL_OFFLOADED_SHAPE: "
                "A=[%ld,%ld,%ld,%ld] "
                "B=[%ld,%ld,%ld,%ld] "
                "node=[%ld,%ld,%ld,%ld]\n",
                (long)A->ne[0],    (long)A->ne[1],    (long)A->ne[2],    (long)A->ne[3],
                (long)B->ne[0],    (long)B->ne[1],    (long)B->ne[2],    (long)B->ne[3],
                (long)node->ne[0], (long)node->ne[1], (long)node->ne[2], (long)node->ne[3]);
        fflush(stderr);
    }
#else
    (void)A;
    (void)B;
    (void)node;
#endif
}


static enum ggml_status ggml_tsavorite_run_tmu_mul_mat(
    struct ggml_backend_tsavorite_context *ctx,
    txe_device_s device,
    struct ggml_tensor *node,
    enum ggml_tsavorite_kernel_type kernel_type,
    int kernel_sub_type) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(kernel_sub_type);

    if (!node || !node->src[0] || !node->src[1] || !node->data) {
        return GGML_STATUS_FAILED;
    }

    const int64_t matrix_start_us = tsavorite_now_us();
    tsavorite_matmul_profile_sample_t profile;

    const struct ggml_tensor *A = node->src[0];
    const struct ggml_tensor *B = node->src[1];

    const int64_t K = A->ne[0];
    const int64_t M = A->ne[1];
    const int64_t N = B->ne[1];

    if (K <= 0 || M <= 0 || N <= 0) {
        return GGML_STATUS_FAILED;
    }

    if (B->ne[0] != K) {
        return GGML_STATUS_FAILED;
    }

    if ((K % 32) != 0) {
        return GGML_STATUS_FAILED;
    }

    const int64_t D2 = node->ne[2];
    const int64_t D3 = node->ne[3];

#if TRITON_DEBUG
    triton_matmul_log_offloaded_shape_once(A, B, node);
#endif

    const int64_t a_nb0 = nb_or_default(A, 0);
    const int64_t a_nb1 = nb_or_default(A, 1);
    const int64_t a_nb2 = nb_or_default(A, 2);
    const int64_t a_nb3 = nb_or_default(A, 3);

    const int64_t b_nb0 = nb_or_default(B, 0);
    const int64_t b_nb1 = nb_or_default(B, 1);
    const int64_t b_nb2 = nb_or_default(B, 2);
    const int64_t b_nb3 = nb_or_default(B, 3);

    const int64_t c_nb0 = nb_or_default(node, 0);
    const int64_t c_nb1 = nb_or_default(node, 1);
    const int64_t c_nb2 = nb_or_default(node, 2);
    const int64_t c_nb3 = nb_or_default(node, 3);

    const int64_t A2 = A->ne[2] > 0 ? A->ne[2] : 1;
    const int64_t A3 = A->ne[3] > 0 ? A->ne[3] : 1;
    const int64_t B2 = B->ne[2] > 0 ? B->ne[2] : 1;
    const int64_t B3 = B->ne[3] > 0 ? B->ne[3] : 1;

    char *A_base = (char *)A->data;
    char *B_base = (char *)B->data;
    char *C_base = (char *)node->data;

    // ============================================================
    // Single-TXE / generated-host-wrapper path.
    //
    // Current shape bucket:
    //   TRITON_MATMUL_SHAPE_1X8
    //
    // Optional small-N transpose optimization:
    //   Original: C[M,N] = A[K,M] x B[K,N]
    //   Swapped : T[N,M] = B[K,N] x A[K,M], then copy back C[m,n] = T[n,m]
    //
    // Why this helps for 1x8:
    //   The 1x8 blob pads N to 64 columns. For decode-style N=1 or small N,
    //   running the original orientation can waste many padded columns. The
    //   swapped orientation can reduce padded work when M is much larger than N.
    //
    // Future 2x4 extension:
    //   Keep this shape-bucket pattern. Select TRITON_MATMUL_SHAPE_2X4 in the
    //   future 2x4 path, then call the same transpose heuristic with that shape.
    // ============================================================
    if (!multi_thread_enable || num_of_txes <= 1) {
        const triton_matmul_txe_shape_t &transpose_decision_shape =
            triton_matmul_select_shape(M, N);
        const bool use_small_n_transpose =
            triton_matmul_should_use_small_n_transpose(transpose_decision_shape, M, N, K);

        const int64_t M_work = use_small_n_transpose ? N : M;
        const int64_t N_work = use_small_n_transpose ? M : N;
        const triton_matmul_txe_shape_t &txe_shape =
            triton_matmul_select_shape(M_work, N_work);
        const int64_t M_pad = tsi_round_up_i64(M_work, txe_shape.m_dim);
        const int64_t N_work_pad = tsi_round_up_i64(N_work, txe_shape.n_dim);

        ensure_triton_full_buffers(M_pad, N_work_pad, K);

        for (int64_t d3 = 0; d3 < D3; ++d3) {
            for (int64_t d2 = 0; d2 < D2; ++d2) {
                const int64_t a_d2 = map_repeat_dim_i64(d2, D2, A2);
                const int64_t a_d3 = map_repeat_dim_i64(d3, D3, A3);
                const int64_t b_d2 = map_repeat_dim_i64(d2, D2, B2);
                const int64_t b_d3 = map_repeat_dim_i64(d3, D3, B3);

                char *A_ptr = A_base + a_d2 * a_nb2 + a_d3 * a_nb3;
                char *B_ptr = B_base + b_d2 * b_nb2 + b_d3 * b_nb3;
                char *C_ptr = C_base + d2 * c_nb2 + d3 * c_nb3;

                int64_t t0 = tsavorite_now_us();
                memset(g_triton_A_full, 0, (size_t)M_pad * (size_t)K * sizeof(float));
                profile.padding_memset_us += tsavorite_elapsed_us(t0);

                t0 = tsavorite_now_us();
                if (use_small_n_transpose) {
                    for (int64_t r = 0; r < N; ++r) {
                        const char *row = B_ptr + r * b_nb1;
                        float *dst = g_triton_A_full + r * K;

                        tsavorite_tensor_copy_k_to_f32(B, row, dst, K, b_nb0);
                    }
                } else {
                    for (int64_t r = 0; r < M; ++r) {
                        const char *row = A_ptr + r * a_nb1;
                        float *dst = g_triton_A_full + r * K;

                        tsavorite_tensor_copy_k_to_f32(A, row, dst, K, a_nb0);
                    }
                }
                profile.pack_a_us += tsavorite_elapsed_us(t0);

                t0 = tsavorite_now_us();
                memset(g_triton_B_full, 0, (size_t)K * (size_t)N_work_pad * sizeof(float));
                profile.padding_memset_us += tsavorite_elapsed_us(t0);

                t0 = tsavorite_now_us();
                if (use_small_n_transpose) {
                    std::vector<float> tmp_k;

                    for (int64_t c = 0; c < M; ++c) {
                        const char *col = A_ptr + c * a_nb1;
                        tsavorite_tensor_scatter_k_to_f32_strided(
                            A,
                            col,
                            g_triton_B_full + c,
                            N_work_pad,
                            K,
                            a_nb0,
                            tmp_k);
                    }
                } else {
                    std::vector<float> tmp_k;

                    for (int64_t c = 0; c < N; ++c) {
                        const char *col = B_ptr + c * b_nb1;
                        tsavorite_tensor_scatter_k_to_f32_strided(
                            B,
                            col,
                            g_triton_B_full + c,
                            N_work_pad,
                            K,
                            b_nb0,
                            tmp_k);
                    }
                }
                profile.pack_b_us += tsavorite_elapsed_us(t0);

                t0 = tsavorite_now_us();
                memset(g_triton_C_full, 0, (size_t)M_pad * (size_t)N_work_pad * sizeof(float));
                profile.padding_memset_us += tsavorite_elapsed_us(t0);

                t0 = tsavorite_now_us();
                call_triton_matmul_full_packed(
                    txe_shape,
                    g_triton_A_full,
                    g_triton_B_full,
                    g_triton_C_full,
                    (int32_t)M_pad,
                    (int32_t)N_work_pad,
                    (int32_t)K);
                profile.launch_us += tsavorite_elapsed_us(t0);

                if (multi_thread_enable) {
                    join_all_workers();
                }

                t0 = tsavorite_now_us();
                if (use_small_n_transpose) {
                    for (int64_t r = 0; r < M; ++r) {
                        for (int64_t c = 0; c < N; ++c) {
                            *(float *)(C_ptr + r * c_nb0 + c * c_nb1) =
                                g_triton_C_full[c * N_work_pad + r];
                        }
                    }
                } else {
                    for (int64_t r = 0; r < M; ++r) {
                        for (int64_t c = 0; c < N; ++c) {
                            *(float *)(C_ptr + r * c_nb0 + c * c_nb1) =
                                g_triton_C_full[r * N_work_pad + c];
                        }
                    }
                }
                profile.copyback_us += tsavorite_elapsed_us(t0);

                if (device) {
                    ++device->stats.op_run_count[kernel_type].num_of_kernel_call;
                }

                ++node->tsi_kernel_runs;
                ++profile.kernel_calls;
            }
        }

        profile.matrix_total_us = tsavorite_elapsed_us(matrix_start_us);
        tsavorite_matmul_profile_record(node, profile);

        return GGML_STATUS_SUCCESS;
    }

    // ============================================================
    // Multi-TXE path: split M dimension across available TXEs.
    //
    // TXEWaitCritical:
    //   For each m0 wave, max(wait_us across launched TXEs).
    //   Sum this max across waves.
    //
    // TXEWaitSum:
    //   Sum wait_us across every launched TXE.
    // ============================================================
    const triton_matmul_txe_shape_t &txe_shape = triton_matmul_select_shape(M, N);
    const int64_t N_pad = tsi_round_up_i64(N, txe_shape.n_dim);

    const int64_t active_txes = (int64_t)num_of_txes;
    const int64_t rows_per_txe_unaligned = (M + active_txes - 1) / active_txes;
    const int64_t rows_per_txe =
        tsi_round_up_i64(rows_per_txe_unaligned, txe_shape.m_dim);

    uint64_t launched_kernel_calls = 0;

    for (int64_t d3 = 0; d3 < D3; ++d3) {
        for (int64_t d2 = 0; d2 < D2; ++d2) {
            const int64_t a_d2 = map_repeat_dim_i64(d2, D2, A2);
            const int64_t a_d3 = map_repeat_dim_i64(d3, D3, A3);
            const int64_t b_d2 = map_repeat_dim_i64(d2, D2, B2);
            const int64_t b_d3 = map_repeat_dim_i64(d3, D3, B3);

            char *A_ptr = A_base + a_d2 * a_nb2 + a_d3 * a_nb3;
            char *B_ptr = B_base + b_d2 * b_nb2 + b_d3 * b_nb3;
            char *C_ptr = C_base + d2 * c_nb2 + d3 * c_nb3;

            for (int64_t m0 = 0; m0 < M; m0 += rows_per_txe * active_txes) {
                uint64_t batch_launched = 0;

                int64_t batch_wait_sum_us = 0;
                int64_t batch_wait_max_us = 0;
                int64_t batch_pack_a_us = 0;
                int64_t batch_pack_b_us = 0;
                int64_t batch_padding_us = 0;
                int64_t batch_launch_us = 0;
                int64_t batch_copyback_us = 0;

                std::mutex batch_profile_mutex;

                for (int64_t t = 0; t < active_txes; ++t) {
                    const int64_t tile_m0 = m0 + t * rows_per_txe;

                    if (tile_m0 >= M) {
                        break;
                    }

                    const int64_t M_valid =
                        (M - tile_m0 > rows_per_txe) ? rows_per_txe : (M - tile_m0);
                    const int64_t M_tile_pad =
                        tsi_round_up_i64(M_valid, txe_shape.m_dim);

                    const int deviceId = acquire_device_blocking();

                    if (deviceId < 0 || (uint32_t)deviceId >= num_of_txes) {
                        fprintf(stderr,
                                "ERROR: Triton MAT_MUL failed to acquire valid deviceId=%d num_of_txes=%u\n",
                                deviceId,
                                (unsigned)num_of_txes);
                        fflush(stderr);
                        tsi_cleanup();
                        abort();
                    }

                    ensure_triton_full_buffers_for_device(
                        deviceId,
                        M_tile_pad,
                        N_pad,
                        K);

                    float *A_tile = g_triton_A_full_mt[deviceId];
                    float *B_tile = g_triton_B_full_mt[deviceId];
                    float *C_tile = g_triton_C_full_mt[deviceId];

                    {
                        std::lock_guard<std::mutex> lk(workers_mutex);

                        workers.emplace_back([=,
                            &batch_profile_mutex,
                            &batch_wait_sum_us,
                            &batch_wait_max_us,
                            &batch_pack_a_us,
                            &batch_pack_b_us,
                            &batch_padding_us,
                            &batch_launch_us,
                            &batch_copyback_us] {

                            int64_t local_pack_a_us = 0;
                            int64_t local_pack_b_us = 0;
                            int64_t local_padding_us = 0;
                            int64_t local_launch_us = 0;
                            int64_t local_wait_us = 0;
        int64_t local_copyback_us = 0;

        int64_t t0 = tsavorite_now_us();

        memset(A_tile, 0, (size_t)M_tile_pad * (size_t)K * sizeof(float));
        local_padding_us += tsavorite_elapsed_us(t0);

        t0 = tsavorite_now_us();

        for (int64_t r = 0; r < M_valid; ++r) {
            const int64_t src_r = tile_m0 + r;
            const char *row = A_ptr + src_r * a_nb1;
            float *dst = A_tile + r * K;

            tsavorite_tensor_copy_k_to_f32(A, row, dst, K, a_nb0);
        }

        local_pack_a_us += tsavorite_elapsed_us(t0);

        t0 = tsavorite_now_us();

        memset(B_tile, 0, (size_t)K * (size_t)N_pad * sizeof(float));
        local_padding_us += tsavorite_elapsed_us(t0);

        t0 = tsavorite_now_us();

        std::vector<float> tmp_k;

        for (int64_t c = 0; c < N; ++c) {
            const char *col = B_ptr + c * b_nb1;
            tsavorite_tensor_scatter_k_to_f32_strided(
                B,
                col,
                B_tile + c,
                N_pad,
                K,
                b_nb0,
                tmp_k);
        }

        local_pack_b_us += tsavorite_elapsed_us(t0);

        t0 = tsavorite_now_us();

        memset(C_tile, 0, (size_t)M_tile_pad * (size_t)N_pad * sizeof(float));
        local_padding_us += tsavorite_elapsed_us(t0);

        triton_matmul_dispatch_profile_t dispatch_prof =
            call_triton_matmul_full_packed_on_device(
                txe_shape,
                deviceId,
                A_tile,
                B_tile,
                C_tile,
                (int32_t)M_tile_pad,
                (int32_t)N_pad,
                (int32_t)K);

        local_launch_us += dispatch_prof.launch_us;
        local_wait_us += dispatch_prof.wait_us;

        t0 = tsavorite_now_us();

        for (int64_t r = 0; r < M_valid; ++r) {
            const int64_t dst_r = tile_m0 + r;

            for (int64_t c = 0; c < N; ++c) {
                *(float *)(C_ptr + dst_r * c_nb0 + c * c_nb1) =
                    C_tile[r * N_pad + c];
            }
        }

        local_copyback_us += tsavorite_elapsed_us(t0);

        {
            std::lock_guard<std::mutex> lk2(batch_profile_mutex);

            batch_pack_a_us += local_pack_a_us;
            batch_pack_b_us += local_pack_b_us;
            batch_padding_us += local_padding_us;
            batch_launch_us += local_launch_us;
            batch_copyback_us += local_copyback_us;

            batch_wait_sum_us += local_wait_us;

            if (local_wait_us > batch_wait_max_us) {
                batch_wait_max_us = local_wait_us;
            }
        }

        release_device(deviceId);
    });
}

                    ++batch_launched;
                    ++launched_kernel_calls;
                }

                if (batch_launched > 0) {
                    join_all_workers();

                    profile.pack_a_us += batch_pack_a_us;
                    profile.pack_b_us += batch_pack_b_us;
                    profile.padding_memset_us += batch_padding_us;
                    profile.launch_us += batch_launch_us;
                    profile.copyback_us += batch_copyback_us;

                    profile.txe_wait_sum_us += batch_wait_sum_us;
                    profile.txe_wait_critical_us += batch_wait_max_us;
                }
            }
        }
    }

    if (device) {
        device->stats.op_run_count[kernel_type].num_of_kernel_call += launched_kernel_calls;
    }

    node->tsi_kernel_runs += launched_kernel_calls;
    profile.kernel_calls += (int64_t)launched_kernel_calls;

    profile.matrix_total_us = tsavorite_elapsed_us(matrix_start_us);
    tsavorite_matmul_profile_record(node, profile);

    return GGML_STATUS_SUCCESS;
}

#else

extern "C" void tmu_mul_mat_k32 (const void *A, const void *B, void *C) {
    call_tmu_blob<32>(A, B, C, _mlir_ciface_txe_mul_mat_tile_f32_k32_host);
}
extern "C" void tmu_mul_mat_k64 (const void *A, const void *B, void *C) {
    call_tmu_blob<64>(A, B, C, _mlir_ciface_txe_mul_mat_tile_f32_k64_host);
}
extern "C" void tmu_mul_mat_k128(const void *A, const void *B, void *C) {
    call_tmu_blob<128>(A, B, C, _mlir_ciface_txe_mul_mat_tile_f32_k128_host);
}
extern "C" void tmu_mul_mat_k256(const void *A, const void *B, void *C) {
    call_tmu_blob<256>(A, B, C, _mlir_ciface_txe_mul_mat_tile_f32_k256_host);
}
extern "C" void tmu_mul_mat_k512(const void *A, const void *B, void *C) {
    call_tmu_blob<512>(A, B, C, _mlir_ciface_txe_mul_mat_tile_f32_k512_host);
}
extern "C" void tmu_mul_mat_k1024(const void *A, const void *B, void *C) {
    call_tmu_blob<1024>(A, B, C, _mlir_ciface_txe_mul_mat_tile_f32_k1024_host);
}
extern "C" void tmu_mul_mat_k2048(const void *A, const void *B, void *C) {
    call_tmu_blob<2048>(A, B, C, _mlir_ciface_txe_mul_mat_tile_f32_k2048_host);
}

// ============================================================================
// DISPATCH (ONE PER K BUCKET) — points to ABI-safe wrappers above
// ============================================================================

typedef void (*tmu_mul_mat_tile_fn)(const void *A_tile, const void *B_tile, void *C_tile);

struct tmu_bucket_dispatch {
    int k;
    tmu_mul_mat_tile_fn fn;
};

static const tmu_bucket_dispatch g_tmu_dispatch[] = {
    // Larger K buckets are temporarily disabled (blob size > 64 KB).
#if 0
    { 2048, tmu_mul_mat_k2048 },
    { 1024, tmu_mul_mat_k1024 },
    {  512, tmu_mul_mat_k512  },
    {  256, tmu_mul_mat_k256  },
    {  128, tmu_mul_mat_k128  },
    {   64, tmu_mul_mat_k64   },
#endif
    {   32, tmu_mul_mat_k32   },
    {    0, nullptr           }
};

static inline const tmu_bucket_dispatch *tmu_find_bucket(int k) {
    for (int i = 0; g_tmu_dispatch[i].k != 0; ++i) {
        if (g_tmu_dispatch[i].k == k) return &g_tmu_dispatch[i];
    }
    return nullptr;
}

static inline int tmu_decompose_k(int64_t k, int *out, int max_parts) {
    if (!out || max_parts <= 0) return -1;
    if (k < 0) return -1;  // defensive: never allow negative K

    int n = 0;

    for (int i = 0; g_tmu_dispatch[i].k != 0 && n < max_parts; ++i) {
        const int bucket = g_tmu_dispatch[i].k;

        // Defensive: bucket must be positive
        if (bucket <= 0) return -1;

        while (k >= (int64_t) bucket && n < max_parts) {
            out[n++] = bucket;
            k -= (int64_t) bucket;
            // Defensive: k should never go negative due to the loop condition
            if (k < 0) return -1;
        }
    }

    // Must decompose exactly; otherwise unsupported remainder
    return (k == 0) ? n : -1;
}

// ============================================================================
// PACKING HELPERS (FP32)
// ============================================================================

static inline void pack_A_tile_f32(
    float *A_pack,             // [TMU_M_TILE_MAX * K_chunk]
    const char *A_base_d23,     // src0->data already offset for d2/d3
    int64_t m0,
    int64_t m_tile,
    int64_t k0,
    int64_t K_chunk,
    int64_t a_nb0,
    int64_t a_nb1
) {
    memset(A_pack, 0, (size_t)(TMU_M_TILE_MAX * K_chunk) * sizeof(float));

    for (int64_t r = 0; r < m_tile; ++r) {
        const int64_t m = m0 + r;
        float *dst = A_pack + r * K_chunk;

        const char *src_row = A_base_d23 + m * a_nb1 + k0 * a_nb0;
        for (int64_t kk = 0; kk < K_chunk; ++kk) {
            dst[kk] = *(const float *)(src_row + kk * a_nb0);
        }
    }
}

static inline void pack_B_tile_f32(
    float *B_pack,             // [TMU_N_BLOCK * K_chunk]
    const char *B_base_d23,     // src1->data already offset for d2/d3
    int64_t n0,
    int64_t n_valid,
    int64_t k0,
    int64_t K_chunk,
    int64_t b_nb0,
    int64_t b_nb1
) {
    memset(B_pack, 0, (size_t)(TMU_N_BLOCK * K_chunk) * sizeof(float));

    for (int64_t c = 0; c < n_valid; ++c) {
        const int64_t n = n0 + c;
        float *dst = B_pack + c * K_chunk;

        const char *src_col = B_base_d23 + n * b_nb1 + k0 * b_nb0;
        for (int64_t kk = 0; kk < K_chunk; ++kk) {
            dst[kk] = *(const float *)(src_col + kk * b_nb0);
        }
    }
}

// ============================================================================
// REUSABLE BUFFERS — allocated once per process
// ============================================================================

static float *g_A_pack = nullptr;   // 64 * 2048
static float *g_B_pack = nullptr;   // 32 * 2048
static float *g_C_tile = nullptr;   // 64 * 32
static int    g_pack_maxK = 0;

static std::once_flag g_tmu_buf_once;

static inline void ensure_tmu_pack_buffers() {
    std::call_once(g_tmu_buf_once, []() {
        const int maxK = 2048; // fixed maximum bucket

        g_pack_maxK = maxK;

        g_A_pack = (float *) tsi_alloc((size_t)TMU_M_TILE_MAX * (size_t)maxK * sizeof(float));
        g_B_pack = (float *) tsi_alloc((size_t)TMU_N_BLOCK     * (size_t)maxK * sizeof(float));
        g_C_tile = (float *) tsi_alloc((size_t)TMU_M_TILE_MAX * (size_t)TMU_N_BLOCK * sizeof(float));

        TSAVORITE_GGML_ASSERT(g_A_pack && g_B_pack && g_C_tile);
    });
}

static enum ggml_status ggml_tsavorite_run_tmu_mul_mat(
    struct ggml_backend_tsavorite_context * /*ctx*/,
    txe_device_s device,
    struct ggml_tensor * node,
    enum ggml_tsavorite_kernel_type kernel_type,
    int /*kernel_sub_type*/) {

    if (!node || !node->src[0] || !node->src[1] || !node->data) {
        return GGML_STATUS_FAILED;
    }

    const struct ggml_tensor * src0 = node->src[0];
    const struct ggml_tensor * src1 = node->src[1];

    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || node->type != GGML_TYPE_F32) {
        return GGML_STATUS_FAILED;
    }

    ensure_tmu_pack_buffers();

    // -------------------------------------------------------------------------
    // Host-only buffer to save previous partial C between K buckets.
    // IMPORTANT: This buffer is NOT passed to the TMU blob, so DO NOT use tsi_alloc
    // (tsi_alloc comes from CMA and is not freed until tsi_finalize).
    // Allocate once from normal host heap and reuse.
    // -------------------------------------------------------------------------
    static float *g_C_prev = nullptr;
    static std::once_flag g_prev_once;
    auto host_alloc_aligned = [](size_t bytes) -> void * {
        void *p = nullptr;
        if (posix_memalign(&p, 64, bytes) != 0) {
            p = malloc(bytes);
        }
        return p;
    };
    std::call_once(g_prev_once, [&]() {
        const size_t bytes = (size_t)TMU_M_TILE_MAX * (size_t)TMU_N_BLOCK * sizeof(float);
        g_C_prev = (float *) host_alloc_aligned(bytes);
        TSAVORITE_GGML_ASSERT(g_C_prev);
        memset(g_C_prev, 0, bytes);
    });

#ifdef TMU_DEBUG_VALIDATE
    // -------------------------------------------------------------------------
    // PR comment fix #1 + your request:
    // 1) Make CPU reference helper take MemRefDescriptor-like structs
    // 2) Actually CALL it here (it was dead previously)
    // -------------------------------------------------------------------------
    static void cpu_ref_mul_mat_f32(
        const MemRefDescriptor<4> *A_desc,
        const MemRefDescriptor<4> *B_desc,
        MemRefDescriptor<4>       *C_desc
    ) {
        if (!A_desc || !B_desc || !C_desc) return;
        if (!A_desc->data || !B_desc->data || !C_desc->data) return;

        const int64_t M = A_desc->shape[2];
        const int64_t K = A_desc->shape[3];
        const int64_t N = B_desc->shape[2];

        if (M <= 0 || N <= 0 || K <= 0) return;
        if (B_desc->shape[3] != K) return;
        if (C_desc->shape[2] != M) return;
        if (C_desc->shape[3] != N) return;

        const float *A = (const float *) A_desc->data;
        const float *B = (const float *) B_desc->data;
        float       *C = (float       *) C_desc->data;

        const int64_t a_s2 = A_desc->strides[2];
        const int64_t a_s3 = A_desc->strides[3];
        const int64_t b_s2 = B_desc->strides[2];
        const int64_t b_s3 = B_desc->strides[3];
        const int64_t c_s2 = C_desc->strides[2];
        const int64_t c_s3 = C_desc->strides[3];

        const int64_t a_off = A_desc->offset;
        const int64_t b_off = B_desc->offset;
        const int64_t c_off = C_desc->offset;

        for (int64_t r = 0; r < M; ++r) {
            const int64_t a_row = a_off + r * a_s2;
            const int64_t c_row = c_off + r * c_s2;

            for (int64_t n = 0; n < N; ++n) {
                const int64_t b_row = b_off + n * b_s2;   // B packed as [N,K]

                float acc = 0.0f;
                for (int64_t kk = 0; kk < K; ++kk) {
                    acc += A[a_row + kk * a_s3] * B[b_row + kk * b_s3];
                }
                C[c_row + n * c_s3] = acc;
            }
        }
    }

    // CPU reference buffers: accumulate chunk-by-chunk in packed space
    static float C_ref_chunk[TMU_M_TILE_MAX * TMU_N_BLOCK];
    static float C_ref_accum[TMU_M_TILE_MAX * TMU_N_BLOCK];
#endif

    const int64_t K = src0->ne[0];
    const int64_t M = src0->ne[1];
    const int64_t N = src1->ne[1];

    if (src1->ne[0] != K) return GGML_STATUS_FAILED;
    if ((K % TMU_K_MULTIPLE) != 0) return GGML_STATUS_FAILED;
    if (src1->ne[1] == 1) return GGML_STATUS_FAILED; // avoid GEMV

    const int64_t a_nb0 = nb_or_default(src0, 0);
    const int64_t a_nb1 = nb_or_default(src0, 1);
    const int64_t a_nb2 = nb_or_default(src0, 2);
    const int64_t a_nb3 = nb_or_default(src0, 3);

    const int64_t b_nb0 = nb_or_default(src1, 0);
    const int64_t b_nb1 = nb_or_default(src1, 1);
    const int64_t b_nb2 = nb_or_default(src1, 2);
    const int64_t b_nb3 = nb_or_default(src1, 3);

    const int64_t c_nb0 = nb_or_default(node, 0);
    const int64_t c_nb1 = nb_or_default(node, 1);
    const int64_t c_nb2 = nb_or_default(node, 2);
    const int64_t c_nb3 = nb_or_default(node, 3);

    // broadcast dims must come from inputs
    const int64_t A2 = src0->ne[2] ? src0->ne[2] : 1;
    const int64_t A3 = src0->ne[3] ? src0->ne[3] : 1;
    const int64_t B2 = src1->ne[2] ? src1->ne[2] : 1;
    const int64_t B3 = src1->ne[3] ? src1->ne[3] : 1;

    const int64_t D2 = (A2 > B2) ? A2 : B2;
    const int64_t D3 = (A3 > B3) ? A3 : B3;

    if (device) {
        ++device->stats.op_run_count[kernel_type].total_tensor_count;
    }

    for (int64_t od3 = 0; od3 < D3; ++od3) {
        const int64_t a_d3 = map_repeat_i64(od3, A3);
        const int64_t b_d3 = map_repeat_i64(od3, B3);

        for (int64_t od2 = 0; od2 < D2; ++od2) {
            const int64_t a_d2 = map_repeat_i64(od2, A2);
            const int64_t b_d2 = map_repeat_i64(od2, B2);

            const char *A_base_d23 = (const char *) src0->data + a_d2 * a_nb2 + a_d3 * a_nb3;
            const char *B_base_d23 = (const char *) src1->data + b_d2 * b_nb2 + b_d3 * b_nb3;

            for (int64_t m0 = 0; m0 < M; m0 += TMU_M_TILE_MAX) {
                const int64_t m_tile = (M - m0 > TMU_M_TILE_MAX) ? TMU_M_TILE_MAX : (M - m0);

                for (int64_t n0 = 0; n0 < N; n0 += TMU_N_BLOCK) {
                    const int64_t n_valid = (N - n0 >= TMU_N_BLOCK) ? TMU_N_BLOCK : (N - n0);

                    memset(g_C_tile, 0, (size_t)TMU_M_TILE_MAX * (size_t)TMU_N_BLOCK * sizeof(float));

#ifdef TMU_DEBUG_VALIDATE
                    // reset CPU ref accumulator for THIS output tile
                    memset(C_ref_accum, 0, sizeof(C_ref_accum));
#endif

                    int parts[128];
                    const int np = tmu_decompose_k(K, parts, (int)(sizeof(parts)/sizeof(parts[0])));
                    if (np < 0) return GGML_STATUS_FAILED;

                    int64_t k0 = 0;

                    for (int pi = 0; pi < np; ++pi) {
                        const int K_chunk = parts[pi];
                        const tmu_bucket_dispatch *bucket = tmu_find_bucket(K_chunk);
                        if (!bucket || !bucket->fn) return GGML_STATUS_FAILED;

                        pack_A_tile_f32(g_A_pack, A_base_d23, m0, m_tile, k0, K_chunk, a_nb0, a_nb1);
                        pack_B_tile_f32(g_B_pack, B_base_d23, n0, n_valid, k0, K_chunk, b_nb0, b_nb1);

                        // Save previous partial before calling blob (because blob overwrites)
                        if (pi > 0) {
                            memcpy(g_C_prev, g_C_tile,
                                   (size_t)TMU_M_TILE_MAX * (size_t)TMU_N_BLOCK * sizeof(float));
                        }

                        // Run blob
                        bucket->fn(g_A_pack, g_B_pack, g_C_tile);

                        // Accumulate back (host-side workaround)
                        if (pi > 0) {
                            const int total = TMU_M_TILE_MAX * TMU_N_BLOCK;
                            for (int i = 0; i < total; ++i) {
                                g_C_tile[i] += g_C_prev[i];
                            }
                        }

                        // Stats per kernel call
                        if (device) ++device->stats.op_run_count[kernel_type].num_of_kernel_call;
                        ++node->tsi_kernel_runs;

#ifdef TMU_DEBUG_VALIDATE
                        // -----------------------------------------------------------------
                        // CPU reference for THIS K_chunk computed from PACKED tiles
                        // and accumulated into C_ref_accum, then compared with g_C_tile.
                        // This makes cpu_ref_mul_mat_f32() actually USED.
                        // -----------------------------------------------------------------
                        memset(C_ref_chunk, 0, sizeof(C_ref_chunk));

                        MemRefDescriptor<4> Aref, Bref, Cref;
                        init_memref_4d(Aref, (void*)g_A_pack, 1, 1, TMU_M_TILE_MAX, (int64_t)K_chunk);
                        init_memref_4d(Bref, (void*)g_B_pack, 1, 1, TMU_N_BLOCK,   (int64_t)K_chunk);
                        init_memref_4d(Cref, (void*)C_ref_chunk, 1, 1, TMU_M_TILE_MAX, TMU_N_BLOCK);

                        cpu_ref_mul_mat_f32(&Aref, &Bref, &Cref);

                        // accumulate CPU reference chunks (only valid region is needed)
                        for (int64_t rr = 0; rr < m_tile; ++rr) {
                            for (int64_t cc = 0; cc < n_valid; ++cc) {
                                C_ref_accum[rr * TMU_N_BLOCK + cc] +=
                                    C_ref_chunk[rr * TMU_N_BLOCK + cc];
                            }
                        }

                        // Compare after each chunk (same tolerance you used before)
                        for (int64_t rr = 0; rr < m_tile; ++rr) {
                            for (int64_t cc = 0; cc < n_valid; ++cc) {
                                const float tmu_v = g_C_tile[rr * TMU_N_BLOCK + cc];
                                const float ref_v = C_ref_accum[rr * TMU_N_BLOCK + cc];
                                if (fabsf(tmu_v - ref_v) > 1e-4f) {
                                    fprintf(stderr,
                                        "\nTMU MISMATCH (packed-ref)\n"
                                        "m0=%ld n0=%ld k0=%ld K_chunk=%d pi=%d\n"
                                        "r=%ld c=%ld TMU=%f REF=%f\n",
                                        (long)m0, (long)n0, (long)k0, K_chunk, pi,
                                        (long)rr, (long)cc, tmu_v, ref_v);
                                    tsi_cleanup();
                                    abort();
                                }
                            }
                        }
#endif

                        k0 += K_chunk;
                    }

                    // Copy tile back to ggml output
                    {
                        char *dst_base = (char *) node->data;
                        const size_t bytes_total = (size_t) ggml_nbytes(node);

                        for (int64_t rr = 0; rr < m_tile; ++rr) {
                            const int64_t m_idx = m0 + rr;
                            const float *src_row = g_C_tile + rr * TMU_N_BLOCK;

                            for (int64_t cc = 0; cc < n_valid; ++cc) {
                                const int64_t n_idx = n0 + cc;

                                const int64_t byte_off =
                                    m_idx * c_nb0 +
                                    n_idx * c_nb1 +
                                    od2  * c_nb2 +
                                    od3  * c_nb3;

                                if (byte_off < 0 ||
                                    (size_t)byte_off + sizeof(float) > bytes_total) {
                                    continue;
                                }
                                *(float *)(dst_base + byte_off) = src_row[cc];
                            }
                        }
                    }

                    // If you decide to USE copy_tileC_to_ggml_f32 instead of inline copy-back:
                    // copy_tileC_to_ggml_f32(g_C_tile, m_tile, 0, node, m0, n0, od2, od3);
                }
            }
        }
    }

    return GGML_STATUS_SUCCESS;
}
#endif /* TRITON_MAT_MUL */


static std::mutex g_tsavorite_compute_mutex;

// nodes are intermediate which has multiple src tensors & operation
// Here we create multiple thread
// Each Thread run the command buffer & pick Tensor and execute and get the result back base on
// async or sync all Compute wil finish all tensors execution
static enum ggml_status ggml_tsavorite_graph_compute(ggml_backend_t backend,
                                                     struct ggml_cgraph *cgraph) {
std::lock_guard<std::mutex> _lk(g_tsavorite_compute_mutex);
#if 0
    GGML_LOG_INFO("Start %s\n", __func__);
    struct ggml_backend_tsavorite_context        * ctx     = backend->context;
    struct ggml_backend_tsavorite_device_context * ctx_dev = backend->device->context;

    // number of nodes encoded by the main thread (empirically determined)
    const int n_main = 128;

    // number of threads in addition to the main thread
    const int n_cb = ctx->n_cb;

    // submit the ggml compute graph to the TXE by creating command buffers and encoding the ops in them
    // the first n_nodes_0 are encoded and submitted for processing directly by the calling thread
    // while these nodes are processing, we start n_cb threads to enqueue the rest of the nodes
    // each thread creates it's own command buffer and enqueues the ops in parallel

    GGML_LOG_INFO("End %s\n", __func__);
    return GGML_STATUS_SUCCESS;
#endif

  struct ggml_backend_tsavorite_context *ctx =
      (struct ggml_backend_tsavorite_context *)backend->context;
  if (!ctx) {
    GGML_LOG_ERROR("\n backend ctx is NULL \n");
    return GGML_STATUS_FAILED;
  }

#if 0
    struct ggml_cplan cplan = ggml_graph_plan(cgraph, ctx->n_threads, ctx->threadpool);

    if (ctx->work_size < cplan.work_size) {
        delete[] ctx->work_data;
        ctx->work_data = new uint8_t[cplan.work_size];
        if (ctx->work_data == NULL) {
            ctx->work_size = 0;
            return GGML_STATUS_ALLOC_FAILED;
        }
        ctx->work_size = cplan.work_size;
    }
    cplan.work_data = (uint8_t *)ctx->work_data;

    cplan.abort_callback      = ctx->abort_callback;
    cplan.abort_callback_data = ctx->abort_callback_data;
#endif

  txe_device_s device = ggml_backend_tsavorite_device_acq(
      (struct ggml_backend_tsavorite_device_context *)backend->device->context);

  if (!device) {
    GGML_TSAVORITE_LOG_ERROR("\n tsavorite device is NULL \n");
    return GGML_STATUS_FAILED;
  }
  // MemRefDescriptor
  MemRefDescriptor<Rank> *srcP0, *srcP1, *nodeP;
  struct ggml_tensor *src0, *src1, *node;
  uint32_t num_elem_src0, num_elem_src1, num_elem_node;
  enum ggml_tsavorite_kernel_type kernel_type;
  // This variable not needed since src0 or node will have max elem size
  //  and src1 size will min elem size
  uint64_t max_num_of_elem, min_num_of_elem;
  enum ggml_tsavorite_input_tensors_count num_of_input_tensors;
  tensor_log log_data;


  for (int i = 0; i < cgraph->n_nodes; i++) {
     int32_t kernel_sub_type=-1;
#if defined(GGML_PERF) || defined(GGML_PERF_RELEASE) || defined(GGML_PERF_DETAIL)
    int64_t t_start = ggml_time_us();
#endif /* GGML_PERF-related flags */
    node = cgraph->nodes[i];
#if defined(GGML_PERF) || defined(GGML_PERF_RELEASE) || defined(GGML_PERF_DETAIL)
    // tsi_kernel_runs must reflect only this pass's real kernel launches (if
    // any of the increment sites below fire for this node). Clear it before
    // dispatch so a node whose kernel_type never launches a real TXE blob
    // (e.g. CONT/SOFT_MAX/GET_ROWS taking the CPU-fallback path) reports 0
    // instead of carrying over an unrelated value.
    node->tsi_kernel_runs = 0;
#endif /* GGML_PERF-related flags */
    src0 = node->src[0];
    src1 = node->src[1];
    min_num_of_elem = 0;
    max_num_of_elem = 0;
    if(node->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && (!src1 || src1->type == GGML_TYPE_F32))
	    kernel_sub_type = DATA_TYPE_F32_INDEX;
    /*
     * FP16 support is being qualified and is work in progress
     */
    if(node->type == GGML_TYPE_F16 && src0->type == GGML_TYPE_F16 && (!src1 || src1->type == GGML_TYPE_F16))
	    kernel_sub_type = DATA_TYPE_F16_INDEX;

    if (node->op == GGML_OP_RMS_NORM ||  node->op == GGML_OP_SOFT_MAX || node->op == GGML_OP_ROPE || node->op == GGML_OP_ROPE_BACK) {
        if (!glob_buf) {
            GGML_TSAVORITE_LOG_ERROR("tsi_alloc failied for creating memory for buf \n");
            return GGML_STATUS_ABORTED;
        }
        glob_buf->offset = 0;
        glob_buf->data   = glob_buf->base = (void *)(glob_buf+1);

        float *vall = (float *)glob_buf->data;
        int ii;
        for(ii=0; ii <= 95; ++ii)
               vall[ii] = 0;
    }
    struct ggml_compute_params params = {
       .ith = 0,
       .nth = 1,
       .wsize = 1,
       .wdata = glob_buf->data,
       .threadpool = global_threadpool,
    };
    switch (node->op) {
    case GGML_OP_SET_ROWS:
      kernel_type = GGML_TSAVORITE_KERNEL_TYPE_SET_ROWS;
      num_of_input_tensors = TSAVORITE_IGNORE_TENSORS;
      ggml_compute_forward_set_rows(&params, node);
      break;
    case GGML_OP_GET_ROWS:
      kernel_type = GGML_TSAVORITE_KERNEL_TYPE_GET_ROWS;
      num_of_input_tensors = TSAVORITE_IGNORE_TENSORS;
      ggml_compute_forward_get_rows(&params, node);
      break;
    case GGML_OP_GET_ROWS_BACK:
      kernel_type = GGML_TSAVORITE_KERNEL_TYPE_GET_ROWS_BACK;
      num_of_input_tensors = TSAVORITE_IGNORE_TENSORS;
      ggml_compute_forward_get_rows_back(&params, node);
      break;
    case GGML_OP_ROPE:
      kernel_type = GGML_TSAVORITE_KERNEL_TYPE_ROPE;
      num_of_input_tensors = TSAVORITE_IGNORE_TENSORS;
      ggml_compute_forward_rope(&params, node);
      break;
    case GGML_OP_ROPE_BACK:
      kernel_type = GGML_TSAVORITE_KERNEL_TYPE_ROPE_BACK;
      num_of_input_tensors = TSAVORITE_IGNORE_TENSORS;
      ggml_compute_forward_rope_back(&params, node);
      break;
    case GGML_OP_ADD:
      kernel_type = GGML_TSAVORITE_KERNEL_TYPE_ADD;
      num_of_input_tensors = TSAVORITE_TWO_INPUT_TENSORS;
      break;
    case GGML_OP_SUB:
      kernel_type = GGML_TSAVORITE_KERNEL_TYPE_SUB;
      num_of_input_tensors = TSAVORITE_TWO_INPUT_TENSORS;
      break;
    case GGML_OP_MUL:
      kernel_type = GGML_TSAVORITE_KERNEL_TYPE_MULT;
      num_of_input_tensors = TSAVORITE_TWO_INPUT_TENSORS;
      break;
    case GGML_OP_DIV:
      kernel_type = GGML_TSAVORITE_KERNEL_TYPE_DIV;
      num_of_input_tensors = TSAVORITE_TWO_INPUT_TENSORS;
      break;
    case GGML_OP_SQRT:
      kernel_type = GGML_TSAVORITE_KERNEL_TYPE_SQRT;
      num_of_input_tensors = TSAVORITE_UNARY_INPUT_TENSORS;
      break;
    case GGML_OP_SQR:
      kernel_type = GGML_TSAVORITE_KERNEL_TYPE_SQR;
      num_of_input_tensors = TSAVORITE_UNARY_INPUT_TENSORS;
      break;
    case GGML_OP_SIN:
      kernel_type = GGML_TSAVORITE_KERNEL_TYPE_SIN;
      num_of_input_tensors = TSAVORITE_UNARY_INPUT_TENSORS;
      break;
    case GGML_OP_RMS_NORM:
      kernel_type = GGML_TSAVORITE_KERNEL_TYPE_RMS_NORM;
      num_of_input_tensors = TSAVORITE_UNARY_INPUT_TENSORS;
      break;
    case GGML_OP_SOFT_MAX:
      kernel_type = GGML_TSAVORITE_KERNEL_TYPE_SOFT_MAX;
      //num_of_input_tensors = TSAVORITE_TWO_INPUT_TENSORS;
      num_of_input_tensors = TSAVORITE_IGNORE_TENSORS;
      ggml_compute_forward_soft_max(&params, node);
      break;
    case GGML_OP_MUL_MAT:
      kernel_type = GGML_TSAVORITE_KERNEL_TYPE_MUL_MAT;
      num_of_input_tensors = TSAVORITE_TWO_INPUT_TENSORS;
      // Disabling the tensor ignore and calling of CPU ops
      // and use Tsavorite backend instead
#ifdef GGML_MUL_MAT_CPU_OPS
      // num_of_input_tensors = TSAVORITE_IGNORE_TENSORS;
      // ggml_compute_forward_mul_mat(&params, node);
#endif
      break;
    case GGML_OP_FLASH_ATTN_EXT:
      kernel_type = GGML_TSAVORITE_KERNEL_TYPE_FLASH_ATTN_EXT;
      //num_of_input_tensors = TSAVORITE_TWO_INPUT_TENSORS;
      num_of_input_tensors = TSAVORITE_IGNORE_TENSORS;
      ggml_compute_forward_flash_attn_ext(&params, node);
      break;
    case GGML_OP_GLU:
      kernel_type = tsi_glu_kernel_type(node);
      if (!src1)
          src1 = src0;
      if (kernel_type == GGML_TSAVORITE_KERNEL_TYPE_COUNT) {
        GGML_TSAVORITE_LOG_ERROR("\n GGML_OP_GLU sub type is not correct \n");
        return GGML_STATUS_ABORTED;
      }
      num_of_input_tensors = TSAVORITE_TWO_INPUT_TENSORS;
      break;
    case GGML_OP_RESHAPE:
      kernel_type = GGML_TSAVORITE_KERNEL_TYPE_RESHAPE;
      num_of_input_tensors = TSAVORITE_IGNORE_TENSORS;
      ggml_compute_forward_reshape(&params, node);
      break;
    case GGML_OP_VIEW:
      kernel_type = GGML_TSAVORITE_KERNEL_TYPE_VIEW;
      num_of_input_tensors = TSAVORITE_IGNORE_TENSORS;
      ggml_compute_forward_reshape(&params, node);
      break;
    case GGML_OP_PERMUTE:
      kernel_type = GGML_TSAVORITE_KERNEL_TYPE_PERMUTE;
      num_of_input_tensors = TSAVORITE_IGNORE_TENSORS;
      ggml_compute_forward_permute(&params, node);
      break;
    case GGML_OP_TRANSPOSE:
      kernel_type = GGML_TSAVORITE_KERNEL_TYPE_TRANSPOSE;
      num_of_input_tensors = TSAVORITE_IGNORE_TENSORS;
      ggml_compute_forward_transpose(&params, node);
      break;
    case GGML_OP_SET:
      kernel_type = GGML_TSAVORITE_KERNEL_TYPE_SET;
      num_of_input_tensors = TSAVORITE_IGNORE_TENSORS;
      ggml_compute_forward_set(&params, node);
      break;
    case GGML_OP_CPY:
      kernel_type = GGML_TSAVORITE_KERNEL_TYPE_CPY;
      num_of_input_tensors = TSAVORITE_IGNORE_TENSORS;
      ggml_compute_forward_cpy(&params, node);
      break;
    case GGML_OP_CONT:
      kernel_type = GGML_TSAVORITE_KERNEL_TYPE_CONT;
      num_of_input_tensors = TSAVORITE_IGNORE_TENSORS;
      ggml_compute_forward_cont(&params, node);
      break;

    case GGML_OP_UNARY:
      switch (ggml_get_unary_op(node)) {
      case GGML_UNARY_OP_NEG:
        kernel_type = GGML_TSAVORITE_KERNEL_TYPE_NEG;
        num_of_input_tensors = TSAVORITE_UNARY_INPUT_TENSORS;
        break;
      case GGML_UNARY_OP_ABS:
        kernel_type = GGML_TSAVORITE_KERNEL_TYPE_ABS;
        num_of_input_tensors = TSAVORITE_UNARY_INPUT_TENSORS;
        break;
      case GGML_UNARY_OP_SIGMOID:
        kernel_type = GGML_TSAVORITE_KERNEL_TYPE_SIGMOID;
        num_of_input_tensors = TSAVORITE_UNARY_INPUT_TENSORS;
        break;
      case GGML_UNARY_OP_SILU:
        kernel_type = GGML_TSAVORITE_KERNEL_TYPE_SILU;
        num_of_input_tensors = TSAVORITE_UNARY_INPUT_TENSORS;
        break;
      default:
        ggml_backend_tsavorite_device_rel(
            (struct ggml_backend_tsavorite_device_context *)backend->device->context);
        return GGML_STATUS_ABORTED;
      }
      break;
    default:
      ggml_backend_tsavorite_device_rel(
          (struct ggml_backend_tsavorite_device_context *)backend->device->context);
      return GGML_STATUS_ABORTED;
    }

    if ((num_of_input_tensors != TSAVORITE_IGNORE_TENSORS) && (!ctx->kernels[kernel_type].pipeline ||
        (!ctx->kernels[kernel_type].pipeline->_mlir_fptr_3_input[kernel_sub_type] &&
         !ctx->kernels[kernel_type].pipeline->_mlir_fptr_2_input[kernel_sub_type] &&
         !ctx->kernels[kernel_type].pipeline->_mlir_fptr_1_input[kernel_sub_type]))) {
      GGML_TSAVORITE_LOG_ERROR("Kernel Type %d, not supported \n", kernel_type);
      return GGML_STATUS_ABORTED;
    }
    ++num_of_op;

    if (num_of_input_tensors == TSAVORITE_TWO_INPUT_TENSORS) {
      if (node->src[0] && node->src[1]) {
        if (!src0->data || !src1->data || !node->data) {
          GGML_TSAVORITE_LOG_ERROR(
              "One of tensor Data doesnt have memory leaf1 %p, leaf2 %p, node %p \n", src0->data,
              src1->data, node->data);
          ggml_backend_tsavorite_device_rel(
              (struct ggml_backend_tsavorite_device_context *)backend->device->context);
          return GGML_STATUS_ABORTED;
        }
        srcP0 = (MemRefDescriptor<Rank> *)src0->data;
        srcP1 = (MemRefDescriptor<Rank> *)src1->data;
        nodeP = (MemRefDescriptor<Rank> *)node->data;
        // This is for tsavorite MemRef Header hence getting header
        --srcP0;
        --srcP1;
        --nodeP;
        srcP0->data = srcP0->base = src0->data;
        srcP1->data = srcP1->base = src1->data;
        nodeP->data = nodeP->base = node->data;
        srcP0->offset = 0;
        srcP1->offset = 0;
        nodeP->offset = 0;

        num_elem_src0 = 1;
        for (int i = 0; i < GGML_MAX_DIMS && src0->nb[i] != 0; ++i)
          num_elem_src0 *= src0->ne[i];

        num_elem_src1 = 1;
        for (int i = 0; i < GGML_MAX_DIMS && src1->nb[i] != 0; ++i)
          num_elem_src1 *= src1->ne[i];

        num_elem_node = 1;
        for (int i = 0; i < GGML_MAX_DIMS && node->nb[i] != 0; ++i)
          num_elem_node *= node->ne[i];

        if (!num_elem_src0 || !num_elem_src1 || !num_elem_node) {
          GGML_TSAVORITE_LOG_ERROR("\nOne or more of Tensor length is zero of kernel_type %d\n",
                                   kernel_type);
          return GGML_STATUS_ABORTED;
        }

        min_num_of_elem = max_num_of_elem = num_elem_src0;

        if (min_num_of_elem > num_elem_src1)
          min_num_of_elem = num_elem_src1;
        if (min_num_of_elem > num_elem_node)
          min_num_of_elem = num_elem_node;

        if (max_num_of_elem < num_elem_src1)
          max_num_of_elem = num_elem_src1;
        if (max_num_of_elem < num_elem_node)
          max_num_of_elem = num_elem_node;

        if (ggml_tsavorite_log_type_val == GGML_TSAVORITE_LOG_DEBUG) {
          bzero((char *)&log_data, sizeof(log_data));
          log_data.leaf1_len = num_elem_src0;
          log_data.leaf2_len = num_elem_src1;
          log_data.node_len = num_elem_node;
          log_data.log_file = tsi_op_log_file;
          log_data.num_of_op = num_of_op;
          //log_data.kernel_type = kernel_type;
          log_data.kernel_type = node->op;

          log_data.data_type = GGML_TSAVORITE_TENSOR_HEADER;
          ggml_tsi_log_tensor_data(log_data);

          log_data.data_type = GGML_TSAVORITE_TENSOR_LEAF1;
          log_data.tensor = src0;
          ggml_tsi_log_tensor_data(log_data);

          log_data.data_type = GGML_TSAVORITE_TENSOR_LEAF2;
          log_data.tensor = src1;
          ggml_tsi_log_tensor_data(log_data);
        }

        ggml_tensor *dst = node;
        const int nr = ggml_nrows(src0);

	/* The current SoftMax implementation does not consider the src2 input,
         * as none of the popular models we currently use require it.
         * However, for future enhancements to SOFT_MAX, we plan to support src2
         * for sinking-based maximization. In that case, src2 will be used to
         * recalculate the maximum value.
         */
        if( kernel_type == GGML_TSAVORITE_KERNEL_TYPE_SOFT_MAX) {
	    const ggml_tensor * src2 = dst->src[2];
	    float scale    = 1.0f;
	    float max_bias = 0.0f;

	    memcpy(&scale,    (float *) dst->op_params + 0, sizeof(float));
	    memcpy(&max_bias, (float *) dst->op_params + 1, sizeof(float));

	    GGML_TENSOR_UNARY_OP_LOCALS

	    const int64_t nb11 = src1 ? src1->nb[1] : 1;
            const int64_t nb12 = src1 ? src1->nb[2] : 1;
            const int64_t nb13 = src1 ? src1->nb[3] : 1;

            const int64_t ne12 = src1 ? src1->ne[2] : 1;
            const int64_t ne13 = src1 ? src1->ne[3] : 1;
        
            // TODO: is this supposed to be ceil instead of floor?
            const uint32_t n_head      = ne02;
            const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

	    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
	    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

	    const bool use_f16 = (src1 && src1->type == GGML_TYPE_F16);

	    // sinks
            const float * sk = src2 ? (float *)((char *) src2->data) : nullptr;
	    //here src2 is NULL for particular model hence u can ignore this for now
	    if (src2) {
		    printf("\n  src2 is not null for SOFT_MAX\n");
	    }
            for (int64_t i03 = 0; i03 < ne03; i03++) {
                for (int64_t i02 = 0; i02 < ne02; i02++) {
                    for (int64_t i01 = 0; i01 < ne01; i01 += 1) {
                        const int64_t i11 = i01;
                        const int64_t i12 = i02%ne12;
                        const int64_t i13 = i03%ne13;

                        // ALiBi
                        const uint32_t h = i02; // head
                        const float slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1) : 1.0f;

                        float * sp = (float *)((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
                        float * dp = (float *)((char *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3);

                        // broadcast the mask across rows
                        ggml_fp16_t * mp_f16 = src1 ? (ggml_fp16_t *)((char *) src1->data + i11*nb11 + i12*nb12 + i13*nb13) : NULL;
                        float       * mp_f32 = src1 ? (float       *)((char *) src1->data + i11*nb11 + i12*nb12 + i13*nb13) : NULL;

                        srcP0->shape[0]   = ne00;
                        srcP1->shape[0]   = ne00;
                        nodeP->shape[0]   = ne00;
                        srcP1->data =  srcP1->base = (void *)(mp_f32);
                        srcP0->data =  srcP0->base = (void *)(sp);
                        nodeP->data =  nodeP->base = (void *)(dp);

                        float *val = (float *)glob_buf->data;
                        val[0] = scale;
                        ctx->kernels[kernel_type].pipeline->_mlir_fptr_3_input[kernel_sub_type](srcP0, srcP1, nodeP, glob_buf);
                        ++device->stats.op_run_count[kernel_type].num_of_kernel_call;
                        ++node->tsi_kernel_runs;
	            }
	        }
	    }
        } else {
            if( kernel_type == GGML_TSAVORITE_KERNEL_TYPE_MUL_MAT) {
                if (ggml_tsavorite_run_tmu_mul_mat(ctx, device, node, kernel_type, kernel_sub_type)  != GGML_STATUS_SUCCESS)
                    return GGML_STATUS_FAILED;

	    } else {

            GGML_TENSOR_BINARY_OP_LOCALS
            for (int ir = 0; ir < nr; ++ir) {
                const int64_t i03 = ir / (ne02 * ne01);
                const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
                const int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

                const int64_t i13 = i03 % ne13;
                const int64_t i12 = i02 % ne12;
                const int64_t i11 = i01 % ne11;
                const int64_t nr0 = ne00 / ne10;

                float *dst_ptr = (float *)((char *)dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1);
                float *src0_ptr = (float *)((char *)src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01);
                float *src1_ptr = (float *)((char *)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11);

                // The following below code operates exclusively on Rank 0
	        // (i.e., the first dimension) for all blob-related processing.

                for (int64_t r = 0; r < nr0; ++r) {
                   memset(srcP0, 0, sizeof(MemRefDescriptor<Rank>));
                   memset(srcP1, 0, sizeof(MemRefDescriptor<Rank>));
                   memset(nodeP, 0, sizeof(MemRefDescriptor<Rank>));



                    srcP0->shape[0]   = ne10;
                    srcP0->offset     = 0;

                    srcP1->shape[0]   = ne10;
                    srcP1->offset     = 0;

                    nodeP->shape[0]   = ne10;
                    nodeP->offset     = 0;

                    srcP1->data =  srcP1->base = (void *)(src1_ptr);
                    srcP0->data =  srcP0->base = (void *)(src0_ptr + r * ne10);
                    nodeP->data =  nodeP->base = (void *)(dst_ptr + r * ne10);
                    // kernel call
#if TRITON_ADD
                    if (kernel_type == GGML_TSAVORITE_KERNEL_TYPE_ADD) {
                        // MemRefDescriptor
                        int32_t *scalar_val; 
                        srcP0->strides[0] = 1;
                        srcP1->strides[0] = 1;
                        nodeP->strides[0] = 1;
                        MemRefDescriptor<Rank> *scalar_loop;
                        MemRefDescriptor<Rank> *scalar_grid1;
                        MemRefDescriptor<Rank> *scalar_grid2;
                        MemRefDescriptor<Rank> *scalar_grid3;

                        scalar_loop = (MemRefDescriptor<Rank> *)scalar_loop_args[0];
                        scalar_grid1 = (MemRefDescriptor<Rank> *)scalar_grid1_args[0];
                        scalar_grid2 = (MemRefDescriptor<Rank> *)scalar_grid2_args[0];
                        scalar_grid3 = (MemRefDescriptor<Rank> *)scalar_grid3_args[0];

                        memset(scalar_loop, 0, sizeof(MemRefDescriptor<Rank>));
                        memset(scalar_grid1, 0, sizeof(MemRefDescriptor<Rank>));
                        memset(scalar_grid2, 0, sizeof(MemRefDescriptor<Rank>));
                        memset(scalar_grid3, 0, sizeof(MemRefDescriptor<Rank>));

                        scalar_loop->shape[0] = (int32_t)srcP0->shape[0] +1;
                        scalar_loop->data = scalar_loop->base = (void *)(scalar_loop+1);
                        scalar_loop->offset = 0;

                        scalar_val = (int32_t *)(scalar_loop+1);
                        *scalar_val = (int32_t)srcP0->shape[0];

                        scalar_grid1->shape[0] = 1;
                        scalar_grid1->data = scalar_grid1->base = (void *)(scalar_grid1 +1);
                        scalar_grid1->offset = 0;

                        scalar_val = (int32_t *)(scalar_grid1+1);
                        *scalar_val = 1;

                        scalar_grid2->shape[0] = 1;
                        scalar_grid2->data = scalar_grid2->base = (void *)(scalar_grid2 +1);
                        scalar_grid2->offset = 0;

                        scalar_val = (int32_t *)(scalar_grid2+1);
                        *scalar_val = 1;

                        scalar_grid3->shape[0] = 1;
                        scalar_grid3->data = scalar_grid3->base = (void *)(scalar_grid3 +1);
                        scalar_grid3->offset = 0;

                        scalar_val = (int32_t *)(scalar_grid3+1);
                        *scalar_val = 1;

                        //ctx->kernels[kernel_type].pipeline->_mlir_fptr_3_input[kernel_sub_type](srcP0, srcP1, nodeP, scalar_loop);
                        static MemRefDescriptor<Rank_Triton> *scalar_max_txes = nullptr;
                        static int32_t *scalar_max_txes_payload = nullptr;

                        if (!scalar_max_txes) {
                            scalar_max_txes = (MemRefDescriptor<Rank_Triton> *) tsi_alloc(sizeof(MemRefDescriptor<Rank_Triton>));
                            scalar_max_txes_payload = (int32_t *) tsi_alloc(TRITON_MATMUL_ALIGNMENT_BYTES);
                            TSAVORITE_GGML_ASSERT(scalar_max_txes);
                            TSAVORITE_GGML_ASSERT(scalar_max_txes_payload);
                        }

                        init_scalar_i32_memref_aligned(scalar_max_txes, scalar_max_txes_payload, (int32_t)num_of_txes);

                        _mlir_ciface_add_kernel_device_wrapper_triton_dispatch(srcP0, srcP1, nodeP,
                                scalar_loop, scalar_grid1, scalar_grid2, scalar_grid3, scalar_max_txes);
                    } else {
#endif /* TRITON_ADD */
                        ctx->kernels[kernel_type].pipeline->_mlir_fptr_2_input[kernel_sub_type](srcP0, srcP1, nodeP);
#if TRITON_ADD
                    }
#endif /* TRITON_ADD */
                    ++device->stats.op_run_count[kernel_type].num_of_kernel_call;
                    ++node->tsi_kernel_runs;
                }
            }
        }
        }

        if (ggml_tsavorite_log_type_val == GGML_TSAVORITE_LOG_DEBUG) {
          log_data.data_type = GGML_TSAVORITE_TENSOR_NODE;
          log_data.tensor = node;
          ggml_tsi_log_tensor_data(log_data);

          log_data.data_type = GGML_TSAVORITE_TENSOR_END_DATA;
          log_data.tensor = NULL;
          ggml_tsi_log_tensor_data(log_data);
        }
      }
    }

    if (num_of_input_tensors == TSAVORITE_UNARY_INPUT_TENSORS) {
      if (node->src[0]) {
        if (!src0->data || !node->data) {
          GGML_TSAVORITE_LOG_ERROR(
              "input or output tensor Data doesnt have memory leaf %p,  node %p \n", src0->data,
              node->data);
          ggml_backend_tsavorite_device_rel(
              (struct ggml_backend_tsavorite_device_context *)backend->device->context);
          return GGML_STATUS_ABORTED;
        }
        srcP0 = (MemRefDescriptor<Rank> *)src0->data;
        nodeP = (MemRefDescriptor<Rank> *)node->data;
        // This is for tsavorite MemRef Header hence getting header
        --srcP0;
        --nodeP;
        srcP0->data = srcP0->base = src0->data;
        nodeP->data = nodeP->base = node->data;
        srcP0->offset = 0;
        nodeP->offset = 0;

        num_elem_src0 = 1;
        for (int i = 0; i < GGML_MAX_DIMS && src0->nb[i] != 0; ++i)
          num_elem_src0 *= src0->ne[i];
        max_num_of_elem = min_num_of_elem = num_elem_src0;

        if (ggml_tsavorite_log_type_val == GGML_TSAVORITE_LOG_DEBUG) {
          bzero((char *)&log_data, sizeof(log_data));
          log_data.leaf1_len = num_elem_src0;
          log_data.leaf2_len = 0;
          log_data.node_len = num_elem_src0;
          log_data.log_file = tsi_op_log_file;
          log_data.num_of_op = num_of_op;
          //log_data.kernel_type = kernel_type;
          log_data.kernel_type = node->op;

          log_data.data_type = GGML_TSAVORITE_TENSOR_HEADER;
          ggml_tsi_log_tensor_data(log_data);

          log_data.data_type = GGML_TSAVORITE_TENSOR_LEAF1;
          log_data.tensor = src0;
          ggml_tsi_log_tensor_data(log_data);
        }

        if (node->op == GGML_OP_SIN) {
          ggml_tsavorite_decompose_unary_kernel(num_elem_src0, src0, node);
        }

        srcP0->data = srcP0->base = (void *)((float *)src0->data);
        nodeP->data = nodeP->base = (void *)((float *)node->data);

        // The following below code operates exclusively on Rank 0
	// (i.e., the first dimension) for all blob-related processing.
        srcP0->shape[0]    = num_elem_src0;
        nodeP->shape[0]    = num_elem_src0;
        srcP0->strides[0]  = 0;
        nodeP->strides[0]  = 0;

	if (kernel_type == GGML_TSAVORITE_KERNEL_TYPE_RMS_NORM) {
	// tsi_alloc is invoked within the function below.
        // We allocate 64 elements for RMS normalization used in the RMS kernel.
        // Although only 32 elements are strictly necessary, reducing this would require changes to the RMS kernel.
        // The remaining 32 elements are used to store src0->ne[0], replicated across each of the last 32 entries.


            float *val = (float *)glob_buf->data;
            int i;
            for(i=64; i <= 95; ++i)
                    val[i] = node->ne[0];

	    int max_dim_index = GGML_MAX_DIMS -1;
	    int strides = 1;
	    bool flag = true;
            for ( i = 0; i <= max_dim_index  && src0->nb[i] != 0; ++i) {
                if (src0->ne[i] == 0) {
                    srcP0->shape[max_dim_index - i]    = 1;
                    nodeP->shape[max_dim_index - i]    = 1;
		    flag = false;
                }
                else  {
                    srcP0->shape[max_dim_index - i]    = src0->ne[i];
                    nodeP->shape[max_dim_index - i]    = node->ne[i];
                }
                srcP0->strides[max_dim_index - i]    = strides;
                nodeP->strides[max_dim_index - i]    = strides;

		// avoiding the case when src0->ne[i] is zero
		if (flag)
			strides = strides * src0->ne[i];
	    }

            ctx->kernels[kernel_type].pipeline->_mlir_fptr_2_input[kernel_sub_type](srcP0, nodeP, glob_buf);

        }
        else {
            // kernel call
            ctx->kernels[kernel_type].pipeline->_mlir_fptr_1_input[kernel_sub_type](srcP0, nodeP);
	}
        ++device->stats.op_run_count[kernel_type].num_of_kernel_call;
        ++node->tsi_kernel_runs;

        if (ggml_tsavorite_log_type_val == GGML_TSAVORITE_LOG_DEBUG) {
          log_data.data_type = GGML_TSAVORITE_TENSOR_NODE;
          log_data.tensor = node;
          ggml_tsi_log_tensor_data(log_data);

          log_data.data_type = GGML_TSAVORITE_TENSOR_END_DATA;
          log_data.tensor = NULL;
          ggml_tsi_log_tensor_data(log_data);
        }
      }
    }
    if (min_num_of_elem > 0
		    || (kernel_type == GGML_TSAVORITE_KERNEL_TYPE_SET_ROWS) || (kernel_type == GGML_TSAVORITE_KERNEL_TYPE_GET_ROWS)
		    || (kernel_type == GGML_TSAVORITE_KERNEL_TYPE_GET_ROWS_BACK) || (kernel_type == GGML_TSAVORITE_KERNEL_TYPE_ROPE)
		    || (kernel_type == GGML_TSAVORITE_KERNEL_TYPE_ROPE_BACK) || (kernel_type == GGML_TSAVORITE_KERNEL_TYPE_TRANSPOSE)
		    || (kernel_type == GGML_TSAVORITE_KERNEL_TYPE_RESHAPE) || (kernel_type == GGML_TSAVORITE_KERNEL_TYPE_VIEW)
		    || (kernel_type == GGML_TSAVORITE_KERNEL_TYPE_CPY) || (kernel_type == GGML_TSAVORITE_KERNEL_TYPE_SET)
		    || (kernel_type == GGML_TSAVORITE_KERNEL_TYPE_CONT) || (kernel_type == GGML_TSAVORITE_KERNEL_TYPE_MUL_MAT)
		    || (kernel_type == GGML_TSAVORITE_KERNEL_TYPE_SOFT_MAX) || (kernel_type == GGML_TSAVORITE_KERNEL_TYPE_FLASH_ATTN_EXT)) {
      ++device->stats.op_run_count[kernel_type].total_tensor_count;

      if (!(device->stats.op_run_count[kernel_type].min_num_of_elem) ||
          device->stats.op_run_count[kernel_type].min_num_of_elem > min_num_of_elem)
        device->stats.op_run_count[kernel_type].min_num_of_elem = min_num_of_elem;

      if (!(device->stats.op_run_count[kernel_type].max_num_of_elem) ||
          device->stats.op_run_count[kernel_type].max_num_of_elem < max_num_of_elem)
        device->stats.op_run_count[kernel_type].max_num_of_elem = max_num_of_elem;
    }
#if defined(GGML_PERF) || defined(GGML_PERF_RELEASE) || defined(GGML_PERF_DETAIL)
    int64_t t_end = ggml_time_us();
    node->perf_runs++;
    node->ggml_compute_backend = GGML_COMPUTE_BACKEND_TSAVORITE;
    if (t_end >= t_start) {
        node->perf_time_us += (t_end - t_start);
    } else {
        // Handle wraparound by assuming timer rolls over at max int64_t value
        node->perf_time_us += (INT64_MAX - t_start + t_end + 1);
    }
#endif /* GGML_PERF-related flags */
    join_all_workers();
  } /* this is main for loop */



  // This this need to implement correctly when we have mixture of CPU and accelerator operation
  // return ggml_graph_compute(cgraph, &cplan);
  ggml_backend_tsavorite_device_rel(
      (struct ggml_backend_tsavorite_device_context *)backend->device->context);

  join_all_workers();
  return GGML_STATUS_SUCCESS;

  GGML_UNUSED(backend);
}

////////////////////////////////////////////////////////////////////////////////

// backend interface

#if 0
static const char * ggml_backend_tsavorite_buffer_get_name(ggml_backend_buffer_t buffer) {
    GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);

    GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
    return "tSavorite";

    TSI_UNUSED(buffer);
}
#endif

static void ggml_backend_tsavorite_buffer_free_buffer(ggml_backend_buffer_t buffer) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  struct ggml_backend_tsavorite_buffer_context *ctx =
      (struct ggml_backend_tsavorite_buffer_context *)buffer->context;

#if 0
    // ctx->all_data & tsi_buffer_free(ctx->buffers[i].data and same memory and created by tsi_alloc
    // tsi_finalize called when ggml call backend free all memory
    // this fucntion called when ggml free backend particular buffer, currently we cant provide this support
    // and just return NoOps
    // But at end there is no memory leak but memory can grow since we free at last once backend is shutdown
    // We need to revisit this hence i kept the stuff under if 0
    for (int i = 0; i < ctx->n_buffers; i++) {
        tsi_buffer_free(ctx->buffers[i].data);
    }
    ggml_backend_tsavorite_device_rel((struct ggml_backend_tsavorite_device_context *)buffer->buft->device->context);

    if (ctx->owned) {
        free(ctx->all_data);
    }
#endif

  free(ctx);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
}

static void *ggml_backend_tsavorite_buffer_get_base(ggml_backend_buffer_t buffer) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  struct ggml_backend_tsavorite_buffer_context *ctx =
      (struct ggml_backend_tsavorite_buffer_context *)buffer->context;

  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return ctx->all_data;
}

static ggml_status ggml_backend_tsavorite_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                                      struct ggml_tensor *tensor) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  MemRefDescriptor<Rank> tensor_data_header;
  tensor->data = (void *)(sizeof(tensor_data_header) + (char *)tensor->data);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return GGML_STATUS_SUCCESS;

  TSI_UNUSED(buffer);
}

static void ggml_backend_tsavorite_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                                        struct ggml_tensor *tensor, uint8_t value,
                                                        size_t offset, size_t size) {
  if (!tensor || !tensor->data) {
    GGML_TSAVORITE_LOG_ERROR("\n tensor or data cant be null under func: %s\n", __func__);
    return;
  }
  memset((char *)tensor->data + offset, value, size);

  GGML_UNUSED(buffer);
}

static void ggml_backend_tsavorite_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                     struct ggml_tensor *tensor, const void *data,
                                                     size_t offset, size_t size) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  memcpy((char *)tensor->data + offset, data, size);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);

  TSI_UNUSED(buffer);
}

static void ggml_backend_tsavorite_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                     const struct ggml_tensor *tensor, void *data,
                                                     size_t offset, size_t size) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  memcpy(data, (const char *)tensor->data + offset, size);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);

  TSI_UNUSED(buffer);
}

static bool ggml_backend_tsavorite_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                                     const struct ggml_tensor *src,
                                                     struct ggml_tensor *dst) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  if (ggml_backend_buffer_is_host(src->buffer)) {
    memcpy(dst->data, src->data, (ggml_nbytes(src)));
    return true;
  }
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return false;

  TSI_UNUSED(buffer);
}

static void ggml_backend_tsavorite_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  struct ggml_backend_tsavorite_buffer_context *ctx =
      (struct ggml_backend_tsavorite_buffer_context *)buffer->context;
  if (!ctx || !ctx->all_data) {
    GGML_TSAVORITE_LOG_ERROR("\n ctx or all_data cant be null under func: %s\n", __func__);
    return;
  }
  memset((char *)ctx->all_data, value, ctx->all_size);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
}

static struct ggml_backend_buffer_i ggml_backend_tsavorite_buffer_i = {
    /* .free_buffer     = */ ggml_backend_tsavorite_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_tsavorite_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_tsavorite_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_tsavorite_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_tsavorite_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_tsavorite_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_tsavorite_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_tsavorite_buffer_clear,
    /* .reset           = */ NULL,
};

// default buffer type

static const char *ggml_backend_tsavorite_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return "tsavorite";

  TSI_UNUSED(buft);
}

static void ggml_backend_tsavorite_log_allocated_size(txe_device_s device, size_t size_aligned) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
#ifndef GGML_TSAVORITE_NDEBUG
#if TARGET_OS_OSX || (TARGET_OS_IOS && __clang_major__ >= 15)
  GGML_TSAVORITE_LOG_INFO("%s: allocated buffer, size = %8.2f MiB, (%8.2f)\n", __func__,
                          size_aligned / 1024.0 / 1024.0,
                          device.currentAllocatedSize / 1024.0 / 1024.0);
#endif
#endif
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  TSI_UNUSED(device);
  TSI_UNUSED(size_aligned);
}

static ggml_backend_buffer_t
ggml_backend_tsavorite_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  tsi_log_setup();
  struct ggml_backend_tsavorite_buffer_context *ctx =
      (struct ggml_backend_tsavorite_buffer_context *)calloc(
          1, sizeof(struct ggml_backend_tsavorite_buffer_context));

  const size_t size_page = sysconf(_SC_PAGESIZE);
  GGML_TSAVORITE_LOG_CONT(
      "ggml_backend_tsavorite_buffer_type_alloc_buffer is called from llama data Loader \n");

  size_t size_aligned = size;
  if ((size_aligned % size_page) != 0) {
    size_aligned += (size_page - (size_aligned % size_page));
  }

  txe_device_s device = ggml_backend_tsavorite_device_acq(
      (struct ggml_backend_tsavorite_device_context *)buft->device->context);
  if (!device)
    return NULL;

  ctx->all_data = ggml_tsavorite_host_malloc(size_aligned);
  ctx->all_size = size_aligned;
  ctx->owned = true;
  ctx->n_buffers = 1;
  GGML_TSAVORITE_LOG_INFO("\n\n\n\n  Memory Starting address %p and size %ld \n\n\n", ctx->all_data,
                          ctx->all_size);

  if (ctx->all_data != NULL) {
    GGML_TSAVORITE_LOG_CONT("\nAddress of Newly Created BUffer %p and size %ld \n", ctx->all_data,
                            ctx->all_size);
    if (ggml_tsavorite_log_type_val == GGML_TSAVORITE_LOG_DEBUG) {
      fprintf(tsi_op_log_file, "Address of Newly Created BUffer %p and size %ld \n", ctx->all_data,
              ctx->all_size);
    }
    ctx->buffers[0].data = NULL;
    ctx->buffers[0].data = ctx->all_data;
    ctx->buffers[0].size = size;
    memset((char *)ctx->all_data, 0, ctx->all_size);
  }

  if (size_aligned > 0 && (ctx->all_data == NULL)) {
    GGML_TSAVORITE_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f MiB\n", __func__,
                             size_aligned / 1024.0 / 1024.0);
    free(ctx);
    ggml_backend_tsavorite_device_rel(
        (struct ggml_backend_tsavorite_device_context *)buft->device->context);
    return NULL;
  }

  // ggml_backend_tsavorite_log_allocated_size(device, size_aligned);
  device->current_allocated_size += ctx->all_size;
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);

  return ggml_backend_buffer_init(buft, ggml_backend_tsavorite_buffer_i, ctx, size);
}

static size_t ggml_backend_tsavorite_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return 32;
  TSI_UNUSED(buft);
}

static size_t ggml_backend_tsavorite_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  txe_device_s device = ggml_backend_tsavorite_device_acq(
      (struct ggml_backend_tsavorite_device_context *)buft->device->context);
  const size_t max_size = device->max_buf_len;
  ggml_backend_tsavorite_device_rel(
      (struct ggml_backend_tsavorite_device_context *)buft->device->context);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);

  return max_size;

  TSI_UNUSED(buft);
}

static size_t ggml_backend_tsavorite_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft,
                                                                const struct ggml_tensor *tensor) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  txe_device_s device = ggml_backend_tsavorite_device_acq(
      (struct ggml_backend_tsavorite_device_context *)buft->device->context);
  if (!device) {
    GGML_TSAVORITE_LOG_ERROR("\n tsavorite device is NULL \n");
    return 0;
  }
  MemRefDescriptor<Rank> tensor_data_header;
  ggml_backend_tsavorite_device_rel(
      (struct ggml_backend_tsavorite_device_context *)buft->device->context);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);

  GGML_TSAVORITE_LOG_INFO(
      "\n\n\n\n Calculating---- Alloc ----Size header %lu  and data %lu \n\n\n\n ",
      sizeof(tensor_data_header), ggml_nbytes(tensor));

  // Add 128-byte buffer to avoid crossing memory boundaries during TVU 1024-bit operations.
  // TVU processes data in 1024-bit chunks, so the last elements may exceed allocated space without this padding.
  const int32_t mem_align = TSI_TVU_MEM_ALIGN;
  // I also added extra Padding buffer
  size_t n =  (((sizeof(tensor_data_header) + ggml_nbytes(tensor))/mem_align +1)*mem_align + mem_align);
  return (n);

  TSI_UNUSED(buft);
}

static bool ggml_backend_tsavorite_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  // For Now CPU is loading all data and then copy some tensor to Tsavorite Backend
  // Once we have most of Operation supported by Tsavorite
  // We will figure out to make tsavorite Backend also host
  return false;

  TSI_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_tsavorite_buffer_type(void) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  static struct ggml_backend_buffer_type ggml_backend_buffer_type_tsavorite = {
      /* .iface = */ {
          /* .get_name         = */ ggml_backend_tsavorite_buffer_type_get_name,
          /* .alloc_buffer     = */ ggml_backend_tsavorite_buffer_type_alloc_buffer,
          /* .get_alignment    = */ ggml_backend_tsavorite_buffer_type_get_alignment,
          /* .get_max_size     = */ ggml_backend_tsavorite_buffer_type_get_max_size,
          /* .get_alloc_size   = */
          ggml_backend_tsavorite_buffer_type_get_alloc_size,  // defaults to ggml_nbytes
          /* .is_host          = */ ggml_backend_tsavorite_buffer_type_is_host,
      },
      /* .device  = */ &g_ggml_backend_tsavorite_device,
      /* .context = */ NULL,
  };
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);

  return &ggml_backend_buffer_type_tsavorite;
}

// backend

static const char *ggml_backend_tsavorite_name(ggml_backend_t backend) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return "Tsavorite";

  TSI_UNUSED(backend);
}

static void ggml_backend_tsavorite_free(ggml_backend_t backend) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  if (!backend || !backend->context || !backend->device || !backend->device->context) {
    GGML_TSAVORITE_LOG_ERROR("At %s One of more pointer among: Backend, backend_context, "
                             "device_context or device are NULL",
                             __func__);
    return;
  }
  struct ggml_backend_tsavorite_context *ctx =
      (struct ggml_backend_tsavorite_context *)backend->context;
  struct ggml_backend_tsavorite_device_context *ctx_dev =
      (struct ggml_backend_tsavorite_device_context *)backend->device->context;
  ggml_tsavorite_disp_stats(ctx, ctx_dev->device);

  ggml_backend_tsavorite_device_rel(ctx_dev);
  ggml_tsavorite_free(ctx);

  free(backend);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
}

static void ggml_backend_tsavorite_synchronize(ggml_backend_t backend) {
    join_all_workers();
    (void)backend;
}

static ggml_backend_buffer_type_t
ggml_backend_tsavorite_get_default_buffer_type(ggml_backend_t backend) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return ggml_backend_tsavorite_buffer_type();

  TSI_UNUSED(backend);
}

static enum ggml_status ggml_backend_tsavorite_graph_compute(ggml_backend_t backend,
                                                             struct ggml_cgraph *cgraph) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return ggml_tsavorite_graph_compute(backend, cgraph);
}

static void ggml_backend_tsavorite_set_n_cb(ggml_backend_t backend, int n_cb) {
  // GGML_ASSERT(ggml_backend_is_tsavorite(backend));
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  struct ggml_backend_tsavorite_context *ctx =
      (struct ggml_backend_tsavorite_context *)backend->context;

  if (ctx->n_cb != n_cb) {
    ctx->n_cb = MIN(n_cb, GGML_TSAVORITE_MAX_COMMAND_BUFFERS);

    if (ctx->n_cb > 2) {
      GGML_TSAVORITE_LOG_WARN("%s: n_cb = %d, using n_cb > 2 is not recommended and can degrade "
                              "the performance in some cases\n",
                              __func__, n_cb);
    }
  }

#if 0
    if (ctx->encode_async) {
        Block_release(ctx->encode_async);
    }
#endif
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
}

#ifdef OLLAMA
void
tsi_log_profile_info() {
    GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
    tsi_unload_all_blobs();
    if(device_free) {
        free(device_free);
        device_free = NULL;
    }
    tsi_reset_per_txe_state_after_teardown();
    printf("\n finalize 4 \n");
    tsi_finalize();
    tsirt::utils::TSIProfiler::finalize();
    // Profiling results already printed during first cleanup
    std::cout << "\nOPU Profiling Results:" << std::endl;
    std::cout << tsirt::utils::TSIProfiler::getFormattedResults(
                  /*truncateFuncNames*/ true)
               << std::endl;
    GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
    fflush(stdout);
    return;
}
#endif /* OLLAMA */

static struct ggml_backend_i ggml_backend_tsavorite_i = {
    /* .get_name                = */ ggml_backend_tsavorite_name,
    /* .free                    = */ ggml_backend_tsavorite_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_tsavorite_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_tsavorite_graph_compute,
    /* .event_record            = */ NULL,
#ifdef OLLAMA
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
    /* .graph_reserve           = */ NULL,
    /* .buffer_size             = */ NULL,
    /* .reset                   = */ NULL,
    /* .profile                 = */ tsi_log_profile_info
#else
    /* .event_wait              = */ NULL
#endif /* OLLAMA */
};

static ggml_guid_t ggml_backend_tsavorite_guid(void) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);

  static ggml_guid guid = {0x81, 0xa1, 0x8b, 0x1e, 0x71, 0xec, 0x79, 0xed,
                           0x2b, 0x85, 0xdc, 0x8a, 0x61, 0x98, 0x30, 0xe6};
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return &guid;
}

// This need to be removed in the future
ggml_backend_t ggml_backend_tsavorite_init(void) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_tsavorite_reg(), 0);
  struct ggml_backend_tsavorite_context *ctx = ggml_tsavorite_init(dev);
  if (ctx == NULL) {
    GGML_TSAVORITE_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
    return NULL;
  }
  ggml_backend_t backend = (ggml_backend_t)malloc(sizeof(struct ggml_backend));
  if (backend) {
    backend->guid = ggml_backend_tsavorite_guid();
    backend->iface = ggml_backend_tsavorite_i;
    backend->device = dev;
    backend->context = ctx;
  }
  // Will enable later
  // ggml_backend_tsavorite_set_n_cb(backend, 1);

  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return backend;
}

bool ggml_backend_is_tsavorite(ggml_backend_t backend) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_tsavorite_guid());
}

void ggml_backend_tsavorite_set_abort_callback(ggml_backend_t backend,
                                               ggml_abort_callback abort_callback,
                                               void *user_data) {
  GGML_ASSERT(ggml_backend_is_tsavorite(backend));
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  struct ggml_backend_tsavorite_context *ctx =
      (struct ggml_backend_tsavorite_context *)backend->context;

  ctx->abort_callback = abort_callback;
  ctx->abort_callback_data = user_data;
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
}

void ggml_backend_tsavorite_capture_next_compute(ggml_backend_t backend) {
  GGML_ASSERT(ggml_backend_is_tsavorite(backend));
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  struct ggml_backend_tsavorite_context *ctx =
      (struct ggml_backend_tsavorite_context *)backend->context;
  ctx->capture_next_compute = true;
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
}

// backend device

static const char *ggml_backend_tsavorite_device_get_name(ggml_backend_dev_t dev) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return "Tsavorite";

  GGML_UNUSED(dev);
}

static const char *ggml_backend_tsavorite_device_get_description(ggml_backend_dev_t dev) {
  // acq/rel just to populate ctx->name in case it hasn't been done yet
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);

  struct ggml_backend_tsavorite_device_context *ctx_dev =
      (struct ggml_backend_tsavorite_device_context *)dev->context;
  ggml_backend_tsavorite_device_acq(ctx_dev);
  ggml_backend_tsavorite_device_rel(ctx_dev);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);

  return ctx_dev->name;
}

static void ggml_backend_tsavorite_device_get_memory(ggml_backend_dev_t dev, size_t *free,
                                                     size_t *total) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  if (!dev || !free || !total) {
    GGML_TSAVORITE_LOG_INFO("One of more pointers(dev, free, total) are NULL\n");
    GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
    return;
  }
  *total = 0;
  *total = 0;
  struct ggml_backend_tsavorite_device_context *ctx_dev =
      (struct ggml_backend_tsavorite_device_context *)dev->context;
  if (ctx_dev) {
    txe_device_s device = ggml_backend_tsavorite_device_acq(ctx_dev);
    *total = device->recommended_max_working_set_size;
    *free = *total - device->current_allocated_size;
    GGML_TSAVORITE_LOG_CONT("\n TXE Device MEMORY Summary total %lu and free %lu \n", *total,
                            *free);
    ggml_backend_tsavorite_device_rel(ctx_dev);
  }
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return;
}

// Currently We are setting our TXE accerator at GPU Type
static enum ggml_backend_dev_type ggml_backend_tsavorite_device_get_type(ggml_backend_dev_t dev) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return GGML_BACKEND_DEVICE_TYPE_GPU;

  GGML_UNUSED(dev);
}

// Need to understand the scope of this API since this is not used
// // use by Structure llama_model_loader
// func llm_load_tensors
// structure lama_new_context_with_model
static void ggml_backend_tsavorite_device_get_props(ggml_backend_dev_t dev,
                                                    struct ggml_backend_dev_props *props) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);

  props->name = ggml_backend_tsavorite_device_get_name(dev);
  props->description = ggml_backend_tsavorite_device_get_description(dev);
  props->type = ggml_backend_tsavorite_device_get_type(dev);
  ggml_backend_tsavorite_device_get_memory(dev, &props->memory_free, &props->memory_total);

  if (props) {
    props->caps.async = false;
    props->caps.host_buffer = false;
    props->caps.buffer_from_host_ptr = true;
    props->caps.buffer_from_host_ptr = false;
    props->caps.events = false;
  }
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
}

static ggml_backend_t ggml_backend_tsavorite_device_init(ggml_backend_dev_t dev,
                                                         const char *params) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);

  struct ggml_backend_tsavorite_context *ctx = ggml_tsavorite_init(dev);
  if (ctx == NULL) {
    GGML_TSAVORITE_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
    return NULL;
  }

  ggml_backend_t backend = (ggml_backend_t)malloc(sizeof(struct ggml_backend));

  if (backend) {
    backend->guid = ggml_backend_tsavorite_guid();
    backend->iface = ggml_backend_tsavorite_i;
    backend->device = dev;
    backend->context = ctx;
  }

  ggml_backend_tsavorite_set_n_cb(backend, 1);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);

  return backend;

  GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t
ggml_backend_tsavorite_device_get_buffer_type(ggml_backend_dev_t dev) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return ggml_backend_tsavorite_buffer_type();

  GGML_UNUSED(dev);
}

// Currently for llama.cpp model below API it seems not used
// llama.cpp is using as part llm_load_tensors
// buffer_from_host_ptr_supported
// is_default_buft
// else they will be using
// ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
// Need to revist when we will look at buffer section implementation
static ggml_backend_buffer_t ggml_backend_tsavorite_device_buffer_from_ptr(ggml_backend_dev_t dev,
                                                                           void *ptr, size_t size,
                                                                           size_t max_tensor_size) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  struct ggml_backend_tsavorite_buffer_context *ctx =
      (struct ggml_backend_tsavorite_buffer_context *)calloc(
          1, sizeof(struct ggml_backend_tsavorite_buffer_context));

  ctx->all_data = ptr;
  ctx->all_size = size;
  ctx->owned = false;
  ctx->n_buffers = 0;

  const size_t size_page = sysconf(_SC_PAGESIZE);

  // page-align the data ptr
  {
    const uintptr_t offs = (uintptr_t)ptr % size_page;
    ptr = (void *)((char *)ptr - offs);
    size += offs;
  }

  size_t size_aligned = size;
  if ((size_aligned % size_page) != 0) {
    size_aligned += (size_page - (size_aligned % size_page));
  }

  struct ggml_backend_tsavorite_device_context *ctx_dev =
      (struct ggml_backend_tsavorite_device_context *)dev->context;
  txe_device_s device = ggml_backend_tsavorite_device_acq(ctx_dev);

  // the buffer fits into the max buffer size allowed by the device
  if (size_aligned <= device->max_buf_len) {
    ctx->buffers[ctx->n_buffers].data = ptr;
    ctx->buffers[ctx->n_buffers].size = size;

    // ggml_backend_tsavorite_log_allocated_size(device, size_aligned);

    ++ctx->n_buffers;
  } else {
    // this overlap between the views will guarantee that the tensor with the maximum size will
    // fully fit into one of the views
    const size_t size_ovlp = ((max_tensor_size + size_page - 1) / size_page + 1) *
                             size_page;  // round-up 2 pages just in case
    const size_t size_step = device->max_buf_len - size_ovlp;
    const size_t size_view = device->max_buf_len;

    for (size_t i = 0; i < size; i += size_step) {
      const size_t size_step_aligned = (i + size_view <= size) ? size_view : (size_aligned - i);

      ctx->buffers[ctx->n_buffers].data = (void *)((uint8_t *)ptr + i);
      ctx->buffers[ctx->n_buffers].size = size_step_aligned;

      // ggml_backend_tsavorite_log_allocated_size(device, size_step_aligned);

      if (i + size_step < size) {
        GGML_TSAVORITE_LOG_INFO("\n");
      }

      ++ctx->n_buffers;
    }
  }
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);

  return ggml_backend_buffer_init(ggml_backend_tsavorite_buffer_type(),
                                  ggml_backend_tsavorite_buffer_i, ctx, size);
}

// llama_build_graph -> ggml_backend_supports_op -> gml_backend_dev_supports_op
// basically if true then it will call ggml_backend_sched_set_tensor_backend(lctx.sched.get(), cur,
// backend.get()); here is cur is tensor
static bool ggml_backend_tsavorite_device_supports_op(ggml_backend_dev_t dev,
                                                      const struct ggml_tensor *op) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);

  struct ggml_backend_tsavorite_device_context *ctx_dev =
      (struct ggml_backend_tsavorite_device_context *)dev->context;

  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return ggml_tsavorite_supports_op(ctx_dev, op);
}

// template<typename F>
// static bool buft_supported(ggml_backend_buffer_type_t buft, ggml_backend_dev_t dev, F & fn) {}
//  ggml_backend_dev_supports_op(dev, op_tensor);
static bool ggml_backend_tsavorite_device_supports_buft(ggml_backend_dev_t dev,
                                                        ggml_backend_buffer_type_t buft) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return buft->iface.get_name == ggml_backend_tsavorite_buffer_type_get_name;
  //return strcmp(buft->iface.get_name(buft), "tsavorite") == 0;


  TSI_UNUSED(dev);
}

// // returns the backend that should be used for the node based on the current locations
// ggml_backend_sched_backend_id_from_cur  -> ggml_backend_offload_op ->
static bool ggml_backend_tsavorite_device_offload_op(ggml_backend_dev_t dev,
                                                     const struct ggml_tensor *op) {
  bool return_value = false;

  return_value = ggml_tsavorite_internal_supports_op(op);

  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  TSI_UNUSED(dev);
  return return_value;
}
#ifdef SYNC_DEBUG
static void ggml_backend_tsavorite_device_synchronize(ggml_backend_dev_t dev,
                                                      ggml_backend_event_t event) {
  usleep(100);
  TSI_UNUSED(dev);
  TSI_UNUSED(event);
}
#endif /* SYNC_DEBUG */

static struct ggml_backend_device_i ggml_backend_tsavorite_device_i = {
    /* .get_name             = */ ggml_backend_tsavorite_device_get_name,
    /* .get_description      = */ ggml_backend_tsavorite_device_get_description,
    /* .get_memory           = */ ggml_backend_tsavorite_device_get_memory,
    /* .get_type             = */ ggml_backend_tsavorite_device_get_type,
    /* .get_props            = */ ggml_backend_tsavorite_device_get_props,
    /* .init_backend         = */ ggml_backend_tsavorite_device_init,
    /* .get_buffer_type      = */ ggml_backend_tsavorite_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_tsavorite_device_buffer_from_ptr,
    /* .supports_op          = */ ggml_backend_tsavorite_device_supports_op,
    /* .supports_buft        = */ ggml_backend_tsavorite_device_supports_buft,
    /* .offload_op           = */ ggml_backend_tsavorite_device_offload_op,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// backend registry

static const char *ggml_backend_tsavorite_reg_get_name(ggml_backend_reg_t reg) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return "Tsavorite";

  GGML_UNUSED(reg);
}

static size_t ggml_backend_tsavorite_reg_device_count(ggml_backend_reg_t reg) {
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);
  return 1;

  GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_tsavorite_reg_device_get(ggml_backend_reg_t reg,
                                                                size_t index) {
  GGML_ASSERT(index == 0);
  GGML_TSAVORITE_LOG_INFO("Start %s\n", __func__);
  GGML_TSAVORITE_LOG_INFO("End %s\n", __func__);

  return &g_ggml_backend_tsavorite_device;

  GGML_UNUSED(reg);
  GGML_UNUSED(index);
}

#ifdef GGML_MUL_MAT_CPU_OPS
void ggml_backend_tsavorite_set_n_threads(ggml_backend_t backend_tsavorite, int n_threads) {
    GGML_ASSERT(ggml_backend_is_tsavorite(backend_tsavorite));

    struct ggml_backend_tsavorite_context * ctx = (struct ggml_backend_tsavorite_context *)backend_tsavorite->context;
    ctx->n_threads = n_threads;
}

std::vector<ggml_backend_buffer_type_t> & ggml_backend_tsavorite_get_extra_buffer_types() {
    static std::vector<ggml_backend_buffer_type_t> bufts = []() {
        std::vector<ggml_backend_buffer_type_t> bufts;
        return bufts;
    }();

    return bufts;
}

static ggml_backend_buffer_type_t * ggml_backend_tsavorite_device_get_extra_buffers_type(ggml_backend_dev_t device) {
    static std::vector<ggml_backend_buffer_type_t> extra_bufts = [] {
        std::vector<ggml_backend_buffer_type_t> bufts = ggml_backend_tsavorite_get_extra_buffer_types();
        bufts.push_back(nullptr);
        return bufts;
    }();

    return extra_bufts.data();

    GGML_UNUSED(device);
}
#endif
static ggml_backend_feature * ggml_backend_tsavorite_get_features(ggml_backend_reg_t reg) {
    static std::vector<ggml_backend_feature> features = []() {
        std::vector<ggml_backend_feature> features;
        if (ggml_cpu_has_sse3()) {
            features.push_back({ "SSE3", "1" });
        }
        if (ggml_cpu_has_ssse3()) {
            features.push_back({ "SSSE3", "1" });
        }
        if (ggml_cpu_has_avx()) {
            features.push_back({ "AVX", "1" });
        }
        if (ggml_cpu_has_avx_vnni()) {
            features.push_back({ "AVX_VNNI", "1" });
        }
        if (ggml_cpu_has_avx2()) {
            features.push_back({ "AVX2", "1" });
        }
        if (ggml_cpu_has_f16c()) {
            features.push_back({ "F16C", "1" });
        }
        if (ggml_cpu_has_fma()) {
            features.push_back({ "FMA", "1" });
        }
        if (ggml_cpu_has_bmi2()) {
            features.push_back({ "BMI2", "1" });
        }
        if (ggml_cpu_has_avx512()) {
            features.push_back({ "AVX512", "1" });
        }
        if (ggml_cpu_has_avx512_vbmi()) {
            features.push_back({ "AVX512_VBMI", "1" });
        }
        if (ggml_cpu_has_avx512_vnni()) {
            features.push_back({ "AVX512_VNNI", "1" });
        }
        if (ggml_cpu_has_avx512_bf16()) {
            features.push_back({ "AVX512_BF16", "1" });
        }
        if (ggml_cpu_has_amx_int8()) {
            features.push_back({ "AMX_INT8", "1" });
        }
        if (ggml_cpu_has_neon()) {
            features.push_back({ "NEON", "1" });
        }
        if (ggml_cpu_has_arm_fma()) {
            features.push_back({ "ARM_FMA", "1" });
        }
        if (ggml_cpu_has_fp16_va()) {
            features.push_back({ "FP16_VA", "1" });
        }
        if (ggml_cpu_has_matmul_int8()) {
            features.push_back({ "MATMUL_INT8", "1" });
        }
        if (ggml_cpu_has_sve()) {
            features.push_back({ "SVE", "1" });
        }
        if (ggml_cpu_has_dotprod()) {
            features.push_back({ "DOTPROD", "1" });
        }
        if (ggml_cpu_get_sve_cnt() > 0) {
            static std::string sve_cnt = std::to_string(ggml_cpu_get_sve_cnt());
            features.push_back({ "SVE_CNT", sve_cnt.c_str() });
        }
        if (ggml_cpu_has_sme()) {
            features.push_back({ "SME", "1" });
        }
        if (ggml_cpu_has_riscv_v()) {
            features.push_back({ "RISCV_V", "1" });
        }
        if (ggml_cpu_has_vsx()) {
            features.push_back({ "VSX", "1" });
        }
        if (ggml_cpu_has_vxe()) {
            features.push_back({ "VXE", "1" });
        }
        if (ggml_cpu_has_wasm_simd()) {
            features.push_back({ "WASM_SIMD", "1" });
        }
        if (ggml_cpu_has_llamafile()) {
            features.push_back({ "LLAMAFILE", "1" });
        }
    #ifdef GGML_USE_ACCELERATE
        features.push_back({ "ACCELERATE", "1" });
    #endif
    #ifdef GGML_USE_CPU_HBM
        features.push_back({ "CPU_HBM", "1" });
    #endif
    #ifdef GGML_USE_OPENMP
        features.push_back({ "OPENMP", "1" });
    #endif
    #ifdef GGML_USE_CPU_KLEIDIAI
        features.push_back({ "KLEIDIAI", "1" });
    #endif
    #ifdef GGML_USE_CPU_REPACK
        features.push_back({ "REPACK", "1" });
    #endif

        features.push_back({ nullptr, nullptr });

        return features;
    }();

    return features.data();

    GGML_UNUSED(reg);
}

static void * ggml_backend_tsavorite_get_proc_address(ggml_backend_reg_t reg, const char * name) {
#ifdef GGML_MUL_MAT_CPU_OPS
    if (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ggml_backend_set_n_threads_t fct = ggml_backend_tsavorite_set_n_threads;
        return (void *)fct;
    }
    if (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0) {
        ggml_backend_dev_get_extra_bufts_t fct = ggml_backend_tsavorite_device_get_extra_buffers_type;
        return (void *)fct;
    }
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_tsavorite_get_features;
    }
    if (strcmp(name, "ggml_backend_set_abort_callback") == 0) {
        return (void *)ggml_backend_tsavorite_set_abort_callback;
    }
    if (strcmp(name, "ggml_backend_cpu_numa_init") == 0) {
        return (void *)ggml_numa_init;
    }
    if (strcmp(name, "ggml_backend_cpu_is_numa") == 0) {
        return (void *)ggml_is_numa;
    }
    if (strcmp(name, "ggml_threadpool_new") == 0) {
        return (void *)ggml_threadpool_new;
    }
    if (strcmp(name, "ggml_threadpool_free") == 0) {
        return (void *)ggml_threadpool_free;
    }
    if (strcmp(name, "ggml_backend_cpu_set_threadpool") == 0) {
        return (void *)ggml_backend_tsavorite_set_threadpool;
    }
#endif
    if (strcmp(name, "ggml_perf_accumulate") == 0) {
        return (void *)ggml_perf_accumulate;
    }
#if defined(GGML_PERF_DETAIL)
    // ggml_perf_log_open/write_detailed_csv only exist in ggml.c under
    // GGML_PERF_DETAIL (see ggml.c's own #if guard around their definitions);
    // taking their address here unconditionally would leave a dangling
    // undefined reference in GGML_PERF_RELEASE/GGML_PERF builds.
    if (strcmp(name, "ggml_perf_log_open") == 0) {
        return (void *)ggml_perf_log_open;
    }
    if (strcmp(name, "ggml_perf_write_detailed_csv") == 0) {
        return (void *)ggml_perf_write_detailed_csv;
    }
#endif
    if (strcmp(name, "ggml_backend_type") == 0) {
        return (void *)ggml_backend_type;
    }
    return NULL;

    GGML_UNUSED(reg);
}

static struct ggml_backend_reg_i ggml_backend_tsavorite_reg_i = {
    /* .get_name         = */ ggml_backend_tsavorite_reg_get_name,
    /* .device_count     = */ ggml_backend_tsavorite_reg_device_count,
    /* .device_get       = */ ggml_backend_tsavorite_reg_device_get,
    /* .get_proc_address = */ ggml_backend_tsavorite_get_proc_address,
};

#ifdef OLLAMA
#define SHM_NAME "/ollama_init_shm"

typedef struct {
  bool init_done;
} shared_state_t;

static shared_state_t *state = NULL;

static bool init_shared_state() {
    int fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (fd == -1) {
        perror("shm_open");
        return false;
    }

    if (ftruncate(fd, sizeof(shared_state_t)) != 0) {
        perror("ftruncate");
        close(fd);
        return false;
    }

    void *p = mmap(NULL,
                   sizeof(shared_state_t),
                   PROT_READ | PROT_WRITE,
                   MAP_SHARED,
                   fd,
                   0);
    close(fd);

    if (p == MAP_FAILED) {
        perror("mmap");
        return false;
    }

    state = (shared_state_t *)p;
    return true;
}
#endif /* OLLAMA */

ggml_backend_reg_t ggml_backend_tsavorite_reg(void) {
    ggml_tsavorite_log_type_val    = GGML_TSAVORITE_LOG_NONE;
    ggml_tsavorite_kernel_mode_flag = GGML_TSAVORITE_KERNEL_MODE_MLIR;

#ifdef OLLAMA
    bool shm_ok = init_shared_state();

    if (!shm_ok || state == NULL) {
        // No shared memory available → per-process init
        ensure_tsi_runtime_initialized();
    } else {
        // Shared memory available → exactly-once init
        if (!state->init_done) {
            state->init_done = true;
            GGML_LOG_DEBUG("%s: Initialization not done, proceeding...\n", __func__);
        } else {
            ensure_tsi_runtime_initialized();
        }
        // else: already initialized in another process
    }
    g_ggml_backend_tsavorite_reg.api_version = GGML_BACKEND_API_VERSION;
#else
    ensure_tsi_runtime_initialized();
    
#endif /* OLLAMA */

    g_ggml_backend_tsavorite_reg.iface = ggml_backend_tsavorite_reg_i;
    g_ggml_backend_tsavorite_reg.context = NULL;

    g_ggml_backend_tsavorite_device.iface   = ggml_backend_tsavorite_device_i;
    g_ggml_backend_tsavorite_device.reg     = &g_ggml_backend_tsavorite_reg;
    g_ggml_backend_tsavorite_device.context = &g_ggml_ctx_dev_main;

    return &g_ggml_backend_tsavorite_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_tsavorite_reg)

