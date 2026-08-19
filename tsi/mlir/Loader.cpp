// Resolves the exporter at runtime instead of linking it.
//
// This file provides exactly the symbols llama-context.cpp and llama-kv-cache.cpp already declare,
// and forwards each to the tsi_mlir_abi_* entry point of libtsi-mlir-driver, dlopen'd on first use.
// So llama.cpp itself needs no change: link this instead of tsi-llama-hook and the same call sites
// work. The point is what this file does NOT pull in - it has no MLIR include path and no MLIR
// libraries, so a consumer that compiles llama.cpp into its own binary (ollama does, via cgo) takes
// on one small translation unit rather than ~400 MLIR archives.
//
// Library path comes from TSI_MLIR_LIB, falling back to a plain soname so a normal loader search
// works after an install. Failure to load is NOT fatal, deliberately: with no driver the hooks all
// report "not handled" and llama computes the graph itself, which is the same outcome as
// TSI_MLIR_EXPORT=0. It is announced once, though - a silent fallback to CPU is the failure mode
// this project keeps having to diagnose, so it must never be silent here.

#include "ggml-backend.h"
#include "ggml-backend-impl.h"   // ggml_backend_buffer_type, built HERE against the host's ggml
#include "ggml-cpu.h"            // ggml_backend_cpu_buffer_from_ptr

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>

struct ggml_cgraph;
struct ggml_tensor;

namespace {

constexpr uint32_t kAbiVersion = 2u;   // must match TSI_MLIR_ABI_VERSION in Abi.cpp

#if defined(__APPLE__)
constexpr const char * kDefaultLib = "libtsi-mlir-driver.dylib";
#else
constexpr const char * kDefaultLib = "libtsi-mlir-driver.so";
#endif

struct Driver {
    bool (*before_compute)(struct ggml_cgraph *)                 = nullptr;
    bool (*after_compute)(struct ggml_cgraph *)                  = nullptr;
    bool (*enabled)(void)                                        = nullptr;
    bool (*eval_cb)(struct ggml_tensor *, bool, void *)          = nullptr;
    void * (*eval_chain)(const void *,
                         bool (*)(struct ggml_tensor *, bool, void *),
                         void *)                                 = nullptr;
    void * (*dram_alloc)(size_t)                                 = nullptr;
};

// One resolve for the process. A failed load is cached as "no driver" so a missing library costs one
// dlopen rather than one per graph.
const Driver & driver() {
    static Driver d = [] {
        Driver out;

        const char * path = std::getenv("TSI_MLIR_LIB");
        if (path == nullptr || path[0] == '\0') {
            path = kDefaultLib;
        }

        void * h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (h == nullptr) {
            fprintf(stderr,
                    "[tsi-mlir] driver not loaded (%s); llama will compute every graph itself.\n"
                    "[tsi-mlir]   set TSI_MLIR_LIB to libtsi-mlir-driver if you meant to export.\n",
                    dlerror());
            return out;
        }

        auto version = (uint32_t (*)(void)) dlsym(h, "tsi_mlir_abi_version");
        if (version == nullptr || version() != kAbiVersion) {
            // Declining beats calling through a signature that moved. The consumer of this shim did
            // not build the library it just opened.
            fprintf(stderr,
                    "[tsi-mlir] driver ABI mismatch in %s (want %u, got %u); not using it.\n",
                    path, kAbiVersion, version ? version() : 0u);
            dlclose(h);
            return out;
        }

        out.before_compute = (bool (*)(struct ggml_cgraph *))        dlsym(h, "tsi_mlir_abi_before_compute");
        out.after_compute  = (bool (*)(struct ggml_cgraph *))        dlsym(h, "tsi_mlir_abi_after_compute");
        out.enabled        = (bool (*)(void))                       dlsym(h, "tsi_mlir_abi_enabled");
        out.eval_cb        = (bool (*)(struct ggml_tensor *, bool, void *)) dlsym(h, "tsi_mlir_abi_eval_cb");
        out.eval_chain     = (void * (*)(const void *,
                                         bool (*)(struct ggml_tensor *, bool, void *),
                                         void *))                   dlsym(h, "tsi_mlir_abi_eval_chain");
        out.dram_alloc     = (void * (*)(size_t))                    dlsym(h, "tsi_mlir_abi_dram_alloc");

        fprintf(stderr, "[tsi-mlir] driver loaded from %s (ABI %u)\n", path, kAbiVersion);
        return out;
    }();
    return d;
}

}  // namespace

// Same names and linkage llama-context.cpp declares, so its call sites are untouched.
bool tsi_mlir_export_before_compute(struct ggml_cgraph * cgraph) {
    const Driver & d = driver();
    return d.before_compute ? d.before_compute(cgraph) : false;
}

bool tsi_mlir_export_after_compute(struct ggml_cgraph * live) {
    const Driver & d = driver();
    return d.after_compute ? d.after_compute(live) : false;
}

extern "C" bool tsi_mlir_export_enabled(void) {
    const Driver & d = driver();
    return d.enabled ? d.enabled() : false;
}

extern "C" bool tsi_mlir_export_eval_cb(struct ggml_tensor * t, bool ask, void * ud) {
    const Driver & d = driver();
    // True keeps llama's own traversal going, which is what a graph nobody is snapshotting wants.
    return d.eval_cb ? d.eval_cb(t, ask, ud) : true;
}

extern "C" void * tsi_mlir_export_eval_chain(const void * sched,
                                             bool (*cb)(struct ggml_tensor *, bool, void *),
                                             void * user_data) {
    const Driver & d = driver();
    return d.eval_chain ? d.eval_chain(sched, cb, user_data) : user_data;
}

// The KV buffer type is built HERE, not in the driver, and that placement is the whole point.
//
// ggml_backend_buffer_i and ggml_backend_buffer_type_i do not have the same members across llama.cpp
// generations - ours carries set_tensor_2d/get_tensor_2d that ollama's vendored ggml lacks, and
// ollama's carries a noalloc_buffer that ours lacks. A struct built inside the .so therefore has its
// function pointers at offsets the host does not expect, and the host calls through the wrong slot.
// That was a SIGSEGV in llama_init_from_model, not a subtle wrong answer.
//
// This file is compiled by the consumer, against the consumer's ggml headers, so the layout is right
// by construction and only a raw pointer crosses the .so boundary.

namespace {

const char * kvName(ggml_backend_buffer_type_t) {
    return "TSI_DRAM";
}

ggml_backend_buffer_t kvAlloc(ggml_backend_buffer_type_t buft, size_t size) {
    const Driver & d = driver();
    if (!d.dram_alloc) {
        return nullptr;
    }

    void * p = d.dram_alloc(size);
    if (!p) {
        fprintf(stderr, "[tsi-mlir] DRAM alloc failed for a %zu-byte KV cache buffer. "
                        "Raise USER_DRAM_SIZE (MiB).\n", size);
        return nullptr;
    }

    // Reuse ggml's own from_ptr buffer: it wraps an existing host pointer and, being CPU memory, gets
    // the standard tensor-access implementations for free. Only the reported type is overridden, so
    // ggml_backend_buft_is_host() answers for this type rather than the generic CPU-mapped one.
    //
    // Not zeroed here: llama clears the buffer itself right after allocating it, and a second full
    // pass costs gigabytes of memset at a large context.
    ggml_backend_buffer_t buf = ggml_backend_cpu_buffer_from_ptr(p, size);
    if (!buf) {
        return nullptr;
    }
    buf->buft = buft;
    fprintf(stderr, "[tsi-mlir] KV cache: %.2f MiB in TSI DRAM at %p\n",
            (double) size / (1024.0 * 1024.0), p);
    return buf;
}

// Matches ggml's CPU alignment; the exporter takes one memref per layer from each tensor's own data
// pointer, so nothing depends on a particular stride.
size_t kvAlignment(ggml_backend_buffer_type_t) { return 32; }

bool kvIsHost(ggml_backend_buffer_type_t) { return true; }

}  // namespace

extern "C" ggml_backend_buffer_type_t tsi_mlir_kv_buffer_type(void) {
    const Driver & d = driver();
    // Null means "use llama's normal CPU cache", the same contract as when the export path is off.
    if (!d.dram_alloc || !d.enabled || !d.enabled()) {
        return nullptr;
    }

    // Designated initializers are avoided so this compiles the same way against either generation of
    // the struct; anything not named here is value-initialized to null, which ggml treats as "not
    // provided" for the optional slots.
    static ggml_backend_buffer_type buft = {};
    static bool init = [] {
        buft.iface.get_name      = kvName;
        buft.iface.alloc_buffer  = kvAlloc;
        buft.iface.get_alignment = kvAlignment;
        buft.iface.is_host       = kvIsHost;
        buft.device              = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        return true;
    }();
    (void) init;
    return &buft;
}
