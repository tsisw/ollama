// Package mlir links the TSI whole-graph MLIR export loader into ollama.
//
// It contributes exactly one C++ translation unit, Loader.cpp, which provides the
// tsi_mlir_export_* symbols that llama-context.cpp and llama-kv-cache.cpp call when built with
// LLAMA_TSI_MLIR_EXPORT. Those forward to libtsi-mlir-driver, resolved with dlopen on first use.
//
// The point of the indirection is what does NOT appear here: no MLIR include path and no MLIR
// libraries. The exporter and its ~400 MLIR archives live behind the .so, so enabling the compiled
// path costs ollama one small object rather than a new build dependency. That is the same
// deployment shape ggml-tsavorite already has - drop a library next to the binary.
//
// Import this package for its side effect; there is no Go API:
//
//	import _ "github.com/ollama/ollama/tsi/mlir"
//
// At runtime, point TSI_MLIR_LIB at libtsi-mlir-driver and set TSI_MLIR_EXPORT=1. With the library
// absent the loader says so once and llama computes every graph itself.
//
// Five things are required under ollama specifically. The first two fail silently, the next two as a
// hang or an abort well into a run, and the last costs the whole decode phase:
//
//	TSI_MLIR_SKIP=0
//	  The driver defaults to discarding the first graph, because under llama-completion the first
//	  graph is llama's warmup rather than the prompt. ollama's runner does not route a warmup graph
//	  through llama_context::graph_compute, so its first graph IS the prompt - leaving the default
//	  in place throws the prefill away and only decode runs compiled.
//
//	the driver must be built against OLLAMA's ggml headers
//	  -DTSI_HOST_GGML_DIR=<ollama>/ml/backend/ggml/ggml when building libtsi-mlir-driver. The driver
//	  reads graphs by op code, and ggml_op values shift between llama.cpp generations: GGML_OP_ROPE
//	  is 52 in ollama's vendored ggml and 53 in mlir-llama.cpp's, whose enum has 102 ops against 91.
//	  A driver built against the wrong headers matches no ops, sees a forward with no rope and no
//	  embedding lookup, and classifies every graph as "skip".
//
//	link ollama with --export-dynamic (ELF only)
//	  go build -ldflags='-extldflags "-Wl,--export-dynamic"'. The driver leaves ggml undefined for
//	  the host to satisfy, and ggml here is compiled into the executable rather than a library, so
//	  without this its symbols are not in the dynamic table and dlopen fails on ggml_graph_node.
//	  Mach-O resolves these through -undefined dynamic_lookup instead, so macOS needs nothing.
//
//	put the TSI kernel libraries on LD_LIBRARY_PATH
//	  The blobs the runtime loads link against <mlir-compiler>/install/<component>/lib, not the
//	  install's top-level lib, and carry no rpath. Missing them aborts the runner at the first
//	  tsi_load_blob, after the graph has already compiled. mlir-llama.cpp's own run targets derive
//	  this path; here it is the caller's to set.
//
//	OLLAMA_FLASH_ATTENTION=1
//	  ollama defaults flash attention off, and llama sets attn_v_trans = !flash_attn, so with it off
//	  the V cache is stored transposed: [n_kv, n_head_kv, head_dim] rather than
//	  [head_dim, n_head_kv, n_kv]. The compiled decode aliases that buffer in place and reads one
//	  declared shape for K and V, so it declines the transposed layout and llama computes every decode
//	  step. Prefill is unaffected, so the run looks half-working: "running compiled prefill" appears
//	  and every decode logs "decode SKIPPED: V cache is transposed".
//
// The first two failure modes produce correct output on CPU, so the only way to know the accelerator
// ran is to check the log for "running compiled" and for the absence of "NOT EXPORTING". Correct
// output is never evidence on its own: llama recomputes whatever the driver declines. Numeric
// agreement needs TSI_MLIR_VERIFY=1, which makes the driver compare against llama's own result and
// log "-> MATCH" or "-> DIFFER" per graph.
package mlir

// The ggml/src include paths below are load-bearing, not convenience: Loader.cpp builds the KV
// cache buffer type itself, and ggml_backend_buffer_type_i has different members in different
// llama.cpp generations. Compiling it here, against ollama's own ggml headers, is what makes the
// layout right.
//
// The blank line after this paragraph is required: cgo treats the comment immediately preceding
// import "C" as C source, so prose must be a separate comment or it reaches the compiler.

// #cgo CXXFLAGS: -std=c++17
// #cgo CPPFLAGS: -I${SRCDIR}/../../ml/backend/ggml/ggml/include
// #cgo CPPFLAGS: -I${SRCDIR}/../../ml/backend/ggml/ggml/src
// #cgo CPPFLAGS: -I${SRCDIR}/../../ml/backend/ggml/ggml/src/ggml-cpu
// #cgo !windows LDFLAGS: -ldl
import "C"
