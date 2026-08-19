package llama

// #cgo CXXFLAGS: -std=c++17
// #cgo CPPFLAGS: -I${SRCDIR}/../include
// #cgo CPPFLAGS: -I${SRCDIR}/../../../ml/backend/ggml/ggml/include
// #cgo CPPFLAGS: -DLLAMA_TSI_MLIR_EXPORT=1
// #cgo windows CPPFLAGS: -D_WIN32_WINNT=0x0602
import "C"

import (
	_ "github.com/ollama/ollama/ml/backend/ggml/ggml/src"
	// Satisfies the tsi_mlir_export_* symbols the hooks in llama-context.cpp and
	// llama-kv-cache.cpp reference. Blank-imported for the link side effect: without it,
	// LLAMA_TSI_MLIR_EXPORT compiles those call sites and nothing defines them.
	_ "github.com/ollama/ollama/tsi/mlir"
)
