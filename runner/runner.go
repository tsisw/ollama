package runner

import (
	"log"
	"log/slog"
	"github.com/ollama/ollama/runner/llamarunner"
	"github.com/ollama/ollama/runner/ollamarunner"
)

func Execute(args []string) error {
	slog.Debug("RUNNER: Execute called", "args", args)

	if args[0] == "runner" {
		args = args[1:]
	}

	var newRunner bool
	if args[0] == "--ollama-engine" {
		args = args[1:]
		newRunner = true
	}
	// NOTE: Intentionally overriding --ollama-engine and forcing llama-runner.
	// This is required to bypass the ollama-runner mmap code path, which skips
	// load_all_data() and prevents backend-specific tensor materialization.
	newRunner = false

	slog.Debug("forcing llama-runner")
	if newRunner {
		return ollamarunner.Execute(args)
	} else {
		return llamarunner.Execute(args)
	}
}
