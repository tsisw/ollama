package runner

import (
        "fmt"
	"github.com/ollama/ollama/runner/llamarunner"
	"github.com/ollama/ollama/runner/ollamarunner"
)

func Execute(args []string) error {
        fmt.Printf("RUNNER: Execute called, args=%v\n", args)
	if args[0] == "runner" {
		args = args[1:]
	}

	var newRunner bool
	if args[0] == "--ollama-engine" {
		args = args[1:]
		newRunner = true
	}
	newRunner = false

       fmt.Printf("RUNNER: forcing llama-runner\n")
	if newRunner {
		return ollamarunner.Execute(args)
	} else {
		return llamarunner.Execute(args)
	}
}
