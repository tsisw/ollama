# Running ollama on the TSI compiled path

`ollama run` normally computes every graph with ggml on the CPU. With this component built in, a
model instead has its whole forward graph intercepted in `llama_context::graph_compute`, lowered to
linalg MLIR, compiled per shape, and executed on TXEs. Nothing about the ollama CLI or its HTTP API
changes; only where the arithmetic happens does.

The accelerator lives behind `libtsi-mlir-driver`, a shared library built in
[mlir-llama.cpp](https://github.com/tsisw/mlir-llama.cpp) and `dlopen`'d at first use. ollama
contributes one translation unit, [`Loader.cpp`](Loader.cpp), which forwards the five hook symbols
llama.cpp calls. That indirection is the point: MLIR and its ~400 archives stay behind the `.so`, so
enabling this costs ollama one small object rather than a new build dependency.

**With the driver absent nothing breaks.** The loader says so once and llama computes every graph
itself, which is the same result as `TSI_MLIR_EXPORT=0`.

---

## Prerequisites

| What | Provides | Typical location |
|---|---|---|
| mlir-compiler install | TSI runtime shim, TXE kernel libraries, `tsi_mlir` bindings | `../mlir-compiler/install` |
| llvm-project-private | the MLIR C++ libraries (`lib/cmake/mlir`) | `../llvm-project-private/install` |
| mlir-llama.cpp checkout | builds `libtsi-mlir-driver` | `../mlir-llama.cpp` |
| Go 1.24+ | builds ollama | |

An aarch64 SDK needs an aarch64 host: the `tsi_mlir` bindings are compiled per architecture and per
CPython minor version.

---

## Step 1 — build the driver

The driver is built in mlir-llama.cpp, but a driver **for ollama** has to be compiled against
**ollama's** ggml headers. That is what `TSI_HOST_GGML_DIR` does below. It defaults to
mlir-llama.cpp's own ggml, so that repo builds and tests standalone with no knowledge of ollama;
only this cross-repo build overrides it.

The override is not optional here. The driver interprets graphs by reading `tensor->op` and `->src[]`
out of the host's own structures, and `ggml_op` values shift between llama.cpp generations:
`GGML_OP_ROPE` is 52 in ollama and 53 in mlir-llama.cpp, whose enum has 102 operations against 91. A
driver built against the wrong headers matches nothing, sees a forward with neither rope nor an
embedding lookup, and classifies every graph as `skip`. Output stays correct because llama recomputes
it, so the failure is silent.

One driver binary per host generation is inherent to interpreting ggml graphs. Only the `.so`
boundary is reusable, not a prebuilt binary.

```sh
cd ../mlir-llama.cpp

# once, ~1 GB, mostly torch. The Python version is detected from the bindings, not assumed.
./tsi_mlir_export/utils/setup-venv.sh <mlir-compiler>/install

cmake -B build-ollama-drv -DLLAMA_CURL=OFF \
      -DMLIR_DIR=<llvm-project-private>/install/lib/cmake/mlir \
      -DTSI_RT_ROOT=<mlir-compiler>/install \
      -DTSI_HOST_GGML_DIR=$PWD/../ollama/ml/backend/ggml/ggml

cmake --build build-ollama-drv --target tsi-mlir-driver -j8
```

Confirm the configure found everything. **A missing dependency is a STATUS line, not an error** —
cmake omits the targets and the build still goes green:

```
-- mlir-export: MLIR 22.0.0git from .../lib/cmake/mlir
-- mlir-export: TSI runtime shim .../install/lib/libTsavRTShimCAPI...
```

`make -C tsi_mlir_export txe-check-deps` reports the same without configuring.

### Finding `MLIR_DIR`

MLIR is never installed into mlir-compiler's own prefix — LLVM is a fetched dependency there. If
`<llvm-project-private>/install` does not exist, it lives in a build tree:

```sh
find <mlir-compiler> <llvm-project-private> -name MLIRConfig.cmake 2>/dev/null
# e.g. <mlir-compiler>/build/_deps/llvm-build/lib/cmake/mlir
```

### Platform notes

| | macOS | Linux |
|---|---|---|
| driver name | `libtsi-mlir-driver.dylib` | `libtsi-mlir-driver.so` |
| ggml resolution | `-undefined dynamic_lookup` | left undefined, resolved from the host |
| symbol binding | two-level namespace, automatic | `-Bsymbolic-functions`, applied by CMake |

Both are handled by the build. They are listed because they are the two places where the platforms
genuinely differ.

---

## Step 2 — build ollama

**macOS:**

```sh
cd ../ollama
go build -o ollama-tsi .
```

**Linux — the link flag is required:**

```sh
cd ../ollama
go build -ldflags='-extldflags "-Wl,--export-dynamic"' -o ollama-tsi .
```

The driver deliberately leaves ggml symbols undefined for the host to satisfy. ggml here is compiled
*into the executable* rather than into a library, so under ELF its symbols are not in the dynamic
symbol table unless you ask for them, and `dlopen` fails with:

```
undefined symbol: ggml_graph_node
```

Mach-O resolves the same references through `-undefined dynamic_lookup`, so macOS needs nothing.

---

## Step 3 — register a model

Any GGUF works. Unquantized is the easier first target; for a quantized model see *Quantized models*
below.

```sh
printf 'FROM /path/to/model.gguf\n' > Modelfile.tsi

./ollama-tsi serve &                       # the server must be up before create
sleep 3
./ollama-tsi create tsi-test -f Modelfile.tsi
```

---

## Step 4 — run

The environment is not optional; four of these settings fail in ways that look like something else.

**macOS:**

```sh
MC=<mlir-compiler>/install

TSI_MLIR_LIB=<mlir-llama.cpp>/build-ollama-drv/bin/libtsi-mlir-driver.dylib \
TSI_MLIR_EXPORT=1 TSI_MLIR_SKIP=0 TSI_MLIR_WEIGHT_ARGS=1 \
TSI_MLIR_DIR=/tmp/tsi-artifacts USER_DRAM_SIZE=2048 TSI_NUM_TXES=1 \
DYLD_LIBRARY_PATH="$MC/lib:$(ls -d $MC/*/lib | paste -sd: -)" \
OLLAMA_NUM_PARALLEL=1 OLLAMA_LOAD_TIMEOUT=30m \
  ./ollama-tsi serve > ollama.log 2>&1 &

sleep 3
./ollama-tsi run tsi-test "hello world"
```

**Linux:**

```sh
MC=<mlir-compiler>/install

env -u LD_LIBRARY_PATH \
  TSI_MLIR_LIB=<mlir-llama.cpp>/build-ollama-drv/bin/libtsi-mlir-driver.so \
  TSI_MLIR_EXPORT=1 TSI_MLIR_SKIP=0 TSI_MLIR_WEIGHT_ARGS=1 \
  TSI_MLIR_DIR=/tmp/tsi-artifacts USER_DRAM_SIZE=2048 TSI_NUM_TXES=1 \
  LD_LIBRARY_PATH="<gcc>/lib64:$MC/lib:$(ls -d $MC/*/lib | paste -sd: -)" \
  OLLAMA_NUM_PARALLEL=1 OLLAMA_LOAD_TIMEOUT=30m \
  ./ollama-tsi serve > ollama.log 2>&1 &

sleep 3
./ollama-tsi run tsi-test "hello world"
```

`<gcc>/lib64` is needed only where the TSI runtime was built with a newer GCC than the system one.
The symptom of getting it wrong is `CXXABI_1.3.15 not found` at `dlopen`.

**The first request is slow.** Two graphs are compiled, one for prefill and one for decode, each
invoking `compile_graph.py`. Budget minutes and do not kill it. Artifacts are written to
`TSI_MLIR_DIR` keyed by graph shape, so later runs of the same prompt length skip compilation
entirely. `OLLAMA_LOAD_TIMEOUT=30m` exists because compilation produces no progress events and
ollama's default stall timeout is five minutes.

---

## Step 5 — verify it actually ran compiled

**Do not judge this by the output.** Every failure mode below produces fluent, correct text, because
llama recomputes any graph the driver declines. These counters are the result:

```sh
# each phase has its own marker; counting only the first is the classic mistake here
echo "prefill compiled : $(grep -c 'running compiled prefill' ollama.log)"
echo "decode steps     : $(grep -c 'decode step'              ollama.log)"
echo "chunks ran       : $(grep -c 'chunk ran in'             ollama.log)"
echo "NOT EXPORTING    : $(grep -c 'NOT EXPORTING'            ollama.log)"   # must be 0
echo "SKIPPED          : $(grep -c 'SKIPPED'                  ollama.log)"   # must be 0
grep -o 'phase=[a-z]*' ollama.log | sort | uniq -c
```

`running compiled` is printed for **prefill only**. Decode logs `decode step N: pos ...` and a batched
chunk logs `chunk ran in ...`, so a run where prefill is compiled and decode silently falls back
still shows `running compiled: 1`. A graph header (`phase=decode`) means the graph was *recognized*,
not that it executed compiled. Require at least one marker from every phase the run produced, and
require both negative counters to be zero.

A healthy first run looks like:

```
[tsi-mlir] driver loaded from .../libtsi-mlir-driver.so (ABI 2)
[tsi-mlir] TSI host runtime initialized (1 TXE)
[tsi-mlir] KV cache: 4.00 MiB in TSI DRAM at 0x...
[tsi-mlir] --- graph: phase=prefill tokens=2 pos=0..1 nodes=294 ---
[tsi-mlir] live dims: layers=8 hidden=64 vocab=32000 ... moe_top_k=0
[tsi-mlir] 78 leafs -> 78 args + 0 baked constants, 17 results
[tsi-mlir] running compiled prefill: 78 args, 17 results, logits [32000 x 2]
```

---

## Environment reference

| Variable | Value | Why |
|---|---|---|
| `TSI_MLIR_LIB` | path to the driver | otherwise the loader searches by soname and reports one line if absent |
| `TSI_MLIR_EXPORT` | `1` | master switch; `0` is plain CPU llama |
| `TSI_MLIR_SKIP` | `0` | **required under ollama.** The driver defaults to discarding the first graph because under `llama-completion` that graph is llama's warmup. ollama's runner routes no warmup graph through `graph_compute`, so its first graph *is* the prompt — the default throws the prefill away and only decode runs compiled |
| `TSI_MLIR_WEIGHT_ARGS` | `1` | weights as arguments rather than baked constants. Required for 1B+ models, whose constant pool overflows Mach-O's 32-bit relocation offsets |
| `TSI_MLIR_DIR` | a writable directory | compiled artifacts, keyed by graph shape. Persist it to skip recompiling |
| `USER_DRAM_SIZE` | MiB | the TSI DRAM pool the KV cache is allocated from. Too small fails at `kvAlloc` with a diagnostic |
| `TSI_NUM_TXES` | `1`..`20` | must be identical at compile and run time; the driver reports both |
| `OMP_NUM_THREADS` | `= TSI_NUM_TXES` | multi-TXE host code is OpenMP-parallel; one thread serializes it |
| `OLLAMA_NUM_PARALLEL` | `1` | graph reconstruction handles one stream; more adds an `n_stream` dimension the driver declines |
| `OLLAMA_LOAD_TIMEOUT` | `30m` | compilation emits no progress events and the default stall timeout is 5 minutes |

---

## Quantized models

Quantized weights are unreadable to the driver unless repacking is off:
`ggml_backend_cpu_repack_buffer_type()` is registered as an extra CPU buffer type whose `is_host` is
null, so every graph is declined with `NOT EXPORTING`. f32 and bf16 models never reach that path.

ollama's runner exposes no `--no-repack` flag, so a quantized model needs one added to the runner
invocation. Until then, prefer unquantized GGUFs on this path.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `undefined symbol: ggml_graph_node` at load | ollama not linked with `--export-dynamic` (Linux) |
| `CXXABI_1.3.15 not found` | system libstdc++ older than the one the TSI runtime was built with; put that GCC's `lib64` first on `LD_LIBRARY_PATH` |
| Runner spins at 100% CPU forever inside `llama_init_from_model` | driver built without `-Bsymbolic-functions`; the host preempts the driver's internal `tsi_mlir_export_*` calls and the two forwarders recurse into each other. Fixed in mlir-llama.cpp's CMake; rebuild the driver |
| `libtxe-ffm-cpp-native.so: cannot open shared object file`, then `SIGABRT` | the TXE kernel libraries are not on the loader path. They live in `<install>/<component>/lib`, not `<install>/lib` |
| `timed out waiting for llama runner to start - progress 1.00` | graph compilation during load; raise `OLLAMA_LOAD_TIMEOUT` |
| Correct output, no phase markers at all | the driver never ran. Check `TSI_MLIR_SKIP=0`, `TSI_HOST_GGML_DIR`, and that `TSI_MLIR_LIB` points at a driver that actually loaded |
| `phase=decode` appears but no `decode step` lines | the decode graph was recognized and then not executed compiled. `decode graph: N nodes, pos P` marks recognition; look for a `decode expects one cell` or similar line after it |
| Every graph `NOT EXPORTING ... offloaded (non-host) memory` | weights are not host-readable. Quantized model without repacking off, or GPU offload |

## Known gaps

- A runtime error inside the TSI runtime throws `std::runtime_error` with nothing catching it, so
  `terminate` kills the runner. ollama reports a 500 and unloads the model, with the real cause only
  in the log.
- Quantized models need a `--no-repack` equivalent on ollama's runner.
- `keep_alive` and concurrency above one stream are untested on this path.
