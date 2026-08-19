#!/usr/bin/env bash
# Checks that tsi-mlir-export-hook.patch still applies to the vendor it targets.
#
# The MLIR export hook cannot live in llama/llama.cpp: Makefile.sync rsyncs that directory from
# llama/vendor with --delete, so the edits are dropped at the next version bump. It lives here as a
# patch instead, applied to the vendor by tsi-ollama-bundle.sh after tsi-consolidated-patches.patch.
#
# What makes this worth automating is the failure mode. A patch that stops applying does not break
# the build: the bundle script logs "already applied or not applicable" and carries on, ollama builds,
# answers correctly, and simply never reaches the accelerator. There is no error to grep for, because
# the code that would print "NOT EXPORTING" is exactly what went missing. The only symptom is the
# absence of "running compiled" in a log nobody is reading.
#
# The order matters and is checked here too: the hook's context lines assume the consolidated patch
# has already been applied, so applying it first is not a stylistic choice.
#
# Deliberately NOT compared against the checked-in llama/llama.cpp. That directory has drifted from
# vendor@FETCH_HEAD + patches in both directions (see the patch header), so it is not a valid
# baseline for anything. The vendor is.
set -uo pipefail

cd "$(git rev-parse --show-toplevel)" || exit 1

VENDOR=llama/vendor
FETCH_HEAD=$(awk -F= '/^FETCH_HEAD=/{print $2}' Makefile.sync)
PATCHES=(tsi-consolidated-patches.patch tsi-ollama-backend-profile.patch tsi-mlir-export-hook.patch)

if [ ! -d "$VENDOR" ]; then
    echo "error: $VENDOR is missing. Run: make -f Makefile.sync checkout" >&2
    exit 1
fi

# A dirty vendor would make this meaningless, so reset to the pin first.
#
# `git clean` as well as `checkout -f`, because the consolidated patch CREATES files
# (ggml/src/mem_nvml.cpp, mem_hip.cpp). checkout -f only restores tracked ones, so without the clean
# a second run fails with "already exists in working directory" - which looks exactly like a broken
# patch and is not one. The vendor is a disposable working copy, so discarding untracked files there
# is safe; that is also why .gitignore lists it.
if ! git -C "$VENDOR" checkout -qf "$FETCH_HEAD" 2>/dev/null; then
    echo "error: cannot check out $FETCH_HEAD in $VENDOR. Run: make -f Makefile.sync checkout" >&2
    exit 1
fi
git -C "$VENDOR" clean -qfd

rc=0
for patch in "${PATCHES[@]}"; do
    if [ ! -f "llama/patches/${patch}" ]; then
        echo "error: llama/patches/${patch} not found" >&2
        rc=1
        continue
    fi
    if ! git -C "$VENDOR" apply "../patches/${patch}" 2>/dev/null; then
        echo "error: ${patch} does not apply to vendor@${FETCH_HEAD}" >&2
        git -C "$VENDOR" apply --check "../patches/${patch}" 2>&1 | head -8 | sed 's/^/       /' >&2
        rc=1
        break
    fi
    echo "ok: ${patch} applies"
done

# Leave the vendor as the bundle script would: patched, ready to rsync.
if [ "$rc" -ne 0 ]; then
    echo >&2
    echo "The hook would be silently dropped at the next sync, disabling the compiled path with no" >&2
    echo "error message. Regenerate: apply the earlier patches to $VENDOR, redo the hook edits there," >&2
    echo "then 'git -C $VENDOR diff' and prepend the existing header." >&2
fi
exit "$rc"
