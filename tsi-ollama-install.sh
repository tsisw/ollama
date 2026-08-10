#!/usr/bin/env bash
# This script installs TSI Ollama on ARM Linux. The script is primarily changes to only work for TSISW OLLAMA software

set -eu

red="$( (/usr/bin/tput bold || :; /usr/bin/tput setaf 1 || :) 2>&-)"
plain="$( (/usr/bin/tput sgr0 || :) 2>&-)"

status() { echo ">>> $*" >&2; }
error() { echo "${red}ERROR:${plain} $*"; exit 1; }
warning() { echo "${red}WARNING:${plain} $*"; }

# Sets ENABLE_PROFILING=1 if the user passes --enable-profiling on the
# command line (e.g. `./install-tsi-ollama.sh --enable-profiling`).
# This value is later used in configure_systemd() to decide whether to
# inject an ENABLE_PROFILING=1 environment variable into the systemd
# service unit.
ENABLE_PROFILING=0
for arg in "$@"; do
    case "$arg" in
        --enable-profiling) ENABLE_PROFILING=1 ;;
        *) ;;
    esac
done

TEMP_DIR=$(mktemp -d)
cleanup() { rm -rf $TEMP_DIR; }
trap cleanup EXIT

available() { command -v $1 >/dev/null; }
require() {
    local MISSING=''
    for TOOL in $*; do
        if ! available $TOOL; then
            MISSING="$MISSING $TOOL"
        fi
    done

    echo $MISSING
}

[ "$(uname -s)" = "Linux" ] || error 'This script is intended to run on Linux only.'

ARCH=$(uname -m)
case "$ARCH" in
    aarch64|arm64) ARCH="arm64" ;;
    *) error "Unsupported architecture: $ARCH" ;;
esac

IS_WSL2=false

KERN=$(uname -r)
case "$KERN" in
    *icrosoft*WSL2 | *icrosoft*wsl2) IS_WSL2=true;;
    *icrosoft) error "Microsoft WSL1 is not currently supported. Please use WSL2 with 'wsl --set-version <distro> 2'" ;;
    *) ;;
esac

SUDO=
if [ "$(id -u)" -ne 0 ]; then
    # Running as root, no need for sudo
    if ! available sudo; then
        error "This script requires superuser permissions. Please re-run as root."
    fi

    SUDO="sudo"
fi

NEEDS=$(require curl awk grep sed tee xargs)
if [ -n "$NEEDS" ]; then
    status "ERROR: The following tools are required but missing:"
    for NEED in $NEEDS; do
        echo "  - $NEED"
    done
    exit 1
fi

for BINDIR in /usr/local/bin /usr/bin /bin; do
    echo $PATH | grep -q $BINDIR && break || continue
done
OLLAMA_INSTALL_DIR=$(dirname ${BINDIR})

if [ -d "$OLLAMA_INSTALL_DIR/lib/ollama" ] ; then
    status "Cleaning up old version at $OLLAMA_INSTALL_DIR/lib/ollama"
    $SUDO rm -rf "$OLLAMA_INSTALL_DIR/lib/ollama"
fi
status "Installing ollama to $OLLAMA_INSTALL_DIR"
$SUDO install -o0 -g0 -m755 -d $BINDIR
$SUDO install -o0 -g0 -m755 -d "$OLLAMA_INSTALL_DIR/lib/ollama"
if [ ! -d "$(pwd)/ollama-arm64-release" ]; then
status "Downloading Linux ${ARCH} bundle"
    wget "https://github.com/tsisw/ollama/releases/download/v0.12.6-tsi-v0.0.17/ollama-arm64-release.tar.gz" \
        -O ollama-arm64-release.tar.gz && \
    $SUDO tar -xvzf ollama-arm64-release.tar.gz -C "$OLLAMA_INSTALL_DIR"
else
    echo "Directory ollama-arm64-release already exists. Skipping download."
    ln -s "$(pwd)/ollama-arm64-release" "$OLLAMA_INSTALL_DIR"
fi

install_aot_ggml_lib_link() {
    # Define source directories
    SOURCE_DIRS=(
        "$OLLAMA_INSTALL_DIR/ollama-arm64-release/lib"
        "/usr/bin/tsi/bin/aot-tests/lib"
    )

    # Loop through each source directory
    for DIR in "${SOURCE_DIRS[@]}"; do
        for LIB in "$DIR"/lib*; do
            if [ -f "$LIB" ]; then
                BASENAME=$(basename "$LIB")
                TARGET="/usr/lib/$BASENAME"

                if [ -L "$TARGET" ]; then
                    echo "Symlink already exists: $TARGET -> $(readlink "$TARGET")"
                elif [ -e "$TARGET" ]; then
                    echo "File exists at $TARGET but is not a symlink. Skipping."
                else
                    echo "Creating symlink: $TARGET -> $LIB"
                    $SUDO ln -s "$LIB" "$TARGET"
                fi
            fi
        done
    done
}

status "Untarred the Tar bundle ${ARCH} "
if [ "$OLLAMA_INSTALL_DIR/ollama-arm64-release/bin/ollama" != "$BINDIR/ollama" ] ; then
    status "Making ollama accessible in the PATH in $BINDIR"
    $SUDO ln -sf "$OLLAMA_INSTALL_DIR/ollama-arm64-release/bin/ollama" "$BINDIR/ollama"
    cd $OLLAMA_INSTALL_DIR/ollama-arm64-release/tsi-ggml/                                                                                  
    $SUDO $OLLAMA_INSTALL_DIR/ollama-arm64-release/tsi-ggml/ggml.sh
    # Copy all the libraries to /usr/lib as the symbolic links as well as the environment
    # varaibles are not working for the libraries when invoked by the service
    # this needs further troubleshoot.
    # $SUDO cp $OLLAMA_INSTALL_DIR/ollama-arm64-release/lib/lib* /usr/lib/
    # $SUDO cp /usr/bin/tsi/bin/aot-tests/lib/lib* /usr/lib/
    install_aot_ggml_lib_link
fi

install_success() {
    status 'The Ollama API is now available at 127.0.0.1:11434.'
    status 'Install complete. Run "ollama" from the command line.'
}
trap install_success EXIT

# Everything from this point onwards is optional.

configure_systemd() {
    if ! id ollama >/dev/null 2>&1; then
        status "Creating ollama user..."
        $SUDO useradd -r -s /bin/false -U -m -d /usr/share/ollama ollama
    fi
    if getent group render >/dev/null 2>&1; then
        status "Adding ollama user to render group..."
        $SUDO usermod -a -G render ollama
    fi
    if getent group video >/dev/null 2>&1; then
        status "Adding ollama user to video group..."
        $SUDO usermod -a -G video ollama
    fi

    status "Adding current user to ollama group..."
    $SUDO usermod -a -G ollama $(whoami)

    status "Creating ollama systemd service..."
    # If --enable-profiling was passed (see ENABLE_PROFILING above), add
    # an Environment="ENABLE_PROFILING=1" line to the generated unit file.
    # Otherwise this stays empty and the heredoc just prints a blank line
    # in its place (harmless - systemd ignores blank lines in unit files).
    PROFILING_ENV_LINE=""
    if [ "$ENABLE_PROFILING" -eq 1 ]; then
        PROFILING_ENV_LINE='Environment="ENABLE_PROFILING=1"'
    fi
    cat <<EOF | $SUDO tee /etc/systemd/system/ollama.service >/dev/null
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=$BINDIR/ollama serve
#User=ollama
#Group=ollama
Environment="OLLAMA_HOST=0.0.0.0"
Environment="HOME=/usr/share/ollama"
Restart=always
RestartSec=3
Environment="PATH=$PATH"
Environment="LD_LIBRARY_PATH=/usr/bin/tsi/bin/aot-tests/lib/:/usr/local/ollama-arm64-release/lib/:/usr/local/ollama-arm64-release/bin/:${LD_LIBRARY_PATH:-}"
Environment="OLLAMA_MODELS=/tsi/ollama-models/"
${PROFILING_ENV_LINE}
[Install]
WantedBy=default.target
EOF
    SYSTEMCTL_RUNNING="$(systemctl is-system-running || true)"
    case $SYSTEMCTL_RUNNING in
        running|degraded)
            status "Enabling and starting ollama service..."
            $SUDO systemctl daemon-reload
            $SUDO systemctl enable ollama

            start_service() { $SUDO systemctl restart ollama; }
            trap start_service EXIT
            ;;
        *)
            warning "systemd is not running"
            if [ "$IS_WSL2" = true ]; then
                warning "see https://learn.microsoft.com/en-us/windows/wsl/systemd#how-to-enable-systemd to enable it"
            fi
            ;;
    esac
}

if available systemctl; then
    status "Configuring ollama service and systemd..."
    configure_systemd
fi

. /etc/os-release

status "TSI OPU ready."
install_success
