#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[install-nsight-systems] %s\n' "$*" >&2
}

die() {
  log "ERROR: $*"
  exit 1
}

if command -v nsys >/dev/null 2>&1; then
  log "nsys already installed: $(command -v nsys)"
  nsys --version || true
  exit 0
fi

[[ "$(uname -s)" == "Linux" ]] || die "Nsight Systems CLI installation is only supported by this helper on Linux."
command -v apt-get >/dev/null 2>&1 || die "apt-get not found. Use the GPU image/provider package flow for Nsight Systems."

if [[ "${EUID}" -eq 0 ]]; then
  SUDO=()
elif command -v sudo >/dev/null 2>&1; then
  SUDO=(sudo)
else
  die "Run as root or install sudo before using this helper."
fi

export DEBIAN_FRONTEND=noninteractive

install_package() {
  local package="$1"
  log "Trying apt install ${package}"
  "${SUDO[@]}" apt-get install -y --no-install-recommends "${package}"
}

verify_nsys() {
  if command -v nsys >/dev/null 2>&1; then
    log "Installed nsys: $(command -v nsys)"
    nsys --version || true
    return 0
  fi
  return 1
}

log "Updating apt package index"
"${SUDO[@]}" apt-get update

if install_package nsight-systems-cli || install_package nsight-systems; then
  verify_nsys || die "Nsight Systems package installed, but nsys is still not on PATH."
  exit 0
fi

log "Nsight Systems package was not available from current apt sources."
log "Adding NVIDIA devtools apt repository for Ubuntu containers."

if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  . /etc/os-release
else
  die "/etc/os-release not found; cannot determine Ubuntu version."
fi

[[ "${ID:-}" == "ubuntu" ]] || die "This helper only adds the NVIDIA devtools repo for Ubuntu."

ubuntu_version="$(printf '%s' "${VERSION_ID}" | tr -d '.')"
arch="$(dpkg --print-architecture)"
keyring="/usr/share/keyrings/nvidia-devtools.gpg"
repo_file="/etc/apt/sources.list.d/nvidia-devtools.list"

"${SUDO[@]}" apt-get install -y --no-install-recommends ca-certificates gnupg wget
"${SUDO[@]}" rm -f "${keyring}"
wget -qO- https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub \
  | "${SUDO[@]}" gpg --dearmor -o "${keyring}"

printf 'deb [signed-by=%s] https://developer.download.nvidia.com/devtools/repos/ubuntu%s/%s/ /\n' \
  "${keyring}" "${ubuntu_version}" "${arch}" \
  | "${SUDO[@]}" tee "${repo_file}" >/dev/null

"${SUDO[@]}" apt-get update

if install_package nsight-systems-cli || install_package nsight-systems; then
  verify_nsys || die "Nsight Systems package installed, but nsys is still not on PATH."
  exit 0
fi

die "Could not install Nsight Systems CLI. Choose a GPU image with nsys preinstalled or follow NVIDIA's Nsight Systems installation guide."
