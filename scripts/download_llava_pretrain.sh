#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data/LLaVA-Pretrain"
DATASET_ID="liuhaotian/LLaVA-Pretrain"
ANNOTATION_FILE="${DATA_DIR}/blip_laion_cc_sbu_558k.json"
IMAGES_ARCHIVE="${DATA_DIR}/images.zip"

log() {
  printf "\n[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf "Error: required command '%s' was not found.\n" "$1" >&2
    exit 1
  fi
}

has_extracted_images() {
  find "${DATA_DIR}" -mindepth 2 -maxdepth 2 -type f -name '*.jpg' -print -quit | grep -q .
}

download_dataset() {
  if [[ -f "${ANNOTATION_FILE}" ]] && { [[ -f "${IMAGES_ARCHIVE}" ]] || has_extracted_images; }; then
    log "Found LLaVA-Pretrain data in ${DATA_DIR}"
    return
  fi

  require_cmd hf
  mkdir -p "${DATA_DIR}"
  log "Downloading ${DATASET_ID} to ${DATA_DIR}"
  hf download "${DATASET_ID}" --repo-type dataset --local-dir "${DATA_DIR}"
}

extract_images() {
  if has_extracted_images; then
    log "Found extracted image shards in ${DATA_DIR}"
    return
  fi

  if [[ ! -f "${IMAGES_ARCHIVE}" ]]; then
    printf "Error: expected archive not found: %s\n" "${IMAGES_ARCHIVE}" >&2
    exit 1
  fi

  require_cmd unzip
  log "Extracting ${IMAGES_ARCHIVE} -> ${DATA_DIR}"
  unzip -q "${IMAGES_ARCHIVE}" -d "${DATA_DIR}"
}

main() {
  download_dataset
  extract_images

  log "Done. Expected layout:"
  printf "%s\n" \
    "  ${ANNOTATION_FILE}" \
    "  ${DATA_DIR}/00000/" \
    "  ${DATA_DIR}/00001/"
}

main "$@"
