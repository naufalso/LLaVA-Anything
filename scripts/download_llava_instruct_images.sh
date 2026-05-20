#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data/LLaVA-Instruct-150K"

COCO_URL="http://images.cocodataset.org/zips/train2017.zip"
GQA_URL="https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip"
TEXTVQA_URL="https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip"
VG_PART1_URL="https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip"
VG_PART2_URL="https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip"
OCR_VQA_GDRIVE_FOLDER="https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing"

log() {
  printf "\n[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf "Error: required command '%s' was not found.\n" "$1" >&2
    exit 1
  fi
}

download_file() {
  local url="$1"
  local output="$2"

  if [[ -f "${output}" ]]; then
    log "Found existing archive: ${output}"
    return
  fi

  mkdir -p "$(dirname "${output}")"
  log "Downloading ${url}"

  if command -v wget >/dev/null 2>&1; then
    wget -c -O "${output}" "${url}"
  elif command -v curl >/dev/null 2>&1; then
    curl -L --fail --continue-at - --output "${output}" "${url}"
  else
    printf "Error: install either wget or curl to download archives.\n" >&2
    exit 1
  fi
}

unzip_if_needed() {
  local archive="$1"
  local destination="$2"
  local expected_path="$3"

  if [[ -d "${expected_path}" ]]; then
    log "Found extracted data: ${expected_path}"
    return
  fi

  require_cmd unzip
  mkdir -p "${destination}"
  log "Extracting ${archive} -> ${destination}"
  unzip -q "${archive}" -d "${destination}"
}

rename_images_to_jpg() {
  local image_dir="$1"

  if [[ ! -d "${image_dir}" ]]; then
    return
  fi

  log "Normalizing OCR-VQA image filenames to .jpg"
  find "${image_dir}" -type f | while IFS= read -r file; do
    local dir base stem lower target
    dir="$(dirname "${file}")"
    base="$(basename "${file}")"
    stem="${base%.*}"
    lower="$(printf '%s' "${base##*.}" | tr '[:upper:]' '[:lower:]')"

    [[ "${lower}" == "jpg" ]] && continue
    target="${dir}/${stem}.jpg"

    if [[ -e "${target}" ]]; then
      printf "Warning: skipping rename because target exists: %s\n" "${target}" >&2
      continue
    fi

    mv "${file}" "${target}"
  done
}

download_coco() {
  local archive="${DATA_DIR}/coco/train2017.zip"
  download_file "${COCO_URL}" "${archive}"
  unzip_if_needed "${archive}" "${DATA_DIR}/coco" "${DATA_DIR}/coco/train2017"
}

download_gqa() {
  local archive="${DATA_DIR}/gqa/images.zip"
  download_file "${GQA_URL}" "${archive}"
  unzip_if_needed "${archive}" "${DATA_DIR}/gqa" "${DATA_DIR}/gqa/images"
}

download_ocr_vqa() {
  local ocr_dir="${DATA_DIR}/ocr_vqa"
  local image_dir="${ocr_dir}/images"
  local loader="${ocr_dir}/loadDataset.py"
  local metadata="${ocr_dir}/dataset.json"

  if [[ ! -f "${loader}" || ! -f "${metadata}" ]]; then
    require_cmd gdown
    mkdir -p "${ocr_dir}"
    log "Downloading OCR-VQA metadata and loader from Google Drive folder"
    gdown --folder "${OCR_VQA_GDRIVE_FOLDER}" --output "${ocr_dir}" --remaining-ok
  else
    log "Found OCR-VQA metadata and loader in ${ocr_dir}"
  fi

  if [[ ! -f "${loader}" || ! -f "${metadata}" ]]; then
    printf "Error: OCR-VQA requires both loadDataset.py and dataset.json in %s.\n" "${ocr_dir}" >&2
    exit 1
  fi

  if [[ -d "${image_dir}" ]] && find "${image_dir}" -type f -name '*.jpg' -print -quit | grep -q .; then
    log "Found OCR-VQA images: ${image_dir}"
    return
  fi

  require_cmd python3
  mkdir -p "${image_dir}"
  log "Running OCR-VQA image downloader from dataset.json"
  (
    cd "${ocr_dir}"
    python3 "${loader}"
  )
  rename_images_to_jpg "${image_dir}"
}

download_textvqa() {
  local archive="${DATA_DIR}/textvqa/train_val_images.zip"
  download_file "${TEXTVQA_URL}" "${archive}"
  unzip_if_needed "${archive}" "${DATA_DIR}/textvqa" "${DATA_DIR}/textvqa/train_images"
}

download_vg() {
  local archive1="${DATA_DIR}/vg/images.zip"
  local archive2="${DATA_DIR}/vg/images2.zip"

  download_file "${VG_PART1_URL}" "${archive1}"
  download_file "${VG_PART2_URL}" "${archive2}"

  unzip_if_needed "${archive1}" "${DATA_DIR}/vg" "${DATA_DIR}/vg/VG_100K"
  unzip_if_needed "${archive2}" "${DATA_DIR}/vg" "${DATA_DIR}/vg/VG_100K_2"
}

main() {
  mkdir -p "${DATA_DIR}"

  # download_coco
  download_gqa
  download_ocr_vqa
  download_textvqa
  download_vg

  log "Done. Expected layout:"
  printf "%s\n" \
    "  ${DATA_DIR}/coco/train2017" \
    "  ${DATA_DIR}/gqa/images" \
    "  ${DATA_DIR}/ocr_vqa/images" \
    "  ${DATA_DIR}/textvqa/train_images" \
    "  ${DATA_DIR}/vg/VG_100K" \
    "  ${DATA_DIR}/vg/VG_100K_2"
}

main "$@"
