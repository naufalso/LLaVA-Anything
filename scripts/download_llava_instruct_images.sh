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
  local complete_marker="${ocr_dir}/.images_complete"

  if [[ ! -f "${loader}" || ! -f "${metadata}" ]]; then
    require_cmd gdown
    mkdir -p "${ocr_dir}"
    log "Downloading OCR-VQA metadata and loader from Google Drive folder"
    gdown --folder "${OCR_VQA_GDRIVE_FOLDER}" --output "${ocr_dir}" --continue
  else
    log "Found OCR-VQA metadata and loader in ${ocr_dir}"
  fi

  if [[ ! -f "${metadata}" ]]; then
    printf "Error: OCR-VQA requires dataset.json in %s.\n" "${ocr_dir}" >&2
    exit 1
  fi

  if [[ -f "${complete_marker}" ]]; then
    log "Found OCR-VQA images: ${image_dir}"
    return
  fi

  require_cmd python3
  mkdir -p "${image_dir}"
  log "Downloading OCR-VQA images from dataset.json with ${OCR_VQA_WORKERS:-8} worker(s)"
  python3 - "${metadata}" "${image_dir}" "${complete_marker}" "${OCR_VQA_WORKERS:-8}" <<'PYTHON'
import concurrent.futures
import contextlib
import json
import shutil
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback for minimal environments.
    tqdm = None

metadata_path = Path(sys.argv[1])
image_dir = Path(sys.argv[2])
complete_marker = Path(sys.argv[3])
try:
    max_workers = max(1, int(sys.argv[4]))
except ValueError:
    raise SystemExit(f"OCR_VQA_WORKERS must be a positive integer, got: {sys.argv[4]!r}")

failure_log = image_dir.parent / "ocr_vqa_failed_downloads.txt"
max_retries = 5
timeout_seconds = 30

with metadata_path.open("r", encoding="utf-8") as handle:
    data = json.load(handle)

records = list(data.items())


def download_one(record):
    image_id, item = record
    url = item.get("imageURL")
    if not url:
        return ("skipped", image_id, url, "missing imageURL")

    suffix = Path(urllib.parse.urlparse(url).path).suffix or ".jpg"
    output = image_dir / f"{image_id}{suffix}"
    if output.exists():
        return ("skipped", image_id, url, "exists")

    tmp_output = output.with_suffix(output.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(1, max_retries + 1):
        try:
            with contextlib.closing(urllib.request.urlopen(request, timeout=timeout_seconds)) as response:
                with tmp_output.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
            tmp_output.replace(output)
            return ("downloaded", image_id, url, "")
        except Exception as exc:
            tmp_output.unlink(missing_ok=True)
            if attempt == max_retries:
                return ("failed", image_id, url, repr(exc))
            time.sleep(min(2 * attempt, 10))
    return ("failed", image_id, url, "exhausted retries")


failures = []
completed = 0
progress = tqdm(total=len(records), desc="OCR-VQA", unit="img") if tqdm is not None else None
record_iter = iter(records)
max_in_flight = max_workers * 4


def handle_result(future):
    global completed
    status, image_id, url, detail = future.result()
    completed += 1
    if progress is not None:
        progress.update(1)
    elif completed == 1 or completed % 1000 == 0:
        print(f"OCR-VQA progress: {completed}/{len(records)}", flush=True)

    if status == "failed":
        failures.append((image_id, url, detail))
        message = f"Warning: failed OCR-VQA image {image_id} after {max_retries} attempts: {detail}"
        if tqdm is not None:
            tqdm.write(message, file=sys.stderr)
        else:
            print(message, file=sys.stderr, flush=True)


with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = set()

    def submit_next():
        try:
            record = next(record_iter)
        except StopIteration:
            return False
        futures.add(executor.submit(download_one, record))
        return True

    for _ in range(max_in_flight):
        if not submit_next():
            break

    while futures:
        done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
        for future in done:
            handle_result(future)
            submit_next()

if progress is not None:
    progress.close()

if failures:
    with failure_log.open("w", encoding="utf-8") as handle:
        for image_id, url, error in failures:
            handle.write(f"{image_id}\t{url}\t{error}\n")
    print(
        f"Warning: {len(failures)} OCR-VQA images failed. See {failure_log}. "
        "Re-run the script later to retry missing files.",
        file=sys.stderr,
        flush=True,
    )
else:
    failure_log.unlink(missing_ok=True)
    complete_marker.touch()
PYTHON
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

  download_coco
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
