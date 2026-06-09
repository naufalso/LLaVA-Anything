#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-data/PangeaInstruct}"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "Dataset directory not found: $DATA_ROOT" >&2
  exit 1
fi

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Required command not found: $cmd" >&2
    exit 1
  fi
}

extract_tar() {
  local archive="$1"
  local dir

  dir="$(dirname "$archive")"
  echo "Extracting tar: $archive"
  tar -xvf "$archive" -C "$dir"
  rm -f -- "$archive"
}

extract_zip() {
  local archive="$1"
  local dir

  dir="$(dirname "$archive")"
  echo "Extracting zip: $archive"
  unzip -o "$archive" -d "$dir"
  rm -f -- "$archive"
}

extract_part_underscore_dir() {
  local dir="$1"

  echo "Combining split tar parts in: $dir"
  (
    shopt -s nullglob
    cd "$dir"
    parts=(part_*)
    if [[ ${#parts[@]} -eq 0 ]]; then
      echo "No part_* files found in: $dir" >&2
      exit 1
    fi
    cat "${parts[@]}" > images.tar
    tar -xvf images.tar
    rm -f -- "${parts[@]}" images.tar
  )
}

extract_tar_part_dir() {
  local dir="$1"
  local part base combined

  while IFS= read -r base; do
    combined="$dir/${base}"
    echo "Combining split tar parts into: $combined"
    (
      shopt -s nullglob
      cd "$dir"
      parts=("${base}".part*)
      if [[ ${#parts[@]} -eq 0 ]]; then
        echo "No ${base}.part* files found in: $dir" >&2
        exit 1
      fi
      cat "${parts[@]}" > "$base"
      tar -xvf "$base"
      rm -f -- "${parts[@]}" "$base"
    )
  done < <(
    find "$dir" -maxdepth 1 -type f -name '*.tar.part*' -printf '%f\n' |
      while IFS= read -r part; do
        echo "${part%%.part*}"
      done |
      sort -u
  )
}

require_cmd find
require_cmd sort
require_cmd tar
require_cmd unzip

echo "Scanning PangeaInstruct archives under: $DATA_ROOT"

# Some subsets store a split tar as part_00, part_01, ... and need images.tar.
while IFS= read -r dir; do
  extract_part_underscore_dir "$dir"
done < <(
  find "$DATA_ROOT" \
    -path "$DATA_ROOT/.cache" -prune -o \
    -type f -name 'part_*' ! -name '*.metadata' -printf '%h\n' |
    sort -u
)

# Other subsets use names such as allava_laion.tar.partaa, allava_laion.tar.partab.
while IFS= read -r dir; do
  extract_tar_part_dir "$dir"
done < <(
  find "$DATA_ROOT" \
    -path "$DATA_ROOT/.cache" -prune -o \
    -type f -name '*.tar.part*' ! -name '*.metadata' -printf '%h\n' |
    sort -u
)

while IFS= read -r archive; do
  extract_tar "$archive"
done < <(
  find "$DATA_ROOT" \
    -path "$DATA_ROOT/.cache" -prune -o \
    -type f \( -name '*.tar' -o -name '*.tar.gz' -o -name '*.tgz' \) -print |
    sort
)

while IFS= read -r archive; do
  extract_zip "$archive"
done < <(
  find "$DATA_ROOT" \
    -path "$DATA_ROOT/.cache" -prune -o \
    -type f -name '*.zip' -print |
    sort
)

echo "Done."
