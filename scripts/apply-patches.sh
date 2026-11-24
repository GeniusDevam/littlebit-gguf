#!/usr/bin/env bash
# Apply the patches in patches/ to a llama.cpp tree. Patches that are already applied are skipped.
# Usage: apply-patches.sh LLAMA_DIR
set -euo pipefail

llama=$1
patches=("$(cd "$(dirname "${BASH_SOURCE[0]}")/../patches" && pwd)"/*.diff)
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT

# Later patches edit lines that earlier ones add, so patches are tried in order on a copy of the files they touch.
fresh() {
    rm -rf "$scratch/tree"
    mkdir "$scratch/tree"
    local file
    for file in $(sed -n 's|^diff --git a/\(.*\) b/.*|\1|p' "${patches[@]}" | sort -u); do
        mkdir -p "$scratch/tree/$(dirname "$file")"
        cp "$llama/$file" "$scratch/tree/$file"
    done
}

# try_apply [--reverse] PATCH...: apply the patches in order to the copy
try_apply() {
    local flag=()
    if [[ $1 == --reverse ]]; then
        flag=(--reverse)
        shift
    fi
    local patch
    for patch in "$@"; do
        (cd "$scratch/tree" && git apply "${flag[@]}" "$patch" 2>/dev/null) || return 1
    done
}

# applied is the largest k for which patches 1..k are all in the tree
applied=0
for ((k = ${#patches[@]}; k >= 1; k--)); do
    fresh
    newest_first=()
    for ((j = k; j >= 1; j--)); do
        newest_first+=("${patches[j - 1]}")
    done
    if try_apply --reverse "${newest_first[@]}"; then
        applied=$k
        break
    fi
done

pending=("${patches[@]:applied}")
fresh
for patch in "${pending[@]}"; do
    if ! try_apply "$patch"; then
        echo "error: $(basename "$patch") does not apply to $llama; no files were changed" >&2
        exit 1
    fi
done
for patch in "${pending[@]}"; do
    git -C "$llama" apply "$patch"
done
