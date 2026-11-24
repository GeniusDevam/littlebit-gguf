#!/usr/bin/env bash
# Apply the patches in patches/ to a llama.cpp tree. Patches that are already applied are skipped.
# Usage: apply-patches.sh LLAMA_DIR
# Written for bash 3.2: no possibly empty array is expanded.
set -euo pipefail

llama=$(cd "$1" && pwd)
patches=("$(cd "$(dirname "${BASH_SOURCE[0]}")/../patches" && pwd)"/*.diff)
count=${#patches[@]}
files=$(sed -n 's|^diff --git a/\(.*\) b/.*|\1|p' "${patches[@]}" | sort -u)
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT

# git apply skips paths outside the current directory when it runs inside a larger repository, and it exits 0.
# The ceiling keeps git from looking above the directory it works in, for the copy and for the real tree.
git_apply() { # DIR [--reverse] PATCH
    local dir=$1
    shift
    (cd "$dir" && GIT_CEILING_DIRECTORIES=$(dirname "$dir") git apply "$@" 2>/dev/null)
}

# Later patches edit lines that earlier ones add, so patches are tried in order on a copy of the files they touch.
fresh() {
    rm -rf "$scratch/tree"
    mkdir "$scratch/tree"
    local file
    for file in $files; do
        mkdir -p "$scratch/tree/$(dirname "$file")"
        cp "$llama/$file" "$scratch/tree/$file"
    done
}

# applied is the largest k for which patches 1..k are all in the tree; the copy is left with them removed
applied=0
for ((k = count; k >= 1; k--)); do
    fresh
    ok=1
    for ((j = k; j >= 1; j--)); do
        if ! git_apply "$scratch/tree" --reverse "${patches[j - 1]}"; then
            ok=0
            break
        fi
    done
    if [[ $ok == 1 ]]; then
        applied=$k
        break
    fi
done
if [[ $applied == 0 ]]; then
    fresh
fi

# the copy now gets every patch, which gives the tree the real one must end up with
for ((j = 0; j < count; j++)); do
    if ! git_apply "$scratch/tree" "${patches[j]}"; then
        echo "error: $(basename "${patches[j]}") does not apply to $llama; no files were changed" >&2
        exit 1
    fi
done

for ((j = applied; j < count; j++)); do
    if ! git_apply "$llama" "${patches[j]}"; then
        echo "error: $(basename "${patches[j]}") could not be applied to $llama" >&2
        exit 1
    fi
done

for file in $files; do
    if ! cmp -s "$llama/$file" "$scratch/tree/$file"; then
        echo "error: $file in $llama does not match the patched file after applying the patches" >&2
        exit 1
    fi
done
