#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
input_path="${1:-$script_dir/solutions.md}"
output_path="${2:-$script_dir/solutions.pdf}"

pandoc "$input_path" \
  --from markdown+tex_math_dollars \
  --pdf-engine=xelatex \
  --variable geometry:margin=1in \
  --output "$output_path"

printf 'Wrote %s\n' "$output_path"
