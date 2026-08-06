#!/usr/bin/env sh
set -eu
if ! command -v nedster >/dev/null 2>&1; then
    printf '%s\n' "Nedster is not installed globally. Run: ./install.sh"
    exit 1
fi
exec nedster --project "$PWD" "$@"
