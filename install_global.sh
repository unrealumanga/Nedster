#!/usr/bin/env sh
# Kept for backwards compatibility: the global install is handled by install.sh.
set -eu
exec sh "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/install.sh" "$@"
