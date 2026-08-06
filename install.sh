#!/usr/bin/env sh
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

printf '%s\n' "Installing the global Nedster command..."
if command -v uv >/dev/null 2>&1; then
    uv tool install --force "$project_root"
    uv tool update-shell >/dev/null 2>&1 || true
    user_bin=$(uv tool dir --bin)
else
    python3 -m pip install --user --upgrade "$project_root"
    user_bin=$(python3 -c "import sysconfig; print(sysconfig.get_path('scripts', scheme='posix_user'))")
fi

case ":${PATH}:" in
    *":${user_bin}:"*) ;;
    *)
        export PATH="${PATH}:${user_bin}"
        profile_file="${NEDSTER_SHELL_PROFILE:-${HOME}/.profile}"
        path_line="export PATH=\"\$PATH:${user_bin}\""
        if [ ! -f "$profile_file" ] || ! grep -F "$path_line" "$profile_file" >/dev/null 2>&1; then
            printf '\n%s\n%s\n' "# Added by the Nedster installer" "$path_line" >>"$profile_file"
        fi
        ;;
esac

if [ ! -x "${user_bin}/nedster" ]; then
    printf '%s\n' "Nedster is installed but its user scripts folder is not on PATH."
    printf '%s\n' "Add this to your shell profile, reopen the terminal, then run nedster setup:"
    printf 'export PATH="$PATH:%s"\n' "$user_bin"
    exit 1
fi

printf 'Global launcher ready: %s\n' "${user_bin}/nedster"
"${user_bin}/nedster" setup
