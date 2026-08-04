#!/usr/bin/env bash
# scripts/migrate_from_aria.sh — retire the Aria layer without losing its state.
#
# Two things are migrated:
#   1. ~/.aria/  ->  ~/.nedster/     (milestones, workflows, journal, skills, alerts)
#   2. the ollama tag mess           (aria-qwen / aria-local / aria-lfm2 -> nedster)
#
# Idempotent. Run it as many times as you like.
# Dry run:  DRY_RUN=1 ./scripts/migrate_from_aria.sh

set -euo pipefail

OLD="$HOME/.aria"
NEW="$HOME/.nedster"
DRY_RUN="${DRY_RUN:-0}"

say()  { printf '%s\n' "$*"; }
step() { printf '\n== %s ==\n' "$*"; }
run()  {
  if [ "$DRY_RUN" = "1" ]; then
    printf '  [dry]'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

step "State directory"

if [ -d "$OLD" ] && [ ! -L "$OLD" ]; then
  run mkdir -p "$NEW"
  # Copy rather than move, so a failed migration is never a data loss event.
  # -n means an existing ~/.nedster file always wins over the ~/.aria copy.
  if [ "$DRY_RUN" = "1" ]; then
    say "  [dry] cp -rn $OLD/. $NEW/"
  else
    cp -rn "$OLD/." "$NEW/" 2>/dev/null || true
  fi
  say "  copied $OLD -> $NEW"

  for f in milestones.md workflows.jsonl; do
    run touch "$NEW/$f"
  done
  run mkdir -p "$NEW/journal" "$NEW/skills/user" "$NEW/daemon_alerts" "$NEW/replays"

  # Preserve the old state as a timestamped backup, but do not leave a legacy
  # symlink behind: all supported runtime paths now use ~/.nedster directly.
  STAMP="$(date +%Y%m%d%H%M%S)"
  run mv "$OLD" "$HOME/.aria.pre-nedster.$STAMP"
  say "  original preserved at ~/.aria.pre-nedster.$STAMP"
elif [ -L "$OLD" ]; then
  run rm "$OLD"
  say "  removed the obsolete ~/.aria symlink"
else
  run mkdir -p "$NEW/journal" "$NEW/skills/user" "$NEW/daemon_alerts" "$NEW/replays"
  run touch "$NEW/milestones.md" "$NEW/workflows.jsonl"
  say "  no ~/.aria found; created a fresh $NEW"
fi

step "Ollama model tag"

if command -v ollama >/dev/null 2>&1; then
  MODELFILE="Modelfile.nedster"
  [ -f "$MODELFILE" ] || MODELFILE="Modelfile"

  if ollama list 2>/dev/null | awk '{print $1}' | grep -qx "nedster:latest\|nedster"; then
    say "  tag 'nedster' already present"
  else
    say "  building 'nedster' from $MODELFILE"
    run ollama create nedster -f "$MODELFILE"
  fi

  # Reclaim VRAM and disk from the retired aliases. Weights are shared layers,
  # so removing a tag does not re-download anything you still use.
  for tag in aria-qwen aria-local aria-lfm2; do
    if ollama list 2>/dev/null | awk '{print $1}' | grep -q "^${tag}"; then
      say "  found legacy tag '$tag'"
      if [ "${REMOVE_LEGACY_TAGS:-0}" = "1" ]; then
        if [ "$DRY_RUN" = "1" ]; then
          say "  [dry] ollama stop $tag"
        else
          ollama stop "$tag" >/dev/null 2>&1 || true
        fi
        run ollama rm "$tag"
        say "    removed"
      else
        say "    kept (set REMOVE_LEGACY_TAGS=1 to remove)"
      fi
    fi
  done
else
  say "  ollama not on PATH — skipping model step"
fi

step "Source references still pointing at Aria"

if command -v grep >/dev/null 2>&1; then
  # Word-boundary match, and skip HTML/ARIA accessibility attributes
  # (aria-label, aria-hidden...) plus substring hits like "variant".
  say "  files that still reference Aria (fix these next):"
  grep -rIlE --exclude-dir=.git --exclude-dir=node_modules \
    -e '\.aria/' -e 'aria-qwen' -e 'aria-local' -e 'aria-lfm2' -e '"Aria"' -e '\bAria\b' . 2>/dev/null \
    | xargs -r grep -lvE 'aria-(label|hidden|describedby|expanded|controls|live|role)' 2>/dev/null \
    | sed 's/^/    /' || say "    none"
fi

step "Done"
say "Set the canonical model for this shell:"
say "  export NEDSTER_MODEL=nedster"
say ""
say "Verify:"
say "  python3 -m core.model_registry"
say "  python3 -m core.vram"
