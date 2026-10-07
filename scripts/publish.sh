#!/usr/bin/env bash
# Publish this repository to GitHub.
#
#   ./scripts/publish.sh                       # create the repo and push
#   ./scripts/publish.sh my-org/nanoh3        # create under a specific owner
#
# Requires the GitHub CLI authenticated (gh auth login) with repo scope.
# This script never runs unattended on its own: it prints what it will do and
# asks for confirmation, because publishing is not reversible in the way a local
# commit is.
set -euo pipefail

REPO_SLUG="${1:-}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ -z "$REPO_SLUG" ]; then
  NAME=$(basename "$PWD")
  OWNER=$(git config --global user.name | tr ' ' '-' | tr '[:upper:]' '[:lower:]')
  REPO_SLUG="${OWNER}/${NAME}"
fi

echo "About to publish:"
echo "  local path : $ROOT"
echo "  github repo: https://github.com/$REPO_SLUG"
echo "  branch     : $(git branch --show-current 2>/dev/null || echo main)"
echo "  files      : $(git ls-files | wc -l | tr -d ' ') tracked"
echo
echo "Checklist before you continue:"
echo "  [ ] no checkpoints or videos are staged (git ls-files | grep -E '\.(pt|mp4)$')"
echo "  [ ] docs/RESEARCH-H3.md licence section still matches your intent"
echo "  [ ] you are happy to publish under Apache-2.0"
echo

read -r -p "Publish? [y/N] " answer
case "$answer" in
  [yY]|[yY][eE][sS]) ;;
  *) echo "aborted"; exit 1 ;;
esac

command -v gh >/dev/null || { echo "gh not found: https://cli.github.com"; exit 1; }
gh auth status >/dev/null || { echo "run: gh auth login"; exit 1; }

if gh repo view "$REPO_SLUG" >/dev/null 2>&1; then
  echo "repo exists, pushing"
  gh repo edit "$REPO_SLUG" --visibility public
else
  gh repo create "$REPO_SLUG" --public --source=. --remote=origin --description \
    "A small, fully open text-to-video model for 8 GB graphics cards"
fi

git push -u origin HEAD
echo "done: https://github.com/$REPO_SLUG"
