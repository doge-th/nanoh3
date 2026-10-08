#!/usr/bin/env bash
# Publish this repository to GitHub.
#
#   ./scripts/publish.sh                       # push to doge-th/nanoh3
#   ./scripts/publish.sh my-org/nanoh3        # push somewhere else
#
# This machine has an SSH key registered with GitHub but no `gh` CLI, and SSH
# cannot create a repository -- only push to one. So:
#
#   * if `gh` is installed and authenticated, this script creates the repo for you;
#   * otherwise it prints the exact URL to create it by hand, then pushes.
#
# It prints what it will do and asks for confirmation, because publishing is not
# reversible in the way a local commit is.
set -euo pipefail

REPO_SLUG="${1:-doge-th/nanoh3}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DESCRIPTION="Research + reference implementation: what MiniMax-H3 actually costs on 8 GB, and why a small open video model freezes"

echo "About to publish:"
echo "  local path : $ROOT"
echo "  github repo: https://github.com/$REPO_SLUG"
echo "  files      : $(git ls-files | wc -l | tr -d ' ') tracked, $(du -sh .git | cut -f1) of history"
echo
echo "Checks:"
echo "  - no checkpoints/videos tracked: $(git ls-files | grep -cE '\.(pt|pth|ckpt|safetensors|mp4)$' || true) found"
echo "  - working tree clean          : $([ -z "$(git status --porcelain)" ] && echo yes || echo NO)"
echo "  - untracked files left behind : $(git status --porcelain --untracked-files=all | grep -c '^??' || true)"
echo
echo "Reminders:"
echo "  - README states up front that the demo model does not animate"
echo "  - no MiniMax weights or outputs are included (licence section V.3)"
echo

read -r -p "Continue? [y/N] " answer
case "$answer" in
  [yY]|[yY][eE][sS]) ;;
  *) echo "aborted"; exit 1 ;;
esac

if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  if gh repo view "$REPO_SLUG" >/dev/null 2>&1; then
    echo "repo exists; pushing"
  else
    gh repo create "$REPO_SLUG" --public --source=. --remote=origin --description "$DESCRIPTION"
  fi
elif git ls-remote "git@github.com:$REPO_SLUG.git" >/dev/null 2>&1; then
  echo "repo already exists on GitHub; pushing"
  git remote remove origin 2>/dev/null || true
  git remote add origin "git@github.com:$REPO_SLUG.git"
else
  echo
  echo "The repo does not exist yet and 'gh' is not installed, so it has to be created"
  echo "by hand. SSH can push but cannot create. Open this, click Create, come back:"
  echo
  echo "    https://github.com/new?name=nanoh3&visibility=public"
  echo
  echo "IMPORTANT: leave 'Add a README file' / .gitignore / licence UNCHECKED, otherwise"
  echo "the first push will conflict with the history in this directory."
  echo
  read -r -p "Press Enter once the repo exists (Ctrl-C to stop)... " _
  git remote remove origin 2>/dev/null || true
  git remote add origin "git@github.com:$REPO_SLUG.git"
fi

git push -u origin HEAD
echo
echo "done: https://github.com/$REPO_SLUG"
