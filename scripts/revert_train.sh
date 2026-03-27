#!/usr/bin/env bash
# Safely revert train.py to the last committed version after a failed run.
# Usage: scripts/revert_train.sh
echo "Reverting train.py to HEAD..."
git checkout HEAD -- train.py
echo "Done. train.py is now at last commit."
