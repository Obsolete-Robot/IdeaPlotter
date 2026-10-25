#!/bin/sh
# Sets up IdeaPlotter: Python environment, packages, the game map and how ideas get written.
# Safe to run again; finished steps are skipped.
cd "$(dirname "$0")" || exit 1
for py in python3 python; do
    if command -v "$py" >/dev/null 2>&1; then
        exec "$py" install.py "$@"
    fi
done
echo "IdeaPlotter needs Python 3.10 or newer: https://www.python.org/downloads/"
exit 1
