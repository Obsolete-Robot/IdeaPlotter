#!/bin/sh
# Starts IdeaPlotter and opens the map in your browser. Extra arguments go to `ideaplotter serve`,
# e.g.  ./run.sh --port 9000
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
    echo "IdeaPlotter is not set up yet. Run ./install.sh first."
    exit 1
fi
export PYTHONWARNINGS=ignore TRANSFORMERS_VERBOSITY=error
echo "Starting IdeaPlotter. Loading the games takes about 30 seconds; your browser opens when it is ready."
echo "Press Ctrl+C to stop."
exec .venv/bin/python -m ideaplotter serve --open "$@"
