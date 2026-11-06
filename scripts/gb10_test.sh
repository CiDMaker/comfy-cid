#!/bin/sh
# Run the H3 node tests that need torch / comfy.* on gb10 (this machine has no torch).
# Syncs the repo to a scratch dir on gb10 (not ComfyUI's custom_nodes) and runs only
# test/test_h3_*.py with ComfyUI's venv; test_cid_nodes.py stubs the core `nodes`
# module and is meant for the local run (npm run test:py).
#   scripts/gb10_test.sh            # all H3 tests
#   scripts/gb10_test.sh prefix     # only tests matching -k prefix
set -e
HOST=${HOST:-dgxspark}
REMOTE=workspace/comfy-cid_dev
HERE=$(cd "$(dirname "$0")/.." && pwd)
ssh "$HOST" "mkdir -p $REMOTE"
rsync -az --delete --exclude .git --exclude __pycache__ --exclude node_modules "$HERE/" "$HOST:$REMOTE/"
ssh "$HOST" "cd $REMOTE && PYTHONPATH=\$HOME/workspace/ComfyUI \$HOME/workspace/ComfyUI/venv/bin/python -m unittest discover -s test -p 'test_h3_*.py' -v ${1:+-k $1}"
