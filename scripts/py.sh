#!/usr/bin/env bash
# Run the project interpreter with a clean Python environment.
# The host app exports PYTHONHOME/PYTHONPATH pointing at its own bundled
# interpreters, which breaks any other virtualenv; this strips them.
exec env -u PYTHONHOME -u PYTHONPATH "${NANOH3_PYTHON:-/Users/doge/.zcode/workspace/h3env/bin/python}" "$@"
