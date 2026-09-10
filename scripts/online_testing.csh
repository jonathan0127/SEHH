#!/bin/csh -f

set PARA_FILE = "src/config/para_setting.h"

sed -i -E 's/(FUNCTION_CHOICE[[:space:]]*=[[:space:]]*)[0-9]+/\11/' "$PARA_FILE"

make release
if ( $status != 0 ) then
    echo "[Error] Compilation failed!"
    exit 1
endif

echo "[Test] Starting SEHH online testing..."
uv run python scripts/run.py
