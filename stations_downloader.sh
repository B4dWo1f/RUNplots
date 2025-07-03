#!/bin/bash
RUN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export RUN_BY_CRON='True'

LOG_FILE="${RUN_DIR}/stations_downloader.log"
ERR_FILE="${RUN_DIR}/stations_downloader.err"

: > "$LOG_FILE"
: > "$ERR_FILE"

exec 1> "$LOG_FILE"
exec 2> "$ERR_FILE"


cd "$RUN_DIR"

source "$RUN_DIR"/.venv3.10/bin/activate
VENV_PY="$RUN_DIR/.venv3.10/bin/python"
echo "$(date): Using Python: $($VENV_PY --version)"

# Prevent concurrent executions with a lock
LOCKFILE="/tmp/stations_download.lock"
exec 200>"$LOCKFILE"
flock -n 200 || {
    echo "$(date): Another instance is already running. Exiting." >> /tmp/stations_download.log
    exit 1
}


time $VENV_PY download_stations_data.py

rm $LOCKFILE
