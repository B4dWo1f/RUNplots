#!/bin/bash
RUN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export RUN_BY_CRON='True'

LOG_FILE="${RUN_DIR}/wrfout_watcher.log"
ERR_FILE="${RUN_DIR}/wrfout_watcher.err"

: > "$LOG_FILE"
: > "$ERR_FILE"

exec 1> "$LOG_FILE"
exec 2> "$ERR_FILE"


# Prevent concurrent executions with a lock
LOCKFILE="/tmp/wrfout_wrapper.lock"
exec 200>"$LOCKFILE"
flock -n 200 || {
    echo "$(date): Another instance is already running. Exiting." >> /tmp/wrfout_wrapper.log
    exit 1
}

cd $RUN_DIR

source "$RUN_DIR"/.venv3.10/bin/activate
VENV_PY="$RUN_DIR/.venv3.10/bin/python"
echo "$(date): Using Python: $($VENV_PY --version)"


# Setup paths
WPS_DOMAIN="Spain6_1"
WRFOUT_DIR="/storage/WRFOUT/$WPS_DOMAIN"
PROCESSED_DIR="$WRFOUT_DIR/processed"
MAIN_SCRIPT="$RUN_DIR/run_postprocess.py"

# Create processed folder if it doesn't exist
mkdir -p "$PROCESSED_DIR"

echo "[$(date)]: wrfout_watcher.sh started"
trap "rm -f \"$RUN_DIR/STOP\"; exit" INT TERM EXIT  # Delete STOP after exit
while [ ! -f $RUN_DIR/STOP ]
do
   for file1 in $(ls ${WRFOUT_DIR}/wrfout_d01* 2> /dev/null)
   do
      sleep 10   # wait in case the files are being written
      echo "==================================="
      echo "= Processing The following files: ="
      echo "==================================="
      file2=`echo $file1 | sed 's/d01/d02/'`
      echo "========"
      ls $file1
      ls $file2
      date
      ############################################################
      MPLCACHE1="/tmp/mplcache_d01"
      MPLCACHE2="/tmp/mplcache_d02"
      (
      export MPLCONFIGDIR="$MPLCACHE1"
      export MPLBACKEND="Agg"
      time $VENV_PY "$MAIN_SCRIPT" "$file1"
      ) &
      (
      export MPLCONFIGDIR="$MPLCACHE2"
      export MPLBACKEND="Agg"
      time $VENV_PY "$MAIN_SCRIPT" "$file2"
      ) &
      wait
      ############################################################
      # time (($VENV_PY "$MAIN_SCRIPT" "$file1") & ($VENV_PY "$MAIN_SCRIPT" "$file2"))
      # time (($VENV_PY "$MAIN_SCRIPT" "$file1" || echo "`date`: Failed to process $file1" >> /tmp/wrfout_wrapper.err) & ($VENV_PY "$MAIN_SCRIPT" "$file2" || echo "`date`: Failed to process $file2" >> /tmp/wrfout_wrapper.err) )
      date
      mv $file1 ${PROCESSED_DIR}
      mv $file2 ${PROCESSED_DIR}
      echo "moved file $file1"
      echo "moved file $file2"
      echo "========"
   done
   sleep 5
done
echo "$(date): wrfout_watcher.sh finished"
