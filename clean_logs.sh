#!/bin/bash
dirscript=`dirname $0`

# Exit on any error
set -e

# Check for required argument
if [ -z "$1" ]; then
  echo "Usage: $0 <batch_id>"
  echo "Example: $0 06"
  exit 1
fi

batch="$1"

# Optionally specify the directory where the log files are
LOG_DIR="$dirscript/logs"  # Change to actual path if needed

# Find and delete matching files
rm $LOG_DIR/run_postprocess_*_GFS${batch}.log
rm $LOG_DIR/run_postprocess_*_GFS${batch}.perform
