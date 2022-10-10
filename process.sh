#!/bin/bash
RUN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# STDOUT to log
LOG_FILE="${RUN_DIR}/process.log"
echo > "${LOG_FILE}"
# Close STDOUT file descriptor
exec 1<&-
# Open STDOUT as $LOG_FILE file for read and write.
exec 1<>"${LOG_FILE}"

# STDERR to err
ERR_FILE="${RUN_DIR}/process.err"
echo > "${ERR_FILE}"
# Close STDERR FD
exec 2<&-
# Redirect STDERR to STDOUT
exec 2<>"${ERR_FILE}"


FOLDER=$1

export RUN_BY_CRON=True

while [ ! -f $RUN_DIR/STOP ]
do
   for file1 in `ls ${FOLDER}/wrfout_d01* 2> /dev/null`
   do
      sleep 30   # wait in case the files are being written
      echo "Processing The following files:"
      file2=`echo $file1 | sed 's/d01/d02/'`
      ls $file1
      ls $file2
      date
      time (python3 web_plots.py $file1 & python3 web_plots.py $file2)
      date
      mv $file1 ${FOLDER}/processed/
      mv $file2 ${FOLDER}/processed/
   done
   # echo "No more files"
done
