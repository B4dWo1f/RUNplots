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
   echo "Processing The following files:"
   for file in `ls ${FOLDER}/wrfout_d01*`
   do
      sleep 45   # wait 45 seconds in case the files are being written
      file1=`echo $file | sed 's/d01/d02/'`
      ls $file
      ls $file1
      date
      time (python3 web_plots.py $file & python3 web_plots.py $file1)
      date
      mv $file ${FOLDER}/processed/
      mv $file1 ${FOLDER}/processed/
   done
   echo "No more files"
   sleep 10  #XXX unnecessary?
done
