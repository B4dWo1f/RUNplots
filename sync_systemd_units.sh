#!/bin/bash
LOGFILE="$HOME/CODES/RUN_plots/logs/sync_systemd_units.log"

echo "$(date): Triggered sync_systemd_units.sh" >> "$LOGFILE"

SRC="$HOME/CODES/RUN_plots"
DST="$HOME/.config/systemd/user"

cp $SRC/*.service $DST
cp $SRC/*.timer $DST
cp $SRC/*.path $DST

echo "$(date): Files copied to $DST" >> "$LOGFILE"

systemctl --user daemon-reload
echo "$(date): Reloaded systemd user daemon" >> "$LOGFILE"

systemctl --user enable --now stations_downloader.timer
# systemctl --user enable --now wrfout_watcher.service
echo "$(date): Enabled timers/services" >> "$LOGFILE"
