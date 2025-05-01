#!/bin/bash

SRC="$HOME/CODES/RUN_plots/wrfout_watcher.service"
DST="$HOME/.config/systemd/user/wrfout_watcher.service"

cp "$SRC" "$DST"
systemctl --user daemon-reload
