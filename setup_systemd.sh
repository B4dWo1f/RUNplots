#!/bin/bash
dirscript=`dirname $0`

mkdir -p ~/.config/systemd/user

cp *.service *.timer *.path ~/.config/systemd/user

systemctl --user daemon-reload
systemctl --user enable --now sync_systemd_units.path
systemctl --user enable --now stations_downloader.timer
systemctl --user enable --now wrfout_watcher.service
