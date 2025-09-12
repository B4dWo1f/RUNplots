#!/bin/bash
# nukes all systemd user units related to RUN_plots

echo "🔴 Nuking all systemd units related to RUN_plots..."

SYSTEMD_USER_DIR="$HOME/.config/systemd/user"
UNIT_PREFIXES=("stations_downloader" "wrfout_watcher" "sync_systemd_units")

for prefix in "${UNIT_PREFIXES[@]}"; do
    for ext in service timer path; do
        UNIT="${prefix}.${ext}"
        UNIT_PATH="$SYSTEMD_USER_DIR/$UNIT"
        if systemctl --user is-enabled --quiet "$UNIT"; then
            echo "Disabling: $UNIT"
            systemctl --user disable --now "$UNIT"
        fi
        if [[ -f "$UNIT_PATH" ]]; then
            echo "Removing: $UNIT_PATH"
            rm -f "$UNIT_PATH"
        fi
    done
done

# Reload systemd daemon to apply changes
systemctl --user daemon-reload

echo "✅ All related systemd units have been nuked."

echo "Maybe you want to rm *.log *.err logs/*"
