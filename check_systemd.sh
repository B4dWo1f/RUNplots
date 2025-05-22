#!/bin/bash
# check_systemd.sh – Show the status of RUN_plots-related systemd units

PROJECT_UNITS=(
  "stations_downloader.service"
  "stations_downloader.timer"
  "wrfout_watcher.service"
  "sync_systemd_units.path"
  "sync_systemd_units.service"
)

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🔍 Checking RUN_plots systemd user units...${NC}"

for unit in "${PROJECT_UNITS[@]}"; do
   # Check if the unit exists
   if systemctl --user list-units --all | grep -q "$unit"; then
      # Get status info
      status=$(systemctl --user is-active "$unit")
      enabled=$(systemctl --user is-enabled "$unit" 2>/dev/null)

      # Format output
      if [[ "$status" == "active" ]]; then
         icon="✅"
         color="${GREEN}"
         status_text="ACTIVE"
      elif [[ "$status" == "inactive" ]]; then
         icon="⚪"
         color="${YELLOW}"
         status_text="INACTIVE"
      else
         icon="❌"
         color="${RED}"
         status_text=$(systemctl --user is-active "$unit" 2>/dev/null || echo "UNKNOWN")
      fi
      status_icon=$(echo -e "${icon} ${color}${status_text}${NC}")
      printf "  %-30s  %s  (%s)\n" "$unit" "$status_icon" "$enabled"
   else
      echo -e "  ❓ ${unit} ${RED}not found${NC}"
   fi
done

echo ""
echo -e "${YELLOW}🕒 Active timers:${NC}"
systemctl --user list-timers --all | grep -E "stations_downloader|wrfout" || echo "  No timers found."
