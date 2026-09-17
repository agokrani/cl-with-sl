#!/bin/bash
# Prints "<jobid> <jobname>" for every failure in the window NOT excused by
# triaged_failures.txt. Empty output == nothing to worry about.
# usage: untriaged.sh [days]   (default 3)
cd /scratch/agokrani/xcountry-20260916 || exit 1
D=${1:-3}
sacct -u "$USER" -S "now-${D}days" -X -n -P -o JobID,JobName,State 2>/dev/null \
| awk -F'|' '$3 ~ /FAILED|TIMEOUT|OUT_OF_ME/ {print $1"|"$2}' \
| while IFS='|' read -r id nm; do
    ok=0
    while IFS='|' read -r pat _; do
      pat=$(echo "$pat" | tr -d ' '); [ -z "$pat" ] && continue
      case "$pat" in \#*) continue;; esac
      if [ "$id" = "$pat" ] || echo "$nm" | grep -qE "$pat"; then ok=1; break; fi
    done < triaged_failures.txt
    [ "$ok" -eq 0 ] && echo "$id $nm"
  done
