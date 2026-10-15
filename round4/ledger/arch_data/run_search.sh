#!/bin/bash
# SCOUT ARCH: reproducible arXiv search log (export.arxiv.org API); slow pacing to respect rate limits.
export PYTHONIOENCODING=utf-8
cd "$(dirname "$0")"
out=search_log.txt
echo "# SCOUT ARCH arXiv search log, run $(date -u +%F) via export.arxiv.org API (arxq.py, newest first; rows >= 2024-01-01)" > $out
while IFS= read -r q; do
  echo "=== QUERY: $q" >> $out
  python arxq.py "$q" 60 2>&1 | awk '/^# retry/ || $1 >= "2024-01-01"' >> $out
  sleep 60
done <<'Q'
abs:conformer AND abs:torsion AND (abs:ring OR abs:"bond angle")
abs:"conformer generation" AND abs:"flow matching"
abs:"conformer generation" AND abs:diffusion AND abs:GEOM
abs:"torsional diffusion"
abs:"internal coordinates" AND abs:conformer AND (abs:diffusion OR abs:flow)
abs:conformer AND abs:"GEOM-QM9"
abs:conformer AND abs:"GEOM-DRUGS"
abs:"Cremer-Pople"
abs:"bond angles" AND abs:diffusion AND abs:conform
abs:macrocycle AND abs:conformer AND (abs:diffusion OR abs:flow)
abs:"local structure" AND abs:conformer AND abs:torsion
Q
echo "# done" >> $out
