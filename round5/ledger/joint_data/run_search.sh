#!/bin/bash
# R-JOINT: reproducible arXiv search log (export.arxiv.org API). Lens: joint L+tau modelling,
# multi-scale / hierarchical conformer generation, Boltzmann generators in internal coordinates.
export PYTHONIOENCODING=utf-8
cd "$(dirname "$0")"
out=search_log.txt
echo "# R-JOINT arXiv search log, run $(date -u +%F) via export.arxiv.org API (arxq.py, newest first; rows >= 2023-01-01)" > $out
while IFS= read -r q; do
  echo "=== QUERY: $q" >> $out
  python arxq.py "$q" 60 2>&1 | awk '/^# retry/ || $1 >= "2023-01-01"' >> $out
  sleep 25
done <<'Q'
abs:hierarchical AND abs:conformer AND (abs:diffusion OR abs:"flow matching")
abs:"multi-scale" AND abs:conformer
abs:multiscale AND abs:conformer AND abs:generative
abs:"coarse-to-fine" AND (abs:conformer OR abs:molecular)
abs:"coarse-grained" AND abs:conformer AND abs:generation
abs:fragment AND abs:conformer AND (abs:diffusion OR abs:"flow matching")
abs:"Boltzmann generator" AND abs:torsion
abs:"Boltzmann generator" AND abs:conformer
abs:"Boltzmann generator" AND abs:"internal coordinates"
abs:torsion AND abs:"bond angle" AND (abs:diffusion OR abs:"flow matching")
abs:puckering AND (abs:generative OR abs:diffusion OR abs:"flow matching")
abs:"product manifold" AND (abs:molecule OR abs:molecular OR abs:conformer)
abs:stereochemistry AND abs:conformer AND abs:generation
abs:"ring closure" AND (abs:generative OR abs:diffusion OR abs:"flow matching")
abs:"redundant internal coordinates" AND abs:generative
abs:conformer AND abs:"joint" AND abs:"local structure"
abs:"torsional diffusion" AND abs:extension
abs:"effective sample size" AND abs:conformer
Q
echo "# done" >> $out
