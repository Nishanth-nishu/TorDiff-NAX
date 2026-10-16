#!/bin/bash
# Round 5: copy the committed flexitors/, tools/ and experiments/ (if present) of HEAD to gnode118
# ($PROJECT/{flexitors,tools,experiments}) and record the commit in $PROJECT/FLEXITORS_COMMIT. The frozen baseline
# torsional-diffusion/ is never touched here (round2/sync_cluster.sh pattern; flexitors/tests/test_baseline_frozen.py
# checks the cluster copy against the pins). Usage: bash round5/sync_cluster.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
if [ -n "$(git status --porcelain -- flexitors tools experiments)" ]; then
  echo "uncommitted changes in flexitors/ tools/ experiments/: commit first" >&2; exit 1
fi
COMMIT=$(git rev-parse HEAD)
T=$(mktemp -d)
DIRS="flexitors tools"; [ -d experiments ] && DIRS="$DIRS experiments"
git archive --format=tar.gz -o "$T/r5sync.tgz" HEAD $DIRS
cat > "$T/r5apply.sh" <<EOS
set -eu
P=/scratch/nishanth.r/tordiff
rm -rf \$P/tmp/r5sync && mkdir -p \$P/tmp/r5sync && tar xzf ~/r5sync.tgz -C \$P/tmp/r5sync
rm -rf \$P/flexitors && cp -a \$P/tmp/r5sync/flexitors \$P/flexitors
cp -a \$P/tmp/r5sync/tools/. \$P/tools/
if [ -d \$P/tmp/r5sync/experiments ]; then rm -rf \$P/experiments && cp -a \$P/tmp/r5sync/experiments \$P/experiments; fi
echo $COMMIT > \$P/FLEXITORS_COMMIT
echo synced $COMMIT
EOS
scp -q "$T/r5sync.tgz" ada:r5sync.tgz
scp -q "$T/r5apply.sh" ada:r5apply.sh
ssh -o BatchMode=yes ada 'srun -n 1 -c 1 --mem=2G -t 00:10:00 -A plafnet2 -p plafnet2 -w gnode118 bash ~/r5apply.sh' 2>&1 \
  | grep -v "post-quantum\|store now\|openssh\|WARNING: conn"
rm -rf "$T"
