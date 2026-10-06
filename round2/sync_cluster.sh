#!/bin/bash
# Sync local commits to gnode118 (round2/IMPLEMENTATION.md §7): TD commits as format-patch into the cluster checkout
# (its own git, equivalent commits), tools/ and slurm/ copied over $PROJECT/{tools,slurm}. Usage: bash sync_cluster.sh BASE
# where BASE = the last local commit already applied on the cluster.
set -euo pipefail
BASE=$1
T=$(mktemp -d)
cd "$(git rev-parse --show-toplevel)"
git format-patch -q --relative=torsional-diffusion "$BASE"..HEAD -o "$T/r2sync/patches" -- torsional-diffusion || true
mkdir -p "$T/r2sync"
git archive --format=tar.gz -o "$T/r2sync/tools_slurm.tgz" HEAD tools slurm
(cd "$T" && tar czf r2sync.tgz r2sync)
cat > "$T/apply.sh" <<'EOS'
set -eu
P=/scratch/nishanth.r/tordiff
cd $P/tmp && rm -rf r2sync && tar xzf ~/r2sync.tgz
cd $P/torsional-diffusion
if ls $P/tmp/r2sync/patches/*.patch >/dev/null 2>&1; then
  git -c user.name="Nishanth" -c user.email="rnishanth2317@gmail.com" am --keep-cr $P/tmp/r2sync/patches/*.patch
fi
git log --oneline -1
mkdir -p $P/tmp/r2sync/ts && tar xzf $P/tmp/r2sync/tools_slurm.tgz -C $P/tmp/r2sync/ts
cp -a $P/tmp/r2sync/ts/tools/. $P/tools/ && cp -a $P/tmp/r2sync/ts/slurm/. $P/slurm/ && chmod +x $P/slurm/*.sh
echo synced
EOS
scp -q "$T/r2sync.tgz" ada:r2sync.tgz
scp -q "$T/apply.sh" ada:r2apply.sh
ssh -o BatchMode=yes ada 'srun -n 1 -c 1 --mem=2G -t 00:10:00 -A plafnet2 -p plafnet2 -w gnode118 bash ~/r2apply.sh' 2>&1 | grep -v "post-quantum\|store now\|openssh\|WARNING: conn"
rm -rf "$T"
