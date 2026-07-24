#!/bin/bash
# Idempotent startup for the extraction GPU VM (runs on every boot, incl. spot restarts).
# Assumes a GCP Deep Learning VM image (NVIDIA driver + conda python preinstalled).
set -euo pipefail
exec > /var/log/emmr-startup.log 2>&1
echo "=== emmr startup $(date -u +%FT%TZ) ==="

BUCKET="gs://emmr-9122a143"
WORKDIR=/opt/emmr
REPO=https://github.com/ThiagoFNJ/e-commerce-multi-modal-retriever.git
BRANCH=feat/review-aspect-mining
MODEL=google/gemma-4-12B-it

mkdir -p "$WORKDIR"
cd "$WORKDIR"

# --- code ---
if [ ! -d repo/.git ]; then
  git clone --depth 1 -b "$BRANCH" "$REPO" repo
else
  git -C repo fetch origin "$BRANCH" && git -C repo reset --hard "origin/$BRANCH"
fi

# --- python env: dedicated venv, fully isolated from debian's system packages ---
# apt may be lock-held by unattended-upgrades right after boot; retry, then verify loudly.
# (a silently-failed install here cost a night of vLLM crash-looping on missing Python.h)
for i in $(seq 1 30); do
  apt-get install -y -q python3-venv python3-dev build-essential ninja-build git && break
  echo "apt attempt $i failed (lock?); retrying in 10s"; sleep 10
done
[ -f "/usr/include/python3.12/Python.h" ] || { echo "FATAL: python3-dev missing"; exit 1; }
command -v ninja >/dev/null || { echo "FATAL: ninja missing"; exit 1; }
[ -d "$WORKDIR/venv" ] || /usr/bin/python3 -m venv "$WORKDIR/venv"
PY="$WORKDIR/venv/bin/python"
$PY -m pip show vllm >/dev/null 2>&1 || $PY -m pip install -q --upgrade pip vllm
$PY -m pip show emmr >/dev/null 2>&1 || {
  $PY -m pip install -q pandas pyarrow httpx pyyaml tqdm ollama
  $PY -m pip install -q --no-deps -e "$WORKDIR/repo"
}

# --- data + checkpoint resume ---
mkdir -p data ckpt
gsutil -q cp -n "$BUCKET/data/reviews_slim.parquet" data/reviews_slim.parquet
gsutil -q cp "$BUCKET/ckpt/review_aspects.jsonl" ckpt/review_aspects.jsonl 2>/dev/null || true
echo "checkpoint lines at boot: $(wc -l < ckpt/review_aspects.jsonl 2>/dev/null || echo 0)"

# --- model weights (cached on the boot disk across restarts of the same VM) ---
$PY -c 'from huggingface_hub import snapshot_download; snapshot_download("google/gemma-4-12B-it")'

# --- services ---
cp "$WORKDIR"/repo/infra/gcp/vllm.service /etc/systemd/system/
cp "$WORKDIR"/repo/infra/gcp/extract.service /etc/systemd/system/
cp "$WORKDIR"/repo/infra/gcp/sync-checkpoint.service /etc/systemd/system/
cp "$WORKDIR"/repo/infra/gcp/sync-checkpoint.timer /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now vllm.service sync-checkpoint.timer
# The full pass only starts once the go-flag exists in GCS -- the gates
# (serving equivalence + throughput pilot) run against bare vLLM first.
# After preemption the flag is already there, so the job resumes unattended.
if gsutil -q stat "$BUCKET/ctl/run_extract"; then
  systemctl enable --now extract.service
  echo "run_extract flag present: extraction started"
else
  systemctl disable --now extract.service 2>/dev/null || true
  echo "no run_extract flag: gates mode (vLLM only)"
fi
echo "=== emmr startup done ==="
