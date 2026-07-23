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
MODEL=google/gemma-4-12B

mkdir -p "$WORKDIR"
cd "$WORKDIR"

# --- code ---
if [ ! -d repo/.git ]; then
  git clone --depth 1 -b "$BRANCH" "$REPO" repo
else
  git -C repo fetch origin "$BRANCH" && git -C repo reset --hard "origin/$BRANCH"
fi

# --- python env (system conda python from the DL image) ---
PIP=/opt/conda/bin/pip
PY=/opt/conda/bin/python
$PIP show vllm >/dev/null 2>&1 || $PIP install -q vllm
$PIP show emmr >/dev/null 2>&1 || {
  $PIP install -q pandas pyarrow httpx pyyaml tqdm ollama
  $PIP install -q --no-deps -e "$WORKDIR/repo"
}

# --- data + checkpoint resume ---
mkdir -p data ckpt
gsutil -q cp -n "$BUCKET/data/reviews_slim.parquet" data/reviews_slim.parquet
gsutil -q cp "$BUCKET/ckpt/review_aspects.jsonl" ckpt/review_aspects.jsonl 2>/dev/null || true
echo "checkpoint lines at boot: $(wc -l < ckpt/review_aspects.jsonl 2>/dev/null || echo 0)"

# --- model weights (cached on the boot disk across restarts of the same VM) ---
$PY - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("google/gemma-4-12B")
EOF

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
