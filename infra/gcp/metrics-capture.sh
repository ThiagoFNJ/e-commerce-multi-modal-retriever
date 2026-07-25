#!/bin/bash
# G2: append one JSON line of vLLM token counters + run progress, sync to GCS.
# Counters are cumulative since vLLM start; ts+vllm_start let us diff windows and
# detect counter resets (vllm restart) post-hoc.
set -u
M=$(curl -sf -m 5 http://localhost:8000/metrics) || exit 0
get() { echo "$M" | awk -v p="$1" '$0 ~ p && $0 !~ /^#/ {print $2; exit}'; }
printf '{"ts":"%s","vllm_start":"%s","prompt_tokens_total":%s,"generation_tokens_total":%s,"requests_running":%s,"requests_waiting":%s,"ckpt_lines":%s}\n' \
  "$(date -u +%FT%TZ)" \
  "$(systemctl show vllm -p ActiveEnterTimestamp --value 2>/dev/null | tr ' ' 'T')" \
  "$(get 'vllm:prompt_tokens_total')" \
  "$(get 'vllm:generation_tokens_total')" \
  "$(get 'vllm:num_requests_running')" \
  "$(get 'vllm:num_requests_waiting\{')" \
  "$(wc -l < /opt/emmr/ckpt/review_aspects.jsonl 2>/dev/null || echo 0)" \
  >> /opt/emmr/ckpt/metrics_timeline.jsonl
gsutil -q cp /opt/emmr/ckpt/metrics_timeline.jsonl gs://emmr-9122a143/ckpt/metrics_timeline.jsonl 2>/dev/null || true
