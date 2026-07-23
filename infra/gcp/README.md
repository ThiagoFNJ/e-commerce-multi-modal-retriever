# GPU extraction infra (GCP, A100)

One spot A100-40GB VM runs vLLM (gemma-4-12B, BF16) + the resumable extraction job.
Everything is idempotent: the startup script reruns on every boot, pulls the latest
checkpoint from GCS, and resumes. A managed instance group of size 1 recreates the VM
after spot preemption.

## Layout

- `startup.sh` — boot-time provisioning (code, deps, data, weights, systemd units)
- `vllm.service` — OpenAI-compatible server on :8000
- `extract.service` — stage-05 runner (`--input` slim parquet, `--workers 32`), waits
  for vLLM health, syncs the checkpoint to GCS on stop
- `sync-checkpoint.{service,timer}` — checkpoint → GCS every 5 min

Bucket: `gs://emmr-9122a143` (`data/reviews_slim.parquet`, `ckpt/review_aspects.jsonl`).

## Bring-up (after A100 quota is granted)

```sh
gcloud compute instance-templates create emmr-extract-tpl \
  --machine-type a2-highgpu-1g \
  --provisioning-model SPOT --instance-termination-action DELETE \
  --image-family common-cu124 --image-project deeplearning-platform-release \
  --boot-disk-size 200GB --boot-disk-type pd-ssd \
  --scopes storage-rw \
  --metadata-from-file startup-script=infra/gcp/startup.sh \
  --region us-central1

gcloud compute instance-groups managed create emmr-extract-mig \
  --template emmr-extract-tpl --size 1 --zone us-central1-a
```

Gates before letting the full pass run (§6 of the stage doc): stop `extract.service`,
run the dev-248 serving-equivalence eval from the laptop against the VM's :8000 (SSH
tunnel), then a ~2k-review throughput pilot; only then `systemctl start extract.service`.

## Monitoring

```sh
gcloud compute ssh <vm> -- tail -f /var/log/emmr-startup.log
gsutil cat gs://emmr-9122a143/ckpt/review_aspects.jsonl | wc -l   # progress
```

## Teardown (job done)

```sh
gcloud compute instance-groups managed delete emmr-extract-mig --zone us-central1-a -q
gcloud compute instance-templates delete emmr-extract-tpl -q
# bucket stays (checkpoint + finalized artifacts), costs cents/month
```
