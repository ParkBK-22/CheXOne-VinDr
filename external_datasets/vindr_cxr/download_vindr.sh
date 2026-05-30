#!/bin/bash
set -euo pipefail

AP="s3://arn:aws:s3:us-east-1:724665945834:accesspoint/vindr-cxr-v1-0-0-01/vindr-cxr/1.0.0"
DEST="${VINDR_DEST:-/workspace/physionet.org/files/vindr-cxr/1.0.0}"
SPLIT="${1:-test}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

if command -v aws >/dev/null 2>&1; then
  AWS=aws
elif [ -x /venv/main/bin/aws ]; then
  AWS=/venv/main/bin/aws
else
  echo "[setup] installing awscli..."
  pip install -q awscli 2>/dev/null || /venv/main/bin/pip install -q awscli
  AWS=$(command -v aws || echo /venv/main/bin/aws)
fi

echo "[info] aws = $AWS ($($AWS --version 2>&1))"

if ! $AWS sts get-caller-identity >/dev/null 2>&1; then
  echo "ERROR: AWS credentials not found."
  echo "Run one of the following first:"
  echo "  A) set -a; source /path/to/.env; set +a"
  echo "  B) $AWS configure"
  exit 1
fi

echo "[info] credentials OK: $($AWS sts get-caller-identity --query Arn --output text)"

mkdir -p "$DEST"

$AWS configure set default.s3.max_concurrent_requests 32 || true
$AWS configure set default.s3.max_queue_size 10000 || true

echo "[1/3] downloading annotations and base files..."
$AWS s3 sync "$AP/annotations/" "$DEST/annotations/"

for f in SHA256SUMS.txt LICENSE.txt supplemental_file_DICOM_tags.pdf; do
  $AWS s3 cp "$AP/$f" "$DEST/$f"
done

echo "[2/3] downloading images: SPLIT=$SPLIT"
case "$SPLIT" in
  test)
    $AWS s3 sync "$AP/test/" "$DEST/test/"
    ;;
  train)
    $AWS s3 sync "$AP/train/" "$DEST/train/"
    ;;
  all)
    $AWS s3 sync "$AP/test/" "$DEST/test/"
    $AWS s3 sync "$AP/train/" "$DEST/train/"
    ;;
  *)
    echo "usage: download_vindr.sh [test|train|all]"
    exit 1
    ;;
esac

echo "[3/3] verifying sha256..."
cd "$DEST"

for s in test train; do
  if [ "$SPLIT" != "$s" ] && [ "$SPLIT" != "all" ]; then
    continue
  fi

  grep " $s/" SHA256SUMS.txt > "/tmp/sums_$s.txt"
  total=$(wc -l < "/tmp/sums_$s.txt")
  ok=$(sha256sum -c "/tmp/sums_$s.txt" 2>/dev/null | grep -c ": OK$" || true)

  echo "  $s: $ok / $total OK"

  if [ "$ok" != "$total" ]; then
    echo "  WARNING: checksum mismatch. Re-run sync to fill missing files."
  fi
done

echo "Done: $DEST"
