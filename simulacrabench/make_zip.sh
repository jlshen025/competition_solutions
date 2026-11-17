#!/usr/bin/env bash
# Rebuild the submitted archive from this directory (run fetch_weights.py first).
# File contents match the Test archive byte for byte (see README); the zip's own
# sha256 differs only through file timestamps.
set -euo pipefail
cd "$(dirname "$0")"
rm -f simulacrabench_v15.zip
zip -X -q simulacrabench_v15.zip main.py requirements.txt LICENSE_TABICL LICENSE_TABPFN3 NOTICE \
    tabicl-classifier-v2-20260212.ckpt tabpfn-v3-classifier-v3_default.ckpt
sha256sum main.py requirements.txt LICENSE_TABICL LICENSE_TABPFN3 NOTICE *.ckpt
