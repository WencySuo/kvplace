#!/usr/bin/env bash
# On the instance: the full suite in the right order. Stops at the first stage that fails.
set -euo pipefail
cd /home/ubuntu/kvplace && . /home/ubuntu/venv/bin/activate
HBM_GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{printf "%.0f", $1/1024}')
python -c "import kvdiag.hostmem as h; h.selftest()"
python -m pytest tests -q
python kvdiag/audit.py --hbm-gb "$HBM_GB"
python kvdiag/bench_link.py --skip nvbandwidth
python kvdiag/bench_attn.py --S 32768 --arms hbm host_direct --R 1 16 128 1024 4096 --reps 1
python kvdiag/summarize.py | grep -A12 "Stage C"      # check hbm effective GB/s at small R before continuing
python kvdiag/bench_attn.py --S 8192 32768 --R 1 16 128 1024 4096 16384 --reps 3
python kvdiag/profile.py --S 8192 32768 --B 1 8
python kvdiag/plot.py && python kvdiag/summarize.py
