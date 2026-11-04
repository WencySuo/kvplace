#!/usr/bin/env bash
# cloud-init user-data: runs once as root on first boot. Installs the GPU python stack for kvdiag.
set -eux
export DEBIAN_FRONTEND=noninteractive
apt-get update -y && apt-get install -y python3-venv git
sudo -u ubuntu bash <<'INNER'
set -eux
cd /home/ubuntu
git clone https://github.com/WencySuo/kvplace.git || true
python3 -m venv venv
. venv/bin/activate
pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install flashinfer-python transformers accelerate numpy matplotlib pytest ninja
nvidia-smi > /home/ubuntu/nvidia-smi.txt
cd kvplace && python -m pytest tests/test_shapes.py -q > /home/ubuntu/bootstrap-tests.txt 2>&1 || true
echo BOOTSTRAP_DONE > /home/ubuntu/BOOTSTRAP_DONE
INNER
