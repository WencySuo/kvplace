#!/usr/bin/env bash
# Launch a single-GPU PCIe benchmark box on EC2 for kvdiag. Idempotent-ish; prints the ssh line.
#
#   aws/launch.sh p5.4xlarge      # 1x H100 80GB, PCIe Gen5 host link, ~$6.9/hr on-demand (needs P quota)
#   aws/launch.sh g6e.xlarge      # 1x L40S 48GB, PCIe Gen4 host link, ~$1.9/hr (needs G quota)
#   aws/launch.sh --terminate     # terminate the instance recorded in aws/.instance
#
# Prereqs: `aws configure` done (or AWS_PROFILE set); region with the instance type.
set -euo pipefail
cd "$(dirname "$0")"
REGION="${AWS_REGION:-$(aws configure get region || echo us-east-1)}"
KEY="${KEY_NAME:-kvdiag}"
TYPE="${1:-g6e.xlarge}"
DISK_GB=200

if [[ "$TYPE" == "--terminate" ]]; then
  ID=$(cat .instance); aws ec2 terminate-instances --region "$REGION" --instance-ids "$ID" --output table; rm -f .instance; exit 0
fi

echo "== region $REGION, type $TYPE"
# 1. quota (P: L-417A185B, G/VT: L-DB2E81BA), in vCPUs
case "$TYPE" in p5*) Q=L-417A185B;; g*) Q=L-DB2E81BA;; *) Q="";; esac
if [[ -n "$Q" ]]; then
  V=$(aws service-quotas get-service-quota --region "$REGION" --service-code ec2 --quota-code $Q --query 'Quota.Value' --output text)
  NEED=$(aws ec2 describe-instance-types --region "$REGION" --instance-types "$TYPE" --query 'InstanceTypes[0].VCpuInfo.DefaultVCpus' --output text)
  echo "== quota $Q = $V vCPUs; $TYPE needs $NEED"
  if (( ${V%.*} < NEED )); then
    echo "!! quota too low. Request with:"
    echo "   aws service-quotas request-service-quota-increase --region $REGION --service-code ec2 --quota-code $Q --desired-value $NEED"
    exit 2
  fi
fi

# 2. AMI: Deep Learning Base (OSS NVIDIA driver, Ubuntu 22.04). We install torch ourselves in a venv.
AMI=$(aws ssm get-parameter --region "$REGION" \
  --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id \
  --query Parameter.Value --output text)
echo "== ami $AMI"

# 3. key pair + security group (ssh from this machine's IP only)
if ! aws ec2 describe-key-pairs --region "$REGION" --key-names "$KEY" >/dev/null 2>&1; then
  aws ec2 create-key-pair --region "$REGION" --key-name "$KEY" --query KeyMaterial --output text > "$HOME/.ssh/$KEY.pem"
  chmod 600 "$HOME/.ssh/$KEY.pem"; echo "== created key ~/.ssh/$KEY.pem"
fi
MYIP=$(curl -s https://checkip.amazonaws.com)/32
SG=$(aws ec2 describe-security-groups --region "$REGION" --filters Name=group-name,Values=kvdiag --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo None)
if [[ "$SG" == "None" ]]; then
  SG=$(aws ec2 create-security-group --region "$REGION" --group-name kvdiag --description "kvdiag ssh" --query GroupId --output text)
fi
aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG" --protocol tcp --port 22 --cidr "$MYIP" >/dev/null 2>&1 || true

# 4. launch
ID=$(aws ec2 run-instances --region "$REGION" --image-id "$AMI" --instance-type "$TYPE" --key-name "$KEY" \
  --security-group-ids "$SG" --user-data file://bootstrap.sh \
  --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$DISK_GB,\"VolumeType\":\"gp3\"}}]" \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=kvdiag}]' \
  --query 'Instances[0].InstanceId' --output text)
echo "$ID" > .instance
echo "== launched $ID; waiting for running"
aws ec2 wait instance-running --region "$REGION" --instance-ids "$ID"
IP=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$ID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "== ssh -i ~/.ssh/$KEY.pem ubuntu@$IP"
echo "   bootstrap runs on first boot (~5 min): tail -f /var/log/cloud-init-output.log"
echo "   REMEMBER: aws/launch.sh --terminate   (billing continues while stopped-but-not-terminated storage exists)"
