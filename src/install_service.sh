#!/usr/bin/env bash
# 과제 8 예측 서비스를 systemd 에 등록 (재부팅하면 자동 시작, 죽으면 다시 시작)
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
PY="$HOME/collector/venv/bin/python"
sudo tee /etc/systemd/system/forecast-service.service > /dev/null <<EOF
[Unit]
Description=team2 forecast service (prediction, anomaly, occupancy)
After=network-online.target collector.service
Wants=network-online.target

[Service]
User=$USER
WorkingDirectory=$DIR
ExecStart=$PY $DIR/service.py
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now forecast-service
sleep 5
systemctl status forecast-service --no-pager | head -8
