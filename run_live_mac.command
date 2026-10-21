#!/bin/bash
set -e

cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
  echo "Creating live virtual environment..."
  python3 -m venv venv
fi

source venv/bin/activate

echo "Installing requirements..."
pip install -r requirements.txt

echo ""
echo "Starting Local Live Helmet Monitor V3 + Collection Dashboard..."
echo "Make sure the iPhone Continuity Camera is available on camera index 0."
echo ""

python3 live_helmet_monitor_v3_dashboard.py
