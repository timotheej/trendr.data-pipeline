#!/bin/bash
# Script pour démarrer le système de monitoring Trendr
# Usage: ./start-monitoring.sh [daemon|foreground]

echo "🚀 Starting Trendr Monitoring System..."

# Check if Python environment is ready
if ! python3 -c "import schedule" 2>/dev/null; then
    echo "❌ Dependencies missing. Please run: pip install -r requirements.txt"
    exit 1
fi

# Check if configuration is present
if [ ! -f "config.py" ]; then
    echo "❌ config.py not found. Please ensure configuration is present."
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Run monitoring system
MODE=${1:-daemon}

if [ "$MODE" = "daemon" ]; then
    echo "🔄 Starting monitoring in daemon mode..."
    nohup python3 monitoring_system.py --daemon > logs/monitoring.log 2>&1 &
    MONITOR_PID=$!
    echo $MONITOR_PID > logs/monitoring.pid
    echo "✅ Monitoring started with PID: $MONITOR_PID"
    echo "📊 View logs: tail -f logs/monitoring.log"
elif [ "$MODE" = "foreground" ]; then
    echo "🔄 Starting monitoring in foreground mode..."
    python3 monitoring_system.py
else
    echo "Usage: $0 [daemon|foreground]"
    exit 1
fi