#!/bin/bash
# Script pour arrêter le système de monitoring Trendr

echo "🛑 Stopping Trendr Monitoring System..."

# Check if monitoring is running
if [ -f "logs/monitoring.pid" ]; then
    MONITOR_PID=$(cat logs/monitoring.pid)
    
    if kill -0 $MONITOR_PID 2>/dev/null; then
        echo "📛 Stopping monitoring process PID: $MONITOR_PID"
        kill $MONITOR_PID
        
        # Wait for graceful shutdown
        sleep 3
        
        if kill -0 $MONITOR_PID 2>/dev/null; then
            echo "⚠️ Process still running, forcing termination..."
            kill -9 $MONITOR_PID
        fi
        
        rm -f logs/monitoring.pid
        echo "✅ Monitoring system stopped"
    else
        echo "ℹ️ Monitoring process not running"
        rm -f logs/monitoring.pid
    fi
else
    echo "ℹ️ No monitoring PID file found"
fi

# Clean up any remaining monitoring processes
pkill -f "monitoring_system.py" 2>/dev/null || true

echo "🧹 Monitoring cleanup complete"