#!/usr/bin/env python3
"""
Trendr Pipeline Monitoring Dashboard
Simple web interface for monitoring pipeline health and statistics
"""
import sys
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
from pathlib import Path
import threading
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from flask import Flask, render_template, jsonify, request
    from utils.database import SupabaseManager
    from utils.api_cache import APICache
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

class PipelineMonitor:
    """Real-time pipeline monitoring system"""
    
    def __init__(self):
        self.db = SupabaseManager()
        self.cache = APICache()
        self.log_path = "/var/log/trendr/pipeline.log"
        self.stats = {}
        
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        try:
            # Database health
            pois = self.db.get_pois_for_city("Montreal", limit=1)
            db_healthy = len(pois) > 0
            
            # Cache health  
            cache_stats = self.cache.get_stats()
            
            # Log file health
            log_exists = os.path.exists(self.log_path)
            
            # Last execution time
            last_execution = self.get_last_execution_time()
            
            return {
                "status": "healthy" if db_healthy and log_exists else "warning",
                "database": "connected" if db_healthy else "error",
                "cache": cache_stats,
                "logs": "available" if log_exists else "missing",
                "last_execution": last_execution,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        try:
            # POI counts by category
            poi_stats = {}
            categories = ["restaurant", "cafe", "bar", "tourist_attraction", "shopping_mall", "store"]
            
            for category in categories:
                pois = self.db.client.table('poi')\
                    .select('id')\
                    .eq('city', 'Montreal')\
                    .eq('category', category)\
                    .execute()
                poi_stats[category] = len(pois.data)
            
            # Collections count
            collections = self.db.client.table('collections')\
                .select('id')\
                .eq('city', 'Montreal')\
                .execute()
            
            # Neighborhoods count
            neighborhoods = self.db.client.table('neighborhoods')\
                .select('id')\
                .eq('city', 'Montreal')\
                .execute()
            
            # Recent activity (last 7 days)
            week_ago = (datetime.now() - timedelta(days=7)).isoformat()
            recent_pois = self.db.client.table('poi')\
                .select('id')\
                .gte('created_at', week_ago)\
                .execute()
            
            return {
                "poi_by_category": poi_stats,
                "total_pois": sum(poi_stats.values()),
                "total_collections": len(collections.data),
                "total_neighborhoods": len(neighborhoods.data),
                "recent_pois_7days": len(recent_pois.data),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_api_usage_stats(self) -> Dict[str, Any]:
        """Get API usage and quota information"""
        try:
            # Parse recent logs for API usage
            api_stats = {
                "google_places_calls": 0,
                "google_search_calls": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "quota_status": "unknown"
            }
            
            if os.path.exists(self.log_path):
                with open(self.log_path, 'r') as f:
                    recent_logs = f.readlines()[-1000:]  # Last 1000 lines
                    
                for line in recent_logs:
                    if "Google Places API" in line:
                        api_stats["google_places_calls"] += 1
                    elif "Google Custom Search" in line:
                        api_stats["google_search_calls"] += 1
                    elif "Cache HIT" in line:
                        api_stats["cache_hits"] += 1
                    elif "Cache MISS" in line:
                        api_stats["cache_misses"] += 1
            
            # Calculate cache efficiency
            total_cache_requests = api_stats["cache_hits"] + api_stats["cache_misses"]
            if total_cache_requests > 0:
                api_stats["cache_efficiency"] = (api_stats["cache_hits"] / total_cache_requests) * 100
            else:
                api_stats["cache_efficiency"] = 0
                
            # Quota status
            if api_stats["google_search_calls"] < 80:
                api_stats["quota_status"] = "healthy"
            elif api_stats["google_search_calls"] < 95:
                api_stats["quota_status"] = "warning" 
            else:
                api_stats["quota_status"] = "critical"
            
            return api_stats
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_last_execution_time(self) -> str:
        """Get timestamp of last pipeline execution"""
        try:
            if os.path.exists(self.log_path):
                with open(self.log_path, 'r') as f:
                    lines = f.readlines()
                    for line in reversed(lines[-100:]):  # Check last 100 lines
                        if "TRENDR DATA PIPELINE EXECUTION SUMMARY" in line:
                            # Extract timestamp from log line
                            parts = line.split(' - ')
                            if len(parts) > 0:
                                return parts[0]
            return "Unknown"
        except:
            return "Error reading logs"
    
    def get_error_summary(self) -> List[Dict[str, str]]:
        """Get recent errors from logs"""
        errors = []
        try:
            if os.path.exists(self.log_path):
                with open(self.log_path, 'r') as f:
                    lines = f.readlines()[-500:]  # Last 500 lines
                    
                for line in lines:
                    if "ERROR" in line or "CRITICAL" in line:
                        parts = line.strip().split(' - ')
                        if len(parts) >= 3:
                            errors.append({
                                "timestamp": parts[0],
                                "level": parts[2],
                                "message": ' - '.join(parts[3:])
                            })
            
            return errors[-10:]  # Return last 10 errors
            
        except Exception as e:
            return [{"error": str(e)}]

# Flask Web Dashboard
if HAS_FLASK:
    app = Flask(__name__)
    monitor = PipelineMonitor()
    
    @app.route('/')
    def dashboard():
        """Main dashboard page"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trendr Pipeline Monitor</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; }
                .card { background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .status { padding: 10px; border-radius: 4px; margin: 10px 0; }
                .status.healthy { background: #d4edda; color: #155724; }
                .status.warning { background: #fff3cd; color: #856404; }
                .status.error { background: #f8d7da; color: #721c24; }
                .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
                .stat { text-align: center; padding: 15px; }
                .stat-value { font-size: 2em; font-weight: bold; color: #007bff; }
                .stat-label { color: #666; margin-top: 5px; }
                .refresh { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
                .log-entry { font-family: monospace; font-size: 12px; padding: 5px; border-left: 3px solid #ccc; margin: 5px 0; }
                .error-entry { border-left-color: #dc3545; background: #f8f9fa; }
            </style>
            <script>
                function refreshData() {
                    fetch('/api/health')
                        .then(response => response.json())
                        .then(data => updateHealth(data));
                    
                    fetch('/api/stats')
                        .then(response => response.json())
                        .then(data => updateStats(data));
                        
                    fetch('/api/api-usage')
                        .then(response => response.json())
                        .then(data => updateApiUsage(data));
                }
                
                function updateHealth(data) {
                    const statusEl = document.getElementById('system-status');
                    statusEl.className = 'status ' + data.status;
                    statusEl.innerHTML = `
                        <strong>System Status: ${data.status.toUpperCase()}</strong><br>
                        Database: ${data.database}<br>
                        Last Execution: ${data.last_execution}<br>
                        Cache Entries: ${data.cache.entries_count || 'N/A'}
                    `;
                }
                
                function updateStats(data) {
                    document.getElementById('total-pois').textContent = data.total_pois || 0;
                    document.getElementById('total-collections').textContent = data.total_collections || 0;
                    document.getElementById('recent-pois').textContent = data.recent_pois_7days || 0;
                    
                    const categoryStats = document.getElementById('category-stats');
                    let categoryHtml = '';
                    for (const [category, count] of Object.entries(data.poi_by_category || {})) {
                        categoryHtml += `<div>${category}: <strong>${count}</strong></div>`;
                    }
                    categoryStats.innerHTML = categoryHtml;
                }
                
                function updateApiUsage(data) {
                    document.getElementById('api-calls').textContent = data.google_search_calls || 0;
                    document.getElementById('cache-efficiency').textContent = Math.round(data.cache_efficiency || 0) + '%';
                    
                    const quotaEl = document.getElementById('quota-status');
                    quotaEl.className = 'status ' + (data.quota_status === 'healthy' ? 'healthy' : data.quota_status === 'warning' ? 'warning' : 'error');
                    quotaEl.textContent = `Quota Status: ${data.quota_status}`;
                }
                
                // Auto-refresh every 30 seconds
                setInterval(refreshData, 30000);
                
                // Initial load
                window.onload = refreshData;
            </script>
        </head>
        <body>
            <div class="container">
                <h1>🚀 Trendr Pipeline Monitor</h1>
                <button class="refresh" onclick="refreshData()">🔄 Refresh</button>
                
                <div class="card">
                    <h2>System Health</h2>
                    <div id="system-status" class="status">Loading...</div>
                </div>
                
                <div class="grid">
                    <div class="card">
                        <div class="stat">
                            <div class="stat-value" id="total-pois">-</div>
                            <div class="stat-label">Total POIs</div>
                        </div>
                    </div>
                    <div class="card">
                        <div class="stat">
                            <div class="stat-value" id="total-collections">-</div>
                            <div class="stat-label">Collections</div>
                        </div>
                    </div>
                    <div class="card">
                        <div class="stat">
                            <div class="stat-value" id="recent-pois">-</div>
                            <div class="stat-label">New POIs (7 days)</div>
                        </div>
                    </div>
                </div>
                
                <div class="grid">
                    <div class="card">
                        <h3>POIs by Category</h3>
                        <div id="category-stats">Loading...</div>
                    </div>
                    <div class="card">
                        <h3>API Usage</h3>
                        <div>Search API Calls: <strong id="api-calls">-</strong></div>
                        <div>Cache Efficiency: <strong id="cache-efficiency">-</strong></div>
                        <div id="quota-status" class="status">Loading...</div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
    
    @app.route('/api/health')
    def api_health():
        return jsonify(monitor.get_system_health())
    
    @app.route('/api/stats') 
    def api_stats():
        return jsonify(monitor.get_pipeline_statistics())
    
    @app.route('/api/api-usage')
    def api_usage():
        return jsonify(monitor.get_api_usage_stats())
    
    @app.route('/api/errors')
    def api_errors():
        return jsonify(monitor.get_error_summary())

def main():
    """Main monitoring interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Trendr Pipeline Monitor')
    parser.add_argument('--web', action='store_true', help='Start web dashboard')
    parser.add_argument('--stats', action='store_true', help='Show current statistics')
    parser.add_argument('--health', action='store_true', help='Check system health')
    parser.add_argument('--port', type=int, default=5000, help='Web dashboard port')
    
    args = parser.parse_args()
    
    monitor = PipelineMonitor()
    
    if args.web:
        if not HAS_FLASK:
            print("❌ Flask not installed. Install with: pip install flask")
            return
        print(f"🌐 Starting web dashboard on http://localhost:{args.port}")
        app.run(host='0.0.0.0', port=args.port, debug=False)
    
    elif args.stats:
        stats = monitor.get_pipeline_statistics()
        print("📊 Pipeline Statistics:")
        print(json.dumps(stats, indent=2))
    
    elif args.health:
        health = monitor.get_system_health()
        print("🏥 System Health:")
        print(json.dumps(health, indent=2))
    
    else:
        print("Trendr Pipeline Monitor")
        print("Usage: python monitoring/dashboard.py --web|--stats|--health")

if __name__ == "__main__":
    main()