#!/usr/bin/env python3
"""
Trendr Monitoring System - Autonomous Pipeline Management
Monitors API usage, system health, and manages automated pipeline execution.
"""
import sys
import os
import logging
import time
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional
import schedule
import threading

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.database import SupabaseManager
from run_pipeline import TrendrDataPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrendrMonitoringSystem:
    """Complete monitoring and automation system for Trendr."""
    
    def __init__(self):
        self.db = SupabaseManager()
        self.config = self.load_monitoring_config()
        self.is_running = False
        
        # System health tracking
        self.health_status = {
            'api_quota_ok': True,
            'database_ok': True,
            'last_successful_run': None,
            'errors_last_24h': 0,
            'pipeline_status': 'idle'
        }
        
        logger.info("🔍 Trendr Monitoring System initialized")
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, 'isoformat'):  # Handle other datetime-like objects
            return obj.isoformat()
        else:
            return obj
    
    def load_monitoring_config(self) -> Dict[str, Any]:
        """Load monitoring configuration."""
        default_config = {
            'daily_api_limit': 95,
            'alert_threshold': 80,  # Alert at 80% quota usage
            'max_errors_per_day': 10,
            'pipeline_schedule': {
                'full_run': '02:00',  # Daily at 2 AM
                'classification_only': '14:00',  # Daily at 2 PM  
                'collections_only': '18:00'  # Daily at 6 PM
            },
            'cities': ['Paris'],
            'health_check_interval': 300,  # 5 minutes
            'cleanup_logs_days': 7
        }
        return default_config
    
    def check_api_quota(self) -> Dict[str, Any]:
        """Check current API quota usage."""
        try:
            today = datetime.now().date().isoformat()
            
            # Get today's API usage
            result = self.db.client.table('api_usage')\
                .select('queries_count')\
                .eq('date', today)\
                .eq('api_type', 'google_search')\
                .execute()
            
            total_queries = sum(row['queries_count'] for row in result.data) if result.data else 0
            quota_percentage = (total_queries / self.config['daily_api_limit']) * 100
            
            quota_status = {
                'total_queries': total_queries,
                'daily_limit': self.config['daily_api_limit'],
                'remaining': max(0, self.config['daily_api_limit'] - total_queries),
                'percentage_used': quota_percentage,
                'status': 'ok' if quota_percentage < self.config['alert_threshold'] else 'warning'
            }
            
            logger.info(f"📊 API Quota: {total_queries}/{self.config['daily_api_limit']} ({quota_percentage:.1f}%)")
            return quota_status
            
        except Exception as e:
            logger.error(f"Error checking API quota: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity and basic health."""
        try:
            # Test basic queries
            poi_count = self.db.client.table('poi').select('id', count='exact').execute()
            collections_count = self.db.client.table('collections').select('id', count='exact').execute()
            
            # Check for recent activity
            recent_pois = self.db.client.table('poi')\
                .select('updated_at')\
                .gte('updated_at', (datetime.now() - timedelta(days=1)).isoformat())\
                .execute()
            
            db_health = {
                'status': 'ok',
                'total_pois': poi_count.count,
                'total_collections': collections_count.count,
                'recent_updates': len(recent_pois.data) if recent_pois.data else 0
            }
            
            logger.info(f"💾 Database Health: {db_health['total_pois']} POIs, {db_health['recent_updates']} recent updates")
            return db_health
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def record_monitoring_report(self, report_data: Dict[str, Any]) -> bool:
        """Record monitoring report to database using existing schema."""
        try:
            # Adapt to existing monitoring_reports schema
            report = {
                'city': self.config['cities'][0] if self.config['cities'] else 'Paris',
                'monitoring_date': datetime.now(timezone.utc).isoformat(),
                'trending_pois': [],  # Will be populated by pipeline results
                'new_entrants': [],
                'significant_changes': [],
                'summary': {
                    'system_health': report_data.get('system_status', 'unknown'),
                    'api_quota_used': report_data.get('api_quota_used', 0),
                    'database_health': report_data.get('database_health', 'unknown'),
                    'errors_count': report_data.get('errors_count', 0),
                    'last_pipeline_run': report_data.get('last_pipeline_run'),
                    'alerts': report_data.get('alerts', []),
                    'monitoring_type': 'system_health'
                }
            }
            
            result = self.db.client.table('monitoring_reports').insert(report).execute()
            return bool(result.data)
            
        except Exception as e:
            logger.error(f"Failed to record monitoring report: {e}")
            return False
    
    def run_automated_pipeline(self, mode: str = 'full') -> Dict[str, Any]:
        """Run automated pipeline execution."""
        logger.info(f"🚀 Starting automated pipeline execution: {mode}")
        
        try:
            self.health_status['pipeline_status'] = 'running'
            
            # Initialize pipeline
            pipeline = TrendrDataPipeline()
            pipeline.config['cities'] = self.config['cities']
            pipeline.config['country'] = 'France'
            
            # Execute based on mode
            if mode == 'full':
                success = pipeline.run_full_pipeline()
            elif mode == 'classification':
                success = pipeline.process_social_proofs_and_classification()
            elif mode == 'collections':
                success = pipeline.generate_collections()
            else:
                logger.error(f"Unknown pipeline mode: {mode}")
                success = False
            
            # Update health status
            if success:
                self.health_status['last_successful_run'] = datetime.now().isoformat()
                self.health_status['pipeline_status'] = 'completed'
                logger.info(f"✅ Pipeline {mode} completed successfully")
            else:
                self.health_status['pipeline_status'] = 'failed'
                self.health_status['errors_last_24h'] += 1
                logger.error(f"❌ Pipeline {mode} failed")
            
            # Convert stats to JSON-serializable format
            serializable_stats = self._make_json_serializable(pipeline.stats)
            
            return {
                'mode': mode,
                'success': success,
                'timestamp': datetime.now().isoformat(),
                'stats': serializable_stats
            }
            
        except Exception as e:
            logger.error(f"Pipeline execution error: {e}")
            self.health_status['pipeline_status'] = 'error'
            self.health_status['errors_last_24h'] += 1
            
            return {
                'mode': mode,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'stats': {}
            }
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        api_status = self.check_api_quota()
        db_status = self.check_database_health()
        
        # Calculate overall system health
        system_health = 'healthy'
        alerts = []
        
        if api_status.get('percentage_used', 0) > self.config['alert_threshold']:
            system_health = 'warning'
            alerts.append(f"API quota at {api_status['percentage_used']:.1f}%")
        
        if db_status.get('status') != 'ok':
            system_health = 'error'
            alerts.append("Database connectivity issues")
        
        if self.health_status['errors_last_24h'] > self.config['max_errors_per_day']:
            system_health = 'error'
            alerts.append(f"Too many errors: {self.health_status['errors_last_24h']}")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_health': system_health,
            'api_status': api_status,
            'database_status': db_status,
            'pipeline_status': self.health_status['pipeline_status'],
            'last_successful_run': self.health_status['last_successful_run'],
            'errors_24h': self.health_status['errors_last_24h'],
            'alerts': alerts
        }
        
        # Record to database
        self.record_monitoring_report({
            'system_status': system_health,
            'api_quota_used': api_status.get('percentage_used', 0),
            'database_health': db_status.get('status', 'unknown'),
            'errors_count': self.health_status['errors_last_24h'],
            'last_pipeline_run': self.health_status['last_successful_run'],
            'api_details': api_status,
            'alerts': alerts
        })
        
        return report
    
    def setup_scheduled_tasks(self):
        """Setup scheduled pipeline tasks."""
        logger.info("📅 Setting up scheduled tasks...")
        
        # Daily full pipeline
        schedule.every().day.at(self.config['pipeline_schedule']['full_run']).do(
            self.run_automated_pipeline, 'full'
        )
        
        # Daily classification update
        schedule.every().day.at(self.config['pipeline_schedule']['classification_only']).do(
            self.run_automated_pipeline, 'classification'
        )
        
        # Daily collections update
        schedule.every().day.at(self.config['pipeline_schedule']['collections_only']).do(
            self.run_automated_pipeline, 'collections'
        )
        
        # Health checks every 5 minutes
        schedule.every(5).minutes.do(self.generate_health_report)
        
        logger.info("✅ Scheduled tasks configured")
    
    def run_monitoring_loop(self):
        """Main monitoring loop."""
        logger.info("🔄 Starting monitoring loop...")
        self.is_running = True
        
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except KeyboardInterrupt:
                logger.info("🛑 Received interrupt signal")
                self.is_running = False
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)  # Continue after error
    
    def start_monitoring(self, daemon: bool = False):
        """Start the monitoring system."""
        self.setup_scheduled_tasks()
        
        # Initial health check
        initial_report = self.generate_health_report()
        logger.info(f"🚀 Initial system status: {initial_report['system_health']}")
        
        if daemon:
            # Run in background thread but keep main process alive
            monitoring_thread = threading.Thread(target=self.run_monitoring_loop, daemon=True)
            monitoring_thread.start()
            logger.info("🔄 Monitoring started in daemon mode")
            
            # Keep main thread alive
            try:
                while self.is_running:
                    time.sleep(30)  # Check every 30 seconds
                    if not monitoring_thread.is_alive():
                        logger.error("❌ Monitoring thread died, restarting...")
                        monitoring_thread = threading.Thread(target=self.run_monitoring_loop, daemon=True)
                        monitoring_thread.start()
            except KeyboardInterrupt:
                logger.info("🛑 Received interrupt signal")
                self.stop_monitoring()
        else:
            # Run in foreground
            self.run_monitoring_loop()
    
    def stop_monitoring(self):
        """Stop the monitoring system."""
        logger.info("🛑 Stopping monitoring system...")
        self.is_running = False

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Trendr Monitoring System')
    parser.add_argument('--daemon', action='store_true', help='Run in daemon mode')
    parser.add_argument('--health-check', action='store_true', help='Run single health check')
    parser.add_argument('--run-pipeline', choices=['full', 'classification', 'collections'], 
                       help='Run pipeline once')
    
    args = parser.parse_args()
    
    monitor = TrendrMonitoringSystem()
    
    try:
        if args.health_check:
            report = monitor.generate_health_report()
            print(json.dumps(report, indent=2))
        elif args.run_pipeline:
            result = monitor.run_automated_pipeline(args.run_pipeline)
            print(json.dumps(result, indent=2))
        else:
            monitor.start_monitoring(daemon=args.daemon)
    
    except KeyboardInterrupt:
        monitor.stop_monitoring()
        logger.info("👋 Monitoring system stopped")
    except Exception as e:
        logger.error(f"Monitoring system error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()