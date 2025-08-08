#!/usr/bin/env python3
"""
Trendr Pipeline Alerting System
Monitors pipeline health and sends notifications when issues occur
"""
import sys
import os
import json
import smtplib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from utils.database import SupabaseManager

class AlertSystem:
    """Pipeline alerting and notification system"""
    
    def __init__(self, config_file: str = "monitoring/alerts_config.json"):
        self.config = self.load_config(config_file)
        self.db = SupabaseManager()
        self.log_path = "/var/log/trendr/pipeline.log"
        
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """Load alerting configuration"""
        default_config = {
            "email": {
                "enabled": False,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "",
                "password": "",
                "to_emails": []
            },
            "discord": {
                "enabled": False,
                "webhook_url": ""
            },
            "slack": {
                "enabled": False,
                "webhook_url": ""
            },
            "thresholds": {
                "max_errors_per_hour": 10,
                "max_api_quota_usage": 90,
                "min_pois_ingested_daily": 5,
                "max_execution_time_minutes": 30
            }
        }
        
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        except FileNotFoundError:
            print(f"⚠️ Config file {config_file} not found, using defaults")
        
        return default_config
    
    def check_system_health(self) -> Dict[str, Any]:
        """Comprehensive system health check"""
        issues = []
        severity = "ok"
        
        # Check database connectivity
        try:
            pois = self.db.get_pois_for_city("Montreal", limit=1)
            if not pois:
                issues.append({
                    "type": "database",
                    "severity": "critical",
                    "message": "Database connection failed or no POIs found"
                })
                severity = "critical"
        except Exception as e:
            issues.append({
                "type": "database", 
                "severity": "critical",
                "message": f"Database error: {str(e)}"
            })
            severity = "critical"
        
        # Check log file and recent errors
        error_count = self.count_recent_errors(hours=1)
        if error_count > self.config["thresholds"]["max_errors_per_hour"]:
            issues.append({
                "type": "errors",
                "severity": "warning",
                "message": f"{error_count} errors in the last hour (threshold: {self.config['thresholds']['max_errors_per_hour']})"
            })
            if severity != "critical":
                severity = "warning"
        
        # Check last execution time
        last_execution = self.get_last_execution_time()
        if last_execution:
            try:
                last_time = datetime.fromisoformat(last_execution.replace('Z', '+00:00'))
                if datetime.now() - last_time > timedelta(days=2):
                    issues.append({
                        "type": "execution",
                        "severity": "critical", 
                        "message": f"Pipeline hasn't run since {last_execution}"
                    })
                    severity = "critical"
            except:
                pass
        
        # Check API quota usage
        api_usage = self.check_api_quota_usage()
        if api_usage > self.config["thresholds"]["max_api_quota_usage"]:
            issues.append({
                "type": "api_quota",
                "severity": "warning",
                "message": f"API quota usage at {api_usage}% (threshold: {self.config['thresholds']['max_api_quota_usage']}%)"
            })
            if severity != "critical":
                severity = "warning"
        
        return {
            "status": severity,
            "issues": issues,
            "timestamp": datetime.now().isoformat(),
            "total_issues": len(issues)
        }
    
    def count_recent_errors(self, hours: int = 1) -> int:
        """Count errors in recent logs"""
        if not os.path.exists(self.log_path):
            return 0
            
        error_count = 0
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        try:
            with open(self.log_path, 'r') as f:
                for line in f.readlines()[-1000:]:  # Last 1000 lines
                    if "ERROR" in line or "CRITICAL" in line:
                        # Try to extract timestamp
                        try:
                            timestamp_str = line.split(' - ')[0]
                            log_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            if log_time > cutoff_time:
                                error_count += 1
                        except:
                            error_count += 1  # Count it anyway if we can't parse timestamp
        except Exception:
            pass
            
        return error_count
    
    def get_last_execution_time(self) -> str:
        """Get last pipeline execution timestamp"""
        if not os.path.exists(self.log_path):
            return None
            
        try:
            with open(self.log_path, 'r') as f:
                lines = f.readlines()
                for line in reversed(lines[-200:]):
                    if "PIPELINE EXECUTED SUCCESSFULLY" in line or "EXECUTION SUMMARY" in line:
                        parts = line.split(' - ')
                        if parts:
                            return parts[0]
        except Exception:
            pass
            
        return None
    
    def check_api_quota_usage(self) -> int:
        """Check API quota usage percentage"""
        if not os.path.exists(self.log_path):
            return 0
            
        search_calls = 0
        try:
            with open(self.log_path, 'r') as f:
                today = datetime.now().date()
                for line in f.readlines()[-2000:]:  # Last 2000 lines
                    try:
                        if "Google Custom Search" in line or "search API" in line:
                            line_date = datetime.fromisoformat(line.split(' - ')[0]).date()
                            if line_date == today:
                                search_calls += 1
                    except:
                        continue
        except Exception:
            pass
            
        return int((search_calls / 100) * 100)  # Percentage of 100 daily quota
    
    def send_email_alert(self, health_status: Dict[str, Any]) -> bool:
        """Send email notification"""
        if not self.config["email"]["enabled"]:
            return False
            
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config["email"]["username"]
            msg['To'] = ", ".join(self.config["email"]["to_emails"])
            msg['Subject'] = f"🚨 Trendr Pipeline Alert - {health_status['status'].upper()}"
            
            body = f"""
Trendr Pipeline Alert

Status: {health_status['status'].upper()}
Timestamp: {health_status['timestamp']}
Total Issues: {health_status['total_issues']}

Issues Detected:
"""
            
            for issue in health_status['issues']:
                body += f"\n• {issue['severity'].upper()}: {issue['message']}"
            
            body += "\n\nPlease check the pipeline logs for more details."
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.config["email"]["smtp_server"], self.config["email"]["smtp_port"])
            server.starttls()
            server.login(self.config["email"]["username"], self.config["email"]["password"])
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to send email: {e}")
            return False
    
    def send_discord_alert(self, health_status: Dict[str, Any]) -> bool:
        """Send Discord webhook notification"""
        if not self.config["discord"]["enabled"] or not HAS_REQUESTS:
            return False
            
        try:
            color = 0x00ff00 if health_status['status'] == 'ok' else 0xffaa00 if health_status['status'] == 'warning' else 0xff0000
            
            embed = {
                "title": "🚨 Trendr Pipeline Alert",
                "color": color,
                "fields": [
                    {"name": "Status", "value": health_status['status'].upper(), "inline": True},
                    {"name": "Issues", "value": str(health_status['total_issues']), "inline": True},
                    {"name": "Timestamp", "value": health_status['timestamp'], "inline": False}
                ]
            }
            
            if health_status['issues']:
                issues_text = "\n".join([f"• **{issue['severity'].upper()}**: {issue['message']}" for issue in health_status['issues']])
                embed["fields"].append({"name": "Details", "value": issues_text[:1024], "inline": False})
            
            payload = {"embeds": [embed]}
            
            response = requests.post(self.config["discord"]["webhook_url"], json=payload)
            return response.status_code == 204
            
        except Exception as e:
            print(f"❌ Failed to send Discord alert: {e}")
            return False
    
    def send_slack_alert(self, health_status: Dict[str, Any]) -> bool:
        """Send Slack webhook notification"""
        if not self.config["slack"]["enabled"] or not HAS_REQUESTS:
            return False
            
        try:
            color = "good" if health_status['status'] == 'ok' else "warning" if health_status['status'] == 'warning' else "danger"
            
            attachment = {
                "color": color,
                "title": "🚨 Trendr Pipeline Alert",
                "fields": [
                    {"title": "Status", "value": health_status['status'].upper(), "short": True},
                    {"title": "Issues", "value": str(health_status['total_issues']), "short": True},
                    {"title": "Timestamp", "value": health_status['timestamp'], "short": False}
                ]
            }
            
            if health_status['issues']:
                issues_text = "\n".join([f"• *{issue['severity'].upper()}*: {issue['message']}" for issue in health_status['issues']])
                attachment["fields"].append({"title": "Details", "value": issues_text, "short": False})
            
            payload = {"attachments": [attachment]}
            
            response = requests.post(self.config["slack"]["webhook_url"], json=payload)
            return response.status_code == 200
            
        except Exception as e:
            print(f"❌ Failed to send Slack alert: {e}")
            return False
    
    def run_health_check_and_alert(self) -> Dict[str, Any]:
        """Run health check and send alerts if needed"""
        health_status = self.check_system_health()
        
        # Only send alerts for warnings and critical issues
        if health_status['status'] in ['warning', 'critical']:
            print(f"🚨 Alerting for status: {health_status['status']}")
            
            # Send notifications
            email_sent = self.send_email_alert(health_status)
            discord_sent = self.send_discord_alert(health_status) 
            slack_sent = self.send_slack_alert(health_status)
            
            health_status['notifications'] = {
                "email": email_sent,
                "discord": discord_sent,
                "slack": slack_sent
            }
        else:
            print(f"✅ System healthy, no alerts sent")
            health_status['notifications'] = {"none": True}
        
        return health_status

def main():
    """Main alerting interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Trendr Pipeline Alerting')
    parser.add_argument('--check', action='store_true', help='Run health check')
    parser.add_argument('--monitor', action='store_true', help='Continuous monitoring mode')
    parser.add_argument('--test-alerts', action='store_true', help='Test alert notifications')
    parser.add_argument('--interval', type=int, default=300, help='Monitoring interval in seconds')
    
    args = parser.parse_args()
    
    alerts = AlertSystem()
    
    if args.test_alerts:
        # Send test alert
        test_status = {
            "status": "warning",
            "issues": [{"type": "test", "severity": "warning", "message": "This is a test alert"}],
            "timestamp": datetime.now().isoformat(),
            "total_issues": 1
        }
        
        print("📧 Testing email alerts...")
        email_result = alerts.send_email_alert(test_status)
        print(f"Email: {'✅' if email_result else '❌'}")
        
        print("📱 Testing Discord alerts...")
        discord_result = alerts.send_discord_alert(test_status)
        print(f"Discord: {'✅' if discord_result else '❌'}")
        
        print("💬 Testing Slack alerts...")
        slack_result = alerts.send_slack_alert(test_status)
        print(f"Slack: {'✅' if slack_result else '❌'}")
    
    elif args.monitor:
        print(f"🔍 Starting continuous monitoring (interval: {args.interval}s)")
        while True:
            try:
                result = alerts.run_health_check_and_alert()
                print(f"Health check: {result['status']} ({result['total_issues']} issues)")
                time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\n👋 Monitoring stopped")
                break
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                time.sleep(60)
    
    elif args.check:
        result = alerts.run_health_check_and_alert()
        print("📊 Health Check Results:")
        print(json.dumps(result, indent=2))
    
    else:
        print("Trendr Pipeline Alerting System")
        print("Usage: python monitoring/alerts.py --check|--monitor|--test-alerts")

if __name__ == "__main__":
    main()