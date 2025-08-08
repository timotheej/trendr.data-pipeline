#!/usr/bin/env python3
"""
Trendr Pipeline Advanced Metrics
Detailed performance and business metrics tracking
"""
import sys
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
from collections import defaultdict
import re

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database import SupabaseManager
from utils.api_cache import APICache

class MetricsCollector:
    """Advanced metrics collection and analysis"""
    
    def __init__(self):
        self.db = SupabaseManager()
        self.cache = APICache()
        self.log_path = "/var/log/trendr/pipeline.log"
    
    def get_poi_growth_metrics(self, days: int = 30) -> Dict[str, Any]:
        """Get POI growth metrics over time"""
        try:
            # Get POIs created in the last N days
            start_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            pois = self.db.client.table('poi')\
                .select('id,name,category,created_at,city')\
                .gte('created_at', start_date)\
                .order('created_at')\
                .execute()
            
            # Group by date and category
            daily_counts = defaultdict(lambda: defaultdict(int))
            category_totals = defaultdict(int)
            
            for poi in pois.data:
                created_date = poi['created_at'][:10]  # YYYY-MM-DD
                category = poi['category']
                
                daily_counts[created_date][category] += 1
                category_totals[category] += 1
            
            # Calculate growth rate
            first_week = sum(sum(daily_counts[date].values()) for date in list(daily_counts.keys())[:7])
            last_week = sum(sum(daily_counts[date].values()) for date in list(daily_counts.keys())[-7:])
            
            growth_rate = ((last_week - first_week) / max(first_week, 1)) * 100 if first_week > 0 else 0
            
            return {
                "total_new_pois": len(pois.data),
                "daily_breakdown": dict(daily_counts),
                "category_breakdown": dict(category_totals),
                "growth_rate_percent": round(growth_rate, 2),
                "average_daily": round(len(pois.data) / days, 2),
                "period_days": days
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_collection_performance_metrics(self) -> Dict[str, Any]:
        """Get collection generation and performance metrics"""
        try:
            # Get all collections with POI counts
            collections = self.db.client.table('collections')\
                .select('*')\
                .execute()
            
            if not collections.data:
                return {"total_collections": 0}
            
            # Analyze collection types and sizes
            type_breakdown = defaultdict(int)
            size_breakdown = {"small": 0, "medium": 0, "large": 0}
            total_pois_in_collections = 0
            
            for collection in collections.data:
                collection_type = collection.get('type', 'unknown')
                type_breakdown[collection_type] += 1
                
                poi_count = len(collection.get('poi_ids', []))
                total_pois_in_collections += poi_count
                
                if poi_count <= 3:
                    size_breakdown["small"] += 1
                elif poi_count <= 8:
                    size_breakdown["medium"] += 1
                else:
                    size_breakdown["large"] += 1
            
            avg_pois_per_collection = total_pois_in_collections / len(collections.data)
            
            return {
                "total_collections": len(collections.data),
                "type_breakdown": dict(type_breakdown),
                "size_breakdown": size_breakdown,
                "avg_pois_per_collection": round(avg_pois_per_collection, 2),
                "total_pois_in_collections": total_pois_in_collections
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_api_efficiency_metrics(self) -> Dict[str, Any]:
        """Get API usage efficiency and cost metrics"""
        try:
            cache_stats = self.cache.get_stats()
            
            # Parse logs for API usage patterns
            api_metrics = {
                "cache_hit_rate": cache_stats.get('cache_hit_rate', 0),
                "cache_entries": cache_stats.get('entries_count', 0),
                "estimated_savings": 0,
                "daily_usage_pattern": defaultdict(int),
                "cost_efficiency": {}
            }
            
            if os.path.exists(self.log_path):
                with open(self.log_path, 'r') as f:
                    recent_lines = f.readlines()[-2000:]  # Last 2000 lines
                    
                    cache_hits = 0
                    cache_misses = 0
                    api_calls = 0
                    
                    for line in recent_lines:
                        if "Cache HIT" in line:
                            cache_hits += 1
                        elif "Cache MISS" in line:
                            cache_misses += 1
                        elif "Google Custom Search API" in line:
                            api_calls += 1
                            # Extract date for pattern analysis
                            try:
                                date = line.split(' - ')[0][:10]  # YYYY-MM-DD
                                api_metrics["daily_usage_pattern"][date] += 1
                            except:
                                pass
                    
                    if cache_hits + cache_misses > 0:
                        actual_hit_rate = (cache_hits / (cache_hits + cache_misses)) * 100
                        api_metrics["actual_cache_hit_rate"] = round(actual_hit_rate, 2)
                        
                        # Estimate cost savings (assuming $5 per 1000 queries)
                        queries_saved = cache_hits
                        cost_saved = (queries_saved * 0.005)  # $0.005 per query
                        api_metrics["estimated_cost_savings_usd"] = round(cost_saved, 2)
                        
                        # Efficiency rating
                        if actual_hit_rate > 70:
                            efficiency = "excellent"
                        elif actual_hit_rate > 50:
                            efficiency = "good"
                        elif actual_hit_rate > 30:
                            efficiency = "fair"
                        else:
                            efficiency = "poor"
                        
                        api_metrics["efficiency_rating"] = efficiency
            
            return api_metrics
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_neighborhood_coverage_metrics(self) -> Dict[str, Any]:
        """Get neighborhood distribution and coverage metrics"""
        try:
            # Get POIs with neighborhood info
            pois_with_neighborhoods = self.db.client.table('poi')\
                .select('id,neighborhood,city')\
                .eq('city', 'Montreal')\
                .execute()
            
            # Get all neighborhoods
            neighborhoods = self.db.client.table('neighborhoods')\
                .select('*')\
                .eq('city', 'Montreal')\
                .execute()
            
            neighborhood_poi_counts = defaultdict(int)
            total_pois = len(pois_with_neighborhoods.data)
            pois_with_neighborhood = 0
            
            for poi in pois_with_neighborhoods.data:
                neighborhood = poi.get('neighborhood')
                if neighborhood:
                    neighborhood_poi_counts[neighborhood] += 1
                    pois_with_neighborhood += 1
            
            coverage_percentage = (pois_with_neighborhood / max(total_pois, 1)) * 100
            
            # Find neighborhoods with no POIs
            all_neighborhood_names = {n['name'] for n in neighborhoods.data}
            covered_neighborhoods = set(neighborhood_poi_counts.keys())
            uncovered_neighborhoods = all_neighborhood_names - covered_neighborhoods
            
            return {
                "total_pois": total_pois,
                "pois_with_neighborhood": pois_with_neighborhood,
                "coverage_percentage": round(coverage_percentage, 2),
                "neighborhood_distribution": dict(neighborhood_poi_counts),
                "total_neighborhoods": len(neighborhoods.data),
                "covered_neighborhoods": len(covered_neighborhoods),
                "uncovered_neighborhoods": list(uncovered_neighborhoods),
                "avg_pois_per_neighborhood": round(pois_with_neighborhood / max(len(covered_neighborhoods), 1), 2)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_execution_performance_metrics(self) -> Dict[str, Any]:
        """Get pipeline execution performance metrics"""
        try:
            performance_data = {
                "recent_executions": [],
                "average_execution_time": 0,
                "success_rate": 0,
                "error_patterns": defaultdict(int)
            }
            
            if not os.path.exists(self.log_path):
                return performance_data
            
            with open(self.log_path, 'r') as f:
                lines = f.readlines()
                
                # Find execution summaries and times
                executions = []
                current_execution = None
                
                for line in lines[-5000:]:  # Last 5000 lines
                    if "BEGINNING OF EXECUTION" in line or "PIPELINE EXECUTION SUMMARY" in line:
                        try:
                            timestamp = line.split(' - ')[0]
                            if current_execution:
                                # Calculate execution time
                                start_time = datetime.fromisoformat(current_execution['start'])
                                end_time = datetime.fromisoformat(timestamp)
                                duration = (end_time - start_time).total_seconds()
                                current_execution['duration_seconds'] = duration
                                executions.append(current_execution)
                            
                            current_execution = {
                                'start': timestamp,
                                'success': "SUCCESS" in line,
                                'errors': []
                            }
                        except:
                            pass
                    
                    elif "ERROR" in line and current_execution:
                        # Extract error type
                        error_match = re.search(r'ERROR.*?:(.+)', line)
                        if error_match:
                            error_type = error_match.group(1).strip().split()[0]
                            current_execution['errors'].append(error_type)
                            performance_data["error_patterns"][error_type] += 1
                
                # Calculate metrics from recent executions
                recent_executions = executions[-10:]  # Last 10 executions
                if recent_executions:
                    avg_duration = sum(e.get('duration_seconds', 0) for e in recent_executions) / len(recent_executions)
                    success_count = sum(1 for e in recent_executions if e.get('success', False))
                    success_rate = (success_count / len(recent_executions)) * 100
                    
                    performance_data.update({
                        "recent_executions": recent_executions,
                        "average_execution_time_seconds": round(avg_duration, 2),
                        "average_execution_time_minutes": round(avg_duration / 60, 2),
                        "success_rate_percent": round(success_rate, 2),
                        "total_recent_executions": len(recent_executions)
                    })
            
            performance_data["error_patterns"] = dict(performance_data["error_patterns"])
            return performance_data
            
        except Exception as e:
            return {"error": str(e)}
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive metrics report"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "poi_growth": self.get_poi_growth_metrics(30),
            "collection_performance": self.get_collection_performance_metrics(),
            "api_efficiency": self.get_api_efficiency_metrics(),
            "neighborhood_coverage": self.get_neighborhood_coverage_metrics(),
            "execution_performance": self.get_execution_performance_metrics()
        }
        
        # Calculate overall health score (0-100)
        health_score = 0
        factors = 0
        
        # API efficiency factor (30 points)
        if 'actual_cache_hit_rate' in report['api_efficiency']:
            hit_rate = report['api_efficiency']['actual_cache_hit_rate']
            health_score += min(hit_rate * 0.3, 30)  # Max 30 points
            factors += 1
        
        # Success rate factor (25 points)
        if 'success_rate_percent' in report['execution_performance']:
            success_rate = report['execution_performance']['success_rate_percent']
            health_score += (success_rate / 100) * 25  # Max 25 points
            factors += 1
        
        # Coverage factor (25 points)
        if 'coverage_percentage' in report['neighborhood_coverage']:
            coverage = report['neighborhood_coverage']['coverage_percentage']
            health_score += (coverage / 100) * 25  # Max 25 points
            factors += 1
        
        # Growth factor (20 points)
        if 'total_new_pois' in report['poi_growth']:
            new_pois = report['poi_growth']['total_new_pois']
            growth_points = min(new_pois / 10, 20)  # Max 20 points, 1 point per 0.5 POIs
            health_score += growth_points
            factors += 1
        
        if factors > 0:
            report['overall_health_score'] = round(health_score, 1)
        else:
            report['overall_health_score'] = 0
        
        return report

def main():
    """Main metrics interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Trendr Pipeline Metrics')
    parser.add_argument('--report', action='store_true', help='Generate comprehensive report')
    parser.add_argument('--poi-growth', action='store_true', help='Show POI growth metrics')
    parser.add_argument('--api-efficiency', action='store_true', help='Show API efficiency metrics')
    parser.add_argument('--collections', action='store_true', help='Show collection metrics')
    parser.add_argument('--neighborhoods', action='store_true', help='Show neighborhood coverage')
    parser.add_argument('--performance', action='store_true', help='Show execution performance')
    parser.add_argument('--days', type=int, default=30, help='Number of days for growth metrics')
    
    args = parser.parse_args()
    
    metrics = MetricsCollector()
    
    if args.report:
        report = metrics.generate_comprehensive_report()
        print("📊 Comprehensive Pipeline Metrics Report")
        print("=" * 50)
        print(json.dumps(report, indent=2))
    
    elif args.poi_growth:
        growth = metrics.get_poi_growth_metrics(args.days)
        print(f"📈 POI Growth Metrics ({args.days} days)")
        print(json.dumps(growth, indent=2))
    
    elif args.api_efficiency:
        efficiency = metrics.get_api_efficiency_metrics()
        print("⚡ API Efficiency Metrics")
        print(json.dumps(efficiency, indent=2))
    
    elif args.collections:
        collections = metrics.get_collection_performance_metrics()
        print("🗂️ Collection Performance Metrics")
        print(json.dumps(collections, indent=2))
    
    elif args.neighborhoods:
        neighborhoods = metrics.get_neighborhood_coverage_metrics()
        print("🏘️ Neighborhood Coverage Metrics")
        print(json.dumps(neighborhoods, indent=2))
    
    elif args.performance:
        performance = metrics.get_execution_performance_metrics()
        print("⏱️ Execution Performance Metrics")
        print(json.dumps(performance, indent=2))
    
    else:
        print("Trendr Pipeline Advanced Metrics")
        print("Usage: python monitoring/metrics.py --report|--poi-growth|--api-efficiency|etc.")

if __name__ == "__main__":
    main()