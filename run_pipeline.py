#!/usr/bin/env python3
"""
Trendr Data Pipeline - Main Orchestrator
Automated execution script for complete POI ingestion and updates.
Optimized to respect API quotas and run autonomously.
"""
import sys
import os
import logging
import time
import argparse
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional
import json

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.database import SupabaseManager
from utils.api_cache import google_search_cache
from scripts.google_places_ingester import GooglePlacesIngester
from scripts.enhanced_proof_scanner import EnhancedProofScanner
from scripts.intelligent_classifier import IntelligentClassifier
# from scripts.dynamic_neighborhoods import DynamicNeighborhoodCalculator  # Not essential
from scripts.photo_processor import PhotoProcessor
from ai.collection_generator import CollectionGenerator

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TrendrDataPipeline:
    """Main orchestrator for Trendr data pipeline"""
    
    def __init__(self, config_file: str = None):
        self.config = self.load_config(config_file)
        self.db = SupabaseManager()
        
        # Initialize components
        self.ingester = GooglePlacesIngester()
        self.proof_scanner = EnhancedProofScanner()
        self.classifier = IntelligentClassifier()
        # self.neighborhood_calc = DynamicNeighborhoodCalculator()  # Not essential
        self.photo_processor = PhotoProcessor()
        self.collection_gen = CollectionGenerator()
        
        # Execution statistics
        self.stats = {
            'start_time': datetime.now(),
            'pois_ingested': 0,
            'pois_classified': 0,
            'pois_updated': 0,
            'photos_processed': 0,
            'api_calls_used': 0,
            'collections_generated': 0,
            'errors': []
        }
        
        logger.info("🚀 Trendr Data Pipeline initialized")
    
    def load_config(self, config_file: str = None) -> Dict[str, Any]:
        """Load pipeline configuration"""
        default_config = {
            'cities': [],  # Will be populated from CLI argument
            'poi_categories': [
                'restaurant', 'cafe', 'bar', 'night_club', 
                'shopping_mall', 'store', 'tourist_attraction',
                'museum', 'park', 'gym', 'spa'
            ],
            'daily_api_limit': 95,
            'batch_size': 20,
            'max_pois_per_category': 100,
            'update_interval_days': 7,
            'social_proof_enabled': True,
            'collections_enabled': True,
            'incremental_mode': True
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                logger.info(f"Configuration loaded from {config_file}")
            except Exception as e:
                logger.warning(f"Error loading config: {e}, using default config")
        
        return default_config
    
    def check_prerequisites(self) -> bool:
        """Check that all prerequisites are met"""
        logger.info("🔍 Checking prerequisites...")
        
        # Check environment variables
        required_env_vars = [
            'SUPABASE_URL', 'SUPABASE_KEY', 
            'GOOGLE_PLACES_API_KEY', 'GOOGLE_CUSTOM_SEARCH_API_KEY'
        ]
        
        missing_vars = []
        for var in required_env_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            logger.error(f"❌ Missing environment variables: {missing_vars}")
            return False
        
        # Check database connection (test with any city that has data)
        try:
            # Test basic table access without hardcoding city
            result = self.db.client.table('poi').select('id').limit(1).execute()
            logger.info(f"✅ DB Connection: Database accessible")
        except Exception as e:
            logger.error(f"❌ DB connection error: {e}")
            return False
        
        # Check cache
        try:
            cache_stats = google_search_cache.get_stats()
            logger.info(f"✅ API Cache: {cache_stats['entries_count']} entries")
        except Exception as e:
            logger.warning(f"⚠️  API cache issue: {e}")
        
        logger.info("✅ All prerequisites are met")
        return True
    
    def get_ingestion_plan(self) -> List[Dict[str, Any]]:
        """Plan ingestion based on quotas and existing data"""
        plan = []
        
        for city in self.config['cities']:
            for category in self.config['poi_categories']:
                # Check existing POIs
                existing_pois = self.db.client.table('poi').select('id').eq('city', city).eq('category', category).execute()
                existing_count = len(existing_pois.data) if existing_pois.data else 0
                
                # Determine number to ingest
                max_pois = self.config['max_pois_per_category']
                to_ingest = max(0, max_pois - existing_count)
                
                if to_ingest > 0:
                    plan.append({
                        'city': city,
                        'category': category,
                        'existing_count': existing_count,
                        'to_ingest': min(to_ingest, 20),  # Limit per batch
                        'priority': self._get_category_priority(category)
                    })
        
        # Sort by priority
        plan.sort(key=lambda x: x['priority'])
        
        logger.info(f"📋 Ingestion plan: {len(plan)} tasks planned")
        return plan
    
    def _get_category_priority(self, category: str) -> int:
        """Define category priority (1 = high priority)"""
        priority_map = {
            'restaurant': 1, 'cafe': 1, 'bar': 2,
            'tourist_attraction': 2, 'shopping_mall': 3,
            'museum': 3, 'park': 3, 'store': 4,
            'night_club': 4, 'gym': 5, 'spa': 5
        }
        return priority_map.get(category, 6)
    
    def execute_neighborhood_rotation_ingestion(self, city: str) -> bool:
        """Execute cost-efficient neighborhood rotation ingestion for new POI detection"""
        logger.info(f"🎯 Starting neighborhood rotation ingestion for {city}")
        
        try:
            country = self.config.get('country', 'France')
            
            # Run neighborhood rotation ingestion
            result = self.ingester.run_neighborhood_rotation_ingestion(city, country)
            
            # Update statistics
            self.stats['pois_ingested'] += result['total_ingested']
            self.stats['new_pois_detected'] = result['new_pois_detected']
            self.stats['current_neighborhood'] = result['neighborhood']
            
            logger.info(f"✅ Neighborhood rotation completed: {result['total_ingested']} POIs ingested, {result['new_pois_detected']} new POIs detected")
            logger.info(f"📍 Today's neighborhood: {result['neighborhood']}")
            
            return True
            
        except Exception as e:
            error_msg = f"Neighborhood rotation ingestion error: {e}"
            logger.error(f"❌ {error_msg}")
            self.stats['errors'].append(error_msg)
            return False
    
    def execute_ingestion_plan(self, plan: List[Dict[str, Any]]) -> bool:
        """Legacy method - now redirects to neighborhood rotation for cost efficiency"""
        logger.info("🔄 Redirecting to cost-efficient neighborhood rotation ingestion")
        
        # Extract city from plan
        city = self.config.get('cities', ['Paris'])[0]
        return self.execute_neighborhood_rotation_ingestion(city)
    
    def process_social_proofs_and_classification(self) -> bool:
        """Process POIs for social proofs and intelligent classification"""
        if not self.config.get('social_proof_enabled', True):
            logger.info("⏭️  Classification disabled in config")
            return True
        
        logger.info("🤖 Starting social proofs and classification processing...")
        
        # Get unclassified or outdated POIs
        cutoff_date = datetime.now() - timedelta(days=self.config.get('update_interval_days', 7))
        
        # For simplicity, take recently ingested POIs
        city = self.config.get('cities', [''])[0] if self.config.get('cities') else 'unknown'
        recent_pois = self.db.get_pois_for_city(city, limit=50)
        unclassified_pois = [poi for poi in recent_pois if not poi.get('primary_mood')]
        
        logger.info(f"🎯 {len(unclassified_pois)} POIs to process for classification")
        
        # Calculate how many POIs can be processed with remaining quota
        api_calls_used = self.proof_scanner.queries_used_today
        remaining_quota = self.config['daily_api_limit'] - api_calls_used
        avg_calls_per_poi = 3  # Average with our optimization
        
        max_processable = remaining_quota // avg_calls_per_poi
        pois_to_process = unclassified_pois[:max_processable]
        
        logger.info(f"📊 Remaining API quota: {remaining_quota}, processable POIs: {len(pois_to_process)}")
        
        processed_count = 0
        for poi in pois_to_process:
            try:
                logger.info(f"🔄 Processing: {poi['name']} ({poi.get('category', 'unknown')})")
                
                # Collect social proofs
                proofs = self.proof_scanner.enhanced_search_for_poi(
                    poi['name'],
                    poi.get('city', city),
                    poi.get('category', '')
                )
                
                # Intelligent classification
                if proofs:
                    classification_result = self.classifier.classify_poi(poi, proofs)
                    
                    # Try to update in DB (may fail due to schema)
                    try:
                        updated = self.classifier.update_poi_in_database(poi['id'], classification_result)
                        if updated:
                            self.stats['pois_updated'] += 1
                    except Exception as db_error:
                        logger.warning(f"DB update failed for {poi['name']}: {db_error}")
                
                processed_count += 1
                
                # Check if approaching limit
                if self.proof_scanner.queries_used_today >= self.config['daily_api_limit']:
                    logger.warning("🚫 API limit reached during classification")
                    break
                
            except Exception as e:
                error_msg = f"Classification error {poi['name']}: {e}"
                logger.error(f"❌ {error_msg}")
                self.stats['errors'].append(error_msg)
                continue
        
        self.stats['pois_classified'] = processed_count
        self.stats['api_calls_used'] = self.proof_scanner.queries_used_today
        
        logger.info(f"🤖 Classification completed: {processed_count} POIs processed")
        return True
    
    def update_dynamic_neighborhoods(self) -> bool:
        """Update dynamic neighborhood calculations"""
        logger.info("🏘️  Updating dynamic neighborhoods...")
        
        try:
            # Skip neighborhood calculations since we simplified to string approach
            logger.info("⏭️ Skipping dynamic neighborhood calculations (simplified approach)")
            return True
        except Exception as e:
            error_msg = f"Neighborhood update error: {e}"
            logger.error(f"❌ {error_msg}")
            self.stats['errors'].append(error_msg)
            return False
        
        return True
    
    def detect_poi_updates(self) -> bool:
        """Detect POIs requiring updates using social proof trends instead of expensive API calls"""
        logger.info("🔍 Detecting necessary POI updates via social proof analysis...")
        
        try:
            update_interval_days = self.config.get('update_interval_days', 7)
            cutoff_date = datetime.now() - timedelta(days=update_interval_days)
            
            # Get POIs without recent social proof analysis (cheaper than Google API calls)
            stale_pois = self.db.client.table('poi')\
                .select('id,name,google_place_id,updated_at,city,category,rating,user_ratings_total')\
                .lt('updated_at', cutoff_date.isoformat())\
                .execute()
            
            if not stale_pois.data:
                logger.info("✅ No POIs requiring updates")
                return True
            
            logger.info(f"📊 {len(stale_pois.data)} POIs have stale data")
            
            # Smart filtering: Only update POIs showing social proof activity
            update_candidates = []
            
            for poi in stale_pois.data:
                try:
                    # Check if POI shows trending signals via social proof (FREE)
                    trending_signals = self._detect_trending_signals_free(poi)
                    
                    if trending_signals['should_update']:
                        update_candidates.append({
                            'poi_id': poi['id'],
                            'poi_name': poi['name'],
                            'google_place_id': poi['google_place_id'],
                            'trending_score': trending_signals['score'],
                            'reason': trending_signals['reason']
                        })
                    
                    # Limit candidates to respect daily quota (conservative)
                    if len(update_candidates) >= 20:
                        break
                        
                except Exception as e:
                    logger.warning(f"Trending detection error {poi['name']}: {e}")
                    continue
            
            logger.info(f"🎯 {len(update_candidates)} POIs show trending activity - will update via Google API")
            
            # Now make expensive Google API calls ONLY for trending POIs
            api_updates_made = 0
            remaining_quota = self.config['daily_api_limit'] - self.stats.get('api_calls_used', 0)
            
            for candidate in update_candidates[:remaining_quota]:
                try:
                    # Expensive Google API call - only for confirmed trending POIs
                    detailed_info = self.ingester.get_place_details(candidate['google_place_id'])
                    if detailed_info:
                        success = self._update_poi_from_google(
                            candidate['poi_id'], 
                            detailed_info
                        )
                        if success:
                            api_updates_made += 1
                            logger.info(f"✅ API Updated: {candidate['poi_name']} - {candidate['reason']}")
                            self.stats['pois_updated'] += 1
                            self.stats['api_calls_used'] = self.stats.get('api_calls_used', 0) + 1
                    
                except Exception as e:
                    logger.error(f"API update error {candidate['poi_name']}: {e}")
                    continue
            
            # Update social proof data for remaining POIs (cheaper)
            social_proof_updates = 0
            remaining_pois = stale_pois.data[len(update_candidates):]
            
            for poi in remaining_pois[:50]:  # Process more via social proof
                try:
                    # Use social proof scanner to refresh data (much cheaper)
                    proofs = self.proof_scanner.enhanced_search_for_poi(
                        poi['name'],
                        poi['city'],
                        poi.get('category', '')
                    )
                    
                    if proofs:
                        # Update social proof metadata without Google API
                        self._update_poi_social_metadata(poi['id'], proofs)
                        social_proof_updates += 1
                    
                except Exception as e:
                    logger.warning(f"Social proof update error {poi['name']}: {e}")
                    continue
            
            logger.info(f"✅ Updates completed:")
            logger.info(f"   • {api_updates_made} POIs updated via Google API")
            logger.info(f"   • {social_proof_updates} POIs updated via social proof")
            logger.info(f"   • API calls saved: {len(stale_pois.data) - api_updates_made}")
            
            return True
            
        except Exception as e:
            error_msg = f"Update detection error: {e}"
            logger.error(f"❌ {error_msg}")
            self.stats['errors'].append(error_msg)
            return False
    
    def _detect_trending_signals_free(self, poi: Dict[str, Any]) -> Dict[str, Any]:
        """Detect trending signals using social proof data WITHOUT Google API calls"""
        
        trending_score = 0
        reasons = []
        
        try:
            # Check social proof activity from our database (FREE)
            recent_proofs = self.db.client.table('proof_sources')\
                .select('created_at,authority_score,source_type')\
                .eq('poi_id', poi['id'])\
                .gte('created_at', (datetime.now() - timedelta(days=7)).isoformat())\
                .execute()
            
            if recent_proofs.data:
                trending_score += len(recent_proofs.data) * 10
                reasons.append(f"Recent social proof activity: {len(recent_proofs.data)} new mentions")
            
            # Check if POI has high authority mentions
            high_authority_proofs = [p for p in recent_proofs.data if p.get('authority_score', 0) > 0.7]
            if high_authority_proofs:
                trending_score += len(high_authority_proofs) * 20
                reasons.append(f"High authority mentions: {len(high_authority_proofs)}")
            
            # Check monitoring reports for trending indicators (FREE)
            recent_reports = self.db.client.table('monitoring_reports')\
                .select('trend_direction,confidence_score')\
                .eq('poi_id', poi['id'])\
                .gte('monitoring_date', (datetime.now() - timedelta(days=3)).isoformat())\
                .order('monitoring_date', desc=True)\
                .limit(1)\
                .execute()
            
            if recent_reports.data:
                report = recent_reports.data[0]
                if report.get('trend_direction') == 'upward' and report.get('confidence_score', 0) > 0.6:
                    trending_score += 30
                    reasons.append(f"Upward trend detected: {report['confidence_score']}")
            
            # Age-based scoring: older POIs need updates less frequently
            poi_age_days = (datetime.now() - datetime.fromisoformat(poi.get('updated_at', '2020-01-01'))).days
            if poi_age_days > 30:
                trending_score += 5  # Slight boost for very stale data
                reasons.append(f"Stale data: {poi_age_days} days old")
            
        except Exception as e:
            logger.warning(f"Error detecting trending signals for {poi.get('name')}: {e}")
        
        should_update = trending_score >= 20  # Threshold for API call
        
        return {
            'should_update': should_update,
            'score': trending_score,
            'reason': ' | '.join(reasons) if reasons else 'No significant activity'
        }
    
    def _update_poi_social_metadata(self, poi_id: str, proof_sources: List[Dict[str, Any]]) -> bool:
        """Update POI with social proof metadata without Google API call"""
        try:
            # Calculate social proof metrics
            total_authority = sum(p.get('authority_score', 0) for p in proof_sources)
            avg_authority = total_authority / len(proof_sources) if proof_sources else 0
            
            # Update POI with social proof data only
            update_data = {
                'social_proof_score': avg_authority,
                'proof_sources_count': len(proof_sources),
                'updated_at': datetime.now().isoformat(),
                'last_social_sync': datetime.now().isoformat()
            }
            
            result = self.db.client.table('poi')\
                .update(update_data)\
                .eq('id', poi_id)\
                .execute()
            
            return bool(result.data)
            
        except Exception as e:
            logger.error(f"Social metadata update error {poi_id}: {e}")
            return False
    
    def _poi_needs_update(self, poi: Dict[str, Any], google_data: Dict[str, Any]) -> Optional[str]:
        """Determine if a POI needs updating and why - SIMPLIFIED to reduce costs"""
        
        # Only check critical changes that affect trending detection
        old_rating = poi.get('rating')
        new_rating = google_data.get('rating')
        if old_rating and new_rating and abs(float(old_rating) - float(new_rating)) > 0.3:
            return f"Significant rating change: {old_rating} → {new_rating}"
        
        # Check review count changes (major indicator for trending)
        old_reviews = poi.get('user_ratings_total', 0)
        new_reviews = google_data.get('user_ratings_total', 0)
        if new_reviews > old_reviews * 1.5:  # 50%+ increase in reviews
            return f"Major review increase: {old_reviews} → {new_reviews}"
        
        # Skip expensive checks like opening_hours and business_status
        # Focus only on trending-relevant data
        
        return None
    
    def _update_poi_from_google(self, poi_id: str, google_data: Dict[str, Any]) -> bool:
        """Update a POI with Google data"""
        try:
            update_data = {
                'rating': google_data.get('rating'),
                'user_ratings_total': google_data.get('user_ratings_total'),
                'updated_at': datetime.now().isoformat(),
                'last_google_sync': datetime.now().isoformat()
            }
            
            # Add opening hours if available
            if 'opening_hours' in google_data:
                opening_hours = google_data['opening_hours']
                if opening_hours:
                    update_data['opening_hours'] = json.dumps(opening_hours.get('weekday_text', []))
                    update_data['is_open_now'] = opening_hours.get('open_now')
            
            # Add business status if available
            if 'business_status' in google_data:
                update_data['business_status'] = google_data['business_status']
            
            # Perform update
            result = self.db.client.table('poi')\
                .update(update_data)\
                .eq('id', poi_id)\
                .execute()
            
            return bool(result.data)
            
        except Exception as e:
            logger.error(f"POI update error {poi_id}: {e}")
            return False
    
    def process_missing_photos(self) -> bool:
        """Process photos for POIs that don't have photos yet"""
        if not self.config.get('photos_enabled', True):
            logger.info("⏭️ Photo processing disabled")
            return True
        
        logger.info("📸 Processing photos for POIs without photos...")
        
        try:
            # Get city from config
            city = self.config.get('cities', [''])[0] if self.config.get('cities') else 'unknown'
            
            # Limit photo processing to avoid API overuse
            photo_batch_limit = self.config.get('photo_batch_limit', 20)
            
            # Run photo backfill for POIs without photos
            result = self.photo_processor.run_photo_backfill(
                city=city,
                limit=photo_batch_limit,
                rate_limit=1.5  # Slower rate to be conservative
            )
            
            if result['status'] == 'complete':
                photos_processed = result['statistics']['photos_downloaded']
                successful_pois = result['statistics']['successful']
                
                self.stats['photos_processed'] += photos_processed
                
                logger.info(f"✅ Photo processing: {successful_pois} POIs, {photos_processed} photos downloaded")
                return True
            else:
                error_msg = f"Photo processing error: {result.get('error', 'Unknown')}"
                logger.error(f"❌ {error_msg}")
                self.stats['errors'].append(error_msg)
                return False
                
        except Exception as e:
            error_msg = f"Error in photo processing: {e}"
            logger.error(f"❌ {error_msg}")
            self.stats['errors'].append(error_msg)
            return False
    
    def generate_collections(self) -> bool:
        """Generate POI collections with smart frequency"""
        if not self.config.get('collections_enabled', True):
            logger.info("⏭️  Collection generation disabled")
            return True
        
        logger.info("🗂️  Generating collections...")
        
        try:
            for city in self.config['cities']:
                # Check if collections need updating (every 3 days or after significant POI changes)
                if self._should_regenerate_collections(city):
                    logger.info(f"🔄 Collections need updating for {city}")
                    
                    # Generate collections with AI agent
                    processed_count, collections = self.collection_gen.generate_collections_for_city(city, use_ai=True)
                    
                    self.stats['collections_generated'] += processed_count
                    logger.info(f"✅ {processed_count} collections processed for {city}")
                else:
                    logger.info(f"⏭️ Collections up-to-date for {city}")
        
        except Exception as e:
            error_msg = f"Collection generation error: {e}"
            logger.error(f"❌ {error_msg}")
            self.stats['errors'].append(error_msg)
            return False
        
        return True
    
    def _should_regenerate_collections(self, city: str) -> bool:
        """Determine if collections need regeneration"""
        try:
            from datetime import timedelta
            
            # Check last collection update
            existing_collections = self.db.client.table('collections')\
                .select('updated_at')\
                .eq('city', city)\
                .order('updated_at', desc=True)\
                .limit(1)\
                .execute()
            
            if not existing_collections.data:
                logger.info(f"No existing collections for {city} - will create")
                return True
            
            last_update = datetime.fromisoformat(existing_collections.data[0]['updated_at'].replace('Z', '+00:00'))
            days_since_update = (datetime.now(timezone.utc) - last_update).days
            
            # Regenerate if:
            # 1. More than 3 days since last update
            # 2. New POIs added today (indicates fresh data)
            if days_since_update >= 3:
                logger.info(f"Collections outdated: {days_since_update} days old")
                return True
            
            if self.stats.get('new_pois_detected', 0) > 0:
                logger.info(f"New POIs detected: {self.stats['new_pois_detected']} - refreshing collections")
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error checking collection freshness: {e}")
            return True  # Err on safe side, regenerate
    
    def cleanup_and_maintenance(self) -> bool:
        """System cleanup and maintenance"""
        logger.info("🧹 Cleanup and maintenance...")
        
        try:
            # Clean expired cache
            cleaned_entries = google_search_cache.clear_expired()
            logger.info(f"🗑️  {cleaned_entries} cache entries cleaned")
            
            # Cache statistics
            cache_stats = google_search_cache.get_stats()
            logger.info(f"💾 Cache: {cache_stats['entries_count']} entries, {cache_stats['cache_hit_rate']:.1f}% hit rate")
            
        except Exception as e:
            logger.warning(f"Maintenance error: {e}")
        
        return True
    
    def print_execution_summary(self):
        """Display execution summary"""
        duration = datetime.now() - self.stats['start_time']
        
        print("\n" + "="*60)
        print("📊 TRENDR DATA PIPELINE EXECUTION SUMMARY")
        print("="*60)
        print(f"⏱️  Duration: {duration}")
        print(f"📍 POIs ingested: {self.stats['pois_ingested']}")
        print(f"🆕 New POIs detected: {self.stats.get('new_pois_detected', 0)}")
        print(f"🏘️  Current neighborhood: {self.stats.get('current_neighborhood', 'N/A')}")
        print(f"🤖 POIs classified: {self.stats['pois_classified']}")
        print(f"🔄 POIs updated: {self.stats['pois_updated']}")
        print(f"📸 Photos processed: {self.stats['photos_processed']}")
        print(f"📞 API calls used: {self.stats['api_calls_used']}/{self.config['daily_api_limit']}")
        print(f"🗂️  Collections generated: {self.stats['collections_generated']}")
        print(f"❌ Errors: {len(self.stats['errors'])}")
        
        if self.stats['errors']:
            print("\n🚨 ERRORS ENCOUNTERED:")
            for error in self.stats['errors'][:5]:  # Montrer les 5 premières
                print(f"  • {error}")
            if len(self.stats['errors']) > 5:
                print(f"  ... and {len(self.stats['errors']) - 5} other errors")
        
        print("="*60)
    
    def run_full_pipeline(self) -> bool:
        """Execute the complete pipeline"""
        logger.info("🚀 STARTING COMPLETE PIPELINE EXECUTION")
        
        try:
            # 1. Prerequisite checks
            if not self.check_prerequisites():
                return False
            
            # 2. Ingestion planning
            ingestion_plan = self.get_ingestion_plan()
            
            # 3. Ingestion execution
            self.execute_ingestion_plan(ingestion_plan)
            
            # 4. Classification and social proofs
            self.process_social_proofs_and_classification()
            
            # 5. POI detection and updates
            self.detect_poi_updates()
            
            # 6. Photo processing for POIs without photos
            self.process_missing_photos()
            
            # 7. Dynamic neighborhood updates
            self.update_dynamic_neighborhoods()
            
            # 8. Collection generation
            self.generate_collections()
            
            # 9. Cleanup and maintenance
            self.cleanup_and_maintenance()
            
            # 10. Final summary
            self.print_execution_summary()
            
            logger.info("✅ PIPELINE EXECUTED SUCCESSFULLY")
            return True
            
        except Exception as e:
            logger.error(f"❌ CRITICAL PIPELINE ERROR: {e}")
            self.stats['errors'].append(f"Critical error: {e}")
            self.print_execution_summary()
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Trendr Data Pipeline - Main Orchestrator')
    parser.add_argument('--config', help='JSON configuration file')
    parser.add_argument('--mode', choices=['full', 'ingestion', 'classification', 'collections'], 
                       default='full', help='Execution mode')
    parser.add_argument('--city', required=True, help='City to process')
    parser.add_argument('--country', help='Country for the city (if not provided, will be auto-detected)')
    parser.add_argument('--dry-run', action='store_true', help='Simulation without modifications')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = TrendrDataPipeline(config_file=args.config)
    
    # Add city and country to config
    pipeline.config['cities'] = [args.city]
    if args.country:
        pipeline.config['country'] = args.country
    else:
        pipeline.config['country'] = 'Japan' if 'tokyo' in args.city.lower() else 'Unknown'
    
    # Execute according to mode
    if args.mode == 'full':
        success = pipeline.run_full_pipeline()
    elif args.mode == 'ingestion':
        plan = pipeline.get_ingestion_plan()
        success = pipeline.execute_ingestion_plan(plan)
    elif args.mode == 'classification':
        success = pipeline.process_social_proofs_and_classification()
    elif args.mode == 'collections':
        success = pipeline.generate_collections()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())