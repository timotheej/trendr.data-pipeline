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
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.database import SupabaseManager
from utils.api_cache import google_search_cache
from scripts.google_places_ingester import GooglePlacesIngester
from scripts.enhanced_proof_scanner import EnhancedProofScanner
from scripts.intelligent_classifier import IntelligentClassifier
from scripts.dynamic_neighborhoods import DynamicNeighborhoodCalculator
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
        self.neighborhood_calc = DynamicNeighborhoodCalculator()
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
            'cities': ['Montreal'],
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
        
        # Check database connection
        try:
            neighborhoods = self.db.get_neighborhoods_for_city('Montreal')
            logger.info(f"✅ DB Connection: {len(neighborhoods)} neighborhoods found")
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
    
    def execute_ingestion_plan(self, plan: List[Dict[str, Any]]) -> bool:
        """Execute ingestion plan while respecting quotas"""
        logger.info("📍 Starting POI ingestion...")
        
        total_api_calls = 0
        
        for task in plan:
            # Check API limit
            if total_api_calls >= self.config['daily_api_limit']:
                logger.warning(f"🚫 API limit reached ({total_api_calls}), stopping ingestion")
                break
            
            try:
                city = task['city']
                category = task['category']
                to_ingest = task['to_ingest']
                
                logger.info(f"🔄 Ingestion: {category} in {city} ({to_ingest} POIs)")
                
                # Build search query
                search_query = f"{category}"
                location = f"{city}, Canada"
                
                # Ingest POIs
                ingested_ids = self.ingester.search_and_ingest(search_query, location)
                
                self.stats['pois_ingested'] += len(ingested_ids)
                logger.info(f"✅ {len(ingested_ids)} POIs ingested for {category} in {city}")
                
                # Small pause to respect rate limits
                time.sleep(1)
                
            except Exception as e:
                error_msg = f"Ingestion error {category} in {city}: {e}"
                logger.error(f"❌ {error_msg}")
                self.stats['errors'].append(error_msg)
                continue
        
        logger.info(f"📍 Ingestion completed: {self.stats['pois_ingested']} POIs ingested")
        return True
    
    def process_social_proofs_and_classification(self) -> bool:
        """Process POIs for social proofs and intelligent classification"""
        if not self.config.get('social_proof_enabled', True):
            logger.info("⏭️  Classification disabled in config")
            return True
        
        logger.info("🤖 Starting social proofs and classification processing...")
        
        # Get unclassified or outdated POIs
        cutoff_date = datetime.now() - timedelta(days=self.config.get('update_interval_days', 7))
        
        # For simplicity, take recently ingested POIs
        recent_pois = self.db.get_pois_for_city('Montreal', limit=50)  # Adjust as needed
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
                    poi.get('city', 'Montreal'),
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
            for city in self.config['cities']:
                result = self.neighborhood_calc.update_city_neighborhoods(city)
                logger.info(f"✅ Neighborhoods updated for {city}: {result}")
        except Exception as e:
            error_msg = f"Neighborhood update error: {e}"
            logger.error(f"❌ {error_msg}")
            self.stats['errors'].append(error_msg)
            return False
        
        return True
    
    def detect_poi_updates(self) -> bool:
        """Detect POIs requiring updates"""
        logger.info("🔍 Detecting necessary POI updates...")
        
        try:
            update_interval_days = self.config.get('update_interval_days', 7)
            cutoff_date = datetime.now() - timedelta(days=update_interval_days)
            
            # Get outdated POIs without recent updates
            outdated_pois = self.db.client.table('poi')\
                .select('id,name,google_place_id,updated_at,city,category')\
                .lt('updated_at', cutoff_date.isoformat())\
                .execute()
            
            if not outdated_pois.data:
                logger.info("✅ No POIs requiring updates")
                return True
            
            logger.info(f"📊 {len(outdated_pois.data)} POIs require updates")
            
            # Prioritize POIs to update (limit to respect quotas)
            high_priority_pois = []
            for poi in outdated_pois.data:
                # Check if POI has recent reviews or activity
                try:
                    detailed_info = self.ingester.get_place_details(poi['google_place_id'])
                    if detailed_info:
                        # Check significant changes
                        needs_update = self._poi_needs_update(poi, detailed_info)
                        if needs_update:
                            high_priority_pois.append({
                                'poi_id': poi['id'],
                                'poi_name': poi['name'],
                                'google_place_id': poi['google_place_id'],
                                'reason': needs_update,
                                'detailed_info': detailed_info
                            })
                    
                    # Limit to 20 updates per execution to manage quotas
                    if len(high_priority_pois) >= 20:
                        break
                        
                except Exception as e:
                    logger.warning(f"POI verification error {poi['name']}: {e}")
                    continue
            
            # Perform updates
            updates_made = 0
            for update_info in high_priority_pois:
                try:
                    success = self._update_poi_from_google(
                        update_info['poi_id'], 
                        update_info['detailed_info']
                    )
                    if success:
                        updates_made += 1
                        logger.info(f"✅ Updated: {update_info['poi_name']} - {update_info['reason']}")
                        self.stats['pois_updated'] += 1
                    
                except Exception as e:
                    logger.error(f"POI update error {update_info['poi_name']}: {e}")
                    continue
            
            logger.info(f"✅ {updates_made} POIs updated successfully")
            return True
            
        except Exception as e:
            error_msg = f"Update detection error: {e}"
            logger.error(f"❌ {error_msg}")
            self.stats['errors'].append(error_msg)
            return False
    
    def _poi_needs_update(self, poi: Dict[str, Any], google_data: Dict[str, Any]) -> Optional[str]:
        """Determine if a POI needs updating and why"""
        
        # Check rating changes
        old_rating = poi.get('rating')
        new_rating = google_data.get('rating')
        if old_rating and new_rating and abs(float(old_rating) - float(new_rating)) > 0.2:
            return f"Rating change: {old_rating} → {new_rating}"
        
        # Check review count changes
        old_reviews = poi.get('user_ratings_total', 0)
        new_reviews = google_data.get('user_ratings_total', 0)
        if new_reviews > old_reviews * 1.2:  # 20%+ increase in reviews
            return f"New reviews: {old_reviews} → {new_reviews}"
        
        # Check schedule changes
        if 'opening_hours' in google_data:
            return "Opening hours potentially modified"
        
        # Check status changes (temporarily closed, etc.)
        business_status = google_data.get('business_status')
        if business_status and business_status != 'OPERATIONAL':
            return f"Status changed: {business_status}"
        
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
            # Limit photo processing to avoid API overuse
            photo_batch_limit = self.config.get('photo_batch_limit', 20)
            
            # Run photo backfill for POIs without photos
            result = self.photo_processor.run_photo_backfill(
                city="Montreal",
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
        """Generate POI collections"""
        if not self.config.get('collections_enabled', True):
            logger.info("⏭️  Collection generation disabled")
            return True
        
        logger.info("🗂️  Generating collections...")
        
        try:
            for city in self.config['cities']:
                collections = self.collection_gen.generate_contextual_collections(city)
                
                # Save collections to DB
                for collection in collections:
                    try:
                        self.db.save_collection(collection)
                        self.stats['collections_generated'] += 1
                    except Exception as e:
                        logger.warning(f"Collection save error: {e}")
                
                logger.info(f"✅ {len(collections)} collections generated for {city}")
        
        except Exception as e:
            error_msg = f"Collection generation error: {e}"
            logger.error(f"❌ {error_msg}")
            self.stats['errors'].append(error_msg)
            return False
        
        return True
    
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
    parser.add_argument('--city', default='Montreal', help='City to process')
    parser.add_argument('--dry-run', action='store_true', help='Simulation without modifications')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = TrendrDataPipeline(config_file=args.config)
    
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