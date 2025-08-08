#!/usr/bin/env python3
"""
POI Photo Processing Script
Batch processes photos for POIs that don't have photos yet
Can be run separately from main pipeline for photo backfill
"""
import sys
import os
import logging
import time
from typing import Dict, List, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database import SupabaseManager
from utils.photo_manager import POIPhotoManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhotoProcessor:
    """Batch photo processing for POIs"""
    
    def __init__(self):
        self.db = SupabaseManager()
        self.photo_manager = POIPhotoManager()
        
        # Processing statistics
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'photos_downloaded': 0
        }
    
    def get_pois_without_photos(self, city: str = "Montreal", limit: int = 100) -> List[Dict[str, Any]]:
        """Get POIs that don't have photos processed yet"""
        try:
            # Get POIs without primary_photo
            result = self.db.client.table('poi')\
                .select('id,google_place_id,name,category')\
                .eq('city', city)\
                .is_('primary_photo', 'null')\
                .limit(limit)\
                .execute()
            
            if result.data:
                logger.info(f"Found {len(result.data)} POIs without photos in {city}")
                return result.data
            else:
                logger.info(f"No POIs found without photos in {city}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching POIs without photos: {e}")
            return []
    
    def get_pois_with_old_photos(self, city: str = "Montreal", days_old: int = 30, limit: int = 50) -> List[Dict[str, Any]]:
        """Get POIs with photos older than X days for refresh"""
        try:
            from datetime import datetime, timedelta
            
            cutoff_date = (datetime.now() - timedelta(days=days_old)).isoformat()
            
            result = self.db.client.table('poi')\
                .select('id,google_place_id,name,category,photos_updated_at')\
                .eq('city', city)\
                .not_.is_('primary_photo', 'null')\
                .lt('photos_updated_at', cutoff_date)\
                .limit(limit)\
                .execute()
            
            if result.data:
                logger.info(f"Found {len(result.data)} POIs with old photos (>{days_old} days) in {city}")
                return result.data
            else:
                logger.info(f"No POIs found with old photos in {city}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching POIs with old photos: {e}")
            return []
    
    def process_poi_batch(self, pois: List[Dict[str, Any]], rate_limit_delay: float = 1.0) -> Dict[str, Any]:
        """Process photos for a batch of POIs with rate limiting"""
        logger.info(f"🖼️ Processing photos for {len(pois)} POIs...")
        
        batch_results = []
        
        for i, poi in enumerate(pois, 1):
            try:
                poi_id = poi['id']
                place_id = poi['google_place_id']
                poi_name = poi['name']
                
                logger.info(f"[{i}/{len(pois)}] Processing {poi_name}...")
                
                # Process photos for this POI
                result = self.photo_manager.process_poi_photos(
                    poi_id, 
                    place_id,
                    max_photos=3  # Process up to 3 photos per POI
                )
                
                # Update statistics
                self.stats['total_processed'] += 1
                
                if result['success']:
                    self.stats['successful'] += 1
                    self.stats['photos_downloaded'] += result['photos_processed']
                    
                    logger.info(f"✅ Success: {poi_name} - {result['photos_processed']} photos (best quality: {result['best_quality']:.3f})")
                else:
                    self.stats['failed'] += 1
                    error_msg = result.get('error', 'Unknown error')
                    logger.warning(f"❌ Failed: {poi_name} - {error_msg}")
                
                batch_results.append({
                    'poi_id': poi_id,
                    'poi_name': poi_name,
                    'success': result['success'],
                    'photos_processed': result['photos_processed'],
                    'best_quality': result['best_quality']
                })
                
                # Rate limiting to avoid hitting API limits
                if i < len(pois):  # Don't delay after the last item
                    time.sleep(rate_limit_delay)
                    
            except Exception as e:
                logger.error(f"Error processing POI {poi.get('name', 'Unknown')}: {e}")
                self.stats['failed'] += 1
                continue
        
        return {
            'processed_count': len(batch_results),
            'results': batch_results,
            'statistics': self.stats
        }
    
    def run_photo_backfill(self, city: str = "Montreal", limit: int = 100, rate_limit: float = 1.0) -> Dict[str, Any]:
        """Run complete photo backfill process for POIs without photos"""
        logger.info(f"🚀 Starting photo backfill for {city} (limit: {limit})")
        
        # Reset statistics
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'photos_downloaded': 0
        }
        
        try:
            # Get POIs without photos
            pois_without_photos = self.get_pois_without_photos(city, limit)
            
            if not pois_without_photos:
                logger.info("✅ All POIs already have photos processed")
                return {
                    'status': 'complete',
                    'message': 'All POIs already have photos',
                    'statistics': self.stats
                }
            
            # Process the batch
            batch_result = self.process_poi_batch(pois_without_photos, rate_limit)
            
            # Calculate success rate
            success_rate = (self.stats['successful'] / max(self.stats['total_processed'], 1)) * 100
            
            logger.info(f"🎉 Photo backfill complete!")
            logger.info(f"📊 Statistics:")
            logger.info(f"   - Total processed: {self.stats['total_processed']}")
            logger.info(f"   - Successful: {self.stats['successful']}")
            logger.info(f"   - Failed: {self.stats['failed']}")
            logger.info(f"   - Photos downloaded: {self.stats['photos_downloaded']}")
            logger.info(f"   - Success rate: {success_rate:.1f}%")
            
            return {
                'status': 'complete',
                'statistics': self.stats,
                'success_rate': success_rate,
                'batch_results': batch_result
            }
            
        except Exception as e:
            logger.error(f"Error in photo backfill process: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'statistics': self.stats
            }
    
    def run_photo_refresh(self, city: str = "Montreal", days_old: int = 30, limit: int = 50, rate_limit: float = 1.0) -> Dict[str, Any]:
        """Refresh photos for POIs with old photos"""
        logger.info(f"🔄 Starting photo refresh for {city} (photos older than {days_old} days)")
        
        try:
            # Get POIs with old photos
            pois_with_old_photos = self.get_pois_with_old_photos(city, days_old, limit)
            
            if not pois_with_old_photos:
                logger.info("✅ No POIs found with old photos")
                return {
                    'status': 'complete',
                    'message': 'No POIs need photo refresh',
                    'statistics': self.stats
                }
            
            # Process the batch
            batch_result = self.process_poi_batch(pois_with_old_photos, rate_limit)
            
            logger.info(f"🔄 Photo refresh complete: {self.stats['successful']}/{self.stats['total_processed']} successful")
            
            return {
                'status': 'complete',
                'statistics': self.stats,
                'batch_results': batch_result
            }
            
        except Exception as e:
            logger.error(f"Error in photo refresh process: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'statistics': self.stats
            }
    
    def analyze_photo_coverage(self, city: str = "Montreal") -> Dict[str, Any]:
        """Analyze current photo coverage for POIs"""
        try:
            # Get total POIs
            total_pois = self.db.client.table('poi')\
                .select('id')\
                .eq('city', city)\
                .execute()
            
            # Get POIs with photos
            pois_with_photos = self.db.client.table('poi')\
                .select('id,primary_photo_quality')\
                .eq('city', city)\
                .not_.is_('primary_photo', 'null')\
                .execute()
            
            # Analyze quality distribution
            quality_distribution = {
                'excellent': 0,  # > 0.8
                'good': 0,       # 0.6 - 0.8
                'fair': 0,       # 0.4 - 0.6
                'poor': 0        # < 0.4
            }
            
            for poi in pois_with_photos.data:
                quality = poi.get('primary_photo_quality', 0)
                if quality > 0.8:
                    quality_distribution['excellent'] += 1
                elif quality > 0.6:
                    quality_distribution['good'] += 1
                elif quality > 0.4:
                    quality_distribution['fair'] += 1
                else:
                    quality_distribution['poor'] += 1
            
            total_count = len(total_pois.data) if total_pois.data else 0
            photos_count = len(pois_with_photos.data) if pois_with_photos.data else 0
            coverage_percentage = (photos_count / max(total_count, 1)) * 100
            
            analysis = {
                'total_pois': total_count,
                'pois_with_photos': photos_count,
                'coverage_percentage': round(coverage_percentage, 2),
                'pois_without_photos': total_count - photos_count,
                'quality_distribution': quality_distribution,
                'average_quality': 0
            }
            
            # Calculate average quality
            if pois_with_photos.data:
                total_quality = sum(poi.get('primary_photo_quality', 0) for poi in pois_with_photos.data)
                analysis['average_quality'] = round(total_quality / len(pois_with_photos.data), 3)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing photo coverage: {e}")
            return {'error': str(e)}

def main():
    """Main CLI interface for photo processing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='POI Photo Processing Script')
    parser.add_argument('--backfill', action='store_true', help='Run photo backfill for POIs without photos')
    parser.add_argument('--refresh', action='store_true', help='Refresh photos for POIs with old photos')
    parser.add_argument('--analyze', action='store_true', help='Analyze current photo coverage')
    parser.add_argument('--city', default='Montreal', help='City to process')
    parser.add_argument('--limit', type=int, default=100, help='Maximum number of POIs to process')
    parser.add_argument('--days-old', type=int, default=30, help='Consider photos older than N days for refresh')
    parser.add_argument('--rate-limit', type=float, default=1.0, help='Delay between API calls in seconds')
    
    args = parser.parse_args()
    
    processor = PhotoProcessor()
    
    if args.backfill:
        result = processor.run_photo_backfill(
            city=args.city,
            limit=args.limit,
            rate_limit=args.rate_limit
        )
        print("📊 Photo Backfill Results:")
        print(f"Status: {result['status']}")
        if 'statistics' in result:
            stats = result['statistics']
            print(f"Processed: {stats['total_processed']}")
            print(f"Successful: {stats['successful']}")
            print(f"Photos downloaded: {stats['photos_downloaded']}")
    
    elif args.refresh:
        result = processor.run_photo_refresh(
            city=args.city,
            days_old=args.days_old,
            limit=args.limit,
            rate_limit=args.rate_limit
        )
        print("🔄 Photo Refresh Results:")
        print(f"Status: {result['status']}")
        if 'statistics' in result:
            stats = result['statistics']
            print(f"Refreshed: {stats['successful']} POIs")
    
    elif args.analyze:
        analysis = processor.analyze_photo_coverage(args.city)
        print(f"📊 Photo Coverage Analysis for {args.city}:")
        print(f"Total POIs: {analysis['total_pois']}")
        print(f"POIs with photos: {analysis['pois_with_photos']}")
        print(f"Coverage: {analysis['coverage_percentage']:.1f}%")
        print(f"Average quality: {analysis['average_quality']:.3f}")
        print("Quality distribution:")
        for quality, count in analysis['quality_distribution'].items():
            print(f"  {quality}: {count}")
    
    else:
        print("POI Photo Processing Script")
        print("Usage:")
        print("  --backfill              # Process photos for POIs without photos")
        print("  --refresh --days-old 30 # Refresh photos older than 30 days")
        print("  --analyze               # Analyze current photo coverage")
        print("  --city Montreal --limit 100  # Process up to 100 POIs in Montreal")

if __name__ == "__main__":
    main()