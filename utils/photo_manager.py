#!/usr/bin/env python3
"""
POI Photo Management System
Downloads, analyzes, and selects the best photos from Google Places API
Optimized for collection covers and visual appeal
"""
import sys
import os
import requests
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from urllib.parse import urljoin

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from utils.database import SupabaseManager

try:
    from PIL import Image, ImageStat, ImageFilter
    import numpy as np
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class POIPhotoManager:
    """Manages POI photos with intelligent selection and optimization"""
    
    def __init__(self, storage_path: str = None):
        self.db = SupabaseManager()
        self.api_key = config.GOOGLE_PLACES_API_KEY
        self.storage_path = Path(storage_path or config.PHOTOS_STORAGE_PATH)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Photo quality scoring weights
        self.quality_weights = {
            'resolution': 0.25,      # Higher resolution = better
            'brightness': 0.20,      # Optimal brightness range
            'contrast': 0.20,        # Good contrast = more appealing
            'saturation': 0.15,      # Vibrant colors
            'composition': 0.10,     # Aspect ratio preferences
            'file_size': 0.10        # Reasonable file size
        }
        
        # Preferred photo dimensions for collections
        self.preferred_dimensions = {
            'min_width': 400,
            'min_height': 300,
            'max_width': 2048,
            'max_height': 1536,
            'aspect_ratios': [
                (16, 9),   # Landscape - best for covers
                (4, 3),    # Standard
                (3, 2),    # Classic photo ratio
                (1, 1)     # Square - acceptable
            ]
        }
        
        if not self.api_key:
            logger.warning("Google Places API key not configured")
    
    def get_photo_references_from_poi(self, place_id: str) -> List[Dict[str, Any]]:
        """Get photo references from Google Places API"""
        if not self.api_key:
            return []
        
        try:
            # Get place details with photos
            url = "https://maps.googleapis.com/maps/api/place/details/json"
            params = {
                'place_id': place_id,
                'fields': 'photos',
                'key': self.api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') != 'OK':
                logger.warning(f"Places API error for {place_id}: {data.get('status')}")
                return []
            
            photos = data.get('result', {}).get('photos', [])
            
            # Sort by size (larger first) and limit to top 5
            photos_sorted = sorted(photos, 
                                 key=lambda x: x.get('width', 0) * x.get('height', 0), 
                                 reverse=True)[:5]
            
            logger.info(f"Found {len(photos_sorted)} photos for place {place_id}")
            return photos_sorted
            
        except Exception as e:
            logger.error(f"Error fetching photo references for {place_id}: {e}")
            return []
    
    def download_photo(self, photo_reference: str, max_width: int = 1600) -> Optional[bytes]:
        """Download photo from Google Places Photo API"""
        if not self.api_key:
            return None
        
        try:
            url = "https://maps.googleapis.com/maps/api/place/photo"
            params = {
                'photo_reference': photo_reference,
                'maxwidth': max_width,
                'key': self.api_key
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            # Check if we got an actual image
            content_type = response.headers.get('content-type', '').lower()
            if not content_type.startswith('image/'):
                logger.warning(f"Invalid content type: {content_type}")
                return None
            
            return response.content
            
        except Exception as e:
            logger.error(f"Error downloading photo {photo_reference}: {e}")
            return None
    
    def analyze_photo_quality(self, image_data: bytes, filename: str = "") -> Dict[str, float]:
        """Analyze photo quality using multiple metrics"""
        if not HAS_PILLOW:
            logger.warning("Pillow not available for photo analysis")
            return {'total_score': 0.5}  # Default score
        
        try:
            from PIL import Image
            import io
            
            # Load image
            image = Image.open(io.BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            width, height = image.size
            
            # Calculate quality metrics
            scores = {}
            
            # 1. Resolution Score (0-1)
            pixel_count = width * height
            optimal_pixels = 800 * 600  # 480k pixels as optimal
            if pixel_count >= optimal_pixels:
                scores['resolution'] = min(1.0, pixel_count / (optimal_pixels * 2))
            else:
                scores['resolution'] = pixel_count / optimal_pixels
            
            # 2. Brightness Score (0-1) - prefer well-lit photos
            stat = ImageStat.Stat(image)
            brightness = sum(stat.mean) / 3  # Average RGB
            # Optimal brightness range: 80-180 (out of 255)
            if 80 <= brightness <= 180:
                scores['brightness'] = 1.0
            elif brightness < 80:
                scores['brightness'] = brightness / 80
            else:
                scores['brightness'] = max(0.1, 1.0 - ((brightness - 180) / 75))
            
            # 3. Contrast Score (0-1)
            contrast = sum(stat.stddev) / 3  # Standard deviation as contrast measure
            # Good contrast: 20-60
            if 20 <= contrast <= 60:
                scores['contrast'] = 1.0
            elif contrast < 20:
                scores['contrast'] = contrast / 20
            else:
                scores['contrast'] = max(0.1, 1.0 - ((contrast - 60) / 40))
            
            # 4. Saturation Score (0-1) - convert to HSV for saturation
            try:
                hsv_image = image.convert('HSV')
                hsv_stat = ImageStat.Stat(hsv_image)
                saturation = hsv_stat.mean[1]  # S channel
                # Prefer moderate to high saturation: 60-200
                if 60 <= saturation <= 200:
                    scores['saturation'] = 1.0
                elif saturation < 60:
                    scores['saturation'] = saturation / 60
                else:
                    scores['saturation'] = max(0.3, 1.0 - ((saturation - 200) / 55))
            except:
                scores['saturation'] = 0.7  # Default if HSV conversion fails
            
            # 5. Composition Score (0-1) - aspect ratio preference
            aspect_ratio = width / height
            composition_score = 0.5  # Default
            
            for pref_w, pref_h in self.preferred_dimensions['aspect_ratios']:
                pref_ratio = pref_w / pref_h
                ratio_diff = abs(aspect_ratio - pref_ratio)
                if ratio_diff < 0.2:  # Close to preferred ratio
                    composition_score = 1.0
                    break
                elif ratio_diff < 0.5:
                    composition_score = max(composition_score, 0.8)
            
            scores['composition'] = composition_score
            
            # 6. File Size Score (0-1) - prefer reasonable sizes
            file_size = len(image_data)
            if 50000 <= file_size <= 500000:  # 50KB to 500KB is optimal
                scores['file_size'] = 1.0
            elif file_size < 50000:
                scores['file_size'] = file_size / 50000
            else:
                scores['file_size'] = max(0.3, 1.0 - ((file_size - 500000) / 1000000))
            
            # Calculate weighted total score
            total_score = sum(score * self.quality_weights[metric] 
                            for metric, score in scores.items())
            
            scores['total_score'] = total_score
            scores['width'] = width
            scores['height'] = height
            scores['file_size'] = file_size
            scores['aspect_ratio'] = aspect_ratio
            
            logger.debug(f"Photo analysis for {filename}: {total_score:.3f}")
            return scores
            
        except Exception as e:
            logger.error(f"Error analyzing photo quality: {e}")
            return {'total_score': 0.1}  # Very low score for failed analysis
    
    def save_photo_to_storage(self, image_data: bytes, poi_id: str, 
                             photo_reference: str, quality_score: float) -> Optional[str]:
        """Save photo to local storage with organized structure"""
        try:
            # Create directory structure: /storage/poi_id/
            poi_dir = self.storage_path / poi_id
            poi_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with quality score and hash
            image_hash = hashlib.md5(image_data).hexdigest()[:8]
            quality_prefix = f"{int(quality_score * 100):02d}"  # 00-99
            filename = f"{quality_prefix}_{image_hash}.jpg"
            
            file_path = poi_dir / filename
            
            # Save image
            with open(file_path, 'wb') as f:
                f.write(image_data)
            
            # Return relative path for database storage
            relative_path = f"{poi_id}/{filename}"
            logger.info(f"Saved photo: {relative_path} (quality: {quality_score:.3f})")
            
            return relative_path
            
        except Exception as e:
            logger.error(f"Error saving photo for POI {poi_id}: {e}")
            return None
    
    def update_poi_photos_in_db(self, poi_id: str, photos_data: List[Dict[str, Any]]) -> bool:
        """Update POI with photo information in database"""
        try:
            if not photos_data:
                return False
            
            # Sort by quality score (best first)
            photos_sorted = sorted(photos_data, key=lambda x: x['quality_score'], reverse=True)
            
            # Prepare photo data for database
            best_photo = photos_sorted[0]
            all_photos = [
                {
                    'file_path': photo['file_path'],
                    'quality_score': photo['quality_score'],
                    'width': photo.get('width', 0),
                    'height': photo.get('height', 0),
                    'file_size': photo.get('file_size', 0)
                }
                for photo in photos_sorted
            ]
            
            # Update POI record
            update_data = {
                'primary_photo': best_photo['file_path'],
                'primary_photo_quality': best_photo['quality_score'],
                'all_photos': json.dumps(all_photos),
                'photos_updated_at': datetime.now().isoformat()
            }
            
            result = self.db.client.table('poi')\
                .update(update_data)\
                .eq('id', poi_id)\
                .execute()
            
            if result.data:
                logger.info(f"Updated POI {poi_id} with {len(all_photos)} photos (best: {best_photo['quality_score']:.3f})")
                return True
            else:
                logger.warning(f"Failed to update POI {poi_id} photos in database")
                return False
                
        except Exception as e:
            logger.error(f"Error updating POI {poi_id} photos in DB: {e}")
            return False
    
    def process_poi_photos(self, poi_id: str, place_id: str, max_photos: int = 3) -> Dict[str, Any]:
        """Complete photo processing pipeline for a POI"""
        logger.info(f"🖼️ Processing photos for POI {poi_id} (place: {place_id})")
        
        result = {
            'poi_id': poi_id,
            'place_id': place_id,
            'photos_processed': 0,
            'best_quality': 0,
            'photos_data': [],
            'success': False
        }
        
        try:
            # 1. Get photo references from Google Places
            photo_refs = self.get_photo_references_from_poi(place_id)
            if not photo_refs:
                logger.warning(f"No photos found for POI {poi_id}")
                return result
            
            # 2. Process each photo (limited by max_photos)
            processed_photos = []
            
            for i, photo_ref in enumerate(photo_refs[:max_photos]):
                try:
                    photo_reference = photo_ref['photo_reference']
                    max_width = min(photo_ref.get('width', 1600), 1600)  # Limit to 1600px
                    
                    # Download photo
                    image_data = self.download_photo(photo_reference, max_width)
                    if not image_data:
                        logger.warning(f"Failed to download photo {i+1} for POI {poi_id}")
                        continue
                    
                    # Analyze quality
                    quality_analysis = self.analyze_photo_quality(
                        image_data, 
                        f"{poi_id}_photo_{i+1}"
                    )
                    
                    # Save to storage
                    file_path = self.save_photo_to_storage(
                        image_data, poi_id, photo_reference, quality_analysis['total_score']
                    )
                    
                    if file_path:
                        photo_data = {
                            'file_path': file_path,
                            'photo_reference': photo_reference,
                            'quality_score': quality_analysis['total_score'],
                            'width': quality_analysis.get('width', 0),
                            'height': quality_analysis.get('height', 0),
                            'file_size': quality_analysis.get('file_size', 0),
                            'analysis': quality_analysis
                        }
                        processed_photos.append(photo_data)
                        
                        logger.info(f"✅ Processed photo {i+1} for POI {poi_id} (quality: {quality_analysis['total_score']:.3f})")
                    
                except Exception as e:
                    logger.error(f"Error processing photo {i+1} for POI {poi_id}: {e}")
                    continue
            
            # 3. Update database if we have photos
            if processed_photos:
                db_success = self.update_poi_photos_in_db(poi_id, processed_photos)
                
                result.update({
                    'photos_processed': len(processed_photos),
                    'best_quality': max(p['quality_score'] for p in processed_photos),
                    'photos_data': processed_photos,
                    'success': db_success
                })
                
                if db_success:
                    logger.info(f"🎉 Successfully processed {len(processed_photos)} photos for POI {poi_id}")
                else:
                    logger.error(f"❌ Failed to update database for POI {poi_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in photo processing pipeline for POI {poi_id}: {e}")
            result['error'] = str(e)
            return result
    
    def get_best_photo_for_collection(self, poi_ids: List[str]) -> Optional[str]:
        """Get the best photo from a list of POIs for collection cover"""
        try:
            best_photo = None
            best_quality = 0
            
            for poi_id in poi_ids:
                # Get POI photo data
                poi_result = self.db.client.table('poi')\
                    .select('primary_photo,primary_photo_quality')\
                    .eq('id', poi_id)\
                    .execute()
                
                if poi_result.data:
                    poi = poi_result.data[0]
                    quality = poi.get('primary_photo_quality', 0)
                    
                    if quality > best_quality:
                        best_quality = quality
                        best_photo = poi.get('primary_photo')
            
            if best_photo:
                logger.info(f"Selected best photo for collection: {best_photo} (quality: {best_quality:.3f})")
            
            return best_photo
            
        except Exception as e:
            logger.error(f"Error selecting best photo for collection: {e}")
            return None
    
    def get_photo_url(self, file_path: str, base_url: str = None) -> str:
        """Get full URL for a photo file path"""
        if not file_path:
            return ""
        
        if base_url:
            return urljoin(base_url.rstrip('/') + '/', file_path)
        else:
            # Return local file path if no base URL provided
            return str(self.storage_path / file_path)
    
    def cleanup_old_photos(self, days_old: int = 30) -> int:
        """Clean up old photo files to save storage space"""
        try:
            import time
            cutoff_time = time.time() - (days_old * 24 * 60 * 60)
            deleted_count = 0
            
            for poi_dir in self.storage_path.iterdir():
                if poi_dir.is_dir():
                    for photo_file in poi_dir.glob('*.jpg'):
                        if photo_file.stat().st_mtime < cutoff_time:
                            photo_file.unlink()
                            deleted_count += 1
                    
                    # Remove empty directories
                    if not any(poi_dir.iterdir()):
                        poi_dir.rmdir()
            
            logger.info(f"Cleaned up {deleted_count} old photo files")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error during photo cleanup: {e}")
            return 0

def main():
    """CLI interface for photo management"""
    import argparse
    
    parser = argparse.ArgumentParser(description='POI Photo Management System')
    parser.add_argument('--process-poi', help='Process photos for specific POI ID')
    parser.add_argument('--place-id', help='Google Place ID for photo processing')
    parser.add_argument('--batch-process', action='store_true', help='Process photos for POIs without photos')
    parser.add_argument('--cleanup', type=int, help='Clean up photos older than N days')
    parser.add_argument('--storage-path', default='/var/trendr/photos', help='Photo storage path')
    parser.add_argument('--limit', type=int, default=50, help='Limit for batch processing')
    
    args = parser.parse_args()
    
    photo_manager = POIPhotoManager(args.storage_path)
    
    if args.process_poi and args.place_id:
        result = photo_manager.process_poi_photos(args.process_poi, args.place_id)
        print("📊 Photo Processing Result:")
        print(json.dumps(result, indent=2, default=str))
    
    elif args.batch_process:
        # Get POIs without photos
        try:
            pois_without_photos = photo_manager.db.client.table('poi')\
                .select('id,google_place_id,name')\
                .is_('primary_photo', 'null')\
                .limit(args.limit)\
                .execute()
            
            if not pois_without_photos.data:
                print("✅ All POIs already have photos processed")
                return
            
            print(f"🖼️ Processing photos for {len(pois_without_photos.data)} POIs...")
            
            success_count = 0
            for i, poi in enumerate(pois_without_photos.data, 1):
                try:
                    print(f"[{i}/{len(pois_without_photos.data)}] Processing {poi['name']}...")
                    
                    result = photo_manager.process_poi_photos(
                        poi['id'], 
                        poi['google_place_id']
                    )
                    
                    if result['success']:
                        success_count += 1
                        print(f"✅ Success: {result['photos_processed']} photos (best: {result['best_quality']:.3f})")
                    else:
                        print(f"❌ Failed: {poi['name']}")
                    
                    # Rate limiting
                    import time
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"❌ Error processing {poi['name']}: {e}")
                    continue
            
            print(f"\n🎉 Batch processing complete: {success_count}/{len(pois_without_photos.data)} successful")
            
        except Exception as e:
            print(f"❌ Batch processing failed: {e}")
    
    elif args.cleanup:
        deleted = photo_manager.cleanup_old_photos(args.cleanup)
        print(f"🧹 Cleaned up {deleted} old photo files")
    
    else:
        print("POI Photo Management System")
        print("Usage:")
        print("  --process-poi POI_ID --place-id PLACE_ID  # Process photos for specific POI")
        print("  --batch-process --limit 50                # Process photos for POIs without photos")
        print("  --cleanup 30                             # Clean up photos older than 30 days")

if __name__ == "__main__":
    main()