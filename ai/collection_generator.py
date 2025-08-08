#!/usr/bin/env python3
"""
AI Collection Generator - Creates smart POI collections using contextual tags.
Modern tag-based collection generation with clean templates and proper naming.
"""
import sys
import os
import logging
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from collections import Counter

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database import SupabaseManager
from utils.photo_manager import POIPhotoManager
import config

# Try to import OpenAI (primary) or Anthropic (fallback)
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CollectionGenerator:
    """Modern AI-powered collection generator using contextual tags."""
    
    def __init__(self):
        self.db = SupabaseManager()
        self.photo_manager = POIPhotoManager()
        self.setup_ai_client()
        
        # SEO-optimized collection templates - English, specific, clickable
        self.collection_templates = {
            # TEMPORAL TRENDING - Critical for Trendr SEO & UX
            'hot_new_spots': {
                'title': 'Hot New Spots',
                'slug': 'hot-new-spots-montreal',
                'description': 'The newest hotspots everyone is talking about - fresh discoveries creating buzz',
                'seo_description': 'Discover Montreal\'s hottest new restaurants, bars and cafes that just opened',
                'required_tags': ['nouveau'],
                'excluded_tags': ['établi'],
                'min_confidence': 0.4,
                'priority': 1
            },
            'rising_stars': {
                'title': 'Rising Stars',
                'slug': 'rising-star-venues-montreal', 
                'description': 'Places gaining momentum and building their reputation - tomorrow\'s classics',
                'seo_description': 'Montreal\'s rising star restaurants and bars gaining popularity fast',
                'required_tags': ['émergent'],
                'excluded_tags': [],
                'min_confidence': 0.5,
                'priority': 2
            },
            'proven_classics': {
                'title': 'Proven Classics',
                'slug': 'classic-montreal-restaurants',
                'description': 'Established favorites that maintain their excellence - trusted quality',
                'seo_description': 'Montreal\'s best established restaurants and bars with proven track records',
                'required_tags': ['établi'],
                'excluded_tags': [],
                'min_confidence': 0.6,
                'priority': 3
            },
            
            # LIFESTYLE-SPECIFIC COLLECTIONS - SEO optimized
            'digital_nomad_cafes': {
                'title': 'Digital Nomad Cafes',
                'slug': 'best-work-cafes-montreal',
                'description': 'Perfect workspaces with reliable WiFi, power outlets, and productive vibes',
                'seo_description': 'Best cafes for remote work in Montreal with WiFi, power outlets and quiet atmosphere',
                'required_tags': ['work-friendly'],
                'excluded_tags': ['vibrant'],
                'min_confidence': 0.6,
                'priority': 4
            },
            'romantic_date_spots': {
                'title': 'Romantic Date Spots',
                'slug': 'romantic-restaurants-montreal', 
                'description': 'Intimate venues perfect for memorable dates and special occasions',
                'seo_description': 'Most romantic restaurants and bars in Montreal for perfect date nights',
                'required_tags': ['date-spot'],
                'excluded_tags': ['group-friendly', 'work-friendly'],
                'min_confidence': 0.5,
                'priority': 5
            },
            'instagram_worthy': {
                'title': 'Instagram-Worthy Spots',
                'slug': 'instagrammable-places-montreal',
                'description': 'Photogenic venues with stunning aesthetics perfect for your feed',
                'seo_description': 'Most Instagram-worthy restaurants and cafes in Montreal with beautiful design',
                'required_tags': ['photo-worthy'],
                'excluded_tags': [],
                'min_confidence': 0.5,
                'priority': 6
            },
            
            # TIME-BASED EXPERIENCES - SEO optimized
            'breakfast_champions': {
                'title': 'Breakfast Champions',
                'slug': 'best-breakfast-montreal',
                'description': 'Perfect morning spots for coffee, brunch and starting your day right',
                'seo_description': 'Best breakfast and brunch spots in Montreal for perfect morning meals',
                'required_tags': ['morning-spot'],
                'excluded_tags': ['evening-spot'],
                'min_confidence': 0.5,
                'priority': 7
            },
            'evening_vibes': {
                'title': 'Evening Vibes',
                'slug': 'best-evening-bars-montreal',
                'description': 'Perfect venues for dinner, drinks and unwinding after a long day',
                'seo_description': 'Best evening bars and restaurants in Montreal for dinner and cocktails',
                'required_tags': ['evening-spot'],
                'excluded_tags': ['morning-spot'],
                'min_confidence': 0.5,
                'priority': 8
            },
            
            # INSIDER COLLECTIONS - SEO optimized  
            'locals_only': {
                'title': 'Locals Only',
                'slug': 'local-favorite-spots-montreal',
                'description': 'Authentic neighborhood gems loved by regulars - off the tourist radar',
                'seo_description': 'Hidden local favorites in Montreal loved by residents but unknown to tourists',
                'required_tags': ['local-favorite'],
                'excluded_tags': ['tourist-friendly'],
                'min_confidence': 0.6,
                'priority': 9
            },
            'scene_setters': {
                'title': 'Scene Setters',
                'slug': 'trendy-hotspots-montreal',
                'description': 'Hip venues defining Montreal\'s contemporary culture - where trends are born',
                'seo_description': 'Montreal\'s trendiest restaurants and bars where the cool crowd gathers',
                'required_tags': ['trendy'],
                'excluded_tags': ['authentic', 'peaceful'],
                'min_confidence': 0.5,
                'priority': 10
            },
            'one_of_a_kind': {
                'title': 'One of a Kind',
                'slug': 'unique-restaurants-montreal',
                'description': 'Distinctive venues with unique character that can\'t be found elsewhere',
                'seo_description': 'Most unique and distinctive restaurants in Montreal with special character',
                'required_tags': ['unique'],
                'excluded_tags': [],
                'min_confidence': 0.5,
                'priority': 11
            }
        }
    
    def setup_ai_client(self):
        """Setup AI client (OpenAI or Anthropic)."""
        self.ai_client = None
        self.ai_provider = 'contextual_tags'
        
        # Try OpenAI first
        if HAS_OPENAI and hasattr(config, 'OPENAI_API_KEY') and config.OPENAI_API_KEY:
            try:
                self.ai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
                self.ai_provider = 'openai'
                logger.info("Using OpenAI for AI generation")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI: {e}")
        
        # Try Anthropic as fallback
        if HAS_ANTHROPIC and hasattr(config, 'ANTHROPIC_API_KEY') and config.ANTHROPIC_API_KEY:
            try:
                self.ai_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
                self.ai_provider = 'anthropic'
                logger.info("Using Anthropic Claude for AI generation")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic: {e}")
        
        # Default to contextual tag-based generation
        logger.info("No AI provider available - using contextual tag-based generation")
        self.ai_client = None
        self.ai_provider = 'contextual_tags'
    
    def get_poi_data_for_analysis(self, city: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get POI data with contextual tags for analysis."""
        try:
            # Get POIs with tags
            pois = self.db.get_pois_for_city(city, limit)
            
            # Filter and enrich POIs that have tags
            enriched_pois = []
            for poi in pois:
                # FIXED: Use tags column as primary (where temporal tags are stored), fallback to contextual_tags
                tags = poi.get('tags') or poi.get('contextual_tags')
                
                # Handle both empty dict and None cases
                if tags and isinstance(tags, dict) and tags != {}:
                    poi['tags_count'] = len(tags)
                    poi['tag_categories'] = {}
                    
                    # Categorize tags
                    for tag_name, tag_data in tags.items():
                        if isinstance(tag_data, dict):
                            category = tag_data.get('category', 'unknown')
                            confidence = tag_data.get('confidence', 0)
                            
                            if category not in poi['tag_categories']:
                                poi['tag_categories'][category] = []
                            poi['tag_categories'][category].append({
                                'tag': tag_name,
                                'confidence': confidence
                            })
                    
                    # Ensure consistency: always use 'tags' as the primary field
                    poi['tags'] = tags
                    
                    enriched_pois.append(poi)
                
                # ALSO include POIs that have primary_mood for fallback collections
                elif poi.get('primary_mood'):
                    # Create synthetic tags based on primary mood for compatibility
                    poi['tags'] = {
                        poi['primary_mood']: {
                            'confidence': poi.get('mood_confidence', 0.5),
                            'category': 'mood',
                            'sources_count': 1
                        }
                    }
                    poi['tags_count'] = 1
                    enriched_pois.append(poi)
            
            logger.info(f"Found {len(enriched_pois)} POIs with tags out of {len(pois)} total")
            return enriched_pois
            
        except Exception as e:
            logger.error(f"Error getting POI data for analysis: {e}")
            return []
    
    def find_pois_by_tag_criteria(self, pois: List[Dict[str, Any]], 
                                 required_tags: List[str], 
                                 excluded_tags: List[str] = None,
                                 min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        """Find POIs that match contextual tag criteria."""
        matching_pois = []
        excluded_tags = excluded_tags or []
        
        for poi in pois:
            # FIXED: Use tags column as primary (where temporal tags are stored)
            tags = poi.get('tags') or {}
            if not tags:
                continue
            
            # Check if POI has at least one required tag with sufficient confidence
            has_required = False
            for required_tag in required_tags:
                if required_tag in tags:
                    tag_data = tags[required_tag]
                    if isinstance(tag_data, dict):
                        confidence = tag_data.get('confidence', 0)
                        if confidence >= min_confidence:
                            has_required = True
                            break
            
            if not has_required:
                continue
            
            # Check if POI doesn't have excluded tags above threshold
            has_excluded = False
            for excluded_tag in excluded_tags:
                if excluded_tag in tags:
                    tag_data = tags[excluded_tag]
                    if isinstance(tag_data, dict):
                        confidence = tag_data.get('confidence', 0)
                        if confidence >= min_confidence:
                            has_excluded = True
                            break
            
            if not has_excluded:
                # Calculate match score based on relevant tags
                match_score = 0.0
                tag_matches = []
                
                for required_tag in required_tags:
                    if required_tag in tags:
                        tag_data = tags[required_tag]
                        if isinstance(tag_data, dict):
                            confidence = tag_data.get('confidence', 0)
                            if confidence >= min_confidence:
                                match_score += confidence
                                tag_matches.append(f"{required_tag}:{confidence:.2f}")
                
                poi['match_score'] = match_score
                poi['matching_tags'] = tag_matches
                matching_pois.append(poi)
        
        # Sort by match score (highest first)
        matching_pois.sort(key=lambda x: x['match_score'], reverse=True)
        return matching_pois
    
    def generate_contextual_collections(self, city: str) -> List[Dict[str, Any]]:
        """Generate collections based on contextual tag templates."""
        logger.info(f"Generating contextual tag-based collections for {city}")
        
        # Get POIs with contextual tags
        pois = self.get_poi_data_for_analysis(city)
        if len(pois) < 5:
            logger.warning(f"Insufficient POIs with contextual tags for {city}: {len(pois)}")
            return []
        
        collections = []
        
        # Get tag usage statistics for prioritization
        tag_stats = Counter()
        for poi in pois:
            # FIXED: Use tags column consistently
            tags = poi.get('tags') or {}
            for tag_name in tags.keys():
                tag_stats[tag_name] += 1
        
        logger.info(f"Most common contextual tags: {dict(tag_stats.most_common(10))}")
        
        # Generate collections for each template
        for template_key, template in self.collection_templates.items():
            try:
                # Find POIs matching this template
                matching_pois = self.find_pois_by_tag_criteria(
                    pois, 
                    template['required_tags'],
                    template['excluded_tags'],
                    template['min_confidence']
                )
                
                if len(matching_pois) >= 2:  # Need at least 2 POIs for a collection
                    # Limit to top 8 POIs
                    selected_pois = matching_pois[:8]
                    
                    # Check if collection already exists and update instead of creating
                    existing_collection = self.get_existing_collection(city, template['title'])
                    
                    # Get best photo for collection cover
                    cover_photo = self.photo_manager.get_best_photo_for_collection(
                        [poi['id'] for poi in selected_pois]
                    )
                    
                    collection_data = {
                        'title': template['title'],
                        'type': 'contextual',
                        'description': template['description'],
                        'city': city,
                        'poi_ids': [poi['id'] for poi in selected_pois],
                        'cover_photo': cover_photo,
                        'required_tags': template['required_tags'],
                        'excluded_tags': template['excluded_tags'],
                        'min_confidence': template['min_confidence'],
                        'metadata': {
                            'generated_by': 'collection_generator_v4_seo',
                            'template_used': template_key,
                            'avg_match_score': sum(poi['match_score'] for poi in selected_pois) / len(selected_pois),
                            'poi_count': len(selected_pois),
                            'generation_timestamp': datetime.now(timezone.utc).isoformat(),
                            'seo_optimized': True,
                            'seo_data': {
                                'slug': template.get('slug', template['title'].lower().replace(' ', '-')),
                                'seo_description': template.get('seo_description', template['description']),
                                'priority': template.get('priority', 99)
                            },
                            'tag_criteria': {
                                'required': template['required_tags'],
                                'excluded': template['excluded_tags'],
                                'min_confidence': template['min_confidence']
                            }
                        }
                    }
                    
                    collection = {
                        'data': collection_data,
                        'existing_id': existing_collection['id'] if existing_collection else None,
                        'is_update': bool(existing_collection)
                    }
                    
                    collections.append(collection)
                    logger.info(f"✅ {template['title']}: {len(selected_pois)} POIs")
                    
                else:
                    logger.info(f"⚠️ {template['title']}: Only {len(matching_pois)} POIs (need ≥2)")
                    
            except Exception as e:
                logger.error(f"Error generating collection {template_key}: {e}")
                continue
        
        logger.info(f"Generated {len(collections)} contextual collections for {city}")
        return collections
    
    def get_existing_collection(self, city: str, title: str) -> Optional[Dict[str, Any]]:
        """Check if a collection with the same title already exists for this city."""
        try:
            existing_collections = self.db.get_collections_for_city(city)
            for collection in existing_collections:
                if collection['title'] == title:
                    return collection
            return None
        except Exception as e:
            logger.error(f"Error checking existing collections: {e}")
            return None
    
    def generate_collections_for_city(self, city: str, use_ai: bool = False) -> Tuple[int, List[str]]:
        """Generate collections for a city using contextual tags."""
        logger.info(f"🚀 Generating collections for {city} (AI: {use_ai})")
        
        try:
            all_collections = self.generate_contextual_collections(city)
            
            # Insert/Update collections into database
            created_collections = []
            updated_collections = []
            
            for collection in all_collections:
                try:
                    collection_data = collection['data']
                    is_update = collection['is_update']
                    existing_id = collection['existing_id']
                    
                    if is_update:
                        # Update existing collection
                        success = self.db.update_collection(existing_id, collection_data)
                        if success:
                            updated_collections.append(collection_data['title'])
                            logger.info(f"🔄 Updated: {collection_data['title']} ({len(collection_data['poi_ids'])} POIs)")
                    else:
                        # Create new collection
                        collection_id = self.db.insert_collection(collection_data)
                        if collection_id:
                            created_collections.append(collection_data['title'])
                            logger.info(f"✅ Created: {collection_data['title']} ({len(collection_data['poi_ids'])} POIs)")
                            
                except Exception as e:
                    logger.error(f"Error processing collection {collection_data['title']}: {e}")
            
            total_processed = len(created_collections) + len(updated_collections)
            logger.info(f"🎯 Successfully processed {total_processed} collections for {city}")
            logger.info(f"   📝 Created: {len(created_collections)}, Updated: {len(updated_collections)}")
            
            return total_processed, created_collections + updated_collections
            
        except Exception as e:
            logger.error(f"Error generating collections for {city}: {e}")
            return 0, []

def main():
    """Main entry point for collection generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Contextual Tag-Based Collections')
    parser.add_argument('--city', default='Montreal', help='City to generate collections for')
    parser.add_argument('--ai', action='store_true', help='Use AI-enhanced generation')
    parser.add_argument('--test', action='store_true', help='Test mode - show collection templates')
    
    args = parser.parse_args()
    
    generator = CollectionGenerator()
    
    try:
        if args.test:
            logger.info("🧪 TEST MODE: Available collection templates")
            for key, template in generator.collection_templates.items():
                print(f"\n📍 {template['title']}")
                print(f"   Required tags: {template['required_tags']}")
                print(f"   Excluded tags: {template['excluded_tags']}")
                print(f"   Min confidence: {template['min_confidence']}")
                print(f"   Description: {template['description'][:100]}...")
            return
        
        count, titles = generator.generate_collections_for_city(
            args.city, 
            use_ai=args.ai
        )
        
        print(f"\n🎉 Collection Generation Complete!")
        print(f"Created {count} collections for {args.city}:")
        for i, title in enumerate(titles, 1):
            print(f"  {i}. {title}")
            
    except Exception as e:
        logger.error(f"Collection generation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()