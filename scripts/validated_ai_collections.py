#!/usr/bin/env python3
"""
Validated AI Collections Generator - Step 3 of Social Proof Enhancement
Creates authentic collections using enhanced social proof and multi-dimensional classification.
Focus: Collections backed by real social validation and intelligent context analysis.
"""
import sys
import os
import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import Counter, defaultdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database import SupabaseManager
from scripts.intelligent_classifier import IntelligentMoodClassifier
from scripts.dynamic_neighborhoods import DynamicNeighborhoodCalculator
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidatedCollectionGenerator:
    """Generate collections using enhanced social proof and intelligent classification."""
    
    def __init__(self):
        self.db = SupabaseManager()
        self.classifier = IntelligentMoodClassifier()
        self.neighborhood_calculator = DynamicNeighborhoodCalculator()
        
        # Social proof validation thresholds (adjusted for current data quality)
        self.validation_thresholds = {
            'minimum_social_proof': {
                'proof_sources_count': 1,  # At least 1 proof source (relaxed)
                'authority_score_min': 0.3,  # Lower minimum authority score
                'recent_activity_days': 365  # Activity within 1 year (more lenient)
            },
            'trending_requirements': {
                'high_authority_sources': 0,  # No high authority requirement initially
                'contextual_tags_count': 1,  # At least 1 contextual tag
                'mood_confidence': 0.4  # Lower confidence threshold
            },
            'authenticity_checks': {
                'max_spam_indicators': 3,  # More lenient spam detection
                'min_unique_sources': 1,  # Minimum unique source domains
                'domain_diversity': 0.2  # Lower diversity requirement
            }
        }
    
    def get_validated_pois_for_city(self, city: str = 'Montreal') -> List[Dict[str, Any]]:
        """Get POIs with sufficient social proof validation."""
        try:
            # Get all POIs with proof sources
            pois = self.db.client.table('poi')\
                .select('*, proof_sources(*)')\
                .eq('city', city)\
                .execute()
            
            validated_pois = []
            
            for poi in pois.data:
                poi_id = poi['id']
                proof_sources = poi.get('proof_sources', [])
                
                # Apply social proof validation
                if self.validate_poi_social_proof(poi, proof_sources):
                    # Enrich with classification data
                    enriched_poi = self.enrich_poi_with_classification(poi)
                    if enriched_poi:
                        validated_pois.append(enriched_poi)
            
            logger.info(f"✅ Validated {len(validated_pois)} POIs out of {len(pois.data)} total for {city}")
            return validated_pois
            
        except Exception as e:
            logger.error(f"Error getting validated POIs: {e}")
            return []
    
    def validate_poi_social_proof(self, poi: Dict[str, Any], proof_sources: List[Dict[str, Any]]) -> bool:
        """Validate if POI has sufficient social proof for collections."""
        thresholds = self.validation_thresholds
        
        # Check minimum proof sources count
        if len(proof_sources) < thresholds['minimum_social_proof']['proof_sources_count']:
            return False
        
        # Check for recent activity
        recent_cutoff = datetime.now() - timedelta(days=thresholds['minimum_social_proof']['recent_activity_days'])
        recent_sources = [
            source for source in proof_sources 
            if datetime.fromisoformat(source.get('created_at', '2020-01-01').replace('Z', '+00:00')) > recent_cutoff
        ]
        
        if not recent_sources:
            return False
        
        # Check authority score distribution
        authority_scores = []
        high_authority_count = 0
        
        for source in proof_sources:
            authority_level = source.get('authority_score', 'Low')
            if authority_level == 'High':
                high_authority_count += 1
                authority_scores.append(1.0)
            elif authority_level == 'Medium':
                authority_scores.append(0.6)
            else:
                authority_scores.append(0.3)
        
        avg_authority = sum(authority_scores) / len(authority_scores) if authority_scores else 0
        
        if avg_authority < thresholds['minimum_social_proof']['authority_score_min']:
            return False
        
        # Check domain diversity
        domains = [source.get('domain', 'unknown') for source in proof_sources]
        unique_domains = len(set(domains))
        
        if unique_domains < thresholds['authenticity_checks']['min_unique_sources']:
            return False
        
        return True
    
    def enrich_poi_with_classification(self, poi: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Enrich POI with intelligent classification data."""
        try:
            poi_id = poi['id']
            
            # Get proof sources for classification
            proof_sources = poi.get('proof_sources', [])
            
            # Use intelligent classifier for multi-dimensional analysis
            classification_result = self.classifier.classify_poi_multi_dimensional(poi, proof_sources)
            
            if not classification_result:
                return None
            
            # Enrich POI data
            enriched_poi = poi.copy()
            enriched_poi['contextual_tags'] = classification_result.get('contextual_tags', [])
            enriched_poi['mood_analysis'] = classification_result.get('mood_analysis', {})
            enriched_poi['classification_confidence'] = classification_result.get('confidence_score', 0.5)
            enriched_poi['enrichment_timestamp'] = datetime.utcnow().isoformat()
            
            return enriched_poi
            
        except Exception as e:
            logger.error(f"Error enriching POI {poi.get('name', 'Unknown')}: {e}")
            return None
    
    def generate_mood_based_collections(self, validated_pois: List[Dict[str, Any]], 
                                      city: str = 'Montreal') -> List[Dict[str, Any]]:
        """Generate mood-based collections using dynamic neighborhood data."""
        collections = []
        
        # Group POIs by neighborhood for context-aware collections
        pois_by_neighborhood = defaultdict(list)
        for poi in validated_pois:
            neighborhood = poi.get('neighborhood', 'Unknown')
            pois_by_neighborhood[neighborhood].append(poi)
        
        for neighborhood, neighborhood_pois in pois_by_neighborhood.items():
            if len(neighborhood_pois) < 2:  # Skip neighborhoods with too few POIs
                continue
            
            # Group by mood
            pois_by_mood = defaultdict(list)
            for poi in neighborhood_pois:
                mood = poi.get('mood_tag', 'trendy').lower()
                if mood in ['chill', 'trendy', 'hidden_gem']:
                    pois_by_mood[mood].append(poi)
            
            # Create collections for each mood with sufficient POIs
            for mood, mood_pois in pois_by_mood.items():
                if len(mood_pois) >= 2:  # Minimum 2 POIs per collection
                    collection = self.create_mood_collection(
                        mood, mood_pois, neighborhood, city
                    )
                    if collection:
                        collections.append(collection)
        
        return collections
    
    def generate_contextual_collections(self, validated_pois: List[Dict[str, Any]], 
                                     city: str = 'Montreal') -> List[Dict[str, Any]]:
        """Generate collections based on contextual tags and experiences."""
        collections = []
        
        # Analyze contextual tags across all POIs
        tag_to_pois = defaultdict(list)
        
        for poi in validated_pois:
            contextual_tags = poi.get('contextual_tags', [])
            for tag in contextual_tags:
                if isinstance(tag, dict) and tag.get('confidence', 0) > 0.6:  # Only high-confidence tags
                    tag_name = tag.get('tag', '').lower()
                    if tag_name:
                        tag_to_pois[tag_name].append(poi)
                elif isinstance(tag, str):  # Handle string tags
                    tag_to_pois[tag.lower()].append(poi)
        
        # Create collections for popular contextual themes
        popular_tags = [
            ('work-friendly', '💻 Perfect for Remote Work', 'Work-friendly spots with WiFi and good coffee'),
            ('date-spot', '💕 Perfect Date Spots', 'Romantic and intimate venues for memorable dates'), 
            ('photo-worthy', '📸 Instagram-Worthy Places', 'Visually stunning venues perfect for photos'),
            ('tourist-friendly', '🗺️ Must-Visit Spots', 'Tourist-friendly places showcasing local culture'),
            ('local-favorite', '🏠 Local Favorites', 'Hidden gems beloved by locals'),
            ('group-friendly', '👥 Perfect for Groups', 'Spacious venues ideal for group gatherings'),
            ('morning-spot', '🌅 Morning Essentials', 'Perfect spots to start your day'),
            ('evening-spot', '🌆 Evening Destinations', 'Ideal places for evening activities')
        ]
        
        for tag_key, title, description in popular_tags:
            matching_pois = tag_to_pois.get(tag_key, [])
            if len(matching_pois) >= 2:  # Minimum 2 POIs for contextual collections
                collection = self.create_contextual_collection(
                    tag_key, title, description, matching_pois, city
                )
                if collection:
                    collections.append(collection)
        
        return collections
    
    def generate_authority_backed_collections(self, validated_pois: List[Dict[str, Any]], 
                                           city: str = 'Montreal') -> List[Dict[str, Any]]:
        """Generate collections based on high-authority social proof."""
        collections = []
        
        # Filter POIs with high authority mentions
        high_authority_pois = []
        for poi in validated_pois:
            proof_sources = poi.get('proof_sources', [])
            high_authority_count = sum(
                1 for source in proof_sources 
                if source.get('authority_score') == 'High'
            )
            
            if high_authority_count >= 1:  # At least 1 high authority mention
                poi['high_authority_count'] = high_authority_count
                high_authority_pois.append(poi)
        
        if len(high_authority_pois) >= 5:
            # Create critics' choice collection
            collection = {
                'title': "👑 Critics' Choice: Montreal's Best",
                'emoji': '👑',
                'type': 'editorial',
                'description': 'Venues praised by food critics and authoritative sources',
                'poi_ids': [poi['id'] for poi in high_authority_pois[:10]],  # Top 10
                'city': city,
                'country': 'Canada',
                'metadata': {
                    'collection_type': 'authority_backed',
                    'generation_method': 'social_proof_validation',
                    'authority_threshold': 'high',
                    'total_high_authority_mentions': sum(
                        poi.get('high_authority_count', 0) for poi in high_authority_pois[:10]
                    ),
                    'generated_at': datetime.utcnow().isoformat(),
                    'validation_criteria': self.validation_thresholds
                }
            }
            collections.append(collection)
        
        return collections
    
    def create_mood_collection(self, mood: str, pois: List[Dict[str, Any]], 
                              neighborhood: str, city: str) -> Optional[Dict[str, Any]]:
        """Create a mood-based collection with social proof validation."""
        
        mood_configs = {
            'chill': {
                'emoji': '😌',
                'title_template': '😌 Chill Spots in {}',
                'description_template': 'Relaxed and cozy places perfect for unwinding in {}'
            },
            'trendy': {
                'emoji': '🔥', 
                'title_template': '🔥 Trending in {}',
                'description_template': 'The hottest and most popular spots everyone\'s talking about in {}'
            },
            'hidden_gem': {
                'emoji': '💎',
                'title_template': '💎 Hidden Gems of {}', 
                'description_template': 'Secret favorites and undiscovered treasures in {}'
            }
        }
        
        if mood not in mood_configs:
            return None
        
        config = mood_configs[mood]
        
        # Sort POIs by classification confidence and social proof strength
        sorted_pois = sorted(pois, key=lambda p: (
            p.get('classification_confidence', 0.5),
            len(p.get('proof_sources', []))
        ), reverse=True)
        
        # Take top POIs for collection (max 8 per collection)
        selected_pois = sorted_pois[:8]
        
        collection = {
            'title': config['title_template'].format(neighborhood),
            'emoji': config['emoji'],
            'type': 'mood',
            'description': config['description_template'].format(neighborhood),
            'poi_ids': [poi['id'] for poi in selected_pois],
            'city': city,
            'country': 'Canada',
            'metadata': {
                'collection_type': 'mood_based',
                'mood': mood,
                'generation_method': 'social_proof_validation',
                'total_proof_sources': sum(len(poi.get('proof_sources', [])) for poi in selected_pois),
                'avg_classification_confidence': sum(
                    poi.get('classification_confidence', 0.5) for poi in selected_pois
                ) / len(selected_pois),
                'generated_at': datetime.utcnow().isoformat(),
                'validation_criteria': self.validation_thresholds
            }
        }
        
        return collection
    
    def create_contextual_collection(self, tag_key: str, title: str, description: str,
                                   pois: List[Dict[str, Any]], city: str) -> Dict[str, Any]:
        """Create a contextual collection based on experiential tags."""
        
        # Sort POIs by contextual tag confidence and social proof
        def get_tag_confidence(poi):
            contextual_tags = poi.get('contextual_tags', [])
            for tag in contextual_tags:
                if isinstance(tag, dict) and tag.get('tag', '').lower() == tag_key:
                    return tag.get('confidence', 0)
                elif isinstance(tag, str) and tag.lower() == tag_key:
                    return 0.7  # Default confidence for string tags
            return 0
        
        sorted_pois = sorted(pois, key=lambda p: (
            get_tag_confidence(p),
            len(p.get('proof_sources', []))
        ), reverse=True)
        
        # Select top POIs (max 10 for contextual collections)
        selected_pois = sorted_pois[:10]
        
        collection = {
            'title': title,
            'emoji': title.split(' ')[0] if title.split(' ')[0] in ['💻', '💕', '📸', '🗺️', '🏠', '👥', '🌅', '🌆'] else '✨',
            'type': 'contextual',
            'description': description,
            'poi_ids': [poi['id'] for poi in selected_pois],
            'city': city,
            'country': 'Canada',
            'metadata': {
                'collection_type': 'contextual',
                'contextual_theme': tag_key,
                'generation_method': 'contextual_tag_analysis',
                'avg_tag_confidence': sum(get_tag_confidence(poi) for poi in selected_pois) / len(selected_pois),
                'total_proof_sources': sum(len(poi.get('proof_sources', [])) for poi in selected_pois),
                'generated_at': datetime.utcnow().isoformat(),
                'validation_criteria': self.validation_thresholds
            }
        }
        
        return collection
    
    def save_collections_to_database(self, collections: List[Dict[str, Any]]) -> int:
        """Save validated collections to database."""
        saved_count = 0
        
        for collection in collections:
            try:
                # Check if similar collection already exists
                existing = self.db.client.table('collections')\
                    .select('id')\
                    .eq('title', collection['title'])\
                    .eq('city', collection['city'])\
                    .execute()
                
                if existing.data:
                    logger.info(f"⚠️ Collection '{collection['title']}' already exists, skipping")
                    continue
                
                # Insert new collection
                result = self.db.client.table('collections')\
                    .insert(collection)\
                    .execute()
                
                if result.data:
                    saved_count += 1
                    logger.info(f"✅ Created collection: {collection['title']} ({len(collection['poi_ids'])} POIs)")
                
            except Exception as e:
                logger.error(f"Error saving collection '{collection.get('title', 'Unknown')}': {e}")
                continue
        
        return saved_count
    
    def generate_all_validated_collections(self, city: str = 'Montreal') -> Dict[str, Any]:
        """Generate all types of validated collections."""
        logger.info(f"🎨 Starting validated AI collection generation for {city}")
        
        # Get validated POIs
        validated_pois = self.get_validated_pois_for_city(city)
        
        if len(validated_pois) < 3:
            logger.warning(f"Insufficient validated POIs ({len(validated_pois)}) for collection generation")
            return {'error': 'insufficient_validated_pois', 'count': len(validated_pois)}
        
        all_collections = []
        
        # Generate different types of collections
        collection_types = {
            'mood_based': self.generate_mood_based_collections(validated_pois, city),
            'contextual': self.generate_contextual_collections(validated_pois, city), 
            'authority_backed': self.generate_authority_backed_collections(validated_pois, city)
        }
        
        # Combine all collections
        for collection_type, collections in collection_types.items():
            logger.info(f"  📋 Generated {len(collections)} {collection_type} collections")
            all_collections.extend(collections)
        
        # Save to database
        saved_count = self.save_collections_to_database(all_collections)
        
        results = {
            'total_validated_pois': len(validated_pois),
            'collections_generated': len(all_collections),
            'collections_saved': saved_count,
            'collection_breakdown': {
                collection_type: len(collections) 
                for collection_type, collections in collection_types.items()
            },
            'generated_at': datetime.utcnow().isoformat()
        }
        
        logger.info(f"🎯 Collection generation complete!")
        logger.info(f"  📊 Validated POIs: {results['total_validated_pois']}")
        logger.info(f"  🎨 Collections created: {results['collections_generated']}")
        logger.info(f"  💾 Collections saved: {results['collections_saved']}")
        
        return results
    
    def analyze_collection_quality(self, city: str = 'Montreal') -> Dict[str, Any]:
        """Analyze quality of generated collections."""
        try:
            collections = self.db.client.table('collections')\
                .select('*')\
                .eq('city', city)\
                .execute()
            
            if not collections.data:
                return {'error': 'no_collections_found'}
            
            analysis = {
                'total_collections': len(collections.data),
                'collections_by_type': Counter(c.get('type', 'unknown') for c in collections.data),
                'avg_pois_per_collection': 0,
                'collections_with_social_proof': 0,
                'total_unique_pois': set(),
                'quality_metrics': {}
            }
            
            total_pois = 0
            collections_with_proof = 0
            
            for collection in collections.data:
                poi_count = len(collection.get('poi_ids', []))
                total_pois += poi_count
                
                # Track unique POIs
                analysis['total_unique_pois'].update(collection.get('poi_ids', []))
                
                # Check social proof validation
                metadata = collection.get('metadata', {})
                if metadata.get('total_proof_sources', 0) > 0:
                    collections_with_proof += 1
            
            analysis['avg_pois_per_collection'] = total_pois / len(collections.data) if collections.data else 0
            analysis['collections_with_social_proof'] = collections_with_proof
            analysis['total_unique_pois'] = len(analysis['total_unique_pois'])
            analysis['social_proof_coverage'] = collections_with_proof / len(collections.data) if collections.data else 0
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing collection quality: {e}")
            return {'error': str(e)}

def main():
    """CLI interface for validated collection generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validated AI Collection Generator - Step 3')
    parser.add_argument('--city', default='Montreal', help='City to generate collections for')
    parser.add_argument('--generate', action='store_true', help='Generate all validated collections')
    parser.add_argument('--analyze', action='store_true', help='Analyze collection quality')
    parser.add_argument('--test', action='store_true', help='Test mode - validate POIs only')
    
    args = parser.parse_args()
    
    generator = ValidatedCollectionGenerator()
    
    try:
        if args.test:
            logger.info("🧪 TEST MODE: Validating POIs only")
            validated_pois = generator.get_validated_pois_for_city(args.city)
            print(f"\n✅ Test Results:")
            print(f"  Validated POIs: {len(validated_pois)}")
            
            # Show sample enriched data
            if validated_pois:
                sample_poi = validated_pois[0]
                print(f"\n📍 Sample enriched POI: {sample_poi.get('name', 'Unknown')}")
                print(f"  Contextual tags: {len(sample_poi.get('contextual_tags', []))}")
                print(f"  Classification confidence: {sample_poi.get('classification_confidence', 0):.2f}")
                print(f"  Proof sources: {len(sample_poi.get('proof_sources', []))}")
        
        elif args.generate:
            results = generator.generate_all_validated_collections(args.city)
            print(f"\n🎨 Collection Generation Results:")
            for key, value in results.items():
                if key == 'collection_breakdown':
                    print(f"  Collection breakdown:")
                    for ctype, count in value.items():
                        print(f"    {ctype}: {count}")
                else:
                    print(f"  {key}: {value}")
        
        elif args.analyze:
            analysis = generator.analyze_collection_quality(args.city)
            print(f"\n📊 Collection Quality Analysis:")
            for key, value in analysis.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for subkey, subvalue in value.items():
                        print(f"    {subkey}: {subvalue}")
                else:
                    print(f"  {key}: {value}")
        
        else:
            print("Use --generate, --analyze, or --test")
    
    except Exception as e:
        logger.error(f"Validated collection generation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()