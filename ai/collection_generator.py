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
        
        # Templates de collections optimisées SEO - Français, spécifiques, engageantes
        self.collection_templates = {
            # TENDANCES TEMPORELLES - Critiques pour SEO & UX Trendr
            'nouveaux_spots_chauds': {
                'title': 'Nouveaux Spots Chauds',
                'slug_template': 'nouveaux-spots-tendance-{city}',
                'description': 'Les derniers lieux dont tout le monde parle - découvertes fraîches qui créent le buzz',
                'seo_description_template': 'Découvrez les restaurants, bars et cafés les plus tendance qui viennent d\'ouvrir à {city}',
                'required_tags': ['new_spot'],
                'excluded_tags': ['established'],
                'min_confidence': 0.4,
                'priority': 1
            },
            'etoiles_montantes': {
                'title': 'Étoiles Montantes',
                'slug_template': 'etoiles-montantes-{city}', 
                'description': 'Lieux en pleine ascension qui construisent leur réputation - les classiques de demain',
                'seo_description_template': 'Restaurants et bars étoiles montantes de {city} qui gagnent en popularité',
                'required_tags': ['emerging'],
                'excluded_tags': [],
                'min_confidence': 0.5,
                'priority': 2
            },
            'classiques_prouves': {
                'title': 'Classiques Prouvés',
                'slug_template': 'restaurants-classiques-{city}',
                'description': 'Favoris établis qui maintiennent leur excellence - qualité de confiance',
                'seo_description_template': 'Meilleurs restaurants et bars établis de {city} avec une réputation éprouvée',
                'required_tags': ['established'],
                'excluded_tags': [],
                'min_confidence': 0.6,
                'priority': 3
            },
            
            # COLLECTIONS LIFESTYLE - SEO optimisées
            'cafes_nomades_digitaux': {
                'title': 'Cafés Nomades Digitaux',
                'slug_template': 'cafes-coworking-{city}',
                'description': 'Espaces de travail parfaits avec WiFi fiable, prises et ambiance productive',
                'seo_description_template': 'Meilleurs cafés pour télétravailler à {city} avec WiFi, prises et atmosphère calme',
                'required_tags': ['work-friendly'],
                'excluded_tags': ['vibrant'],
                'min_confidence': 0.6,
                'priority': 4
            },
            'spots_romantiques': {
                'title': 'Spots Romantiques',
                'slug_template': 'restaurants-romantiques-{city}', 
                'description': 'Lieux intimistes parfaits pour des rendez-vous mémorables et occasions spéciales',
                'seo_description_template': 'Restaurants et bars les plus romantiques de {city} pour soirées en amoureux parfaites',
                'required_tags': ['date-spot'],
                'excluded_tags': ['group-friendly', 'work-friendly'],
                'min_confidence': 0.5,
                'priority': 5
            },
            'spots_instagrammables': {
                'title': 'Spots Instagrammables',
                'slug_template': 'lieux-instagrammables-{city}',
                'description': 'Lieux photogéniques à l\'esthétique époustouflante, parfaits pour votre feed',
                'seo_description_template': 'Restaurants et cafés les plus instagrammables de {city} au design magnifique',
                'required_tags': ['photo-worthy'],
                'excluded_tags': [],
                'min_confidence': 0.5,
                'priority': 6
            },
            
            # TIME-BASED EXPERIENCES - SEO optimized
            'champions_petit_dejeuner': {
                'title': 'Champions Petit-Déjeuner',
                'slug_template': 'best-breakfast-{city}',
                'description': 'Spots matinaux parfaits pour café, brunch et bien commencer la journée',
                'seo_description_template': 'Best breakfast and brunch spots in {city} for perfect morning meals',
                'required_tags': ['morning-spot'],
                'excluded_tags': ['evening-spot'],
                'min_confidence': 0.5,
                'priority': 7
            },
            'ambiances_soiree': {
                'title': 'Ambiances Soirée',
                'slug_template': 'best-evening-bars-{city}',
                'description': 'Lieux parfaits pour dîner, boire un verre et se détendre après une longue journée',
                'seo_description_template': 'Best evening bars and restaurants in {city} for dinner and cocktails',
                'required_tags': ['evening-spot'],
                'excluded_tags': ['morning-spot'],
                'min_confidence': 0.5,
                'priority': 8
            },
            
            # INSIDER COLLECTIONS - SEO optimized  
            'locaux_seulement': {
                'title': 'Locaux Seulement',
                'slug_template': 'local-favorite-spots-{city}',
                'description': 'Pépites de quartier authentiques adorées des habitués - loin des radars touristiques',
                'seo_description_template': 'Hidden local favorites in {city} loved by residents but unknown to tourists',
                'required_tags': ['local-favorite'],
                'excluded_tags': ['tourist-friendly'],
                'min_confidence': 0.6,
                'priority': 9
            },
            'createurs_ambiance': {
                'title': 'Créateurs d\'Ambiance',
                'slug_template': 'trendy-hotspots-{city}',
                'description': 'Lieux branchés qui définissent la culture contemporaine de la ville - là où naissent les tendances',
                'seo_description_template': '{city}\'s trendiest restaurants and bars where the cool crowd gathers',
                'required_tags': ['trendy'],
                'excluded_tags': ['authentic', 'peaceful'],
                'min_confidence': 0.5,
                'priority': 10
            },
            'uniques_en_leur_genre': {
                'title': 'Uniques en Leur Genre',
                'slug_template': 'unique-restaurants-{city}',
                'description': 'Lieux distinctifs au caractère unique qu\'on ne trouve nulle part ailleurs',
                'seo_description_template': 'Most unique and distinctive restaurants in {city} with special character',
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
                    
                    # Generate city-specific slug and description
                    city_slug = city.lower().replace(' ', '-').replace('ème', 'e')
                    slug = template.get('slug_template', template['title'].lower().replace(' ', '-')).format(city=city_slug)
                    seo_description = template.get('seo_description_template', template['description']).format(city=city)
                    
                    # Determine country based on city
                    country = self._get_country_for_city(city)
                    
                    collection_data = {
                        'title': template['title'],
                        'type': 'contextual',
                        'description': template['description'],
                        'city': city,
                        'country': country,
                        'poi_ids': [poi['id'] for poi in selected_pois],
                        'cover_photo': cover_photo,
                        'required_tags': template['required_tags'],
                        'excluded_tags': template['excluded_tags'],
                        'min_confidence': template['min_confidence'],
                        'metadata': {
                            'generated_by': 'collection_generator_v5_fixed',
                            'template_used': template_key,
                            'avg_match_score': sum(poi['match_score'] for poi in selected_pois) / len(selected_pois),
                            'poi_count': len(selected_pois),
                            'generation_timestamp': datetime.now(timezone.utc).isoformat(),
                            'seo_optimized': True,
                            'seo_data': {
                                'slug': slug,
                                'seo_description': seo_description,
                                'priority': template.get('priority', 99),
                                'seo_title': f"{template['title']} in {city} - Trendr",
                                'keywords': self._generate_seo_keywords(template_key, city),
                                'og_image': cover_photo,
                                'canonical_url': f"https://trendr.app/collections/{slug}",
                                'breadcrumbs': [
                                    {'name': 'Home', 'url': '/'},
                                    {'name': city, 'url': f'/city/{city_slug}'},
                                    {'name': 'Collections', 'url': f'/city/{city_slug}/collections'},
                                    {'name': template['title'], 'url': f'/collections/{slug}'}
                                ]
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
        """Generate collections for a city using contextual tags + AI enhancement."""
        logger.info(f"🚀 Generating collections for {city} (AI: {use_ai})")
        
        try:
            if use_ai and self.ai_client:
                logger.info("🧠 Using AI-enhanced collection generation")
                all_collections = self.generate_ai_enhanced_collections(city)
            else:
                logger.info("📋 Using template-based collection generation")
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
    
    def _get_country_for_city(self, city: str) -> str:
        """Determine country based on city name"""
        city_lower = city.lower()
        
        # French cities
        french_cities = ['paris', 'lyon', 'marseille', 'toulouse', 'nice', 'nantes', 'strasbourg', 'montpellier']
        if any(french_city in city_lower for french_city in french_cities):
            return 'France'
        
        # Canadian cities  
        canadian_cities = ['montreal', 'toronto', 'vancouver', 'ottawa', 'quebec', 'calgary', 'edmonton']
        if any(canadian_city in city_lower for canadian_city in canadian_cities):
            return 'Canada'
        
        # Default fallback
        return 'Unknown'
    
    def generate_ai_enhanced_collections(self, city: str) -> List[Dict[str, Any]]:
        """Generate AI-enhanced collections with real intelligence and trend analysis."""
        logger.info(f"🧠 Starting AI-enhanced collection generation for {city}")
        
        try:
            # Get all POIs with their social proof data for AI analysis
            pois_data = self._get_pois_with_social_context(city)
            recent_trends = self._analyze_recent_trends(city)
            seasonal_context = self._get_seasonal_context()
            
            logger.info(f"📊 Analyzing {len(pois_data)} POIs with AI")
            
            # Generate AI-powered collections
            ai_collections = []
            
            # 1. Dynamic Trending Collections (AI-detected)
            trending_collections = self._generate_trending_collections_ai(pois_data, recent_trends, city)
            ai_collections.extend(trending_collections)
            
            # 2. Seasonal Smart Collections (AI-curated)
            seasonal_collections = self._generate_seasonal_collections_ai(pois_data, seasonal_context, city)
            ai_collections.extend(seasonal_collections)
            
            # 3. Micro-Trend Collections (AI-discovered)
            micro_trend_collections = self._generate_micro_trend_collections_ai(pois_data, city)
            ai_collections.extend(micro_trend_collections)
            
            # 4. Enhanced Template Collections with AI insights
            enhanced_template_collections = self._enhance_template_collections_with_ai(city, pois_data)
            ai_collections.extend(enhanced_template_collections)
            
            logger.info(f"🎯 Generated {len(ai_collections)} AI-enhanced collections")
            return ai_collections
            
        except Exception as e:
            logger.error(f"AI collection generation failed, falling back to templates: {e}")
            return self.generate_contextual_collections(city)
    
    def _get_pois_with_social_context(self, city: str) -> List[Dict[str, Any]]:
        """Get POIs with their social proof context for AI analysis."""
        try:
            # Get POIs with tags and recent social proofs
            result = self.db.client.table('poi')\
                .select('''
                    id, name, address, neighborhood, category, rating, user_ratings_total,
                    price_level, website, tags, classified_tags, social_proof_score,
                    created_at, updated_at
                ''')\
                .eq('city', city)\
                .order('updated_at', desc=True)\
                .execute()
            
            pois = result.data
            
            # Enrich with recent social proofs
            for poi in pois:
                try:
                    proofs_result = self.db.client.table('proof_sources')\
                        .select('source_url, authority_score, content_snippet, found_at')\
                        .eq('poi_id', poi['id'])\
                        .order('found_at', desc=True)\
                        .limit(5)\
                        .execute()
                    
                    poi['recent_social_proofs'] = proofs_result.data
                except:
                    poi['recent_social_proofs'] = []
            
            return pois
            
        except Exception as e:
            logger.error(f"Error getting POIs with social context: {e}")
            return []
    
    def _analyze_recent_trends(self, city: str) -> Dict[str, Any]:
        """Analyze recent trends from social proof data."""
        try:
            # Get recent social proofs (last 30 days)
            thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()
            
            result = self.db.client.table('proof_sources')\
                .select('poi_id, source_url, authority_score, content_snippet, found_at')\
                .gte('found_at', thirty_days_ago)\
                .order('found_at', desc=True)\
                .execute()
            
            recent_proofs = result.data
            
            # Analyze trending keywords and POIs
            trending_keywords = {}
            trending_pois = {}
            
            for proof in recent_proofs:
                # Extract keywords from content
                content = proof.get('content_snippet', '').lower()
                keywords = re.findall(r'\b(nouveau|ouvert|tendance|viral|populaire|branché|hot|amazing|incredible)\b', content)
                
                for keyword in keywords:
                    trending_keywords[keyword] = trending_keywords.get(keyword, 0) + 1
                
                # Count POI mentions
                poi_id = proof['poi_id']
                trending_pois[poi_id] = trending_pois.get(poi_id, 0) + proof.get('authority_score', 0.5)
            
            return {
                'trending_keywords': dict(sorted(trending_keywords.items(), key=lambda x: x[1], reverse=True)[:10]),
                'trending_pois': dict(sorted(trending_pois.items(), key=lambda x: x[1], reverse=True)[:20]),
                'total_mentions': len(recent_proofs),
                'analysis_period': '30_days'
            }
            
        except Exception as e:
            logger.warning(f"Could not analyze recent trends: {e}")
            return {'trending_keywords': {}, 'trending_pois': {}, 'total_mentions': 0}
    
    def _get_seasonal_context(self) -> Dict[str, Any]:
        """Get current seasonal context for AI."""
        now = datetime.now()
        month = now.month
        day = now.day
        
        # Seasonal contexts
        seasonal_contexts = {
            'winter': {'months': [12, 1, 2], 'themes': ['cozy', 'warm', 'comfort', 'indoor']},
            'spring': {'months': [3, 4, 5], 'themes': ['fresh', 'outdoor', 'terrace', 'renewal']},
            'summer': {'months': [6, 7, 8], 'themes': ['terrace', 'festival', 'outdoor', 'vacation']},
            'autumn': {'months': [9, 10, 11], 'themes': ['cozy', 'wine', 'harvest', 'comfort']}
        }
        
        current_season = 'spring'  # Default
        for season, data in seasonal_contexts.items():
            if month in data['months']:
                current_season = season
                break
        
        # Special events/periods
        special_events = []
        if month == 12 and day >= 15:
            special_events.append('christmas_holiday')
        elif month == 2 and 10 <= day <= 20:
            special_events.append('valentine_period')
        elif month == 9 and day >= 20:
            special_events.append('back_to_school')
        
        return {
            'season': current_season,
            'themes': seasonal_contexts[current_season]['themes'],
            'special_events': special_events,
            'month': month,
            'day': day
        }
    
    def _generate_trending_collections_ai(self, pois_data: List[Dict], trends: Dict, city: str) -> List[Dict]:
        """Generate dynamic trending collections using AI analysis."""
        collections = []
        
        try:
            if not self.ai_client or not trends.get('trending_pois'):
                return []
            
            # Get top trending POIs
            trending_poi_ids = list(trends['trending_pois'].keys())[:15]
            trending_pois = [poi for poi in pois_data if poi['id'] in trending_poi_ids]
            
            if len(trending_pois) < 5:
                return []
            
            # AI prompt for dynamic collection creation
            ai_prompt = f"""
            Analyse ces POIs tendance à {city} et crée une collection captivante en FRANÇAIS.
            
            Données POIs: {json.dumps([{
                'name': poi['name'], 
                'category': poi['category'],
                'neighborhood': poi['neighborhood'],
                'social_proof_score': poi.get('social_proof_score', 0),
                'recent_mentions': len(poi.get('recent_social_proofs', []))
            } for poi in trending_pois[:10]], indent=2)}
            
            Mots-clés tendance: {trends['trending_keywords']}
            
            Crée une collection avec:
            1. Titre accrocheur (2-4 mots, engageant) EN FRANÇAIS
            2. Description SEO optimisée (50-80 mots) EN FRANÇAIS
            3. Angle unique qui capture POURQUOI ces lieux sont tendance
            
            Réponds en JSON format:
            {{
                "title": "Titre Collection",
                "description": "Description captivante expliquant pourquoi ces lieux sont spéciaux en ce moment",
                "angle": "perspective_unique_ou_thème"
            }}
            """
            
            # Call AI
            ai_response = self._call_ai(ai_prompt)
            if ai_response:
                try:
                    ai_data = json.loads(ai_response)
                    
                    collection = {
                        'data': {
                            'title': ai_data['title'],
                            'type': 'ai_trending',
                            'description': ai_data['description'],
                            'city': city,
                            'country': self._get_country_for_city(city),
                            'poi_ids': [poi['id'] for poi in trending_pois],
                            'cover_photo': self._select_best_cover_photo([poi['id'] for poi in trending_pois]),
                            'metadata': {
                                'generated_by': 'ai_trending_analysis',
                                'ai_angle': ai_data.get('angle'),
                                'trending_score': sum(trends['trending_pois'].values()),
                                'generation_timestamp': datetime.now(timezone.utc).isoformat(),
                                'seo_data': {
                                    'slug': f"trending-{city.lower()}-{datetime.now().strftime('%Y-%m')}",
                                    'seo_description': ai_data['description'][:160],
                                    'priority': 1
                                }
                            }
                        },
                        'is_update': False,
                        'existing_id': None
                    }
                    
                    collections.append(collection)
                    logger.info(f"🧠 AI created trending collection: {ai_data['title']}")
                    
                except json.JSONDecodeError:
                    logger.warning("AI response was not valid JSON")
            
        except Exception as e:
            logger.error(f"Error generating AI trending collections: {e}")
        
        return collections
    
    def _generate_seasonal_collections_ai(self, pois_data: List[Dict], seasonal_context: Dict, city: str) -> List[Dict]:
        """Generate seasonal collections using AI contextual understanding."""
        collections = []
        
        try:
            if not self.ai_client:
                return []
            
            season = seasonal_context['season']
            themes = seasonal_context['themes']
            special_events = seasonal_context['special_events']
            
            # Filter POIs that match seasonal themes
            seasonal_pois = []
            for poi in pois_data:
                poi_tags = poi.get('classified_tags', '[]')
                if isinstance(poi_tags, str):
                    poi_tags = json.loads(poi_tags) if poi_tags.startswith('[') else []
                
                # Check if POI matches seasonal themes
                if any(theme in str(poi_tags).lower() or theme in poi.get('category', '').lower() 
                       for theme in themes):
                    seasonal_pois.append(poi)
            
            if len(seasonal_pois) < 5:
                return []
            
            # AI prompt for seasonal collection
            events_context = f" Special period: {', '.join(special_events)}" if special_events else ""
            
            ai_prompt = f"""
            Crée une collection {season} parfaite pour {city} EN FRANÇAIS.{events_context}
            
            Thèmes saison actuelle: {themes}
            POIs disponibles: {json.dumps([{
                'name': poi['name'],
                'category': poi['category'], 
                'neighborhood': poi['neighborhood'],
                'tags': poi.get('classified_tags', [])
            } for poi in seasonal_pois[:15]], indent=2)}
            
            Crée une collection saisonnière qui:
            1. Capture l'essence de {season} à {city}
            2. Utilise un langage et des émotions spécifiques à la saison
            3. Plaît aux locaux et aux visiteurs
            4. Est SEO-optimisée pour les recherches "{season} {city}" EN FRANÇAIS
            
            Réponse JSON EN FRANÇAIS:
            {{
                "title": "Titre Collection Saisonnière",
                "description": "Description saisonnière (60-100 mots, émotionnelle, spécifique au {season})",
                "seasonal_angle": "pourquoi_parfait_pour_cette_saison"
            }}
            """
            
            ai_response = self._call_ai(ai_prompt)
            if ai_response:
                try:
                    ai_data = json.loads(ai_response)
                    
                    collection = {
                        'data': {
                            'title': ai_data['title'],
                            'type': 'ai_seasonal',
                            'description': ai_data['description'],
                            'city': city,
                            'country': self._get_country_for_city(city),
                            'poi_ids': [poi['id'] for poi in seasonal_pois[:12]],
                            'cover_photo': self._select_best_cover_photo([poi['id'] for poi in seasonal_pois[:12]]),
                            'metadata': {
                                'generated_by': 'ai_seasonal_analysis',
                                'season': season,
                                'seasonal_angle': ai_data.get('seasonal_angle'),
                                'seasonal_themes': themes,
                                'generation_timestamp': datetime.now(timezone.utc).isoformat(),
                                'seo_data': {
                                    'slug': f"{season}-{city.lower()}-{datetime.now().year}",
                                    'seo_description': ai_data['description'][:160],
                                    'priority': 2
                                }
                            }
                        },
                        'is_update': False,
                        'existing_id': None
                    }
                    
                    collections.append(collection)
                    logger.info(f"🌟 AI created seasonal collection: {ai_data['title']}")
                    
                except json.JSONDecodeError:
                    logger.warning("AI seasonal response was not valid JSON")
        
        except Exception as e:
            logger.error(f"Error generating AI seasonal collections: {e}")
        
        return collections
    
    def _generate_micro_trend_collections_ai(self, pois_data: List[Dict], city: str) -> List[Dict]:
        """Generate micro-trend collections using AI pattern detection."""
        collections = []
        
        try:
            if not self.ai_client or len(pois_data) < 10:
                return []
            
            # Analyze POI patterns for micro-trends
            neighborhood_clusters = {}
            category_patterns = {}
            
            for poi in pois_data:
                neighborhood = poi.get('neighborhood', 'Unknown')
                category = poi.get('category', 'Unknown')
                
                if neighborhood not in neighborhood_clusters:
                    neighborhood_clusters[neighborhood] = []
                neighborhood_clusters[neighborhood].append(poi)
                
                if category not in category_patterns:
                    category_patterns[category] = []
                category_patterns[category].append(poi)
            
            # Find interesting patterns
            interesting_clusters = {k: v for k, v in neighborhood_clusters.items() 
                                  if len(v) >= 5 and k != 'Unknown'}
            
            if not interesting_clusters:
                return []
            
            # AI analysis for micro-trends
            for neighborhood, neighborhood_pois in list(interesting_clusters.items())[:2]:  # Limit to 2 micro-trends
                ai_prompt = f"""
                Découvre une micro-tendance cachée dans {neighborhood}, {city} EN FRANÇAIS.
                
                POIs dans ce quartier: {json.dumps([{
                    'name': poi['name'],
                    'category': poi['category'],
                    'rating': poi.get('rating'),
                    'social_proof_score': poi.get('social_proof_score', 0)
                } for poi in neighborhood_pois[:10]], indent=2)}
                
                Trouve une micro-tendance ou un thème captivant qui relie ces lieux.
                Exemples: "Révolution café", "Renaissance vintage", "Hub bien-être"
                
                Réponse JSON EN FRANÇAIS:
                {{
                    "title": "Titre Micro-Tendance (2-4 mots)",
                    "description": "Pourquoi ce quartier/tendance est spécial (40-70 mots)",
                    "micro_trend": "tendance_spécifique_identifiée"
                }}
                """
                
                ai_response = self._call_ai(ai_prompt)
                if ai_response:
                    try:
                        ai_data = json.loads(ai_response)
                        
                        collection = {
                            'data': {
                                'title': f"{ai_data['title']} - {neighborhood}",
                                'type': 'ai_micro_trend',
                                'description': ai_data['description'],
                                'city': city,
                                'country': self._get_country_for_city(city),
                                'poi_ids': [poi['id'] for poi in neighborhood_pois[:8]],
                                'cover_photo': self._select_best_cover_photo([poi['id'] for poi in neighborhood_pois[:8]]),
                                'metadata': {
                                    'generated_by': 'ai_micro_trend_analysis',
                                    'neighborhood_focus': neighborhood,
                                    'micro_trend': ai_data.get('micro_trend'),
                                    'generation_timestamp': datetime.now(timezone.utc).isoformat(),
                                    'seo_data': {
                                        'slug': f"{ai_data['title'].lower().replace(' ', '-')}-{neighborhood.lower().replace(' ', '-')}",
                                        'seo_description': ai_data['description'],
                                        'priority': 3
                                    }
                                }
                            },
                            'is_update': False,
                            'existing_id': None
                        }
                        
                        collections.append(collection)
                        logger.info(f"🔍 AI discovered micro-trend: {ai_data['title']} in {neighborhood}")
                        
                    except json.JSONDecodeError:
                        logger.warning(f"AI micro-trend response was not valid JSON for {neighborhood}")
        
        except Exception as e:
            logger.error(f"Error generating AI micro-trend collections: {e}")
        
        return collections
    
    def _enhance_template_collections_with_ai(self, city: str, pois_data: List[Dict]) -> List[Dict]:
        """Enhance standard template collections with AI insights."""
        try:
            # Generate standard template collections first
            template_collections = self.generate_contextual_collections(city)
            
            if not self.ai_client or not template_collections:
                return template_collections
            
            # Enhance 2-3 key collections with AI
            enhanced_collections = []
            key_templates = ['romantic_date_spots', 'locals_only', 'instagram_worthy']
            
            for collection in template_collections[:3]:  # Enhance top 3 collections
                try:
                    collection_data = collection['data']
                    collection_pois = [poi for poi in pois_data if poi['id'] in collection_data['poi_ids']]
                    
                    if len(collection_pois) < 3:
                        enhanced_collections.append(collection)
                        continue
                    
                    # AI enhancement prompt
                    ai_prompt = f"""
                    Améliore cette description de collection pour être plus captivante et spécifique à {city} EN FRANÇAIS.
                    
                    Original: "{collection_data['description']}"
                    
                    POIs dans la collection: {json.dumps([{
                        'name': poi['name'],
                        'neighborhood': poi['neighborhood'],
                        'category': poi['category']
                    } for poi in collection_pois[:8]], indent=2)}
                    
                    Crée une meilleure description qui:
                    1. Est spécifique à {city} (pas générique)
                    2. Mentionne des quartiers réels ou la culture locale
                    3. Fait 60-90 mots, captivante et naturelle EN FRANÇAIS
                    4. Fait appel aux émotions et expériences
                    
                    Réponse JSON EN FRANÇAIS:
                    {{
                        "enhanced_description": "Description améliorée ici",
                        "local_angle": "ce_qui_rend_ça_spécifique_à_{city}"
                    }}
                    """
                    
                    ai_response = self._call_ai(ai_prompt)
                    if ai_response:
                        try:
                            ai_data = json.loads(ai_response)
                            collection_data['description'] = ai_data['enhanced_description']
                            collection_data['metadata']['ai_enhanced'] = True
                            collection_data['metadata']['local_angle'] = ai_data.get('local_angle')
                            collection_data['metadata']['generated_by'] = 'ai_enhanced_template'
                            
                            logger.info(f"🧠 AI enhanced: {collection_data['title']}")
                            
                        except json.JSONDecodeError:
                            logger.warning(f"AI enhancement failed for {collection_data['title']}")
                    
                    enhanced_collections.append(collection)
                    
                except Exception as e:
                    logger.warning(f"Error enhancing collection: {e}")
                    enhanced_collections.append(collection)
            
            # Add remaining collections as-is
            enhanced_collections.extend(template_collections[3:])
            return enhanced_collections
            
        except Exception as e:
            logger.error(f"Error enhancing template collections: {e}")
            return template_collections
    
    def _call_ai(self, prompt: str, max_tokens: int = 500) -> Optional[str]:
        """Call AI API with fallback handling."""
        try:
            if not self.ai_client:
                return None
            
            if hasattr(self.ai_client, 'chat'):  # OpenAI
                response = self.ai_client.chat.completions.create(
                    model="gpt-4o-mini",  # Cost-efficient model
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.7
                )
                return response.choices[0].message.content
                
            elif hasattr(self.ai_client, 'messages'):  # Anthropic
                response = self.ai_client.messages.create(
                    model="claude-3-haiku-20240307",  # Cost-efficient model
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            
            return None
            
        except Exception as e:
            logger.warning(f"AI call failed: {e}")
            return None
    
    def _generate_seo_keywords(self, template_key: str, city: str) -> List[str]:
        """Generate SEO keywords for collections."""
        city_lower = city.lower()
        
        base_keywords = {
            'spots_romantiques': [
                f"restaurants romantiques {city_lower}",
                f"soirée en amoureux {city_lower}",
                f"dîner intime {city_lower}",
                f"restaurants couples {city_lower}",
                f"bars romantiques {city_lower}"
            ],
            'spots_instagrammables': [
                f"lieux instagrammables {city_lower}",
                f"spots photos {city_lower}",
                f"cafés esthétiques {city_lower}",
                f"restaurants photogéniques {city_lower}",
                f"cafés instagram {city_lower}"
            ],
            'locaux_seulement': [
                f"favoris locaux {city_lower}",
                f"pépites cachées {city_lower}",
                f"restaurants authentiques {city_lower}",
                f"secrets locaux {city_lower}",
                f"hors sentiers battus {city_lower}"
            ],
            'champions_petit_dejeuner': [
                f"meilleur petit déjeuner {city_lower}",
                f"spots brunch {city_lower}",
                f"cafés matinaux {city_lower}",
                f"coffee shops {city_lower}",
                f"restaurants petit déjeuner {city_lower}"
            ],
            'cafes_nomades_digitaux': [
                f"cafés coworking {city_lower}",
                f"espaces travail {city_lower}",
                f"cafés wifi {city_lower}",
                f"lieux étude {city_lower}",
                f"télétravail {city_lower}"
            ]
        }
        
        # Get specific keywords or default
        keywords = base_keywords.get(template_key, [
            f"meilleurs restaurants {city_lower}",
            f"gastronomie {city_lower}",
            f"cuisine {city_lower}"
        ])
        
        # Add city-specific keywords
        keywords.extend([
            f"guide {city_lower}",
            f"scène culinaire {city_lower}",
            f"trendr {city_lower}",
            f"que faire {city_lower}"
        ])
        
        return keywords[:8]  # Limit to 8 keywords
    
    def generate_seo_pages_for_collections(self, city: str) -> List[Dict[str, Any]]:
        """Generate SEO page data for all collections in a city."""
        try:
            # Get all collections for the city
            result = self.db.client.table('collections')\
                .select('*')\
                .eq('city', city)\
                .execute()
            
            collections = result.data
            seo_pages = []
            
            for collection in collections:
                metadata = collection.get('metadata', {})
                seo_data = metadata.get('seo_data', {})
                
                if not seo_data.get('slug'):
                    continue
                
                # Generate comprehensive SEO page
                seo_page = {
                    'slug': seo_data['slug'],
                    'title': seo_data.get('seo_title', f"{collection['title']} - Trendr"),
                    'description': seo_data.get('seo_description', collection['description'])[:160],
                    'keywords': seo_data.get('keywords', []),
                    'canonical_url': seo_data.get('canonical_url'),
                    'og_title': seo_data.get('seo_title', collection['title']),
                    'og_description': seo_data.get('seo_description', collection['description'])[:160],
                    'og_image': seo_data.get('og_image'),
                    'og_type': 'website',
                    'twitter_card': 'summary_large_image',
                    'breadcrumbs': json.dumps(seo_data.get('breadcrumbs', [])),
                    'structured_data': json.dumps(self._generate_structured_data(collection)),
                    'city': city,
                    'country': collection.get('country', 'Unknown'),
                    'collection_id': collection['id'],
                    'priority': seo_data.get('priority', 99),
                    'last_modified': collection.get('updated_at', datetime.now().isoformat()),
                    'content_type': 'collection',
                    'estimated_read_time': self._calculate_read_time(collection),
                    'poi_count': len(collection.get('poi_ids', [])),
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'updated_at': datetime.now(timezone.utc).isoformat()
                }
                
                seo_pages.append(seo_page)
            
            # Insert/update SEO pages in database
            for seo_page in seo_pages:
                try:
                    self.db.insert_seo_page(seo_page)
                except Exception as e:
                    logger.warning(f"Could not insert SEO page for {seo_page['slug']}: {e}")
            
            logger.info(f"Generated {len(seo_pages)} SEO pages for {city} collections")
            return seo_pages
            
        except Exception as e:
            logger.error(f"Error generating SEO pages for {city}: {e}")
            return []
    
    def _generate_structured_data(self, collection: Dict[str, Any]) -> Dict[str, Any]:
        """Generate JSON-LD structured data for collections."""
        try:
            metadata = collection.get('metadata', {})
            seo_data = metadata.get('seo_data', {})
            
            # Get POI details for the collection
            poi_ids = collection.get('poi_ids', [])
            structured_pois = []
            
            if poi_ids:
                # Get first few POIs for structured data
                for poi_id in poi_ids[:5]:  # Limit to 5 POIs for performance
                    try:
                        poi_result = self.db.client.table('poi')\
                            .select('name, address, latitude, longitude, rating, category')\
                            .eq('id', poi_id)\
                            .single()\
                            .execute()
                        
                        if poi_result.data:
                            poi = poi_result.data
                            structured_poi = {
                                "@type": "Restaurant",
                                "name": poi['name'],
                                "address": poi.get('address'),
                                "geo": {
                                    "@type": "GeoCoordinates",
                                    "latitude": poi.get('latitude'),
                                    "longitude": poi.get('longitude')
                                }
                            }
                            
                            if poi.get('rating'):
                                structured_poi["aggregateRating"] = {
                                    "@type": "AggregateRating",
                                    "ratingValue": poi['rating']
                                }
                            
                            structured_pois.append(structured_poi)
                    except:
                        continue
            
            # Main structured data
            structured_data = {
                "@context": "https://schema.org",
                "@type": "CollectionPage",
                "name": collection['title'],
                "description": collection['description'],
                "url": seo_data.get('canonical_url'),
                "author": {
                    "@type": "Organization",
                    "name": "Trendr",
                    "url": "https://trendr.app"
                },
                "publisher": {
                    "@type": "Organization",
                    "name": "Trendr",
                    "url": "https://trendr.app"
                },
                "dateCreated": collection.get('created_at'),
                "dateModified": collection.get('updated_at'),
                "about": {
                    "@type": "City",
                    "name": collection['city']
                }
            }
            
            if structured_pois:
                structured_data["mainEntity"] = structured_pois
            
            if seo_data.get('og_image'):
                structured_data["image"] = seo_data['og_image']
            
            return structured_data
            
        except Exception as e:
            logger.warning(f"Error generating structured data: {e}")
            return {}
    
    def _calculate_read_time(self, collection: Dict[str, Any]) -> int:
        """Calculate estimated reading time in minutes."""
        try:
            description_words = len(collection.get('description', '').split())
            poi_count = len(collection.get('poi_ids', []))
            
            # Assume 200 words per minute reading speed
            # Description + estimated POI names/details
            total_words = description_words + (poi_count * 10)  # 10 words per POI on average
            
            read_time = max(1, round(total_words / 200))  # Minimum 1 minute
            return read_time
            
        except:
            return 2  # Default 2 minutes

def main():
    """Main entry point for collection generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Contextual Tag-Based Collections')
    parser.add_argument('--city', default='Paris', help='City to generate collections for')
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