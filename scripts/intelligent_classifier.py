#!/usr/bin/env python3
"""
Intelligent POI Classifier - Multi-Tag System
Complete modern classification using contextual tags with confidence scores.
Replaces simple mood classification with rich multi-dimensional tagging.
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
# from scripts.dynamic_neighborhoods import DynamicNeighborhoodCalculator  # Not essential
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntelligentClassifier:
    """Modern POI classifier using multi-dimensional contextual tags."""
    
    def __init__(self):
        self.db = SupabaseManager()
        # self.neighborhood_calculator = DynamicNeighborhoodCalculator()  # Not essential
        
        # Enhanced contextual tag definitions with social proof focus
        self.contextual_tag_definitions = {
            # SOCIAL PROOF AUTHORITY TAGS - NEW: Exploit our competitive advantage
            'michelin_mentioned': {
                'keywords': ['michelin', 'étoile', 'star', 'guide michelin', 'starred', 'étoilé'],
                'weight': 2.0,  # Very high weight - premium indicator
                'min_confidence': 0.4,  # Réduit pour être plus permissif
                'category': 'authority',
                'domains_boost': ['guide.michelin.com']
            },
            'media_featured': {
                'keywords': ['le figaro', 'time out', 'conde nast', 'vogue', 'elle', 'grazia', 'featured', 'article', 'press'],
                'weight': 1.5,
                'min_confidence': 0.6,
                'category': 'authority',
                'domains_boost': ['timeout.com', 'condenast.com', 'lefigaro.fr']
            },
            'food_critic_approved': {
                'keywords': ['critique', 'food critic', 'reviewed', 'évaluation', 'chronique gastronomique'],
                'weight': 1.4,
                'min_confidence': 0.7,
                'category': 'authority'
            },
            
            # TEMPORAL TAGS - Enhanced with social proof patterns
            'new_spot': {
                'keywords': ['new', 'nouveau', 'just opened', 'vient d\'ouvrir', 'recently opened', 'fresh', 'latest', 'opening'],
                'weight': 1.2,  # High weight for trend detection
                'min_confidence': 0.3,
                'category': 'temporal'
            },
            'emerging': {
                'keywords': ['emerging', 'up and coming', 'rising', 'gaining popularity', 'buzz', 'talk of the town'],
                'weight': 1.1,
                'min_confidence': 0.4,
                'category': 'temporal'
            },
            'established': {
                'keywords': ['established', 'institution', 'longtime', 'classic', 'traditional', 'institution'],
                'weight': 0.9,
                'min_confidence': 0.5,
                'category': 'temporal'
            },
            
            # CULINARY EXPERTISE TAGS - NEW: Based on social proof analysis
            'chef_driven': {
                'keywords': ['chef', 'owner-chef', 'head chef', 'executive chef', 'culinary', 'gastronomic', 'gastronomy', 
                           'chef propriétaire', 'chef étoilé', 'culinaire', 'gastronomique', 'gastronomie'],
                'weight': 1.3,
                'min_confidence': 0.6,
                'category': 'culinary'
            },
            'wine_specialist': {
                'keywords': ['wine', 'sommelier', 'wine list', 'natural wine', 'vin', 'carte des vins', 'wine bar', 'wine selection',
                           'vins naturels', 'cave à vin', 'accord mets vins', 'dégustation', 'vignoble'],
                'weight': 1.2,
                'min_confidence': 0.5,
                'category': 'culinary'
            },
            'innovative_cuisine': {
                'keywords': ['innovative', 'creative', 'inventive', 'modern', 'contemporary', 'fusion', 'avant-garde',
                           'innovant', 'créatif', 'inventif', 'moderne', 'contemporain', 'créativité culinaire'],
                'weight': 1.1,
                'min_confidence': 0.6,
                'category': 'culinary'
            },
            
            # BUSINESS VALUE TAGS - NEW: Actionable for collections
            'reservation_essential': {
                'keywords': ['reservation', 'réservation', 'booking', 'fully booked', 'complet', 'book ahead', 'réserver',
                           'réserver une table', 'indispensable de réserver', 'obligatoire de réserver'],
                'weight': 1.0,
                'min_confidence': 0.5,
                'category': 'business'
            },
            'good_value': {
                'keywords': ['value', 'affordable', 'reasonable', 'worth it', 'rapport qualité prix', 'bon rapport', 'price',
                           'abordable', 'raisonnable', 'ça vaut le coup', 'bon plan', 'pas cher', 'tarif correct'],
                'weight': 0.9,
                'min_confidence': 0.4,
                'category': 'business'
            },
            
            # Experience-based tags
            'work-friendly': {
                'keywords': ['wifi', 'laptop', 'study', 'work', 'quiet', 'productive', 'desk', 'plugs', 'internet'],
                'weight': 1.0,
                'min_confidence': 0.4,
                'category': 'experience'
            },
            'date-spot': {
                'keywords': ['romantic', 'intimate', 'cozy', 'couple', 'date', 'wine', 'candlelit', 'atmosphere'],
                'weight': 0.9,
                'min_confidence': 0.5,
                'category': 'experience'
            },
            'photo-worthy': {
                'keywords': ['beautiful', 'instagram', 'aesthetic', 'photogenic', 'decor', 'interior', 'views', 'gorgeous'],
                'weight': 0.8,
                'min_confidence': 0.4,
                'category': 'experience'
            },
            'group-friendly': {
                'keywords': ['spacious', 'large', 'groups', 'party', 'booking', 'tables', 'accommodate', 'events'],
                'weight': 0.8,
                'min_confidence': 0.5,
                'category': 'experience'
            },
            
            # Temporal tags
            'morning-spot': {
                'keywords': ['breakfast', 'coffee', 'early', 'morning', 'croissant', 'bagel', 'espresso', 'sunrise'],
                'weight': 0.9,
                'min_confidence': 0.4,
                'category': 'temporal'
            },
            'evening-spot': {
                'keywords': ['dinner', 'cocktail', 'wine', 'nightlife', 'late', 'evening', 'sunset', 'drinks'],
                'weight': 0.9,
                'min_confidence': 0.4,
                'category': 'temporal'
            },
            'weekend-spot': {
                'keywords': ['brunch', 'weekend', 'saturday', 'sunday', 'leisure', 'relaxed'],
                'weight': 0.7,
                'min_confidence': 0.5,
                'category': 'temporal'
            },
            
            # Social context tags
            'tourist-friendly': {
                'keywords': ['must-visit', 'landmark', 'popular', 'famous', 'guide', 'recommendation', 'tourist', 'top 10', 'best places', 'recommended', 'featured', 'listed'],
                'weight': 0.8,
                'min_confidence': 0.5,
                'category': 'social'
            },
            'local-favorite': {
                'keywords': ['locals', 'neighborhood', 'authentic', 'community', 'regulars', 'hidden', 'insider'],
                'weight': 0.9,
                'min_confidence': 0.6,
                'category': 'social'
            },
            'family-friendly': {
                'keywords': ['family', 'kids', 'children', 'playground', 'stroller', 'high chair', 'child'],
                'weight': 0.8,
                'min_confidence': 0.6,
                'category': 'social'
            },
            
            # Atmosphere tags
            'vibrant': {
                'keywords': ['lively', 'energetic', 'vibrant', 'bustling', 'dynamic', 'animated', 'spirited'],
                'weight': 0.8,
                'min_confidence': 0.5,
                'category': 'atmosphere'
            },
            'peaceful': {
                'keywords': ['calm', 'peaceful', 'serene', 'tranquil', 'zen', 'meditation', 'quiet'],
                'weight': 0.8,
                'min_confidence': 0.5,
                'category': 'atmosphere'
            },
            'trendy': {
                'keywords': ['hip', 'trendy', 'modern', 'stylish', 'contemporary', 'fashionable', 'chic'],
                'weight': 0.9,
                'min_confidence': 0.4,
                'category': 'atmosphere'
            },
            'authentic': {
                'keywords': ['traditional', 'authentic', 'genuine', 'original', 'classic', 'heritage'],
                'weight': 0.9,
                'min_confidence': 0.6,
                'category': 'atmosphere'
            },
            
            # Quality indicators
            'high-quality': {
                'keywords': ['excellent', 'outstanding', 'exceptional', 'premium', 'top-notch', 'superb', 'amazing', 'best', 'top 10', 'top-rated', 'highly rated', 'great', 'fantastic', 'wonderful'],
                'weight': 1.0,
                'min_confidence': 0.6,
                'category': 'quality'
            },
            'budget-friendly': {
                'keywords': ['affordable', 'cheap', 'budget', 'inexpensive', 'value', 'reasonable'],
                'weight': 0.7,
                'min_confidence': 0.5,
                'category': 'quality'
            },
            'unique': {
                'keywords': ['unique', 'unusual', 'special', 'different', 'distinctive', 'one-of-a-kind'],
                'weight': 0.9,
                'min_confidence': 0.5,
                'category': 'quality'
            }
        }
        
        # Primary mood definitions (backward compatibility)
        self.primary_mood_keywords = {
            'chill': ['cozy', 'quiet', 'relaxed', 'intimate', 'peaceful', 'calm', 'comfortable', 'laid-back'],
            'trendy': ['hip', 'popular', 'buzzing', 'vibrant', 'hotspot', 'trendy', 'stylish', 'modern'],
            'hidden_gem': ['hidden', 'gem', 'secret', 'undiscovered', 'local', 'authentic', 'unique', 'special']
        }
    
    def extract_all_contextual_tags(self, text_content: str, poi_name: str = "", 
                                  proof_sources: List[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        """Extract all contextual tags from text content and social proof sources with confidence scores."""
        detected_tags = {}
        text_lower = text_content.lower()
        
        # Split into sentences for context analysis
        sentences = re.split(r'[.!?]+', text_content)
        
        # NEW: Analyze social proof sources for domain authority and richer content
        domain_authority_boost = {}
        source_content = text_content  # Start with base content
        
        if proof_sources:
            for proof in proof_sources:
                domain = proof.get('domain', '')
                snippet = proof.get('snippet', '')
                page_title = proof.get('page_title', '')
                
                # Add proof source content to analysis
                source_content += f" {snippet} {page_title}"
                
                # Domain authority boosting
                if domain in ['guide.michelin.com', 'timeout.com', 'zagat.com']:
                    domain_authority_boost[domain] = 2.0
                elif domain in ['tripadvisor.com', 'yelp.com', 'foursquare.com']:
                    domain_authority_boost[domain] = 1.3
                elif domain.endswith(('.fr', '.com')) and 'food' in domain:
                    domain_authority_boost[domain] = 1.2
        
        # Enhanced text analysis with social proof content
        enhanced_text_lower = source_content.lower()
        
        for tag_name, tag_config in self.contextual_tag_definitions.items():
            tag_score = 0.0
            matched_keywords = []
            sources_count = 0
            domain_boost_applied = False
            
            # DEBUG pour Michelin
            if tag_name == 'michelin_mentioned':
                print(f"DEBUG: Testing {tag_name} with keywords: {tag_config['keywords']}")
                print(f"DEBUG: Text contains: {[kw for kw in tag_config['keywords'] if kw in enhanced_text_lower]}")
            
            for keyword in tag_config['keywords']:
                if keyword in enhanced_text_lower:
                    # Context validation - ensure it's about the POI
                    poi_context_weight = 0.4  # Default weight
                    
                    if poi_name:
                        for sentence in sentences:
                            sentence_lower = sentence.lower().strip()
                            if not sentence_lower:
                                continue
                            
                            if keyword in sentence_lower:
                                # Check if POI name is in the same sentence
                                poi_words = [word.lower() for word in poi_name.split() 
                                           if len(word) > 3 and word.lower() not in ['cafe', 'café', 'bar', 'restaurant']]
                                
                                if (poi_name.lower() in sentence_lower or 
                                    (poi_words and any(word in sentence_lower for word in poi_words))):
                                    poi_context_weight = 1.0
                                    break
                    else:
                        poi_context_weight = 0.7  # Default when no POI name
                    
                    # NEW: Apply domain boost for authority tags
                    domain_multiplier = 1.0
                    if tag_config.get('domains_boost') and proof_sources:
                        for proof in proof_sources:
                            if (proof.get('domain') in tag_config['domains_boost'] and 
                                keyword in proof.get('snippet', '').lower()):
                                domain_multiplier = 2.0
                                domain_boost_applied = True
                                break
                    
                    # General domain authority boost
                    for domain, boost in domain_authority_boost.items():
                        if proof_sources:
                            for proof in proof_sources:
                                if (proof.get('domain') == domain and 
                                    keyword in proof.get('snippet', '').lower()):
                                    domain_multiplier = max(domain_multiplier, boost)
                                    break
                    
                    keyword_contribution = tag_config['weight'] * poi_context_weight * domain_multiplier
                    tag_score += keyword_contribution
                    matched_keywords.append(keyword)
                    sources_count += 1
            
            # Calculate confidence based on score and keyword diversity
            if matched_keywords and tag_score > 0:
                # Normalize confidence: more keywords and higher scores = higher confidence
                raw_confidence = min(1.0, tag_score / 2.0)
                keyword_diversity = len(set(matched_keywords)) / len(tag_config['keywords'])
                
                # Final confidence combines raw score with keyword diversity
                final_confidence = (raw_confidence * 0.7) + (keyword_diversity * 0.3)
                
                # DEBUG: Pour identifier le problème
                if tag_name == 'michelin_mentioned' and matched_keywords:
                    print(f"DEBUG {tag_name}: score={tag_score:.2f}, raw_conf={raw_confidence:.2f}, diversity={keyword_diversity:.2f}, final={final_confidence:.2f}, min_req={tag_config['min_confidence']:.2f}")
                
                if final_confidence >= tag_config['min_confidence']:
                    detected_tags[tag_name] = {
                        'confidence': round(final_confidence, 3),
                        'score': round(tag_score, 3),
                        'matched_keywords': list(set(matched_keywords)),
                        'sources_count': sources_count,
                        'category': tag_config['category']
                    }
        
        return detected_tags
    
    def analyze_poi_temporal_status(self, poi: Dict[str, Any], proof_sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze POI temporal status based on age and social proof patterns."""
        
        # Extract POI age
        poi_age_days = None
        poi_creation_date = None
        
        if poi.get('created_at'):
            try:
                poi_creation_date = datetime.fromisoformat(poi['created_at'].replace('Z', '+00:00'))
                poi_age_days = (datetime.now(timezone.utc) - poi_creation_date).days
            except:
                pass
        
        # Analyze proof source freshness patterns
        recent_mentions = 0
        total_mentions = len(proof_sources)
        fresh_high_authority = 0
        
        for source in proof_sources:
            # Check if source has freshness data (from enhanced scanner temporal_data)
            temporal_data = {}
            if source.get('temporal_data'):
                try:
                    temporal_data = json.loads(source['temporal_data'])
                except:
                    pass
            
            if temporal_data.get('content_date'):
                try:
                    content_date = datetime.fromisoformat(temporal_data['content_date'])
                    days_old = (datetime.now(timezone.utc) - content_date).days
                    
                    if days_old <= 90:  # Recent content
                        recent_mentions += 1
                        if source.get('authority_score') == 'High':
                            fresh_high_authority += 1
                except:
                    pass
            
            # Also check freshness_score from temporal_data if available
            elif temporal_data.get('freshness_score', 0) > 0.7:
                recent_mentions += 1
                if source.get('authority_score') == 'High':
                    fresh_high_authority += 1
        
        # Determine temporal status
        temporal_status = 'established'  # Default
        confidence = 0.5
        reasoning = []
        
        if poi_age_days is not None:
            if poi_age_days < 90:  # < 3 months
                if recent_mentions >= 2:
                    temporal_status = 'new_spot'
                    confidence = 0.9
                    reasoning.append(f"Very recent POI ({poi_age_days} days) with {recent_mentions} recent mentions")
                else:
                    temporal_status = 'new_spot'
                    confidence = 0.6
                    reasoning.append(f"Recent POI ({poi_age_days} days) but little buzz")
                    
            elif poi_age_days < 365:  # 3-12 months
                if recent_mentions >= 3:
                    temporal_status = 'emerging'
                    confidence = 0.8
                    reasoning.append(f"Emerging POI ({poi_age_days} days) with recent momentum")
                elif recent_mentions >= 1:
                    temporal_status = 'emerging'
                    confidence = 0.6
                    reasoning.append(f"Emerging POI with continuous mentions")
                
            else:  # > 1 year
                if recent_mentions >= 2:
                    temporal_status = 'established'
                    confidence = 0.7
                    reasoning.append(f"Established POI ({poi_age_days} days) maintaining relevance")
                else:
                    temporal_status = 'established'
                    confidence = 0.4
                    reasoning.append(f"Established POI with little recent activity")
        
        else:
            # No POI age data - infer from social proof patterns
            if recent_mentions >= 3:
                temporal_status = 'emerging'
                confidence = 0.6
                reasoning.append("Unknown age but recent buzz suggests emergence")
            elif total_mentions <= 2:
                temporal_status = 'new_spot'  
                confidence = 0.4
                reasoning.append("Few total mentions - probably new")
        
        # Boost confidence if high-authority recent mentions
        if fresh_high_authority >= 2:
            confidence = min(confidence + 0.2, 1.0)
            reasoning.append(f"{fresh_high_authority} recent high authority mentions")
        
        return {
            'temporal_status': temporal_status,
            'confidence': confidence,
            'poi_age_days': poi_age_days,
            'recent_mentions_count': recent_mentions,
            'total_mentions_count': total_mentions,
            'fresh_high_authority_count': fresh_high_authority,
            'reasoning': ' | '.join(reasoning)
        }
    
    def determine_primary_mood(self, contextual_tags: Dict[str, Dict[str, Any]], 
                             social_proof_text: str) -> Tuple[str, float]:
        """Determine primary mood from contextual tags and social proof."""
        
        # Score each primary mood based on contextual tags and keywords
        mood_scores = {'chill': 0.0, 'trendy': 0.0, 'hidden_gem': 0.0}
        
        # Map contextual tags to primary moods
        tag_to_mood_mapping = {
            'peaceful': 'chill',
            'work-friendly': 'chill',
            'authentic': 'hidden_gem',
            'local-favorite': 'hidden_gem',
            'unique': 'hidden_gem',
            'trendy': 'trendy',
            'vibrant': 'trendy',
            'tourist-friendly': 'trendy',
            'photo-worthy': 'trendy'
        }
        
        # Score based on contextual tags
        for tag_name, tag_data in contextual_tags.items():
            confidence = tag_data['confidence']
            if tag_name in tag_to_mood_mapping:
                target_mood = tag_to_mood_mapping[tag_name]
                mood_scores[target_mood] += confidence * 0.5
        
        # Score based on direct keyword analysis
        text_lower = social_proof_text.lower()
        for mood, keywords in self.primary_mood_keywords.items():
            keyword_score = sum(0.2 for keyword in keywords if keyword in text_lower)
            mood_scores[mood] += keyword_score
        
        # Determine winning mood
        if all(score < 0.1 for score in mood_scores.values()):
            # Fallback based on contextual tag categories
            category_scores = Counter()
            for tag_data in contextual_tags.values():
                category_scores[tag_data['category']] += tag_data['confidence']
            
            if category_scores:
                top_category = category_scores.most_common(1)[0][0]
                if top_category in ['experience', 'social']:
                    winning_mood = 'trendy'
                elif top_category in ['atmosphere', 'quality']:
                    winning_mood = 'hidden_gem'  
                else:
                    winning_mood = 'chill'
            else:
                winning_mood = 'chill'  # Default
            confidence = 0.3  # Low confidence fallback
        else:
            winning_mood = max(mood_scores.keys(), key=lambda k: mood_scores[k])
            max_score = mood_scores[winning_mood]
            confidence = min(1.0, max_score / 2.0) if max_score > 0 else 0.3
        
        return winning_mood, confidence
    
    def classify_poi(self, poi: Dict[str, Any], 
                    proof_sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Modern POI classification with temporal analysis and contextual tags."""
        
        poi_id = poi['id']
        poi_name = poi.get('name', '')
        
        # Aggregate all text from proof sources
        all_text = []
        source_analysis = []
        
        for source in proof_sources:
            title = source.get('title', '')
            snippet = source.get('snippet', '')
            domain = source.get('domain', '')
            authority = source.get('authority_score', 'Low')
            
            text_content = f"{title} {snippet}".strip()
            if text_content:
                all_text.append(text_content)
                analysis_item = {
                    'domain': domain,
                    'authority': authority,
                    'text_length': len(text_content),
                    'content': text_content,
                    'freshness_score': 0.3,  # Default
                    'content_date': None
                }
                
                # Extract temporal data if available
                if source.get('temporal_data'):
                    try:
                        temporal_data = json.loads(source['temporal_data'])
                        analysis_item['freshness_score'] = temporal_data.get('freshness_score', 0.3)
                        analysis_item['content_date'] = temporal_data.get('content_date')
                    except:
                        pass
                
                source_analysis.append(analysis_item)
        
        combined_text = ' '.join(all_text)
        
        if not combined_text.strip():
            # No social proof - return minimal classification with temporal analysis
            temporal_analysis = self.analyze_poi_temporal_status(poi, proof_sources)
            return {
                'primary_mood': 'chill',  # Default
                'mood_confidence': 0.2,
                'contextual_tags': {},
                'temporal_status': temporal_analysis['temporal_status'],
                'temporal_confidence': temporal_analysis['confidence'],
                'classification_metadata': {
                    'sources_analyzed': 0,
                    'classification_method': 'fallback_no_social_proof',
                    'reasoning': 'No social proof available - assigned default chill mood',
                    'temporal_analysis': temporal_analysis,
                    'classified_at': datetime.now(timezone.utc).isoformat()
                }
            }
        
        # NEW: Analyze temporal status first
        temporal_analysis = self.analyze_poi_temporal_status(poi, proof_sources)
        
        # Extract all contextual tags with social proof analysis
        contextual_tags = self.extract_all_contextual_tags(combined_text, poi_name, proof_sources)
        
        # ADD temporal tag based on analysis
        if temporal_analysis['confidence'] > 0.5:
            temporal_tag = temporal_analysis['temporal_status']
            contextual_tags[temporal_tag] = {
                'confidence': temporal_analysis['confidence'],
                'score': temporal_analysis['confidence'],
                'matched_keywords': ['temporal_analysis'],
                'sources_count': temporal_analysis['total_mentions_count'],
                'category': 'temporal'
            }
        
        # Determine primary mood from contextual tags
        primary_mood, mood_confidence = self.determine_primary_mood(contextual_tags, combined_text)
        
        # Get neighborhood context for additional insights
        neighborhood_name = poi.get('neighborhood', '')
        neighborhood_context = {}
        if neighborhood_name:
            try:
                neighborhood_dist = self.neighborhood_calculator.get_neighborhood_distribution_by_name(neighborhood_name)
                if neighborhood_dist:
                    neighborhood_context = {
                        'name': neighborhood_name,
                        'mood_distribution': neighborhood_dist,
                        'dominant_mood': max(neighborhood_dist.keys(), key=lambda k: neighborhood_dist[k])
                    }
            except Exception as e:
                logger.debug(f"Could not get neighborhood context: {e}")
        
        # Generate comprehensive reasoning
        reasoning_parts = [
            f"Classified as '{primary_mood}' (confidence: {mood_confidence:.2f})",
            f"Analyzed {len(proof_sources)} social proof sources",
            f"Extracted {len(contextual_tags)} contextual tags"
        ]
        
        if contextual_tags:
            top_tags = sorted(contextual_tags.items(), key=lambda x: x[1]['confidence'], reverse=True)[:3]
            tag_list = [f"{tag} ({data['confidence']:.2f})" for tag, data in top_tags]
            reasoning_parts.append(f"Top contextual tags: {', '.join(tag_list)}")
        
        if neighborhood_context:
            reasoning_parts.append(f"Neighborhood context: {neighborhood_context['name']} ({neighborhood_context['dominant_mood']} dominant)")
        
        reasoning = '\n'.join(reasoning_parts)
        
        return {
            'primary_mood': primary_mood,
            'mood_confidence': mood_confidence,
            'contextual_tags': contextual_tags,
            'temporal_status': temporal_analysis['temporal_status'],
            'temporal_confidence': temporal_analysis['confidence'],
            'classification_metadata': {
                'sources_analyzed': len(proof_sources),
                'total_text_length': len(combined_text),
                'classification_method': 'multi_tag_temporal_v4',
                'neighborhood_context': neighborhood_context,
                'source_breakdown': {
                    'high_authority': len([s for s in proof_sources if s.get('authority_score') == 'High']),
                    'medium_authority': len([s for s in proof_sources if s.get('authority_score') == 'Medium']),
                    'low_authority': len([s for s in proof_sources if s.get('authority_score') == 'Low'])
                },
                'temporal_analysis': temporal_analysis,
                'reasoning': reasoning,
                'classified_at': datetime.now(timezone.utc).isoformat()
            }
        }
    
    def update_poi_in_database(self, poi_id: str, classification_result: Dict[str, Any]) -> bool:
        """Update POI in database with classification results - FIXED to use correct columns."""
        try:
            # Use the CORRECT column names that exist in the schema
            update_data = {
                'primary_mood': classification_result['primary_mood'],
                'mood_confidence': int(classification_result['mood_confidence'] * 100),  # Convert to 0-100
                'tags': classification_result['contextual_tags'],
                'classification_data': classification_result['classification_metadata'], 
                'last_classified': datetime.now(timezone.utc).isoformat()
            }
            
            result = self.db.client.table('poi')\
                .update(update_data)\
                .eq('id', poi_id)\
                .execute()
            
            if result.data:
                logger.info(f"✅ Updated POI {poi_id} with mood: {classification_result['primary_mood']}")
                return True
            else:
                logger.warning(f"❌ No rows updated for POI {poi_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Database update failed for POI {poi_id}: {e}")
            return False
    
    def classify_single_poi(self, poi_id: str) -> Optional[Dict[str, Any]]:
        """Classify a single POI using modern multi-tag system."""
        try:
            # Get POI data
            poi_data = self.db.client.table('poi')\
                .select('*')\
                .eq('id', poi_id)\
                .single()\
                .execute()
            
            if not poi_data.data:
                logger.error(f"POI {poi_id} not found")
                return None
                
            poi = poi_data.data
            
            # Get proof sources
            proof_sources = self.db.get_proof_sources_for_poi(poi_id)
            
            # Classify using modern system
            classification_result = self.classify_poi(poi, proof_sources)
            
            # Update database
            success = self.update_poi_in_database(poi_id, classification_result)
            
            if success:
                logger.info(f"✅ {poi['name']}: Classified as {classification_result['primary_mood']} with {len(classification_result['contextual_tags'])} tags")
                return classification_result
            else:
                logger.error(f"❌ Failed to update {poi['name']} in database")
                return None
                
        except Exception as e:
            logger.error(f"Error classifying POI {poi_id}: {e}")
            return None
    
    def bulk_classify_city(self, city: str = 'Montreal', limit: int = 50) -> Dict[str, Any]:
        """Bulk classify POIs using modern multi-tag system."""
        logger.info(f"🚀 Starting modern bulk classification for {city}")
        
        # Get POIs with social proof
        pois = self.db.get_pois_for_city(city, limit)
        
        results = {
            'pois_processed': 0,
            'pois_updated': 0,
            'pois_failed': 0,
            'total_contextual_tags': 0,
            'primary_mood_distribution': Counter(),
            'contextual_tag_distribution': Counter(),
            'avg_mood_confidence': 0.0,
            'classification_details': []
        }
        
        total_confidence = 0.0
        
        for poi in pois:
            try:
                poi_id = poi['id']
                poi_name = poi['name']
                
                # Get proof sources
                proof_sources = self.db.get_proof_sources_for_poi(poi_id)
                
                if not proof_sources:
                    logger.warning(f"⚠️ {poi_name}: No social proof sources, skipping")
                    continue
                
                # Classify with modern system
                classification = self.classify_poi(poi, proof_sources)
                
                # Update database
                success = self.update_poi_in_database(poi_id, classification)
                
                # Update statistics
                results['pois_processed'] += 1
                
                if success:
                    results['pois_updated'] += 1
                    results['primary_mood_distribution'][classification['primary_mood']] += 1
                    results['total_contextual_tags'] += len(classification['contextual_tags'])
                    
                    # Track contextual tag usage
                    for tag_name in classification['contextual_tags'].keys():
                        results['contextual_tag_distribution'][tag_name] += 1
                    
                    total_confidence += classification['mood_confidence']
                    
                    logger.info(f"✅ {poi_name}: {classification['primary_mood']} ({len(classification['contextual_tags'])} tags)")
                else:
                    results['pois_failed'] += 1
                    logger.error(f"❌ {poi_name}: Database update failed")
                
                # Store detailed results
                results['classification_details'].append({
                    'poi_id': poi_id,
                    'poi_name': poi_name,
                    'primary_mood': classification['primary_mood'],
                    'mood_confidence': classification['mood_confidence'],
                    'contextual_tags_count': len(classification['contextual_tags']),
                    'top_contextual_tags': list(classification['contextual_tags'].keys())[:3],
                    'success': success
                })
                
            except Exception as e:
                results['pois_failed'] += 1
                logger.error(f"❌ Error processing {poi.get('name', 'Unknown')}: {e}")
                continue
        
        # Calculate averages
        if results['pois_updated'] > 0:
            results['avg_mood_confidence'] = total_confidence / results['pois_updated']
            results['avg_contextual_tags_per_poi'] = results['total_contextual_tags'] / results['pois_updated']
        
        # Final summary
        logger.info(f"🎯 Modern classification complete!")
        logger.info(f"  📊 POIs processed: {results['pois_processed']}")
        logger.info(f"  ✅ POIs updated: {results['pois_updated']}")
        logger.info(f"  ❌ POIs failed: {results['pois_failed']}")
        logger.info(f"  🏷️ Total contextual tags: {results['total_contextual_tags']}")
        logger.info(f"  📈 Avg mood confidence: {results['avg_mood_confidence']:.2f}")
        
        # Show top contextual tags
        if results['contextual_tag_distribution']:
            top_tags = results['contextual_tag_distribution'].most_common(5)
            logger.info(f"  🔝 Top contextual tags: {dict(top_tags)}")
        
        return results

def main():
    """CLI interface for intelligent classification system."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Intelligent POI Classifier - Multi-Tag System')
    parser.add_argument('--city', default='Montreal', help='City to classify')
    parser.add_argument('--poi-id', help='Classify specific POI by ID')
    parser.add_argument('--poi-name', help='Classify specific POI by name')
    parser.add_argument('--bulk', action='store_true', help='Bulk classify all POIs')
    parser.add_argument('--limit', type=int, default=50, help='Limit for bulk classification')
    parser.add_argument('--test', action='store_true', help='Test mode - classify only 3 POIs')
    
    args = parser.parse_args()
    
    classifier = IntelligentClassifier()
    
    try:
        if args.test:
            logger.info("🧪 TEST MODE: Classification on 3 POIs")
            results = classifier.bulk_classify_city(args.city, limit=3)
            
            print(f"\n🚀 Classification Test Results:")
            print(f"  📊 POIs updated: {results['pois_updated']}")
            print(f"  🏷️ Total contextual tags: {results['total_contextual_tags']}")
            print(f"  📈 Avg confidence: {results['avg_mood_confidence']:.2f}")
            
            if results['classification_details']:
                print(f"\n📍 Sample classifications:")
                for detail in results['classification_details'][:3]:
                    tags = ', '.join(detail['top_contextual_tags'])
                    print(f"    {detail['poi_name']}: {detail['primary_mood']} + [{tags}]")
                    
        elif args.bulk:
            results = classifier.bulk_classify_city(args.city, args.limit)
            
            print(f"\n🚀 Bulk Classification Results:")
            for key, value in results.items():
                if key not in ['classification_details']:
                    print(f"  {key}: {value}")
            
        elif args.poi_id:
            result = classifier.classify_single_poi(args.poi_id)
            if result:
                print(f"\n✅ POI Classification:")
                print(f"  Primary mood: {result['primary_mood']} ({result['mood_confidence']:.2f})")
                print(f"  Contextual tags: {len(result['contextual_tags'])}")
                for tag, data in result['contextual_tags'].items():
                    print(f"    - {tag}: {data['confidence']:.2f}")
            else:
                print("❌ Classification failed")
                
        elif args.poi_name:
            # Find POI by name
            pois = classifier.db.get_pois_by_name(args.poi_name, args.city)
            if pois:
                poi = pois[0]
                result = classifier.classify_single_poi(poi['id'])
                if result:
                    print(f"\n✅ {args.poi_name} Classification:")
                    print(f"  Primary mood: {result['primary_mood']} ({result['mood_confidence']:.2f})")
                    print(f"  Contextual tags ({len(result['contextual_tags'])}):") 
                    for tag, data in result['contextual_tags'].items():
                        print(f"    - {tag}: {data['confidence']:.2f} ({data['category']})")
                else:
                    print("❌ Classification failed")
            else:
                print(f"❌ POI '{args.poi_name}' not found in {args.city}")
        
        else:
            print("Use --bulk, --poi-id, --poi-name, or --test")
    
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()