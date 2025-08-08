#!/usr/bin/env python3
"""
Enhanced Proof Sources Scanner - Step 1 of Social Proof Enhancement
Improved multi-source web crawling with better authority scoring.
Focus: Simple but effective improvement over basic scanner.
"""
import sys
import os
import logging
import requests
import json
import time
import re
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse
from dateutil import parser as date_parser
import dateutil

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database import SupabaseManager
from utils.api_cache import google_search_cache
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedProofScanner:
    """Enhanced social proof scanner with multi-source and better scoring."""
    
    def __init__(self):
        self.db = SupabaseManager()
        self.api_key = config.GOOGLE_CUSTOM_SEARCH_API_KEY
        self.search_engine_id = config.GOOGLE_CUSTOM_SEARCH_ENGINE_ID
        
        # API optimization configuration
        self.max_queries_per_poi = 3  # Reduced from 7 to 3
        self.daily_api_limit = 95     # Keep 5 requests as margin
        self.queries_used_today = 0
        
        # Priority sources - STRICT high-authority only approach
        self.priority_sources = {
            'high_authority': [
                'cbc.ca', 'thestar.com', 'timeout.com', 'eater.com', 
                'theglobeandmail.com', 'lapresse.ca', 'montreal.eater.com'
            ],
            'local_authority': [
                'blogto.com', 'narcity.com', 'mtlblog.com', 'dailyhive.com',
                'curiocity.com', 'cultmtl.com'
            ],
            'food_lifestyle': [
                'foodnetwork.ca', 'chatelaine.com', 'buzzfeed.com',
                'refinery29.com'
            ]
        }
        
        # Low-value domains to STRICTLY avoid
        self.blacklisted_domains = [
            'restomontreal.ca', 'th3rdwave.coffee', 'yelp.com', 'tripadvisor.com',
            'foursquare.com', 'zomato.com', 'yellowpages.ca', 'google.com',
            'facebook.com', 'instagram.com', 'tiktok.com', 'reddit.com',
            'mustdocanada.com', 'wheree.com', 'dessertadvisor.com', 'mindtrip.ai'
        ]
        
        # Authority scoring with DB-compatible values
        self.authority_weights = {
            'high_authority': 1.0,
            'local_authority': 0.8,
            'food_lifestyle': 0.6,
            'other': 0.3  # Low but not zero for classification
        }
        
        if not self.api_key or not self.search_engine_id:
            logger.warning("Google Custom Search API not configured")
    
    def get_source_category(self, domain: str) -> str:
        """Categorize domain by authority level."""
        domain = domain.lower()
        
        for source in self.priority_sources['high_authority']:
            if source in domain:
                return 'high_authority'
        
        for source in self.priority_sources['local_authority']:
            if source in domain:
                return 'local_authority'
                
        for source in self.priority_sources['food_lifestyle']:
            if source in domain:
                return 'food_lifestyle'
                
        return 'other'
    
    def calculate_enhanced_authority_score(self, domain: str, content_snippet: str = "") -> Dict[str, Any]:
        """Enhanced authority scoring with multiple factors."""
        category = self.get_source_category(domain)
        base_weight = self.authority_weights[category]
        
        # Content quality indicators
        quality_bonus = 0.0
        quality_keywords = ['review', 'recommendation', 'best', 'guide', 'must-visit', 'featured']
        
        if content_snippet:
            snippet_lower = content_snippet.lower()
            quality_matches = sum(1 for keyword in quality_keywords if keyword in snippet_lower)
            quality_bonus = min(quality_matches * 0.1, 0.3)  # Max 30% bonus
        
        final_score = min(base_weight + quality_bonus, 1.0)
        
        # Map to database-compatible authority levels
        if final_score >= 0.8:
            authority_level = 'High'
        elif final_score >= 0.5:
            authority_level = 'Medium'
        else:
            authority_level = 'Low'  # Changed from 'Rejected' to 'Low' for DB compatibility
        
        return {
            'category': category,
            'base_weight': base_weight,
            'quality_bonus': quality_bonus,
            'final_score': final_score,
            'authority_level': authority_level
        }
    
    def enhanced_search_for_poi(self, poi_name: str, city: str, category: str = '') -> List[Dict[str, Any]]:
        """OPTIMIZED search with caching and reduced API calls."""
        if not self.api_key or not self.search_engine_id:
            return []
        
        # Check daily limit
        if self.queries_used_today >= self.daily_api_limit:
            logger.warning(f"Daily API limit reached ({self.daily_api_limit})")
            return []
        
        try:
            # OPTIMIZED: Only 3 essential queries instead of 7
            search_queries = [
                # 1. Basic authority search - most important
                f'"{poi_name}" {city}',
                # 2. Review search - for social proof
                f'"{poi_name}" {city} review',
                # 3. Best/quality search - for ranking context
                f'best {category} {city}' if category else f'best places {city}'
            ]
            
            all_results = []
            
            for query in search_queries[:self.max_queries_per_poi]:
                # Check API limit before each query
                if self.queries_used_today >= self.daily_api_limit:
                    logger.warning("API limit reached during processing")
                    break
                
                try:
                    # CACHE CHECK FIRST
                    cached_results = google_search_cache.search_cached(query)
                    if cached_results is not None:
                        # Cache hit - no API call
                        for item in cached_results:
                            item['search_query'] = query
                        all_results.extend(cached_results)
                        logger.info(f"CACHED Query '{query}': {len(cached_results)} results")
                        continue
                    
                    # Cache miss - make API call
                    url = "https://www.googleapis.com/customsearch/v1"
                    params = {
                        'key': self.api_key,
                        'cx': self.search_engine_id,
                        'q': query,
                        'num': 10
                    }
                    
                    response = requests.get(url, params=params)
                    response.raise_for_status()
                    
                    data = response.json()
                    self.queries_used_today += 1
                    
                    if 'items' in data:
                        # CACHE RESULTS for future use
                        google_search_cache.cache_search_results(query, data['items'])
                        
                        # Add query context to results
                        for item in data['items']:
                            item['search_query'] = query
                        all_results.extend(data['items'])
                        logger.info(f"API Query '{query}': {len(data['items'])} results (Total API calls today: {self.queries_used_today})")
                    
                    # Rate limiting - more conservative
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.warning(f"Search query '{query}' failed: {e}")
                    continue
            
            # Deduplicate by URL
            seen_urls = set()
            unique_results = []
            for item in all_results:
                url = item.get('link', '')
                if url not in seen_urls:
                    seen_urls.add(url)
                    unique_results.append(item)
            
            logger.info(f"Found {len(unique_results)} unique mentions for {poi_name}")
            return unique_results
            
        except Exception as e:
            logger.error(f"Enhanced search failed: {e}")
            return []
    
    def extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower().replace('www.', '')
        except:
            return 'unknown'
    
    def extract_content_date(self, title: str, snippet: str, url: str) -> Optional[datetime]:
        """Extract publication date from content using multiple strategies."""
        text_content = f"{title} {snippet}".lower()
        
        # Strategy 1: Look for explicit date patterns
        date_patterns = [
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{4})',  # MM/DD/YYYY or DD/MM/YYYY
            r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})',  # YYYY/MM/DD
            r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}',
            r'(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{1,2},?\s+\d{4}',
            r'\d{1,2}\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}',
            r'\d{1,2}\s+(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{4}'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text_content, re.IGNORECASE)
            if matches:
                try:
                    # Try to parse the first match
                    date_str = matches[0] if isinstance(matches[0], str) else ' '.join(matches[0])
                    parsed_date = date_parser.parse(date_str, fuzzy=True)
                    
                    # Sanity check - date should be reasonable (not in future, not too old)
                    now = datetime.now(timezone.utc)
                    if parsed_date.replace(tzinfo=timezone.utc) <= now and (now - parsed_date.replace(tzinfo=timezone.utc)).days <= 365 * 5:
                        return parsed_date.replace(tzinfo=timezone.utc)
                except:
                    continue
        
        # Strategy 2: Look for temporal keywords
        recent_keywords = {
            'today': 0, 'yesterday': 1, 'this week': 3, 'last week': 7,
            'this month': 15, 'last month': 30, 'recently': 14,
            'nouveau': 7, 'new': 7, 'just opened': 3, 'vient d\'ouvrir': 3
        }
        
        for keyword, days_ago in recent_keywords.items():
            if keyword in text_content:
                estimated_date = datetime.now(timezone.utc) - timedelta(days=days_ago)
                return estimated_date
        
        # Strategy 3: Extract from URL if possible (some sites include date in URL)
        url_date_patterns = [
            r'/(\d{4})/(\d{1,2})/(\d{1,2})/',  # /YYYY/MM/DD/
            r'/(\d{4})-(\d{1,2})-(\d{1,2})-',  # /YYYY-MM-DD-
        ]
        
        for pattern in url_date_patterns:
            match = re.search(pattern, url)
            if match:
                try:
                    year, month, day = map(int, match.groups())
                    url_date = datetime(year, month, day, tzinfo=timezone.utc)
                    now = datetime.now(timezone.utc)
                    if url_date <= now and (now - url_date).days <= 365 * 3:
                        return url_date
                except:
                    continue
        
        return None  # Could not determine date
    
    def calculate_freshness_score(self, content_date: Optional[datetime], poi_creation_date: Optional[datetime] = None) -> float:
        """Calculate freshness score based on content age and POI age."""
        if not content_date:
            return 0.3  # Unknown date = low freshness
        
        now = datetime.now(timezone.utc)
        
        # Ensure content_date has timezone info
        if content_date.tzinfo is None:
            content_date = content_date.replace(tzinfo=timezone.utc)
        
        content_age_days = (now - content_date).days
        
        # Base freshness score by content age
        if content_age_days <= 7:
            base_freshness = 1.0      # Very fresh
        elif content_age_days <= 30:
            base_freshness = 0.9      # Fresh
        elif content_age_days <= 90:
            base_freshness = 0.7      # Recent
        elif content_age_days <= 180:
            base_freshness = 0.5      # Somewhat recent
        elif content_age_days <= 365:
            base_freshness = 0.3      # Getting old
        else:
            base_freshness = 0.1      # Old content
        
        # Adjust based on POI age if available
        if poi_creation_date:
            # Ensure POI creation date has timezone info
            if poi_creation_date.tzinfo is None:
                poi_creation_date = poi_creation_date.replace(tzinfo=timezone.utc)
            
            poi_age_days = (now - poi_creation_date).days
            
            # New POI (< 6 months) - recent content is critical
            if poi_age_days < 180:
                if content_age_days <= 30:
                    base_freshness *= 1.3  # Boost fresh content for new POIs
                elif content_age_days > 90:
                    base_freshness *= 0.7  # Penalize old content for new POIs
            
            # Established POI - consistency matters more than freshness
            elif poi_age_days > 365:
                base_freshness *= 0.9  # Slight preference for established places
        
        return min(base_freshness, 1.0)
    
    def filter_quality_results(self, results: List[Dict[str, Any]], poi_name: str) -> List[Dict[str, Any]]:
        """RELAXED filtering - accept more sources to improve classification rate."""
        quality_results = []
        
        for result in results:
            domain = self.extract_domain(result.get('link', ''))
            title = result.get('title', '').lower()
            snippet = result.get('snippet', '').lower()
            
            # STRICT blacklist - reject immediately
            if any(blacklisted in domain for blacklisted in self.blacklisted_domains):
                continue
            
            # Must contain POI name in title or snippet (or key parts of POI name)
            poi_words = poi_name.lower().split()
            main_poi_words = [word for word in poi_words if len(word) > 3 and word not in ['cafe', 'café', 'bar', 'restaurant']]
            
            has_poi_mention = (poi_name.lower() in title or poi_name.lower() in snippet or
                              (main_poi_words and any(word in title or word in snippet for word in main_poi_words)))
            
            if not has_poi_mention:
                continue
            
            # RELAXED: Accept authority sources OR unknown domains with strong signals
            source_category = self.get_source_category(domain)
            
            # Quality indicators (broader than just trend)
            quality_indicators = [
                'trending', 'hotspot', 'must visit', 'best', 'buzzing', 'popular', 'recommended',
                'featured', 'top', 'essential', 'trendy', 'trending_now', 'excellent', 'great',
                'favorite', 'love', 'amazing', 'perfect', 'review', 'guide', 'new_opening', 'new'
            ]
            
            has_quality_indicator = any(indicator in title or indicator in snippet for indicator in quality_indicators)
            
            # RELAXED acceptance criteria:
            # 1. Known authority sources (always accept)
            # 2. Unknown sources with quality indicators
            # 3. Any source mentioning the POI with substantial content (length > 100 chars)
            if (source_category in ['high_authority', 'local_authority', 'food_lifestyle'] or
                has_quality_indicator or 
                len(snippet) > 100):
                quality_results.append(result)
        
        return quality_results
    
    def create_enhanced_proof_record(self, poi_id: str, search_result: Dict[str, Any], poi_creation_date: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        """Create enhanced proof source record with temporal analysis."""
        try:
            url = search_result.get('link', '')
            domain = self.extract_domain(url)
            title = search_result.get('title', '')
            snippet = search_result.get('snippet', '')
            
            # Enhanced authority scoring
            authority_analysis = self.calculate_enhanced_authority_score(domain, snippet)
            
            # NEW: Extract content publication date
            content_date = self.extract_content_date(title, snippet, url)
            
            # NEW: Calculate freshness score  
            freshness_score = self.calculate_freshness_score(content_date, poi_creation_date)
            
            # Extract context and keywords
            context = self.extract_mention_context(snippet, poi_id)
            keywords = self.extract_quality_keywords(title + ' ' + snippet)
            
            # Combine authority + freshness for final relevance score
            final_relevance = (authority_analysis['final_score'] * 0.6) + (freshness_score * 0.4)
            
            proof_data = {
                'poi_id': poi_id,
                'source_name': title[:100],  # Limit length
                'url': url,
                'domain': domain,
                'page_title': title,
                'snippet': snippet,
                'authority_score': authority_analysis['authority_level'],
                'mention_count': 1,
                'last_crawled': datetime.now(timezone.utc).isoformat(),
                'created_at': datetime.now(timezone.utc).isoformat(),
                
                # Store temporal analysis in snippet temporarily (backward compatibility)
                'snippet': f"{snippet[:400]} | TEMPORAL_SCORE:{round(freshness_score, 2)}"
            }
            
            return proof_data
            
        except Exception as e:
            logger.error(f"Error creating enhanced proof record: {e}")
            return None
    
    def extract_mention_context(self, text: str, poi_id: str) -> str:
        """Extract meaningful context around POI mention."""
        sentences = re.split(r'[.!?]+', text)
        
        # Find the most relevant sentence
        best_sentence = ""
        max_relevance = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Score sentence relevance
            relevance_keywords = ['best', 'recommend', 'must-visit', 'favorite', 'popular', 'excellent', 'amazing']
            relevance_score = sum(1 for keyword in relevance_keywords if keyword in sentence.lower())
            
            if relevance_score > max_relevance:
                max_relevance = relevance_score
                best_sentence = sentence
        
        return best_sentence if best_sentence else text[:200]
    
    def extract_quality_keywords(self, text: str) -> List[str]:
        """Extract quality and mood keywords from text."""
        text_lower = text.lower()
        
        quality_keywords = {
            'positive': ['best', 'excellent', 'amazing', 'outstanding', 'incredible', 'perfect', 'love', 'favorite'],
            'atmosphere': ['cozy', 'trendy', 'hip', 'chill', 'vibrant', 'intimate', 'bustling', 'quiet'],
            'experience': ['must-visit', 'hidden gem', 'popular', 'crowded', 'peaceful', 'lively']
        }
        
        found_keywords = []
        for category, keywords in quality_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    found_keywords.append(f"{category}:{keyword}")
        
        return found_keywords[:10]  # Limit to top 10
    
    def enhanced_scan_poi(self, poi: Dict[str, Any]) -> int:
        """Enhanced scan for a single POI with temporal analysis."""
        poi_id = poi['id']
        poi_name = poi['name']
        city = poi.get('city', 'Montreal')
        category = poi.get('category', '')
        
        # Extract POI creation date - SIMPLIFIED to avoid timezone issues
        poi_creation_date = None
        poi_age_days = None
        
        if poi.get('created_at'):
            try:
                # Try multiple parsing strategies
                created_str = poi['created_at']
                if created_str.endswith('Z'):
                    created_str = created_str.replace('Z', '+00:00')
                elif '+' not in created_str and 'T' in created_str:
                    created_str = created_str + '+00:00'
                
                poi_creation_date = datetime.fromisoformat(created_str)
                if poi_creation_date.tzinfo is None:
                    poi_creation_date = poi_creation_date.replace(tzinfo=timezone.utc)
                    
                poi_age_days = (datetime.now(timezone.utc) - poi_creation_date).days
            except Exception as e:
                # Fallback - assume POI is reasonably recent if no date
                logger.debug(f"Could not parse POI creation date {poi.get('created_at')}: {e}")
                poi_age_days = 180  # Assume 6 months old
        
        logger.info(f"🔍 Enhanced temporal scanning for: {poi_name}")
        if poi_age_days is not None:
            logger.info(f"  📅 POI age: {poi_age_days} days")
        
        # Enhanced search
        search_results = self.enhanced_search_for_poi(poi_name, city, category)
        
        if not search_results:
            logger.warning(f"No search results found for {poi_name}")
            return 0
        
        # Filter for quality
        quality_results = self.filter_quality_results(search_results, poi_name)
        logger.info(f"  📊 {len(quality_results)} quality results from {len(search_results)} total")
        
        proof_sources_added = 0
        total_freshness = 0.0
        recent_mentions = 0
        
        for result in quality_results:
            try:
                proof_data = self.create_enhanced_proof_record(poi_id, result, poi_creation_date)
                if proof_data:
                    # Check if we already have this source
                    existing = self.check_existing_proof_source(poi_id, proof_data['url'])
                    if not existing:
                        proof_id = self.db.insert_proof_sources([proof_data])
                        if proof_id:
                            proof_sources_added += 1
                            authority = proof_data['authority_score']
                            # Extract freshness from snippet
                            freshness = freshness_score  # Use calculated value
                            relevance = final_relevance
                            
                            total_freshness += freshness
                            if freshness > 0.7:  # Recent content
                                recent_mentions += 1
                            
                            logger.info(f"  ✅ Added {authority} proof: {proof_data['domain']} (fresh: {freshness:.2f}, relevance: {relevance:.2f})")
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error processing result: {e}")
                continue
        
        # Analyze temporal patterns
        avg_freshness = total_freshness / proof_sources_added if proof_sources_added > 0 else 0
        
        temporal_signal = ""
        if recent_mentions >= 3:
            temporal_signal = " 🔥 HOT - Multiple recent mentions!"
        elif recent_mentions >= 1 and poi_age_days and poi_age_days < 90:
            temporal_signal = " ⭐ NEW HOTSPOT - Recent POI with fresh buzz!"
        
        logger.info(f"📈 Added {proof_sources_added} proof sources for {poi_name}{temporal_signal}")
        if proof_sources_added > 0:
            logger.info(f"  🕒 Avg freshness: {avg_freshness:.2f}, Recent mentions: {recent_mentions}")
        
        return proof_sources_added
    
    def check_existing_proof_source(self, poi_id: str, url: str) -> bool:
        """Check if proof source already exists."""
        try:
            result = self.db.client.table('proof_sources')\
                .select('id')\
                .eq('poi_id', poi_id)\
                .eq('url', url)\
                .execute()
            
            return len(result.data) > 0
        except Exception as e:
            logger.error(f"Error checking existing proof source: {e}")
            return False
    
    def enhanced_bulk_scan(self, city: str = 'Montreal', limit: int = None) -> Dict[str, int]:
        """Enhanced bulk scan with better results tracking."""
        logger.info(f"🚀 Starting enhanced bulk proof scan for {city}")
        
        # Get POIs from database
        pois = self.db.get_pois_for_city(city, limit or 100)
        
        if not pois:
            logger.warning(f"No POIs found for {city}")
            return {'error': 'no_pois'}
        
        results = {
            'pois_scanned': 0,
            'proof_sources_added': 0,
            'high_authority_sources': 0,
            'medium_authority_sources': 0,
            'errors': 0,
            'improved_pois': 0  # POIs that got significantly better proof
        }
        
        for poi in pois:
            try:
                # Get current proof count
                current_proof_count = len(self.db.get_proof_sources_for_poi(poi['id']))
                
                # Enhanced scan
                new_proof_added = self.enhanced_scan_poi(poi)
                
                results['pois_scanned'] += 1
                results['proof_sources_added'] += new_proof_added
                
                # Check if significantly improved (3+ new sources)
                if new_proof_added >= 3:
                    results['improved_pois'] += 1
                
                # Count authority levels (simplified for now)
                if new_proof_added > 0:
                    # Get the new proof sources to count authority levels
                    new_proofs = self.get_recent_proof_sources(poi['id'], new_proof_added)
                    for proof in new_proofs:
                        if proof.get('authority_score') == 'High':
                            results['high_authority_sources'] += 1
                        elif proof.get('authority_score') == 'Medium':
                            results['medium_authority_sources'] += 1
                
                # Rate limiting between POIs
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error scanning POI {poi.get('name', 'Unknown')}: {e}")
                results['errors'] += 1
                continue
        
        # Final results
        logger.info(f"🎯 Enhanced scan complete!")
        logger.info(f"  📊 POIs scanned: {results['pois_scanned']}")
        logger.info(f"  📈 Proof sources added: {results['proof_sources_added']}")
        logger.info(f"  👑 High authority: {results['high_authority_sources']}")
        logger.info(f"  🏆 Medium authority: {results['medium_authority_sources']}")
        logger.info(f"  ⚡ Significantly improved POIs: {results['improved_pois']}")
        logger.info(f"  ❌ Errors: {results['errors']}")
        
        return results
    
    def get_recent_proof_sources(self, poi_id: str, limit: int) -> List[Dict[str, Any]]:
        """Get the most recently added proof sources for a POI."""
        try:
            result = self.db.client.table('proof_sources')\
                .select('*')\
                .eq('poi_id', poi_id)\
                .order('created_at', desc=True)\
                .limit(limit)\
                .execute()
            
            return result.data or []
        except Exception as e:
            logger.error(f"Error getting recent proof sources: {e}")
            return []

def main():
    """CLI interface for enhanced proof scanning."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Social Proof Scanner - Step 1')
    parser.add_argument('--city', default='Montreal', help='City to scan')
    parser.add_argument('--poi-name', help='Specific POI to scan')
    parser.add_argument('--bulk', action='store_true', help='Enhanced bulk scan')
    parser.add_argument('--limit', type=int, default=10, help='Limit POIs to scan (for testing)')
    parser.add_argument('--test', action='store_true', help='Test mode - scan only 3 POIs')
    
    args = parser.parse_args()
    
    scanner = EnhancedProofScanner()
    
    try:
        if args.test:
            logger.info("🧪 TEST MODE: Scanning only 3 POIs")
            results = scanner.enhanced_bulk_scan(args.city, limit=3)
            print(f"\n✅ Test Results: {results}")
        elif args.bulk:
            results = scanner.enhanced_bulk_scan(args.city, args.limit)
            print(f"\n📊 Enhanced Scan Results:")
            for key, value in results.items():
                print(f"  {key}: {value}")
        elif args.poi_name:
            # Scan specific POI
            pois = scanner.db.get_pois_by_name(args.poi_name, args.city)
            if pois:
                poi = pois[0]
                proof_added = scanner.enhanced_scan_poi(poi)
                print(f"\n✅ Added {proof_added} proof sources for {args.poi_name}")
            else:
                print(f"❌ POI '{args.poi_name}' not found in {args.city}")
        else:
            print("Use --bulk, --poi-name, or --test")
    
    except Exception as e:
        logger.error(f"Enhanced proof scanning failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()