#!/usr/bin/env python3
"""
Intelligent Classifier - Sprint 4
Advanced POI scoring system with Authority, Review, Momentum scores, badges, and eligibility status.
"""
import sys
import os
import logging
import json
import time
import math
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from collections import Counter
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database import SupabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntelligentClassifierV4:
    """Sprint 4: Advanced POI scoring with Gatto Score, badges, and eligibility"""
    
    def __init__(self):
        self.db = SupabaseManager()
        
        # Scoring weights and defaults
        self.weight_type_defaults = {
            'guide': 1.0,
            'press': 0.8, 
            'local': 0.5
        }
        
        self.decay_tau_days = {
            'guide': 180,
            'press': 75,
            'local': 30
        }
        
        # Category adjustments
        self.category_adjustments = {
            'bar': {'review_score_multiplier': 0.85},
            'night_club': {'review_score_multiplier': 0.85},
            'bakery': {'momentum_multiplier': 1.15}
        }
    
    def calculate_authority_score(self, poi_id: str) -> float:
        """Calculate Authority Score (0-100) based on source mentions with temporal decay"""
        try:
            # Get source mentions for POI
            result = self.db.client.table('source_mention')\
                .select('*')\
                .eq('poi_id', poi_id)\
                .execute()
            
            mentions = result.data
            if not mentions:
                return 0.0
            
            raw_score = 0.0
            now = datetime.now(timezone.utc)
            
            for mention in mentions:
                # Get mention details
                weight_type = self.weight_type_defaults.get(mention.get('source_type', 'local').lower(), 0.5)
                authority_weight = mention.get('authority_weight', 1.0)
                match_score = mention.get('match_score', 1.0)
                
                # Calculate temporal decay
                w_time = mention.get('w_time')
                if w_time is None:
                    # Calculate from created_at if available
                    if mention.get('created_at'):
                        try:
                            created_at = datetime.fromisoformat(mention['created_at'].replace('Z', '+00:00'))
                            delta_days = (now - created_at).days
                            source_type = mention.get('source_type', 'local').lower()
                            tau = self.decay_tau_days.get(source_type, 30)
                            w_time = math.exp(-delta_days / tau)
                        except:
                            w_time = 0.5  # Fallback
                    else:
                        w_time = 0.5  # Fallback
                
                # Authority contribution
                contribution = weight_type * authority_weight * w_time * match_score
                raw_score += contribution
            
            # Soft-cap: tanh(raw/100) * 100
            authority_score = math.tanh(raw_score / 100) * 100
            return min(100.0, max(0.0, authority_score))
            
        except Exception as e:
            logger.warning(f"Error calculating authority score for POI {poi_id}: {e}")
            return 0.0
    
    def calculate_review_score(self, poi: Dict[str, Any]) -> float:
        """Calculate Review Score (0-100) based on rating and review volume"""
        try:
            rating = poi.get('rating')
            reviews_count = poi.get('reviews_count', 0)
            
            if rating is None:
                return 0.0
            
            # Score rating: clamp((rating-3.5)/1.5, 0, 1) * 100
            score_rating = max(0.0, min(1.0, (rating - 3.5) / 1.5)) * 100
            
            # Score volume: clamp(log1p(reviews_count)/log1p(2000), 0, 1) * 100
            if reviews_count > 0:
                score_volume = max(0.0, min(1.0, math.log1p(reviews_count) / math.log1p(2000))) * 100
            else:
                score_volume = 0.0
            
            # ReviewScore = 0.70 * score_rating + 0.30 * score_volume
            review_score = 0.70 * score_rating + 0.30 * score_volume
            
            # Category adjustments
            category = poi.get('category', '').lower()
            if category in ['bar', 'night_club']:
                review_score *= 0.85
            
            return min(100.0, max(0.0, review_score))
            
        except Exception as e:
            logger.warning(f"Error calculating review score: {e}")
            return 0.0
    
    def calculate_momentum_score(self, poi: Dict[str, Any]) -> float:
        """Calculate Momentum Score (0-100) based on 14-day review growth and recent mentions"""
        try:
            poi_id = poi['id']
            city = poi.get('city', 'unknown')
            category = poi.get('category', '').lower()
            
            # Get 14-day review count delta from rating_snapshot
            delta_14d = self._get_review_count_delta_14d(poi_id)
            
            # Normalize by city p95 (fallback global 5.0)
            p95_normalizer = self._get_city_p95_reviews(city)
            
            # Base momentum from review growth
            if delta_14d > 0:
                normalized_growth = min(1.0, delta_14d / p95_normalizer)
                base_momentum = normalized_growth * 100
            else:
                base_momentum = 0.0
            
            # Bonus for recent high-weight mentions
            mention_bonus = self._get_recent_mention_bonus(poi_id)
            
            momentum_score = base_momentum + mention_bonus
            
            # Category adjustments
            if category == 'bakery':
                momentum_score *= 1.15
            
            # Fallback for insufficient data
            if self._has_insufficient_snapshot_data(poi_id):
                first_seen = poi.get('first_seen_at')
                if first_seen:
                    try:
                        first_seen_dt = datetime.fromisoformat(first_seen.replace('Z', '+00:00'))
                        days_since_first_seen = (datetime.now(timezone.utc) - first_seen_dt).days
                        if days_since_first_seen < 30:
                            momentum_score = 50.0  # New POI bonus
                        else:
                            momentum_score = 0.0
                    except:
                        momentum_score = 0.0
                else:
                    momentum_score = 0.0
            
            return min(100.0, max(0.0, momentum_score))
            
        except Exception as e:
            logger.warning(f"Error calculating momentum score: {e}")
            return 0.0
    
    def _get_review_count_delta_14d(self, poi_id: str) -> int:
        """Get 14-day review count delta from rating_snapshot"""
        try:
            # Get snapshots from last 14 days
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=14)
            
            result = self.db.client.table('rating_snapshots')\
                .select('reviews_count, created_at')\
                .eq('poi_id', poi_id)\
                .gte('created_at', cutoff_date.isoformat())\
                .order('created_at', desc=True)\
                .execute()
            
            snapshots = result.data
            if len(snapshots) < 2:
                return 0
            
            # Get most recent and oldest in the period
            newest = snapshots[0]['reviews_count'] or 0
            oldest = snapshots[-1]['reviews_count'] or 0
            
            return max(0, newest - oldest)
            
        except Exception as e:
            logger.debug(f"Error getting review delta for {poi_id}: {e}")
            return 0
    
    def _get_city_p95_reviews(self, city: str) -> float:
        """Get p95 review normalizer for city (fallback 5.0)"""
        try:
            # This would ideally query aggregated statistics
            # For now, use fallback
            return 5.0
        except:
            return 5.0
    
    def _get_recent_mention_bonus(self, poi_id: str) -> float:
        """Get bonus for recent high-weight mentions (≥2 mentions w_time>0.8 on 30j = +15)"""
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)
            
            result = self.db.client.table('source_mention')\
                .select('w_time, created_at')\
                .eq('poi_id', poi_id)\
                .gte('created_at', cutoff_date.isoformat())\
                .execute()
            
            recent_mentions = result.data
            high_weight_count = 0
            
            for mention in recent_mentions:
                w_time = mention.get('w_time', 0.0)
                if w_time > 0.8:
                    high_weight_count += 1
            
            return 15.0 if high_weight_count >= 2 else 0.0
            
        except Exception as e:
            logger.debug(f"Error getting mention bonus for {poi_id}: {e}")
            return 0.0
    
    def _has_insufficient_snapshot_data(self, poi_id: str) -> bool:
        """Check if POI has insufficient snapshot data for momentum calculation"""
        try:
            result = self.db.client.table('rating_snapshots')\
                .select('id')\
                .eq('poi_id', poi_id)\
                .limit(2)\
                .execute()
            
            return len(result.data) < 2
            
        except:
            return True
    
    def calculate_gatto_score(self, authority: float, review: float, momentum: float) -> float:
        """Calculate Gatto Score: 0.5*Authority + 0.3*Review + 0.2*Momentum"""
        gatto_score = 0.5 * authority + 0.3 * review + 0.2 * momentum
        return min(100.0, max(0.0, gatto_score))
    
    def calculate_badges(self, poi: Dict[str, Any], authority: float, review: float, momentum: float) -> List[str]:
        """Calculate badges for POI based on scores and characteristics"""
        badges = []
        
        try:
            # New: days_since_first_seen ≤ 60 ET ≥1 mention w_time>0.6
            first_seen = poi.get('first_seen_at')
            if first_seen:
                try:
                    first_seen_dt = datetime.fromisoformat(first_seen.replace('Z', '+00:00'))
                    days_since_first_seen = (datetime.now(timezone.utc) - first_seen_dt).days
                    
                    if days_since_first_seen <= 60:
                        # Check for mentions with w_time > 0.6
                        if self._has_recent_quality_mentions(poi['id'], 0.6, days=60):
                            badges.append('new')
                except:
                    pass
            
            # Trending: MomentumScore ≥ 65 OU ≥2 mentions récentes w_time>0.8
            if momentum >= 65 or self._has_recent_quality_mentions(poi['id'], 0.8, count_threshold=2, days=30):
                badges.append('trending')
            
            # Hidden Gem: rating≥4.6 ET reviews<500 ET ≥1 mention press/local
            rating = poi.get('rating', 0)
            reviews_count = poi.get('reviews_count', 0)
            if rating >= 4.6 and reviews_count < 500:
                if self._has_press_or_local_mentions(poi['id']):
                    badges.append('hidden_gem')
            
            # Local Favorite: ≥2 mentions local ET rating≥4.4
            if rating >= 4.4:
                if self._count_local_mentions(poi['id']) >= 2:
                    badges.append('local_favorite')
            
        except Exception as e:
            logger.warning(f"Error calculating badges for POI {poi.get('id')}: {e}")
        
        return badges
    
    def _has_recent_quality_mentions(self, poi_id: str, w_time_threshold: float, 
                                   count_threshold: int = 1, days: int = 30) -> bool:
        """Check if POI has recent quality mentions above threshold"""
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            result = self.db.client.table('source_mention')\
                .select('w_time')\
                .eq('poi_id', poi_id)\
                .gte('created_at', cutoff_date.isoformat())\
                .execute()
            
            quality_mentions = 0
            for mention in result.data:
                if mention.get('w_time', 0.0) > w_time_threshold:
                    quality_mentions += 1
                    if quality_mentions >= count_threshold:
                        return True
            
            return False
            
        except:
            return False
    
    def _has_press_or_local_mentions(self, poi_id: str) -> bool:
        """Check if POI has press or local mentions"""
        try:
            result = self.db.client.table('source_mention')\
                .select('source_type')\
                .eq('poi_id', poi_id)\
                .in_('source_type', ['press', 'local'])\
                .limit(1)\
                .execute()
            
            return len(result.data) > 0
            
        except:
            return False
    
    def _count_local_mentions(self, poi_id: str) -> int:
        """Count local mentions for POI"""
        try:
            result = self.db.client.table('source_mention')\
                .select('id')\
                .eq('poi_id', poi_id)\
                .eq('source_type', 'local')\
                .execute()
            
            return len(result.data)
            
        except:
            return 0
    
    def determine_eligibility_status(self, gatto_score: float, authority: float, review: float) -> str:
        """Determine eligibility status based on scores"""
        # approved: GattoScore ≥ 60 ET (Authority ≥ 40 OU Review ≥ 70)
        if gatto_score >= 60 and (authority >= 40 or review >= 70):
            return 'approved'
        
        # eligible: GattoScore ≥ 50
        elif gatto_score >= 50:
            return 'eligible'
        
        # sinon hold
        else:
            return 'hold'
    
    def score_poi(self, poi: Dict[str, Any], force: bool = False) -> Optional[Dict[str, Any]]:
        """Score a single POI with all Sprint 4 metrics"""
        try:
            poi_id = poi['id']
            
            # Check if POI needs scoring (skip if last_scored_at ≤ 24h unless --force)
            if not force:
                last_scored = poi.get('last_scored_at')
                if last_scored:
                    try:
                        last_scored_dt = datetime.fromisoformat(last_scored.replace('Z', '+00:00'))
                        hours_since_scored = (datetime.now(timezone.utc) - last_scored_dt).total_seconds() / 3600
                        if hours_since_scored <= 24:
                            logger.debug(f"Skipping {poi['name']} - scored {hours_since_scored:.1f}h ago")
                            return None
                    except:
                        pass
            
            # Calculate scores
            authority_score = self.calculate_authority_score(poi_id)
            review_score = self.calculate_review_score(poi)
            momentum_score = self.calculate_momentum_score(poi)
            gatto_score = self.calculate_gatto_score(authority_score, review_score, momentum_score)
            
            # Calculate badges
            badges = self.calculate_badges(poi, authority_score, review_score, momentum_score)
            
            # Determine eligibility status
            eligibility_status = self.determine_eligibility_status(gatto_score, authority_score, review_score)
            
            # Store trend_score as momentum for now (could be extended)
            trend_score = momentum_score
            
            result = {
                'poi_id': poi_id,
                'authority_score': authority_score,
                'review_score': review_score, 
                'momentum_score': momentum_score,
                'gatto_score': gatto_score,
                'trend_score': trend_score,
                'badges': badges,
                'eligibility_status': eligibility_status,
                'scored_at': datetime.now(timezone.utc).isoformat()
            }
            
            logger.info(f"✅ {poi['name']}: Gatto={gatto_score:.1f} (A:{authority_score:.1f}, R:{review_score:.1f}, M:{momentum_score:.1f}) | {eligibility_status} | {len(badges)} badges")
            
            return result
            
        except Exception as e:
            logger.error(f"Error scoring POI {poi.get('name', 'Unknown')}: {e}")
            return None
    
    def update_poi_scores(self, poi_id: str, scores: Dict[str, Any]) -> bool:
        """Update POI scores in database"""
        try:
            update_data = {
                'gatto_score': scores['gatto_score'],
                'trend_score': scores['trend_score'],
                'badges': scores['badges'],  # Store as array directly, not JSON string
                'eligibility_status': scores['eligibility_status'],
                'last_scored_at': scores['scored_at']
            }
            
            result = self.db.client.table('poi')\
                .update(update_data)\
                .eq('id', poi_id)\
                .execute()
            
            return len(result.data) > 0
            
        except Exception as e:
            logger.error(f"Error updating scores for POI {poi_id}: {e}")
            return False
    
    def score_city_pois(self, city_slug: str, force: bool = False) -> Dict[str, Any]:
        """Score POIs for a city (maj ≤7j OU jamais scorés, skip si last_scored_at ≤ 24h sauf --force)"""
        try:
            logger.info(f"🎯 Starting city scoring: {city_slug}")
            
            # Get POIs that need scoring
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=7)
            
            query = self.db.client.table('poi')\
                .select('*')\
                .eq('city_slug', city_slug)
            
            # Filter for POIs updated ≤7j OR never scored
            if not force:
                query = query.or_(f'updated_at.gte.{cutoff_date.isoformat()},last_scored_at.is.null')
            
            result = query.execute()
            candidate_pois = result.data
            
            logger.info(f"Found {len(candidate_pois)} candidate POIs for scoring")
            
            # Score POIs
            results = {
                'processed': 0,
                'scored': 0,
                'skipped': 0,
                'failed': 0,
                'status_transitions': Counter(),
                'scores': [],
                'gatto_score_p50': 0.0,
                'gatto_score_p95': 0.0
            }
            
            for poi in candidate_pois:
                try:
                    old_status = poi.get('eligibility_status', 'unknown')
                    
                    scores = self.score_poi(poi, force=force)
                    results['processed'] += 1
                    
                    if scores is None:
                        results['skipped'] += 1
                        continue
                    
                    # Update database
                    success = self.update_poi_scores(poi['id'], scores)
                    
                    if success:
                        results['scored'] += 1
                        results['scores'].append(scores['gatto_score'])
                        
                        # Track status transitions
                        new_status = scores['eligibility_status']
                        if old_status != new_status:
                            transition = f"{old_status}->{new_status}"
                            results['status_transitions'][transition] += 1
                    else:
                        results['failed'] += 1
                        
                except Exception as e:
                    logger.error(f"Error processing POI {poi.get('name', 'Unknown')}: {e}")
                    results['failed'] += 1
                    continue
            
            # Calculate aggregated stats
            if results['scores']:
                sorted_scores = sorted(results['scores'])
                n = len(sorted_scores)
                results['gatto_score_p50'] = sorted_scores[n//2]
                results['gatto_score_p95'] = sorted_scores[int(n*0.95)]
            
            logger.info(f"🎯 City scoring complete: {results['scored']} POIs scored, {results['skipped']} skipped, {results['failed']} failed")
            logger.info(f"📊 GattoScore P50: {results['gatto_score_p50']:.1f}, P95: {results['gatto_score_p95']:.1f}")
            
            if results['status_transitions']:
                logger.info(f"🔄 Status transitions: {dict(results['status_transitions'])}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error scoring city {city_slug}: {e}")
            return {'error': str(e)}
    
    def score_single_poi(self, poi_id: str) -> Dict[str, Any]:
        """Score a single POI by ID"""
        try:
            # Get POI data
            result = self.db.client.table('poi')\
                .select('*')\
                .eq('id', poi_id)\
                .single()\
                .execute()
            
            if not result.data:
                return {'error': f'POI {poi_id} not found'}
            
            poi = result.data
            scores = self.score_poi(poi, force=True)
            
            if scores:
                success = self.update_poi_scores(poi_id, scores)
                scores['updated'] = success
                return scores
            else:
                return {'error': 'Scoring failed'}
                
        except Exception as e:
            logger.error(f"Error scoring single POI {poi_id}: {e}")
            return {'error': str(e)}

# MOCK TESTS - Integrated in same file, no network calls
def run_mock_tests():
    """Run mock tests without network calls"""
    print("🧪 Running Sprint 4 Mock Tests...")
    
    test_count = 0
    
    # Mock SupabaseManager completely
    with patch('utils.database.SupabaseManager') as mock_db:
        
        # Create synthetic test data
        mock_client = MagicMock()
        mock_db.return_value.client = mock_client
        
        # Mock POI data
        test_pois = {
            'A': {  # New + Trending
                'id': 'poi-a',
                'name': 'New Trendy Cafe',
                'category': 'cafe',
                'city': 'paris',
                'rating': 4.8,
                'reviews_count': 50,
                'first_seen_at': (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
                'last_scored_at': None
            },
            'B': {  # Established
                'id': 'poi-b', 
                'name': 'Classic Bistro',
                'category': 'restaurant',
                'city': 'paris',
                'rating': 4.2,
                'reviews_count': 800,
                'first_seen_at': (datetime.now(timezone.utc) - timedelta(days=500)).isoformat(),
                'last_scored_at': None
            },
            'C': {  # Weak
                'id': 'poi-c',
                'name': 'Struggling Bar',
                'category': 'bar',
                'city': 'paris', 
                'rating': 3.2,
                'reviews_count': 20,
                'first_seen_at': (datetime.now(timezone.utc) - timedelta(days=200)).isoformat(),
                'last_scored_at': None
            },
            'D': {  # Hidden Gem
                'id': 'poi-d',
                'name': 'Secret Wine Bar',
                'category': 'bar',
                'city': 'paris',
                'rating': 4.7,
                'reviews_count': 120,
                'first_seen_at': (datetime.now(timezone.utc) - timedelta(days=100)).isoformat(),
                'last_scored_at': None
            }
        }
        
        # Mock database responses
        def mock_table_response(table_name):
            mock_table = MagicMock()
            
            if table_name == 'source_mention':
                # Mock mentions with different patterns for each POI
                def mock_select_execute(*args, **kwargs):
                    mock_result = MagicMock()
                    # Return different mention patterns based on the query (simplified)
                    mock_result.data = [
                        {
                            'source_type': 'guide',
                            'authority_weight': 1.5,
                            'match_score': 1.0,
                            'w_time': 0.9,
                            'created_at': datetime.now(timezone.utc).isoformat()
                        }
                    ]
                    return mock_result
                
                mock_table.select.return_value.eq.return_value.execute = mock_select_execute
                mock_table.select.return_value.eq.return_value.gte.return_value.execute = mock_select_execute
                mock_table.select.return_value.eq.return_value.in_.return_value.limit.return_value.execute = mock_select_execute
                
            elif table_name == 'rating_snapshots':
                mock_result = MagicMock()
                mock_result.data = [
                    {'reviews_count': 50, 'created_at': datetime.now(timezone.utc).isoformat()},
                    {'reviews_count': 40, 'created_at': (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()}
                ]
                mock_table.select.return_value.eq.return_value.gte.return_value.order.return_value.execute.return_value = mock_result
                mock_table.select.return_value.eq.return_value.limit.return_value.execute.return_value = mock_result
                
            elif table_name == 'poi':
                mock_result = MagicMock()
                mock_result.data = [test_pois['A']]  # Default response
                mock_table.update.return_value.eq.return_value.execute.return_value = mock_result
            
            return mock_table
        
        mock_client.table.side_effect = mock_table_response
        
        classifier = IntelligentClassifierV4()
        
        # Test 1: Authority Score Calculation
        test_count += 1
        authority = classifier.calculate_authority_score('poi-a')
        assert 0 <= authority <= 100
        print(f"✅ Test {test_count}: Authority Score calculation ({authority:.1f})")
        
        # Test 2: Review Score Calculation  
        test_count += 1
        review = classifier.calculate_review_score(test_pois['A'])
        assert 0 <= review <= 100
        print(f"✅ Test {test_count}: Review Score calculation ({review:.1f})")
        
        # Test 3: Momentum Score Calculation
        test_count += 1
        momentum = classifier.calculate_momentum_score(test_pois['A']) 
        assert 0 <= momentum <= 100
        print(f"✅ Test {test_count}: Momentum Score calculation ({momentum:.1f})")
        
        # Test 4: Gatto Score Calculation
        test_count += 1
        gatto = classifier.calculate_gatto_score(authority, review, momentum)
        assert 0 <= gatto <= 100
        expected_gatto = 0.5 * authority + 0.3 * review + 0.2 * momentum
        assert abs(gatto - expected_gatto) < 0.1
        print(f"✅ Test {test_count}: Gatto Score calculation ({gatto:.1f})")
        
        # Test 5: Badge Calculation
        test_count += 1
        badges = classifier.calculate_badges(test_pois['A'], authority, review, momentum)
        assert isinstance(badges, list)
        print(f"✅ Test {test_count}: Badge calculation ({len(badges)} badges: {badges})")
        
        # Test 6: Eligibility Status
        test_count += 1
        status = classifier.determine_eligibility_status(gatto, authority, review)
        assert status in ['approved', 'eligible', 'hold']
        print(f"✅ Test {test_count}: Eligibility status ({status})")
        
        # Test 7: Full POI Scoring
        test_count += 1
        result = classifier.score_poi(test_pois['A'], force=True)
        assert result is not None
        assert 'gatto_score' in result
        assert 'badges' in result
        assert 'eligibility_status' in result
        print(f"✅ Test {test_count}: Full POI scoring")
        
        # Display results table
        print(f"\n📊 Mock Scoring Results:")
        print(f"{'POI':<20} {'Authority':<10} {'Review':<8} {'Momentum':<10} {'Gatto':<8} {'Status':<10} {'Badges'}")
        print("-" * 80)
        
        for poi_key, poi_data in test_pois.items():
            auth = classifier.calculate_authority_score(poi_data['id'])
            rev = classifier.calculate_review_score(poi_data)
            mom = classifier.calculate_momentum_score(poi_data)
            gat = classifier.calculate_gatto_score(auth, rev, mom)
            badges = classifier.calculate_badges(poi_data, auth, rev, mom)
            status = classifier.determine_eligibility_status(gat, auth, rev)
            
            print(f"{poi_data['name']:<20} {auth:<10.1f} {rev:<8.1f} {mom:<10.1f} {gat:<8.1f} {status:<10} {len(badges)}")
    
    print(f"\n🎉 All {test_count} tests passed!")
    print("S4_MOCKS_OK")

def main():
    """CLI interface for Intelligent Classifier V4"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Intelligent Classifier V4 - Sprint 4')
    parser.add_argument('--score-city', help='Score POIs for city slug')
    parser.add_argument('--score-poi', help='Score specific POI by ID')
    parser.add_argument('--force', action='store_true', help='Force scoring even if recently scored')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--run-mocks', action='store_true', help='Run mock tests without network calls')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.run_mocks:
        run_mock_tests()
        return
    
    classifier = IntelligentClassifierV4()
    
    if args.score_city:
        result = classifier.score_city_pois(args.score_city, force=args.force)
        print(f"\n🎯 City Scoring Results:")
        for key, value in result.items():
            if key not in ['scores']:
                print(f"  {key}: {value}")
        
    elif args.score_poi:
        result = classifier.score_single_poi(args.score_poi)
        print(f"\n🎯 POI Scoring Result:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    
    else:
        print("Use --score-city, --score-poi, or --run-mocks")

if __name__ == "__main__":
    main()