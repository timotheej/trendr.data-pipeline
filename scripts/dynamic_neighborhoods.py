#!/usr/bin/env python3
"""
Dynamic Neighborhood Calculator - Replace static neighborhood tendencies
Calculates neighborhood mood distributions based on actual POIs within each neighborhood.
This replaces hardcoded percentages with real data-driven calculations.
"""
import sys
import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import Counter

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database import SupabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DynamicNeighborhoodCalculator:
    """Calculate neighborhood characteristics dynamically from POI data."""
    
    def __init__(self):
        self.db = SupabaseManager()
        
        # Default fallback distribution when no data available
        self.default_distribution = {
            'chill': 0.33,
            'trendy': 0.34, 
            'hidden_gem': 0.33
        }
    
    def calculate_neighborhood_mood_distribution(self, neighborhood_id: str) -> Optional[Dict[str, float]]:
        """Calculate mood distribution for a neighborhood based on its POIs."""
        try:
            # Get all POIs in the neighborhood
            pois = self.db.get_pois_for_neighborhood(neighborhood_id)
            
            if not pois:
                logger.warning(f"No POIs found for neighborhood {neighborhood_id}")
                return self.default_distribution
            
            # Count mood tags (use both current mood_tag and intelligent classifications)
            mood_counts = {'chill': 0, 'trendy': 0, 'hidden_gem': 0}
            total_pois = len(pois)
            
            for poi in pois:
                mood = poi.get('mood_tag', 'trendy')  # Default fallback
                
                # Normalize mood names to our standard 3
                if mood and isinstance(mood, str):
                    mood_lower = mood.lower()
                    if 'chill' in mood_lower or 'calm' in mood_lower or 'relax' in mood_lower:
                        mood_counts['chill'] += 1
                    elif 'trendy' in mood_lower or 'hip' in mood_lower or 'popular' in mood_lower:
                        mood_counts['trendy'] += 1
                    elif 'hidden' in mood_lower or 'gem' in mood_lower or 'secret' in mood_lower:
                        mood_counts['hidden_gem'] += 1
                    else:
                        # Default fallback for unclear moods
                        mood_counts['trendy'] += 1
                else:
                    # No mood tag - default to trendy
                    mood_counts['trendy'] += 1
            
            # Calculate percentages
            distribution = {
                mood: count / total_pois for mood, count in mood_counts.items()
            }
            
            logger.info(f"Calculated distribution for neighborhood {neighborhood_id}:")
            logger.info(f"  Total POIs: {total_pois}")
            logger.info(f"  Distribution: {distribution}")
            
            return distribution
            
        except Exception as e:
            logger.error(f"Error calculating mood distribution for neighborhood {neighborhood_id}: {e}")
            return self.default_distribution
    
    def calculate_neighborhood_contextual_profile(self, neighborhood_id: str) -> Dict[str, Any]:
        """Calculate comprehensive neighborhood profile including contextual tags."""
        try:
            pois = self.db.get_pois_for_neighborhood(neighborhood_id)
            
            if not pois:
                return {
                    'mood_distribution': self.default_distribution,
                    'dominant_categories': [],
                    'contextual_characteristics': [],
                    'poi_count': 0,
                    'calculated_at': datetime.utcnow().isoformat()
                }
            
            # Basic mood distribution
            mood_distribution = self.calculate_neighborhood_mood_distribution(neighborhood_id)
            
            # Category analysis
            categories = [poi.get('category', 'unknown') for poi in pois]
            category_counts = Counter(categories)
            dominant_categories = [cat for cat, count in category_counts.most_common(3)]
            
            # Extract contextual characteristics from POI names and types
            contextual_chars = self.extract_neighborhood_characteristics(pois)
            
            return {
                'mood_distribution': mood_distribution,
                'dominant_categories': dominant_categories,
                'category_distribution': dict(category_counts),
                'contextual_characteristics': contextual_chars,
                'poi_count': len(pois),
                'calculated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating neighborhood profile for {neighborhood_id}: {e}")
            return {
                'mood_distribution': self.default_distribution,
                'error': str(e),
                'calculated_at': datetime.utcnow().isoformat()
            }
    
    def extract_neighborhood_characteristics(self, pois: List[Dict[str, Any]]) -> List[str]:
        """Extract contextual characteristics from POI data."""
        characteristics = []
        
        # Analyze POI names for patterns
        names = [poi.get('name', '').lower() for poi in pois]
        all_text = ' '.join(names)
        
        # Characteristic indicators
        if any('vintage' in name or 'antique' in name or 'classic' in name for name in names):
            characteristics.append('vintage-focused')
        
        if any('art' in name or 'gallery' in name or 'studio' in name for name in names):
            characteristics.append('artistic')
        
        if any('market' in name or 'local' in name for name in names):
            characteristics.append('local-market')
        
        if any('bistro' in name or 'brasserie' in name for name in names):
            characteristics.append('french-influence')
        
        # Category-based characteristics
        categories = [poi.get('category', '') for poi in pois]
        category_counts = Counter(categories)
        
        if category_counts.get('cafe', 0) > len(pois) * 0.4:
            characteristics.append('coffee-culture')
        
        if category_counts.get('bar', 0) > len(pois) * 0.3:
            characteristics.append('nightlife-focused')
        
        if category_counts.get('restaurant', 0) > len(pois) * 0.5:
            characteristics.append('dining-destination')
        
        return characteristics[:5]  # Top 5 characteristics max
    
    def update_neighborhood_distribution(self, neighborhood_id: str) -> bool:
        """Update a single neighborhood's mood distribution in the database."""
        try:
            new_distribution = self.calculate_neighborhood_mood_distribution(neighborhood_id)
            
            if new_distribution:
                success = self.db.update_neighborhood_mood_distribution(neighborhood_id, new_distribution)
                if success:
                    logger.info(f"✅ Updated neighborhood {neighborhood_id} with distribution: {new_distribution}")
                return success
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating neighborhood {neighborhood_id}: {e}")
            return False
    
    def update_all_neighborhood_distributions(self) -> Dict[str, Any]:
        """Update mood distributions for all neighborhoods."""
        logger.info("🏘️ Starting dynamic neighborhood calculations")
        
        try:
            # Get all neighborhoods
            neighborhoods = self.db.client.table('neighborhoods')\
                .select('id, name, city')\
                .execute()
            
            results = {
                'neighborhoods_processed': 0,
                'successful_updates': 0,
                'failed_updates': 0,
                'neighborhood_details': []
            }
            
            for neighborhood in neighborhoods.data:
                neighborhood_id = neighborhood['id']
                neighborhood_name = neighborhood['name']
                
                try:
                    # Calculate comprehensive profile
                    profile = self.calculate_neighborhood_contextual_profile(neighborhood_id)
                    
                    # Update mood distribution in database
                    success = self.db.update_neighborhood_mood_distribution(
                        neighborhood_id, 
                        profile['mood_distribution']
                    )
                    
                    if success:
                        results['successful_updates'] += 1
                        logger.info(f"✅ {neighborhood_name}: Updated with {profile['poi_count']} POIs")
                        
                        # Store detailed results
                        results['neighborhood_details'].append({
                            'name': neighborhood_name,
                            'poi_count': profile['poi_count'],
                            'mood_distribution': profile['mood_distribution'],
                            'dominant_categories': profile['dominant_categories'],
                            'characteristics': profile.get('contextual_characteristics', []),
                            'success': True
                        })
                    else:
                        results['failed_updates'] += 1
                        results['neighborhood_details'].append({
                            'name': neighborhood_name,
                            'success': False,
                            'error': 'Database update failed'
                        })
                
                except Exception as e:
                    logger.error(f"❌ Error processing {neighborhood_name}: {e}")
                    results['failed_updates'] += 1
                    results['neighborhood_details'].append({
                        'name': neighborhood_name,
                        'success': False,
                        'error': str(e)
                    })
                
                results['neighborhoods_processed'] += 1
            
            # Final summary
            logger.info(f"🎯 Dynamic neighborhood calculation complete!")
            logger.info(f"  📊 Processed: {results['neighborhoods_processed']} neighborhoods")
            logger.info(f"  ✅ Successful: {results['successful_updates']}")
            logger.info(f"  ❌ Failed: {results['failed_updates']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in bulk neighborhood update: {e}")
            return {'error': str(e)}
    
    def compare_static_vs_dynamic(self) -> Dict[str, Any]:
        """Compare old static distributions with new dynamic ones."""
        
        # Static distributions (what we had before)
        static_distributions = {
            'Mile End': {'chill': 0.6, 'trendy': 0.3, 'hidden_gem': 0.1},
            'Plateau Mont-Royal': {'chill': 0.4, 'trendy': 0.5, 'hidden_gem': 0.1},
            'Old Montreal': {'chill': 0.3, 'trendy': 0.4, 'hidden_gem': 0.3},
            'Downtown': {'chill': 0.2, 'trendy': 0.7, 'hidden_gem': 0.1},
            'Griffintown': {'chill': 0.4, 'trendy': 0.5, 'hidden_gem': 0.1},
            'Little Italy': {'chill': 0.7, 'trendy': 0.2, 'hidden_gem': 0.1}
        }
        
        comparison = {}
        
        try:
            neighborhoods = self.db.client.table('neighborhoods')\
                .select('id, name')\
                .execute()
            
            for neighborhood in neighborhoods.data:
                name = neighborhood['name']
                neighborhood_id = neighborhood['id']
                
                # Calculate dynamic distribution
                dynamic_dist = self.calculate_neighborhood_mood_distribution(neighborhood_id)
                static_dist = static_distributions.get(name, self.default_distribution)
                
                # Calculate differences
                differences = {}
                for mood in ['chill', 'trendy', 'hidden_gem']:
                    diff = dynamic_dist[mood] - static_dist[mood]
                    differences[mood] = {
                        'static': static_dist[mood],
                        'dynamic': dynamic_dist[mood],
                        'difference': diff,
                        'change_pct': (diff / static_dist[mood] * 100) if static_dist[mood] > 0 else 0
                    }
                
                comparison[name] = differences
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error in static vs dynamic comparison: {e}")
            return {'error': str(e)}

def main():
    """CLI interface for dynamic neighborhood calculations."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Dynamic Neighborhood Calculator')
    parser.add_argument('--update-all', action='store_true', help='Update all neighborhood distributions')
    parser.add_argument('--neighborhood-id', help='Update specific neighborhood by ID')
    parser.add_argument('--neighborhood-name', help='Update specific neighborhood by name')
    parser.add_argument('--compare', action='store_true', help='Compare static vs dynamic distributions')
    parser.add_argument('--profile', help='Show detailed profile for neighborhood (by name)')
    
    args = parser.parse_args()
    
    calculator = DynamicNeighborhoodCalculator()
    
    try:
        if args.update_all:
            results = calculator.update_all_neighborhood_distributions()
            
            print(f"\n🏘️ Dynamic Neighborhood Update Results:")
            print(f"  Processed: {results.get('neighborhoods_processed', 0)}")
            print(f"  Successful: {results.get('successful_updates', 0)}")
            print(f"  Failed: {results.get('failed_updates', 0)}")
            
            # Show details for successful updates
            for detail in results.get('neighborhood_details', []):
                if detail.get('success'):
                    name = detail['name']
                    poi_count = detail['poi_count']
                    dist = detail['mood_distribution']
                    chars = ', '.join(detail.get('characteristics', []))
                    
                    print(f"\n  📍 {name} ({poi_count} POIs):")
                    print(f"    Chill: {dist['chill']:.1%}, Trendy: {dist['trendy']:.1%}, Hidden: {dist['hidden_gem']:.1%}")
                    if chars:
                        print(f"    Characteristics: {chars}")
        
        elif args.compare:
            comparison = calculator.compare_static_vs_dynamic()
            
            print(f"\n📊 Static vs Dynamic Distribution Comparison:")
            for neighborhood, diffs in comparison.items():
                print(f"\n  🏘️ {neighborhood}:")
                for mood, data in diffs.items():
                    static_val = data['static']
                    dynamic_val = data['dynamic']
                    change = data['change_pct']
                    arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
                    
                    print(f"    {mood}: {static_val:.1%} → {dynamic_val:.1%} {arrow} ({change:+.1f}%)")
        
        elif args.profile:
            # Find neighborhood by name
            neighborhoods = calculator.db.client.table('neighborhoods')\
                .select('id, name')\
                .ilike('name', f'%{args.profile}%')\
                .execute()
            
            if neighborhoods.data:
                neighborhood = neighborhoods.data[0]
                profile = calculator.calculate_neighborhood_contextual_profile(neighborhood['id'])
                
                print(f"\n🏘️ {neighborhood['name']} Profile:")
                print(f"  POI Count: {profile['poi_count']}")
                print(f"  Mood Distribution:")
                for mood, pct in profile['mood_distribution'].items():
                    print(f"    {mood}: {pct:.1%}")
                print(f"  Dominant Categories: {', '.join(profile['dominant_categories'])}")
                print(f"  Characteristics: {', '.join(profile.get('contextual_characteristics', []))}")
            else:
                print(f"❌ Neighborhood '{args.profile}' not found")
        
        elif args.neighborhood_id:
            success = calculator.update_neighborhood_distribution(args.neighborhood_id)
            print(f"Update neighborhood {args.neighborhood_id}: {'✅ Success' if success else '❌ Failed'}")
        
        else:
            print("Use --update-all, --compare, --profile, or --neighborhood-id")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Dynamic neighborhood calculation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()